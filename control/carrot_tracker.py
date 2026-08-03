"""
carrot_tracker.py

Path-following carrot (look-ahead) tracker.

Algorithm
---------
For the current segment  r0 → r1  (unit tangent ea):

  1.  Project drone position onto the segment:
        lambda_L = (pos - r0) · ea          [m along segment]

  2.  Place the carrot c metres ahead of the foot:
        rC = foot + c * ea  (= r0 + (lambda_L + c) * ea)

      Commit phase: once the remaining distance to r1 (seg_len - lambda_L)
      drops to commit_dist_m or below, the aim point is linearly blended from
      the ordinary lookahead point toward r1 (reaching r1 exactly at
      remaining=0) instead of snapping straight to it. A hard snap to r1 the
      instant commit starts is a real position jump of (c - commit_dist_m)
      metres — confirmed in practice, and worse, the carrot then sits frozen
      at r1 for the whole commit window while the drone keeps closing in,
      since r1 doesn't move. The blend removes both: it matches the lookahead
      formula's value exactly at the commit boundary and still ends exactly at
      r1 for the final approach, just without the intermediate teleport/freeze.
      If the drone overshoots r1 along-track before wp advances (remaining
      goes negative — e.g. a late gate_passed signal), the aim point keeps
      advancing forward from the foot instead of staying pinned to r1 — aiming
      at a point that's now behind can flip v_ned_ref by ~2*v_ref in one tick,
      and continues from the same offset the blend ended on, so that boundary
      is also jump-free.

  3.  Reference velocity:  v_ned_ref = v_ref * unit(rC - pos)  (NED frame)
      Reference yaw:       psi_ref   = atan2(ea_E, ea_N)

  4.  Waypoint advance is driven externally: the controller calls
      tracker.wp += 1 on gate_passed (COLLISION) only. Switching segments
      means the whole lookahead geometry (r0, r1, ea) is recomputed from
      scratch, which is a real, unavoidable jump in rC/v_ned_ref for any
      turn that isn't dead straight. wp_transition_blend_tau blends the
      output from its pre-transition value over that short window instead
      of snapping, so both carrot_pos and v_ned_ref cross the transition
      smoothly (see update()'s blend block).

Note: velocities and positions are all in the NED frame.
      NED z is positive downward → negative altitude.
"""

import numpy as np


class CarrotTracker:

    def __init__(self, param):
        self.waypoints      = [np.asarray(w, dtype=float) for w in param['waypoints']]
        self.v_ref          = float(param['v_ref'])
        self.c              = float(param['lookahead_gain']) * self.v_ref
        # Distance-to-gate at which the tracker stops advancing the lookahead
        # carrot and commits to the gate centre directly. Defaults to the
        # lookahead distance itself — inside one carrot-length of the gate,
        # the dynamic carrot would already be at-or-past it anyway.
        self.commit_dist    = float(param.get('commit_dist_m', self.c))
        # Corner-cutting: during commit, blend the aim point/yaw toward the
        # NEXT segment's tangent as the drone closes on the current gate, so
        # it arrives already leaning into the next leg instead of flying
        # dead-on-line then snap-turning right after passage. 0 = old
        # behaviour (straight at the gate centre, no anticipation). Kept
        # small by default — this bounds the maximum lean, ramped in from
        # zero at the start of commit (see update()'s blend factor s), so a
        # conservative value only nudges the approach rather than risking a
        # miss the way a discrete helper-waypoint version of this idea did
        # (that also desynced tracker.waypoints from real gate indices,
        # which active_gate_index/next_gate_ned-driven logic assumes stay
        # 1:1 — this blend never touches the waypoint list, so that failure
        # mode doesn't apply here).
        self.corner_cut_frac = float(param.get('corner_cut_frac', 0.2))
        # Below this aim-point distance, blend the commanded direction toward
        # the path tangent instead of the raw bearing-to-point (see the
        # catch-up-singularity guard in update()).
        self._CATCHUP_FLOOR_M = float(param.get('catchup_floor_m', 1.0))
        self.n_waypoints    = len(self.waypoints)
        self.wp             = 1        # index of current target waypoint
        self.carrot_pos     = np.zeros(3)
        self.committed      = False    # True while pinned to the current gate centre
        self.finished       = False

        # Post wp-advance blend: bridges the segment-switch jump described in
        # the module docstring's point 4. 0 disables (snap immediately, old
        # behaviour). Kept short — this only needs to cover the single-tick
        # geometry discontinuity, not act as a general reference filter (that
        # job belongs to tau_ref_smooth/the PT2 filter in controller.py).
        self._blend_tau   = float(param.get('wp_transition_blend_tau', 0.25))
        self._wp_prev     = self.wp
        self._blend_w     = 0.0            # weight on the pre-transition value, decays to 0
        self._blend_rC    = None
        self._blend_v_ref = None
        self._last_rC     = None
        self._last_v_ref  = None

        # Live-target EMA state for set_live_target() below — smooths an
        # externally-supplied (vision-derived) update to a waypoint slot
        # instead of the caller overwriting it raw every call.
        self._live_target_tau  = float(param.get('wp_live_update_tau', 0.2))   # [s]
        self._live_target_idx  = None   # which waypoints[] index the filter currently belongs to
        self._live_target_filt = None   # filtered NED position, or None until first call

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, pos_ned, dt=None):
        """
        Compute NED reference velocity and yaw for one timestep.

        Parameters
        ----------
        pos_ned : array-like (3,)  current NED position [m]
        dt      : float, optional  seconds since the last update() call, used
                  only to decay the post-transition blend below. Falls back to
                  the nominal control period if omitted.

        Returns
        -------
        v_ned_ref : np.ndarray (3,)  reference velocity in NED  [m/s]
        psi_ref   : float            reference yaw [rad]  0=North, +ve CW
        """
        pos = np.asarray(pos_ned, dtype=float)
        _dt = float(dt) if dt is not None else 0.004

        if self.finished:
            return np.zeros(3), self._last_psi

        # wp advanced externally (controller sets tracker.wp on gate_passed)
        # since the last call — snapshot the pre-transition output and start
        # blending it out, instead of snapping straight to the new segment's
        # freshly-computed geometry.
        if self.wp != self._wp_prev and self._last_rC is not None:
            self._blend_w     = 1.0
            self._blend_rC    = self._last_rC.copy()
            self._blend_v_ref = self._last_v_ref.copy()
        self._wp_prev = self.wp

        WP = self.wp
        r0 = self.waypoints[WP - 1]
        r1 = self.waypoints[WP]

        ea, _ = self._unit(r1 - r0)

        # Carrot is always exactly c metres ahead of the drone's projection along the
        # current segment — no clamping to segment end.  When the drone is within c of
        # the gate, the carrot sits past the gate on the old segment direction, which is
        # fine: the drone still flies through the gate toward the carrot.  When past the
        # gate (gate_pass_dist_m delay window), the carrot keeps pointing forward along
        # the old segment so vN_ref stays at -v_ref rather than collapsing to ~0.
        # Clamping is no longer needed because gate_pass_dist_m controls wp advance timing.
        lambda_L  = float(np.dot(pos - r0, ea))
        seg_len   = float(np.linalg.norm(r1 - r0))
        remaining = seg_len - lambda_L

        # Commit phase: within commit_dist of the gate, blend the aim point
        # from the ordinary lookahead formula toward the gate centre.
        self.committed = remaining <= self.commit_dist

        # Corner-cutting blend: lean the aim direction toward the NEXT segment's
        # tangent as the drone closes on r1, ramped from s=0 at the start of
        # commit to s=1 at (and past) the gate, capped by corner_cut_frac (see
        # __init__'s note). s is clipped to 1 rather than left to grow past it
        # once remaining goes negative (overshoot, see below) — ea_aim must stay
        # at its terminal blend, not keep rotating further toward ea_next.
        # Computed unconditionally (not just pre-overshoot) so ea_aim is
        # continuous across the remaining>=0 / remaining<0 boundary below; this
        # used to jump back to raw `ea` the instant remaining went negative,
        # undoing the whole corner-cut lean in one tick right at gate passage.
        if self.committed:
            s = float(np.clip(1.0 - remaining / self.commit_dist, 0.0, 1.0))
            ea_next = self._next_tangent()
        else:
            s, ea_next = 0.0, None
        if ea_next is not None and self.corner_cut_frac > 0.0:
            k = self.corner_cut_frac
            ea_aim, _ = self._unit((1.0 - s * k) * ea + s * k * ea_next)
        else:
            k = 0.0
            ea_aim = ea

        # `foot` is the projection of pos onto the (infinite) segment line, so
        # (foot - pos) has zero along-track component by construction — adding
        # any forward-leaning offset to it can never point behind the drone,
        # unlike offsetting from the fixed point r1. This is what keeps the
        # overshoot branch below safe for arbitrarily large overshoot without
        # needing to special-case its magnitude.
        foot   = r0 + lambda_L * ea
        P_look = foot + self.c * ea              # ordinary forward lookahead
        # Terminal corner-cut offset from r1 (== 0 when corner-cutting is off,
        # matching the plain "pin to gate centre" behaviour).
        gate_offset = k * self.c * ea_next if ea_next is not None else np.zeros(3)
        P_pin  = r1 + gate_offset

        if not self.committed:
            rC = P_look
        elif remaining >= 0.0:
            # Linearly blend from P_look (matches the not-committed branch
            # exactly at remaining == commit_dist, so there's no seam there)
            # down to P_pin (reached exactly at remaining == 0).
            frac = remaining / self.commit_dist
            rC = frac * P_look + (1.0 - frac) * P_pin
        else:
            # Already past r1 along-track (remaining < 0) with wp-advance not
            # fired yet — e.g. RACE_STATUS/COLLISION lands a tick or two late.
            # Pinning to r1 here would aim BEHIND the drone: unit(r1 - pos) can
            # reverse once the drone has overshot the pin point, flipping
            # v_ned_ref by ~2*v_ref in one tick. Confirmed in a GT-mode flight
            # log: vc_N held at +4.9 (backward) for the last ~140ms of commit
            # phase, then flipped to -4.8 at the wp transition — a ~9.6 m/s
            # reference reversal causing visible jitter.
            #
            # Offset grows from gate_offset (matches P_pin exactly at
            # remaining == 0, so this boundary is jump-free) toward the full
            # c*ea_aim lookahead as overshoot deepens. gate_offset alone isn't
            # enough to fall back on here: with corner_cut_frac == 0 (the
            # current default) gate_offset is exactly zero, and foot's
            # along-track component is zero by construction (see foot's note
            # above) — so foot + gate_offset would carry *no* forward-pointing
            # component at all, degenerating to whatever residual cross-track
            # error happens to be (near-zero on a well-tracked path), which
            # would aim the reference almost purely sideways instead of ahead
            # for however long the external wp-advance is delayed.
            frac2  = float(np.clip(-remaining / self.commit_dist, 0.0, 1.0))
            offset = (1.0 - frac2) * gate_offset + frac2 * (self.c * ea_aim)
            rC = foot + offset

        # Velocity points from drone to carrot → automatic cross-track correction.
        # Yaw uses path tangent (ea_aim — blended toward the next segment during
        # commit, see above) so heading stays stable even with altitude error.
        to_carrot, dist_to_carrot = self._unit(rC - pos)

        # Pure-pursuit "catch-up" singularity guard: rC -> r1 as remaining -> 0
        # (see the commit-phase blend above), and the drone is, by design,
        # also arriving at r1 at that same moment — so (rC - pos) shrinks
        # toward zero right at gate passage and the bearing-to-point direction
        # becomes dominated by numerical noise (confirmed in a closed-loop
        # sim: |v_ned_ref| stayed pinned at v_ref while its direction swung
        # by several m/s tick-to-tick, right as the drone converged on the
        # gate). Blend toward the stable path tangent (ea_aim) as dist shrinks
        # below _CATCHUP_FLOOR_M — at that range the tangent IS the correct
        # direction to fly anyway, and it doesn't depend on the ill-conditioned
        # rC-pos difference.
        blend_t   = float(np.clip(dist_to_carrot / self._CATCHUP_FLOOR_M, 0.0, 1.0))
        to_carrot, _ = self._unit(blend_t * to_carrot + (1.0 - blend_t) * ea_aim)
        v_ned_ref     = self.v_ref * to_carrot

        # Post wp-advance blend (see __init__ / module docstring point 4):
        # ease from the pre-transition rC/v_ned_ref toward this tick's freshly
        # computed (new-segment) values instead of snapping. self._blend_w
        # decays exponentially each tick; once it's negligible this is a
        # no-op and rC/v_ned_ref pass through unchanged.
        if self._blend_w > 1e-3:
            w = self._blend_w
            rC        = w * self._blend_rC    + (1.0 - w) * rC
            v_ned_ref = w * self._blend_v_ref + (1.0 - w) * v_ned_ref
            if self._blend_tau > 0.0:
                self._blend_w *= np.exp(-_dt / self._blend_tau)
            else:
                self._blend_w = 0.0
        else:
            self._blend_w = 0.0

        self.carrot_pos = rC
        self._last_rC     = rC.copy()
        self._last_v_ref  = v_ned_ref.copy()
        psi_ref       = self._yaw(ea_aim)
        self._last_psi = psi_ref

        return v_ned_ref, psi_ref

    # ── Live target update (vision-driven, external caller) ────────────────

    def set_live_target(self, idx, ned, dt=None):
        """Blend a fresh externally-supplied (vision-derived) NED position
        into waypoints[idx] via EMA, instead of the caller overwriting it
        raw every call — a reacquisition-after-vision-loss jump landing
        here unfiltered gets frozen forever as r0 once wp advances past it
        (confirmed in a flight log: an unfiltered snap produced a ~1.5 m/s
        one-tick reference jump right after a gate passage).

        On the first call for a given idx (a new segment just became
        active, or this is the first live update ever), seed the filter
        directly from the CURRENT waypoints[idx] value — no smoothing lag
        on a fresh target — rather than snapping straight to this single
        fresh reading. Blending (not resetting) is always safe: even a
        crude static default converges to live detections at the same
        wp_live_update_tau rate as any later update, never slower.

        idx : which waypoints[] slot to update. Caller is responsible for
              only calling this for the CURRENTLY active target (e.g.
              gated on gate-identity match upstream) — this method has no
              way to know whether idx is the right one.
        ned : fresh externally-measured NED position for that target.
        dt  : seconds since the last call for THIS idx; falls back to the
              nominal control period if omitted (mirrors update()'s own dt
              fallback).
        """
        ned = np.asarray(ned, dtype=float)
        _dt = float(dt) if dt is not None else 0.004
        if self._live_target_idx != idx:
            self._live_target_filt = np.asarray(self.waypoints[idx], dtype=float).copy()
            self._live_target_idx  = idx
        alpha = (np.exp(-_dt / self._live_target_tau)
                 if self._live_target_tau > 0.0 else 0.0)
        self._live_target_filt = (alpha * self._live_target_filt
                                   + (1.0 - alpha) * ned)
        self.waypoints[idx] = self._live_target_filt.copy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_tangent(self):
        """Unit tangent of the segment AFTER the current one (wp -> wp+1),
        or None if the current gate is the last waypoint (nothing to
        anticipate) or that segment is degenerate."""
        if self.wp + 1 >= self.n_waypoints:
            return None
        r1 = self.waypoints[self.wp]
        r2 = self.waypoints[self.wp + 1]
        ea_next, length = self._unit(r2 - r1)
        if length < 1e-6:
            return None
        return ea_next

    @staticmethod
    def _unit(v):
        """Return (unit_vector, length). Handles near-zero gracefully."""
        length = np.linalg.norm(v)
        if length < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return v / length, float(length)

    @staticmethod
    def _yaw(ea):
        """Yaw from NED direction vector: atan2(East, North) → CW from North."""
        return np.arctan2(ea[1], ea[0])
