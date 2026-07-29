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
      drops to commit_dist_m or below, rC is pinned to r1 directly instead of
      continuing to advance past it. The lookahead carrot's position along the
      segment is already immune to lateral position noise (it only depends on
      lambda_L, a projection onto the line), but a dynamically-advancing carrot
      still means the aim point keeps shifting forward as the drone closes in —
      pinning to the actual gate centre for the final approach commits to
      flying through that specific point rather than a carrot that's always
      sliding further down the line. If the drone overshoots r1 along-track
      before wp advances (remaining goes negative), the pin is dropped back to
      the normal foot + c*ea lookahead so the reference never aims behind the
      drone — aiming at a point that's now behind can flip v_ned_ref by ~2*v_ref
      in one tick.

  3.  Reference velocity:  v_ned_ref = v_ref * unit(rC - pos)  (NED frame)
      Reference yaw:       psi_ref   = atan2(ea_E, ea_N)

  4.  Waypoint advance is driven externally: the controller calls
      tracker.wp += 1 on gate_passed (COLLISION) only.

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
        self.n_waypoints    = len(self.waypoints)
        self.wp             = 1        # index of current target waypoint
        self.carrot_pos     = np.zeros(3)
        self.committed      = False    # True while pinned to the current gate centre
        self.finished       = False

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, pos_ned):
        """
        Compute NED reference velocity and yaw for one timestep.

        Parameters
        ----------
        pos_ned : array-like (3,)  current NED position [m]

        Returns
        -------
        v_ned_ref : np.ndarray (3,)  reference velocity in NED  [m/s]
        psi_ref   : float            reference yaw [rad]  0=North, +ve CW
        """
        pos = np.asarray(pos_ned, dtype=float)

        if self.finished:
            return np.zeros(3), self._last_psi

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

        # Commit phase: within commit_dist of the gate, stop advancing the
        # lookahead point and pin it to the gate centre itself.
        self.committed = remaining <= self.commit_dist
        if self.committed and remaining >= 0.0:
            rC = r1
        else:
            # Either not yet committed, or already past r1 along-track (remaining
            # < 0) with wp-advance not fired yet — e.g. RACE_STATUS/COLLISION
            # lands a tick or two late. Pinning to r1 here would aim BEHIND the
            # drone: unit(r1 - pos) can reverse once the drone has overshot the
            # pin point, flipping v_ned_ref by ~2*v_ref in one tick. Confirmed in
            # a GT-mode flight log: vc_N held at +4.9 (backward) for the last
            # ~140ms of commit phase, then flipped to -4.8 at the wp transition —
            # a ~9.6 m/s reference reversal causing visible jitter. Falling back
            # to the same forward-lookahead-from-foot used pre-commit keeps the
            # reference pointing along ea regardless of overshoot.
            foot = r0 + lambda_L * ea
            rC   = foot + self.c * ea
        self.carrot_pos = rC

        # Velocity points from drone to carrot → automatic cross-track correction.
        # Yaw uses path tangent (ea) so heading stays stable even with altitude error.
        to_carrot, _ = self._unit(rC - pos)
        v_ned_ref     = self.v_ref * to_carrot
        psi_ref       = self._yaw(ea)
        self._last_psi = psi_ref

        return v_ned_ref, psi_ref

    # ── Helpers ───────────────────────────────────────────────────────────────

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
