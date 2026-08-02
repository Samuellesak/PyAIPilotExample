"""
vision_mode.py — Layer 2 (reference shaping, controller.py) mode/confidence
state machine and vertical-assist blend.

Part of the gate-transition/vision-reacquisition rewrite (see the plan doc).
Replaces controller.py's old vis_ctrl_mode string state machine (TRANSITION/
PNP/HOLD/CARROT, independently re-derived each tick from a mixture of this-
tick-fresh and last-tick-stale sub-signals: tracker.committed, _fresh_tvec,
and the previous tick's _vis_tvec_hold) with one state machine driven
directly by Layer 1's single PoseEstimate.state.

Confidence (TRACKING/REACQUIRING/BLIND, from PoseEstimate.state) and commit-
phase (tracker.committed) are two orthogonal signals, combined only at the
point of use — "committed" is only knowable AFTER CarrotTracker.update()
runs, so it can't be folded into an earlier-computed mode value.

Also owns VerticalAssist, which merges the old blind-search descend nudge
and vertical visual-centering nudge (previously two independent,
uncoordinated += onto v_ref_for_gains[2]) under one explicit, bounded
combination.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from pose_estimate import LockState


def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def path_convergence_weight(pos_ned, seg_r0, seg_ea, recover_dist_m):
    """1.0 when pos_ned sits on the segment line through seg_r0 along unit
    tangent seg_ea, ramping linearly down to 0.0 at recover_dist_m of
    perpendicular (cross-track) distance from that line.

    Gates how much weight controller.py's pursuit override gives its
    direct-at-the-gate direction, alongside (via min()) the existing
    trust_w. trust_w alone is a vision-confidence ramp on a fixed time
    schedule (vision_reacq_blend_tau) — it reflects "how long has vision
    looked stable," not "has the drone actually gotten back near the
    path," so it can reach full weight before cross-track error from a
    BLIND stretch has actually closed, letting direct gate-bearing
    pursuit cut a diagonal across to the gate instead of first rejoining
    the line. Stateless by design: position is already smooth tick to
    tick (unlike raw vision bearings), so unlike trust_w this needs no
    ramp of its own — it's recomputed fresh from the current cross-track
    distance every call.
    """
    along   = np.dot(np.asarray(pos_ned, dtype=float) - seg_r0, seg_ea)
    on_line = seg_r0 + along * seg_ea
    cross_track = float(np.linalg.norm(np.asarray(pos_ned, dtype=float) - on_line))
    if recover_dist_m <= 0.0:
        return 1.0
    return float(np.clip(1.0 - cross_track / recover_dist_m, 0.0, 1.0))


class Mode(Enum):
    TRACKING    = "TRACKING"
    REACQUIRING = "REACQUIRING"
    BLIND       = "BLIND"
    RECOVERY    = "RECOVERY"
    # Wire value deliberately kept as the legacy literal string 'TRANSITION'
    # (not 'COMMIT') — log.py's _plot_carrot hardcodes 'TRANSITION' to shade
    # carrot.png's commit-phase span, and vision_rx.py's _in_transit check
    # (`ctrl_mode_at_capture == 'TRANSITION'`) gates flight-safety-relevant
    # detection suppression during commit. Neither file is touched by this
    # rewrite; keeping this member's value unchanged means both keep working
    # with zero changes. The other members are free to use their own names
    # since nothing outside controller.py hardcodes 'PNP'/'HOLD'/'CARROT'
    # (verified repo-wide).
    COMMIT = "TRANSITION"


@dataclass
class VisionModeTracker:
    """Tracks the reacquisition trust ramp and derives Mode from
    PoseEstimate.state + CarrotTracker.committed.

    Level-triggered on lock_state alone — no separate edge-detection is
    needed the way the old vis_ctrl_mode-transition check required, because
    GateLock's own miss-streak hysteresis already absorbs ordinary brief
    flicker without ever leaving LOCKED; leaving LOCKED IS the "real loss"
    signal this ramp resets on.
    """
    reacq_tau_s: float
    trust_w: float = 1.0

    def update(self, lock_state: LockState, dt: float) -> None:
        if lock_state == LockState.LOCKED:
            a = np.exp(-dt / self.reacq_tau_s) if self.reacq_tau_s > 0.0 else 0.0
            self.trust_w = 1.0 - a * (1.0 - self.trust_w)
        else:
            self.trust_w = 0.0

    def base_mode(self, lock_state: LockState) -> Mode:
        if lock_state == LockState.LOCKED:
            return Mode.TRACKING if self.trust_w > 0.999 else Mode.REACQUIRING
        return Mode.BLIND   # covers both UNLOCKED and ACQUIRING

    def effective_mode(self, lock_state: LockState, committed: bool) -> Mode:
        """Commit takes absolute priority, matching today's
        TRANSITION > PNP > HOLD > CARROT precedence."""
        return Mode.COMMIT if committed else self.base_mode(lock_state)


@dataclass
class VerticalAssist:
    """Merges the blind-search descend nudge (fires once vision is BLIND
    near the target for a sustained period) and the vertical visual-
    centering nudge (fires proactively while still tracking but the gate is
    drifting toward the top/bottom of frame) under one owner.

    Trigger conditions for each sub-term stay exactly as they were — they
    are not physically mutually exclusive, so forcing exclusivity isn't
    correct — only the final combination and cap gain a single owner
    instead of each independently `+=`-ing onto the D reference.
    """
    search_range_m:      float
    search_trigger_s:    float
    search_max_m:         float
    search_rate_mps:      float
    vnudge_border_frac:   float
    vnudge_max_mps:       float
    cam_cy:                float

    _blind_since:      Optional[float] = field(default=None, repr=False)
    _search_offset_m:  float           = 0.0

    def update(self, mode: Mode, dist_to_target: float,
               centre_px, now: float, dt: float,
               debug_waypoints_only: bool) -> Tuple[float, dict]:
        """Returns (combined_D_velocity_bias, diagnostics_dict).

        debug_waypoints_only must stay a separate, explicit condition here
        (not folded into Mode itself) — in that config vis-driven control is
        switched off by design, so Mode is permanently BLIND regardless of
        whether the gate is actually visible; without this guard the search
        nudge misreads that as a permanent vision loss and ramps to its full
        descend cap on every approach, even with the gate in plain view
        (confirmed in a flight log — this is the exact bug that motivated
        adding this guard in the first place).
        """
        # ── Blind-search sub-term ───────────────────────────────────────
        if mode == Mode.BLIND:
            if self._blind_since is None:
                self._blind_since = now
        else:
            self._blind_since = None

        searching = (
            not debug_waypoints_only
            and self._blind_since is not None
            and (now - self._blind_since) > self.search_trigger_s
            and dist_to_target < self.search_range_m
        )
        # _search_offset_m (m) is the bounded target descent distance; the
        # velocity bias actually commanded is its derivative, so the bias is
        # +search_rate_mps while ramping toward the cap, 0 once capped, and
        # -search_rate_mps while unwinding — the accumulated commanded
        # descent never exceeds search_max_m.
        target = self.search_max_m if searching else 0.0
        step   = self.search_rate_mps * dt
        prev   = self._search_offset_m
        if prev < target:
            self._search_offset_m = min(target, prev + step)
        elif prev > target:
            self._search_offset_m = max(target, prev - step)
        search_term = (self._search_offset_m - prev) / max(dt, 1e-6)

        # ── Vertical visual-centering sub-term ──────────────────────────
        # cam_cy is the image half-height (principal point), so dy_px/cam_cy
        # is ~0 at vertical center and ~±1 at the top/bottom border. Only
        # activates past vnudge_border_frac of the way to the edge, so
        # ordinary comfortably-centered flight is unaffected.
        vnudge_term = 0.0
        if centre_px is not None:
            dy_px = float(centre_px[1]) - self.cam_cy
            border_frac = abs(dy_px) / self.cam_cy if self.cam_cy > 0 else 0.0
            if border_frac > self.vnudge_border_frac:
                excess = ((border_frac - self.vnudge_border_frac)
                          / (1.0 - self.vnudge_border_frac))
                excess = float(np.clip(excess, 0.0, 1.0))
                # dy_px > 0: gate below center (near bottom border) ->
                # descend (positive Down) to bring it back toward center.
                # dy_px < 0: gate above center (near top border) -> climb.
                vnudge_term = float(np.sign(dy_px) * excess * self.vnudge_max_mps)

        cap = self.search_rate_mps + self.vnudge_max_mps
        combined = float(np.clip(search_term + vnudge_term, -cap, cap))
        diag = dict(search_active=bool(searching),
                    search_offset=self._search_offset_m,
                    vnudge_bias=vnudge_term)
        return combined, diag


@dataclass
class PursuitGuidance:
    """Rate-limits the pursuit-override bearing direction (NED-frame unit
    vector) so a large one-off correction — e.g. the bearing step right
    after a long BLIND coast — is flown as a gradual turn-in instead of the
    override's raw bearing-following law, which points straight at the
    gate's current bearing every tick with no damping term of its own.

    Confirmed in a flight log: after a ~3s BLIND coast, reacquisition
    produced an overshoot-then-correct swing (drone_E off by several
    metres, oscillating rather than decaying smoothly) that a short
    reacquisition-to-commit runway didn't leave enough time to settle
    before COMMIT locked the approach in — this is the direct cause of an
    oblique gate crossing observed downstream. Rate-limiting the direction
    itself (not just the trust-ramp blend weight, which only scales
    magnitude toward this same unlimited direction) turns that single
    violent swing into a gradual turn-in.

    Resets whenever the caller isn't feeding it a live pursuit direction
    (pursuit override not active this tick) — see reset() — so a stale
    direction left over from a previous gate can't leak across a
    BLIND/COMMIT gap into the next gate's reacquisition.
    """
    max_turn_rate: float   # rad/s cap on how fast the direction angle can slew
    _dir: Optional[np.ndarray] = field(default=None, repr=False)

    def update(self, dir_n: float, dir_e: float, dt: float) -> Tuple[float, float]:
        target = np.array([float(dir_n), float(dir_e)])
        norm = float(np.linalg.norm(target))
        if norm < 1e-9:
            return dir_n, dir_e
        target = target / norm
        if self._dir is None:
            self._dir = target
        else:
            cur_ang = float(np.arctan2(self._dir[1], self._dir[0]))
            tgt_ang = float(np.arctan2(target[1], target[0]))
            step = float(np.clip(_wrap_pi(tgt_ang - cur_ang),
                                  -self.max_turn_rate * dt, self.max_turn_rate * dt))
            new_ang = cur_ang + step
            self._dir = np.array([np.cos(new_ang), np.sin(new_ang)])
        return float(self._dir[0]), float(self._dir[1])

    def reset(self):
        self._dir = None


@dataclass
class RecoveryGuard:
    """Vetoes a carrot-tracker COMMIT trigger that vision evidence directly
    contradicts, and manages the resulting recovery episode.

    tracker.committed is computed purely from the drone's own EKF position
    projected onto the current segment (carrot_tracker.py, untouched by
    this rewrite) — nothing checks it against an independent source before
    it fires. Confirmed in a flight log: a drifted position estimate
    tripped committed early, vision got suppressed for the whole commit-
    phase flight-through (by design, to avoid near-gate bearing noise), and
    by the time transit ended the gate measured 27.75m away against the 3m
    commit_dist_m that triggered entry — nothing upstream ever found out
    committed was wrong until GateLock's own spike check started rejecting
    the (correct) recovery data, because its reference range was now stale
    from before the mis-triggered commit.

    Entry: committed is true, but a FRESH, locked, gate_id-matching pose
    reports a range well beyond commit_dist_m — not just PnP noise at
    close range. Latches active rather than requiring the mismatch to keep
    re-firing every tick, since fresh vision data can be sparse exactly
    when this matters most (that sparseness is usually why committed got
    triggered wrong in the first place). Exit: the drone reaches the
    recovery point, or a timeout elapses so a persistently uncooperative
    vision feed can't strand the drone in recovery forever — whichever
    comes first.
    """
    sanity_margin_m: float
    exit_dist_m:     float
    timeout_s:       float

    active:      bool            = False
    _entered_at: Optional[float] = field(default=None, repr=False)

    def update(self, raw_committed: bool, pose, gate_id_match: bool,
               commit_dist_m: float, dist_to_recovery_point: float,
               now: float) -> bool:
        """Returns True iff RECOVERY should override this tick's mode."""
        if self.active:
            timed_out = (self._entered_at is not None
                         and (now - self._entered_at) > self.timeout_s)
            if dist_to_recovery_point < self.exit_dist_m or timed_out:
                self.active = False
                self._entered_at = None
            return self.active

        mismatch = (
            raw_committed and gate_id_match
            and pose.fresh and pose.has_pnp and pose.range_m is not None
            and pose.range_m > commit_dist_m + self.sanity_margin_m
        )
        if mismatch:
            self.active = True
            self._entered_at = now
        return self.active
