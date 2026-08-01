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


class Mode(Enum):
    TRACKING    = "TRACKING"
    REACQUIRING = "REACQUIRING"
    BLIND       = "BLIND"
    # Wire value deliberately kept as the legacy literal string 'TRANSITION'
    # (not 'COMMIT') — log.py's _plot_carrot hardcodes 'TRANSITION' to shade
    # carrot.png's commit-phase span, and vision_rx.py's _in_transit check
    # (`ctrl_mode_at_capture == 'TRANSITION'`) gates flight-safety-relevant
    # detection suppression during commit. Neither file is touched by this
    # rewrite; keeping this member's value unchanged means both keep working
    # with zero changes. The other three members are free to use their new
    # names since nothing outside controller.py hardcodes 'PNP'/'HOLD'/
    # 'CARROT' (verified repo-wide).
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
