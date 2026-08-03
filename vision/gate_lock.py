"""
gate_lock.py — the gate lock/confidence state machine for vision_rx.py.

Replaces `_locked_dist`/`_consec_det`/`_consec_miss` plus the previously
side-channel gate-identity plausibility recheck (which used to mutate lock
state from outside the state machine, in a separate code block) with one
explicit machine: UNLOCKED -> ACQUIRING -> LOCKED, with the identity recheck
as a guarded transition condition rather than an external side effect.

Behavior preserved exactly from the code this replaces (see on_frame's
docstring for the specific invariants and why each one is load-bearing, not
incidental): identity-check-before-spike-check ordering; hit_count surviving
isolated misses (only a sustained miss streak resets acquisition progress);
a LOCKED-but-wrong identity recheck dropping the lock entirely rather than
just rejecting one frame; the periodic (not every-locked-frame) recheck
cadence to avoid the ordinary-EKF-drift self-reinforcing-lockout risk that
motivated scoping the original check to acquisition-time in the first place.

Dependency-free (stdlib only) so this can be unit-tested without pulling in
vision_rx.py's camera/model machinery.
"""

from dataclasses import dataclass, field
from typing import Optional

from vision.pose_estimate import LockState


@dataclass
class GateLock:
    lock_time_s:          float   # sustained good-detection duration required to LOCK
    lock_min_hits:        int     # floor on accepted-frame count, independent of elapsed time
    miss_max:             int
    spike_tol_m:          float
    id_tol_m:             float
    id_recheck_period_s:  float

    state:               LockState       = LockState.UNLOCKED
    hit_count:           int             = 0
    miss_streak:         int             = 0
    last_good_range_m:   Optional[float] = None
    last_id_recheck_t:   Optional[float] = None
    # Wall-clock time of the first accepted frame in the current ACQUIRING
    # run — set once on UNLOCKED->ACQUIRING and left alone by isolated
    # misses (same "isolated miss doesn't wipe progress" invariant as
    # hit_count), so acquisition is gated on how long good detection has
    # actually been sustained rather than on a fixed frame count. A fixed
    # count is a proxy for "roughly N seconds at the assumed camera rate" —
    # confirmed in a flight log that once the real rate dropped well below
    # what lock_frames (the old field) was tuned against, acquisition took
    # proportionally longer in wall-clock terms, directly extending BLIND
    # periods. lock_min_hits is a small floor kept alongside lock_time_s so
    # a single lucky frame plus a long gap can't lock on time alone.
    _acquiring_since:    Optional[float] = field(default=None, repr=False)
    # Diagnostics only, set by the most recent on_frame() call — lets the
    # caller (vision_rx.py) pick the right skip_reason/gate_id_rejected
    # values without re-deriving the identity-tolerance comparison itself.
    # One of 'no_measurement' | 'identity' | 'spike' | None (accepted).
    last_reject_reason:  Optional[str]   = None

    def reset(self) -> None:
        """Full reset to UNLOCKED. Called both internally (miss-streak
        exhausted, identity recheck failed while LOCKED) and externally by
        vision_rx.py on an active_gate_index change."""
        self.state = LockState.UNLOCKED
        self.hit_count = 0
        self.miss_streak = 0
        self.last_good_range_m = None
        self._acquiring_since = None
        # last_id_recheck_t is NOT cleared here: it tracks wall-clock recheck
        # cadence, not lock identity, and clearing it would make the very
        # next frame after a reset skip straight to "due" regardless of how
        # recently a recheck ran — harmless either way, but leaving it alone
        # matches the narrower scope of what actually needs to reset.

    def on_frame(self, measured_range_m: Optional[float],
                 expected_range_m: Optional[float], now: float) -> bool:
        """Advance the state machine by one frame. Returns True iff this
        frame's measurement is ACCEPTED (safe for the caller to use for a
        position/attitude/velocity fix this tick).

        measured_range_m=None means no PnP candidate at all this frame
        (YOLO miss or solvePnP failure) — goes straight to miss handling,
        the same way a candidate that fails the identity check below does.

        expected_range_m=None means the identity check can't run this frame
        (no independent position anchor available yet) — the check stays
        "due" (last_id_recheck_t untouched) until one appears, exactly
        mirroring the code this replaces rather than silently treating an
        unavailable anchor as a pass.
        """
        id_due = (
            self.state != LockState.LOCKED
            or self.last_id_recheck_t is None
            or (now - self.last_id_recheck_t) >= self.id_recheck_period_s
        )

        accepted = measured_range_m
        if accepted is not None and id_due and expected_range_m is not None:
            self.last_id_recheck_t = now
            if abs(accepted - expected_range_m) > self.id_tol_m:
                # A lock this far off its expected target is wrong, not just
                # noisy — drop it entirely (not just this frame) so the next
                # successful detection goes through full acquisition instead
                # of being folded into the same bad lock's history. Only
                # meaningful once actually LOCKED; during ACQUIRING there is
                # no "lock" yet to drop, so hit_count is deliberately left
                # untouched here (an isolated bad-identity frame mid-
                # acquisition doesn't wipe progress, same as an isolated miss).
                if self.state == LockState.LOCKED:
                    self.reset()
                accepted = None

        if accepted is None:
            self.last_reject_reason = 'no_measurement' if measured_range_m is None else 'identity'
            self.miss_streak += 1
            if self.miss_streak >= self.miss_max:
                self.reset()
            return False

        if self.state == LockState.LOCKED:
            if accepted > self.last_good_range_m + self.spike_tol_m:
                # Distance spike — almost certainly a different, farther
                # gate. Folded into the ordinary miss-streak budget, not an
                # instant drop: a single spike shouldn't cost the whole lock.
                self.last_reject_reason = 'spike'
                self.miss_streak += 1
                if self.miss_streak >= self.miss_max:
                    self.reset()
                return False
            self.last_reject_reason = None
            self.miss_streak = 0
            self.last_good_range_m = accepted   # track distance as drone approaches
            return True

        # UNLOCKED or ACQUIRING: every identity-plausible frame counts,
        # consecutive or not (an isolated miss above never reset hit_count
        # or _acquiring_since). Locks once BOTH the elapsed-time and
        # min-hits thresholds are satisfied — see _acquiring_since's
        # comment for why time, not just a frame count, is load-bearing.
        self.last_reject_reason = None
        self.miss_streak = 0
        self.hit_count += 1
        if self.state == LockState.UNLOCKED:
            self.state = LockState.ACQUIRING
            self._acquiring_since = now
        if (self.hit_count >= self.lock_min_hits
                and self._acquiring_since is not None
                and (now - self._acquiring_since) >= self.lock_time_s):
            self.state = LockState.LOCKED
            self.last_good_range_m = accepted
        return True
