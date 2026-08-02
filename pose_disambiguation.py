"""
pose_disambiguation.py — the single, non-self-referential pose-disambiguation
policy for vision_rx.py's two ambiguity sites (the IPPE 2-solution front/back
flip in _pnp_gate, and the 4-fold corner-labelling relabelling in
_attitude_from_pnp).

Why this file exists
---------------------
Both ambiguities used to be resolved (independently, with two separately
hand-written and separately patched implementations) by scoring candidates
against the EKF's OWN current attitude (mav_state['quat']). That is
self-referential: the EKF's attitude is itself corrected by the very PnP
measurement being disambiguated, so once the EKF has drifted even a little,
a wrong candidate can look "closer" and get picked — which then feeds back
into the EKF and drifts it further next frame. This was found and patched
twice this session, at two different layers, because the first patch (a
yaw-rate plausibility check on the corner-labelling search) didn't fix the
symptom — the real cause was the SAME architectural mistake one layer
upstream, in the IPPE disambiguation.

The fix here is structural, not another patch: this function never looks at
a live EKF/attitude estimate. It disambiguates using only:
  1. Position-consistency against an independent anchor (mav_state['pos_ned'] —
     proven far more reliable than attitude in this codebase's own flight
     logs: bounded ~0.3-0.5m error vs. attitude's demonstrated 100+ degree
     excursions), when position actually differs across candidates.
  2. For position-degenerate ambiguity (always true for corner-labelling,
     since relabelling doesn't move tvec — confirmed by the object model
     itself), rate-limited temporal continuity against THIS PIPELINE's own
     last-accepted output — never the EKF's.
  3. True cold start (no prior accepted output at all): a fixed nominal
     reference that never changes in response to this pipeline's own past
     output, so it cannot self-reinforce.

Dependency-free (numpy only) so this can be unit-tested (see
test_pose_disambiguation.py) without pulling in vision_rx.py's camera/model
machinery.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

import rotations


@dataclass(frozen=True)
class PoseCandidate:
    R_b2n:           np.ndarray   # (3,3) candidate body->NED rotation
    implied_pos_ned: np.ndarray   # (3,) candidate's implied drone NED position
    payload:         Any          # opaque — caller's own (rvec, tvec) or (roll, pitch, yaw) etc.


@dataclass(frozen=True)
class DisambiguationResult:
    index:  int    # which candidate was chosen
    method: str    # 'unambiguous' | 'position' | 'continuity' | 'continuity_implausible' | 'cold_start'
                    # — callers must only persist this pick as the new continuity anchor when
                    # method != 'continuity_implausible' (see disambiguate()'s docstring)
    margin: float  # diagnostics only: chosen-vs-runner-up gap, in the tier's native units (m or rad)


def disambiguate(
    candidates: Sequence[PoseCandidate],
    mav_pos_ned: Optional[np.ndarray],
    last_accepted_R: Optional[np.ndarray],
    last_accepted_t: Optional[float],
    now: Optional[float],
    position_degenerate_tol_m: float,
    max_rotation_rate_rad_s: float,
    nominal_R: np.ndarray = np.eye(3),
) -> DisambiguationResult:
    """Pick which of `candidates` is correct. Never consults a live EKF
    attitude estimate — see module docstring for why.

    Tier 1 (position): used when >=2 candidates imply meaningfully different
    positions (spread > position_degenerate_tol_m) AND mav_pos_ned is
    available — pick whichever implied position is closest to mav_pos_ned.

    Tier 2 (continuity): position-degenerate, or no independent position
    anchor available — pick whichever candidate's rotation is closest to
    last_accepted_R. If that implies a rotation faster than
    max_rotation_rate_rad_s since last_accepted_t, the pick is still
    returned (it's the best available among a small discrete set — there is
    no second self-consistent strategy to fall back to once the EKF is out
    of the picture) but flagged via method='continuity_implausible'.

    That flag is not just diagnostics: it is also the caller's contract for
    whether to persist this pick as the NEW last_accepted_R/_t. Confirmed in
    a flight log that a caller which persists unconditionally lets a single
    implausible pick (e.g. a 4-fold corner-labelling relabelling that
    briefly wins on raw angle during a fast real turn) become the anchor for
    every subsequent frame — the SAME wrong branch then stays closest to its
    own now-wrong anchor indefinitely, since angle-order vs. last_accepted_R
    is unchanged by which branch last_accepted_R itself sits on. A caller
    that only persists on method=='continuity' (never on
    'continuity_implausible') breaks that loop: the anchor stays on the last
    trusted pick, so the true candidate keeps a fair (indeed growing, as
    now - last_accepted_t widens the plausible-rate window) chance to win
    again on the next frame instead of being permanently disadvantaged.

    Tier 3 (cold start): no last_accepted_R at all (true flight start, or a
    long-enough gap that continuity itself would be groundless) — pick
    whichever candidate is closest to a FIXED nominal_R. This never
    self-reinforces because nominal_R does not change in response to this
    pipeline's own prior output.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("disambiguate() requires at least one candidate")
    if n == 1:
        return DisambiguationResult(index=0, method='unambiguous', margin=float('inf'))

    if mav_pos_ned is not None:
        positions = np.stack([c.implied_pos_ned for c in candidates])
        spread = float(np.max(np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1)))
        if spread > position_degenerate_tol_m:
            dists = np.linalg.norm(positions - np.asarray(mav_pos_ned), axis=1)
            order = np.argsort(dists)
            best, runner_up = int(order[0]), float(dists[order[1]] - dists[order[0]])
            return DisambiguationResult(index=best, method='position', margin=runner_up)

    if last_accepted_R is not None:
        angles = np.array([
            rotations.rotation_angle_distance(c.R_b2n, last_accepted_R) for c in candidates
        ])
        order = np.argsort(angles)
        best, runner_up = int(order[0]), float(angles[order[1]] - angles[order[0]])
        method = 'continuity'
        if (last_accepted_t is not None and now is not None
                and now > last_accepted_t):
            implied_rate = angles[best] / (now - last_accepted_t)
            if implied_rate > max_rotation_rate_rad_s:
                method = 'continuity_implausible'
        return DisambiguationResult(index=best, method=method, margin=runner_up)

    angles = np.array([
        rotations.rotation_angle_distance(c.R_b2n, nominal_R) for c in candidates
    ])
    order = np.argsort(angles)
    best, runner_up = int(order[0]), float(angles[order[1]] - angles[order[0]])
    return DisambiguationResult(index=best, method='cold_start', margin=runner_up)
