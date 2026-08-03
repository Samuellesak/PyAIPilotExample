"""
test_pose_disambiguation.py — regression tests for pose_disambiguation.py's
candidate-selection policy. Run: `python -m tests.test_pose_disambiguation`
from the repo root.

Same bare-script style as test_rotations.py (no framework). Because
pose_disambiguation.py has zero live-pipeline dependency (no cv2, no
threading, no model loading), these run in milliseconds and can be exercised
before ever flying the rewrite.
"""

import numpy as np

from flight_model import rotations as rot
from vision.pose_disambiguation import PoseCandidate, disambiguate


def _yaw_R(yaw_rad):
    return rot.quat_to_R_body2ned(rot.euler_to_quat(0.0, 0.0, yaw_rad))


def main():
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        status = "OK" if cond else "FAIL"
        if not cond:
            n_fail += 1
        print(f"[{status}] {name}")

    # 1. Single candidate is always unambiguous, regardless of any other input.
    c = [PoseCandidate(R_b2n=np.eye(3), implied_pos_ned=np.zeros(3), payload='only')]
    r = disambiguate(c, mav_pos_ned=np.array([50.0, 50.0, 50.0]),
                      last_accepted_R=_yaw_R(2.0), last_accepted_t=0.0, now=100.0,
                      position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0)
    check("single candidate -> unambiguous, index 0", r.index == 0 and r.method == 'unambiguous')

    # 2. Two position-DISTINGUISHABLE candidates: position-consistency must win,
    #    regardless of how either candidate's ROTATION compares to any
    #    "drifted EKF"-like reference — there is no such parameter in this
    #    function's signature at all, so this also structurally proves the
    #    policy never consults a live attitude estimate.
    mav_pos = np.array([10.0, 0.0, -5.0])
    near = PoseCandidate(R_b2n=_yaw_R(3.0), implied_pos_ned=np.array([10.1, 0.05, -5.0]), payload='near')
    far  = PoseCandidate(R_b2n=_yaw_R(0.0), implied_pos_ned=np.array([10.1, 6.0, -5.0]), payload='far')
    for i, cands in enumerate([[near, far], [far, near]]):
        r = disambiguate(cands, mav_pos_ned=mav_pos,
                          last_accepted_R=None, last_accepted_t=None, now=None,
                          position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0)
        chosen = cands[r.index]
        check(f"position-distinguishable candidates pick the near one (order {i})",
              chosen.payload == 'near' and r.method == 'position')

    # 3. Four position-IDENTICAL candidates (corner-labelling shape): must
    #    resolve via continuity against last_accepted_R, never position
    #    (position can't discriminate at all here) and never an EKF estimate
    #    (again: no such parameter exists).
    same_pos = np.array([20.0, -3.0, -8.0])
    corner_candidates = [
        PoseCandidate(R_b2n=_yaw_R(k * np.pi / 2.0), implied_pos_ned=same_pos.copy(), payload=k)
        for k in range(4)
    ]
    last_R = _yaw_R(np.pi / 2.0 + 0.05)   # close to candidate k=1
    r = disambiguate(corner_candidates, mav_pos_ned=mav_pos,
                      last_accepted_R=last_R, last_accepted_t=10.0, now=10.05,
                      position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0)
    check("position-identical candidates resolve via continuity, picks nearest to last_accepted_R",
          corner_candidates[r.index].payload == 1 and r.method == 'continuity')

    # 4. Same setup, but the only plausible-looking pick still implies a
    #    faster rotation than max_rotation_rate_rad_s allows -> flagged, but
    #    still returned (best available among a small discrete set).
    last_R_far = _yaw_R(np.pi / 2.0 + 0.05)
    r_fast = disambiguate(corner_candidates, mav_pos_ned=mav_pos,
                           last_accepted_R=last_R_far, last_accepted_t=10.0, now=10.001,
                           position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0)
    check("implausibly-fast continuity pick is flagged, not silently accepted",
          r_fast.method == 'continuity_implausible' and corner_candidates[r_fast.index].payload == 1)

    # 5. True cold start (no last_accepted_R at all): falls back to a FIXED
    #    nominal reference, never a live estimate — passing an attitude-like
    #    reference here would be a caller bug (there is no live-attitude
    #    parameter to accidentally wire up), so nominal_R is explicitly the
    #    only thing this tier can ever consult.
    nominal = _yaw_R(0.0)
    r = disambiguate(corner_candidates, mav_pos_ned=mav_pos,
                      last_accepted_R=None, last_accepted_t=None, now=None,
                      position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0,
                      nominal_R=nominal)
    check("cold start (no history) resolves against the fixed nominal reference",
          corner_candidates[r.index].payload == 0 and r.method == 'cold_start')

    # 6. Position-degenerate but WITHOUT an independent position anchor
    #    (mav_pos_ned=None) must skip tier 1 entirely and go straight to
    #    continuity, not silently treat "no anchor" as "candidates agree."
    r = disambiguate(corner_candidates, mav_pos_ned=None,
                      last_accepted_R=last_R, last_accepted_t=10.0, now=10.05,
                      position_degenerate_tol_m=1.0, max_rotation_rate_rad_s=3.0)
    check("no independent position anchor -> falls through to continuity",
          r.method == 'continuity' and corner_candidates[r.index].payload == 1)

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
