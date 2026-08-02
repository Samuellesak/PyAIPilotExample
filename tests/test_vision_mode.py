"""
test_vision_mode.py — regression tests for vision_mode.py's Mode/
VisionModeTracker/VerticalAssist/PursuitGuidance/RecoveryGuard. Run:
`python -m tests.test_vision_mode` from the repo root.
"""

import numpy as np

from vision.pose_estimate import LockState, PoseEstimate
from vision.vision_mode import (Mode, VisionModeTracker, VerticalAssist, PursuitGuidance,
                          RecoveryGuard, path_convergence_weight)


def main():
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        status = "OK" if cond else "FAIL"
        if not cond:
            n_fail += 1
        print(f"[{status}] {name}")

    # 1. VisionModeTracker: not locked -> trust_w forced to 0, mode BLIND.
    vm = VisionModeTracker(reacq_tau_s=0.4)
    vm.update(LockState.UNLOCKED, 0.02)
    check("UNLOCKED forces trust_w to 0", vm.trust_w == 0.0)
    check("UNLOCKED -> base_mode BLIND", vm.base_mode(LockState.UNLOCKED) == Mode.BLIND)
    vm.update(LockState.ACQUIRING, 0.02)
    check("ACQUIRING also forces trust_w to 0", vm.trust_w == 0.0)
    check("ACQUIRING -> base_mode BLIND", vm.base_mode(LockState.ACQUIRING) == Mode.BLIND)

    # 2. Continuous LOCKED ramps trust_w 0->1 over reacq_tau_s, mode flips
    #    REACQUIRING -> TRACKING once it crosses the 0.999 threshold. Start
    #    from an explicit loss (trust_w default is 1.0, matching a fresh
    #    flight's initial trust) so this is a genuine "just reacquired" ramp,
    #    the same as how a real loss->reacquire sequence would drive it.
    vm2 = VisionModeTracker(reacq_tau_s=0.4)
    vm2.update(LockState.UNLOCKED, 0.01)
    check("setup: trust_w is 0 right after a loss", vm2.trust_w == 0.0)
    saw_reacquiring = False
    t = 0.0
    # exp(-t/tau) < 0.001 needs t > tau*ln(1000) =~ 2.76s at tau=0.4 -> 500
    # steps of 0.01s (5.0s) comfortably clears that with margin to spare.
    for _ in range(500):
        vm2.update(LockState.LOCKED, 0.01)
        t += 0.01
        m = vm2.base_mode(LockState.LOCKED)
        if m == Mode.REACQUIRING:
            saw_reacquiring = True
        if vm2.trust_w > 0.999:
            break
    check("ramping trust_w passes through REACQUIRING before TRACKING", saw_reacquiring)
    check("trust_w eventually reaches ~1.0", vm2.trust_w > 0.999)
    check("base_mode is TRACKING once trust_w > 0.999",
          vm2.base_mode(LockState.LOCKED) == Mode.TRACKING)

    # 3. A real loss (leaving LOCKED) resets trust_w to 0 even after it had
    #    fully ramped up — ordinary brief flicker never leaves LOCKED at
    #    all (that's GateLock's own miss-streak hysteresis), so this reset
    #    only ever fires on a genuine loss.
    vm2.update(LockState.UNLOCKED, 0.01)
    check("leaving LOCKED resets trust_w to 0 even after full ramp", vm2.trust_w == 0.0)

    # 4. effective_mode: COMMIT overrides everything, regardless of lock state.
    vm3 = VisionModeTracker(reacq_tau_s=0.4)
    check("committed + LOCKED -> COMMIT", vm3.effective_mode(LockState.LOCKED, True) == Mode.COMMIT)
    check("committed + UNLOCKED -> COMMIT", vm3.effective_mode(LockState.UNLOCKED, True) == Mode.COMMIT)
    check("COMMIT wire value is the legacy 'TRANSITION' string", Mode.COMMIT.value == "TRANSITION")
    check("not committed + UNLOCKED -> BLIND", vm3.effective_mode(LockState.UNLOCKED, False) == Mode.BLIND)

    # 5. VerticalAssist: debug_waypoints_only suppresses the search term even
    #    when mode is BLIND for a long sustained period within range — this
    #    is the exact historical bug (search misfiring under
    #    debug_waypoints_only) this guard exists to prevent.
    va = VerticalAssist(search_range_m=15.0, search_trigger_s=0.5, search_max_m=3.0,
                         search_rate_mps=0.6, vnudge_border_frac=0.6, vnudge_max_mps=1.5,
                         cam_cy=180.0)
    now = 0.0
    for _ in range(100):
        bias, diag = va.update(Mode.BLIND, dist_to_target=5.0, centre_px=None,
                                now=now, dt=0.02, debug_waypoints_only=True)
        now += 0.02
    check("search stays inactive under debug_waypoints_only despite sustained BLIND",
          diag['search_active'] is False and va._search_offset_m == 0.0)

    # 6. VerticalAssist: search activates once BLIND persists past
    #    search_trigger_s within range, and ramps at search_rate_mps.
    va2 = VerticalAssist(search_range_m=15.0, search_trigger_s=0.5, search_max_m=3.0,
                          search_rate_mps=0.6, vnudge_border_frac=0.6, vnudge_max_mps=1.5,
                          cam_cy=180.0)
    now = 0.0
    bias = 0.0
    for _ in range(20):   # 20*0.05 = 1.0s, past the 0.5s trigger
        bias, diag = va2.update(Mode.BLIND, dist_to_target=5.0, centre_px=None,
                                 now=now, dt=0.05, debug_waypoints_only=False)
        now += 0.05
    check("search activates after sustained BLIND within range", diag['search_active'] is True)
    check("search offset ramped up (non-zero, below cap)",
          0.0 < va2._search_offset_m <= 3.0)
    check("commanded bias while ramping matches search_rate_mps", np.isclose(bias, 0.6, atol=1e-6))

    # 7. VerticalAssist: search unwinds back to 0 once mode leaves BLIND.
    for _ in range(200):
        bias, diag = va2.update(Mode.TRACKING, dist_to_target=5.0, centre_px=None,
                                 now=now, dt=0.05, debug_waypoints_only=False)
        now += 0.05
        if va2._search_offset_m <= 0.0:
            break
    check("search offset fully unwinds once mode leaves BLIND", va2._search_offset_m == 0.0)
    check("search_active reports False once unwound", diag['search_active'] is False)

    # 8. VerticalAssist: vnudge only activates past the border fraction, and
    #    the sign matches "descend when gate is below center."
    va3 = VerticalAssist(search_range_m=15.0, search_trigger_s=0.5, search_max_m=3.0,
                          search_rate_mps=0.6, vnudge_border_frac=0.6, vnudge_max_mps=1.5,
                          cam_cy=180.0)
    # Comfortably centered (dy small) -> no nudge.
    _, diag_c = va3.update(Mode.TRACKING, dist_to_target=5.0, centre_px=(320.0, 190.0),
                            now=0.0, dt=0.02, debug_waypoints_only=False)
    check("vnudge inactive when comfortably centered", diag_c['vnudge_bias'] == 0.0)
    # Near bottom border (dy_px large positive) -> positive (descend) bias.
    _, diag_b = va3.update(Mode.TRACKING, dist_to_target=5.0, centre_px=(320.0, 350.0),
                            now=0.02, dt=0.02, debug_waypoints_only=False)
    check("vnudge is positive (descend) when gate is near the bottom border",
          diag_b['vnudge_bias'] > 0.0)
    # Near top border (dy_px large negative) -> negative (climb) bias.
    _, diag_t = va3.update(Mode.TRACKING, dist_to_target=5.0, centre_px=(320.0, 10.0),
                            now=0.04, dt=0.02, debug_waypoints_only=False)
    check("vnudge is negative (climb) when gate is near the top border",
          diag_t['vnudge_bias'] < 0.0)
    check("vnudge never exceeds vnudge_max_mps", abs(diag_b['vnudge_bias']) <= 1.5 + 1e-9)

    # 9. PursuitGuidance: first call after construction (no cached direction
    #    yet) seeds directly from the target, unlimited — there's nothing to
    #    rate-limit against yet.
    pg = PursuitGuidance(max_turn_rate=np.deg2rad(60.0))
    n0, e0 = pg.update(0.0, 1.0, dt=0.01)   # dead east, far outside a 0.6deg step
    check("first call seeds directly from the target direction",
          np.isclose(n0, 0.0, atol=1e-9) and np.isclose(e0, 1.0, atol=1e-9))

    # 10. A large subsequent swing is capped to max_turn_rate * dt in a
    #     single call, not applied instantly.
    n1, e1 = pg.update(1.0, 0.0, dt=0.01)   # target swings 90 deg to dead north
    ang0 = np.arctan2(e0, n0)
    ang1 = np.arctan2(e1, n1)
    step = abs(((ang1 - ang0) + np.pi) % (2 * np.pi) - np.pi)
    check("single-tick direction change is capped near max_turn_rate*dt",
          step <= np.deg2rad(60.0) * 0.01 + 1e-6)
    check("a 90deg target swing is NOT applied in one 0.01s tick",
          step < np.deg2rad(89.0))

    # 11. Sustained calls at a fixed new target converge to it over time.
    pg2 = PursuitGuidance(max_turn_rate=np.deg2rad(90.0))
    pg2.update(1.0, 0.0, dt=0.001)   # seed pointing north
    n, e = 1.0, 0.0
    for _ in range(500):   # 500*0.01 = 5.0s, comfortably enough at 90 deg/s for a 90 deg turn
        n, e = pg2.update(0.0, 1.0, dt=0.01)   # target: dead east
    check("sustained updates converge to a new target direction",
          np.isclose(n, 0.0, atol=1e-3) and np.isclose(e, 1.0, atol=1e-3))

    # 12. Output stays a unit vector throughout.
    check("converged direction is still unit-norm", np.isclose(n**2 + e**2, 1.0, atol=1e-6))

    # 13. reset() drops the cached direction, so the next call snaps
    #     immediately again (matches first-call/cold-start behaviour).
    pg2.reset()
    n2, e2 = pg2.update(1.0, 0.0, dt=0.01)   # target swings back to north
    check("reset() makes the next call snap directly to the new target",
          np.isclose(n2, 1.0, atol=1e-9) and np.isclose(e2, 0.0, atol=1e-9))

    # 14. Degenerate (zero-norm) input passes through unchanged rather than
    #     raising (e.g. a division by zero on normalization).
    pg3 = PursuitGuidance(max_turn_rate=np.deg2rad(60.0))
    pg3.update(1.0, 0.0, dt=0.01)
    try:
        n3, e3 = pg3.update(0.0, 0.0, dt=0.01)
        check("zero-norm input does not raise and passes through unchanged",
              n3 == 0.0 and e3 == 0.0)
    except Exception as e:
        check(f"zero-norm input does not raise (raised {e!r})", False)

    # 15. RecoveryGuard: a fresh, locked, gate_id-matching pose reporting a
    #     range far beyond commit_dist_m + sanity_margin_m while committed
    #     is true -> mismatch -> active.
    def _pose(fresh, range_m, has_pnp=True):
        return PoseEstimate(state=LockState.LOCKED, fresh=fresh,
                             position_ned=(np.zeros(3) if has_pnp else None),
                             range_m=range_m)

    rg = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    active = rg.update(raw_committed=True, pose=_pose(True, 30.0), gate_id_match=True,
                        commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0)
    check("large range mismatch while committed triggers recovery", active is True and rg.active)

    # 16. No mismatch: range within commit_dist_m + margin is ordinary PnP
    #     noise, not evidence committed is wrong.
    rg2 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    active2 = rg2.update(raw_committed=True, pose=_pose(True, 4.0), gate_id_match=True,
                          commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0)
    check("range within commit_dist_m+margin does not trigger recovery", active2 is False)

    # 17. Not committed at all -> nothing to veto, regardless of range.
    rg3 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    active3 = rg3.update(raw_committed=False, pose=_pose(True, 30.0), gate_id_match=True,
                          commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0)
    check("not committed -> no recovery regardless of range", active3 is False)

    # 18. Stale (not fresh) or wrong-gate pose can't be trusted to veto a
    #     commit — no entry.
    rg4 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    check("stale pose does not trigger recovery",
          rg4.update(raw_committed=True, pose=_pose(False, 30.0), gate_id_match=True,
                      commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0) is False)
    rg5 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    check("gate_id mismatch does not trigger recovery",
          rg5.update(raw_committed=True, pose=_pose(True, 30.0), gate_id_match=False,
                      commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0) is False)

    # 19. Latching: once active, stays active on a later tick even with a
    #     stale pose (fresh vision data can be sparse exactly when this
    #     matters) as long as neither exit condition is met yet.
    rg6 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=15.0)
    rg6.update(raw_committed=True, pose=_pose(True, 30.0), gate_id_match=True,
               commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0)
    still_active = rg6.update(raw_committed=True, pose=_pose(False, None, has_pnp=False),
                               gate_id_match=True, commit_dist_m=3.0,
                               dist_to_recovery_point=15.0, now=1.0)
    check("recovery latches active across a stale-pose tick", still_active is True)

    # 20. Exit via distance: reaching the recovery point ends recovery.
    ended_by_distance = rg6.update(raw_committed=True, pose=_pose(False, None, has_pnp=False),
                                    gate_id_match=True, commit_dist_m=3.0,
                                    dist_to_recovery_point=1.0, now=2.0)
    check("recovery exits once within exit_dist_m of the recovery point",
          ended_by_distance is False and rg6.active is False)

    # 21. Exit via timeout even while still far from the recovery point.
    rg7 = RecoveryGuard(sanity_margin_m=5.0, exit_dist_m=2.5, timeout_s=5.0)
    rg7.update(raw_committed=True, pose=_pose(True, 30.0), gate_id_match=True,
               commit_dist_m=3.0, dist_to_recovery_point=20.0, now=0.0)
    timed_out = rg7.update(raw_committed=True, pose=_pose(False, None, has_pnp=False),
                            gate_id_match=True, commit_dist_m=3.0,
                            dist_to_recovery_point=18.0, now=10.0)
    check("recovery exits via timeout even while still far from the recovery point",
          timed_out is False and rg7.active is False)

    # 22. path_convergence_weight: exactly on the line -> full weight,
    #     regardless of how far along the (infinite) line the point is.
    r0 = np.array([0.0, 0.0, 0.0])
    ea = np.array([1.0, 0.0, 0.0])   # line running along North
    check("on the line at the origin -> weight 1.0",
          np.isclose(path_convergence_weight(np.array([0.0, 0.0, 0.0]), r0, ea, 4.0), 1.0))
    check("on the line far along it -> still weight 1.0",
          np.isclose(path_convergence_weight(np.array([50.0, 0.0, 0.0]), r0, ea, 4.0), 1.0))
    check("on the line behind the segment start -> still weight 1.0 (infinite line, not clamped)",
          np.isclose(path_convergence_weight(np.array([-20.0, 0.0, 0.0]), r0, ea, 4.0), 1.0))

    # 23. At or beyond recover_dist_m of cross-track distance -> weight 0.
    check("cross-track == recover_dist_m -> weight 0.0",
          np.isclose(path_convergence_weight(np.array([10.0, 4.0, 0.0]), r0, ea, 4.0), 0.0))
    check("cross-track beyond recover_dist_m -> clamped to weight 0.0",
          path_convergence_weight(np.array([10.0, 9.0, 0.0]), r0, ea, 4.0) == 0.0)

    # 24. Linear ramp in between, and only the PERPENDICULAR component
    #     matters — along-track offset doesn't affect the weight.
    check("halfway to recover_dist_m -> weight 0.5",
          np.isclose(path_convergence_weight(np.array([0.0, 2.0, 0.0]), r0, ea, 4.0), 0.5))
    check("same cross-track, different along-track -> same weight",
          np.isclose(path_convergence_weight(np.array([37.0, 2.0, 0.0]), r0, ea, 4.0), 0.5))

    # 25. recover_dist_m <= 0 is a degenerate "no recovery gating" config
    #     -> always full weight rather than dividing by zero.
    check("recover_dist_m == 0 -> always weight 1.0",
          path_convergence_weight(np.array([0.0, 99.0, 0.0]), r0, ea, 0.0) == 1.0)

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
