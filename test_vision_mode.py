"""
test_vision_mode.py — regression tests for vision_mode.py's Mode/
VisionModeTracker/VerticalAssist. Run: `python test_vision_mode.py`.
"""

import numpy as np

from pose_estimate import LockState
from vision_mode import Mode, VisionModeTracker, VerticalAssist


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

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
