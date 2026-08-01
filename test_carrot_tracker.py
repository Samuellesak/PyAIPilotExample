"""
test_carrot_tracker.py — regression tests for CarrotTracker.set_live_target(),
the one method added to this file by the gate-transition rewrite (Phase B).
Everything else in carrot_tracker.py is untouched and out of scope here — it
has been flight-validated extensively this session. Run:
`python test_carrot_tracker.py`.
"""

import numpy as np

from carrot_tracker import CarrotTracker


def _make_tracker():
    param = dict(
        waypoints=[[0.0, 0.0, 0.0], [10.0, 0.0, -1.0], [20.0, 5.0, -2.0]],
        v_ref=5.0, lookahead_gain=1.0,
    )
    return CarrotTracker(param)


def main():
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        status = "OK" if cond else "FAIL"
        if not cond:
            n_fail += 1
        print(f"[{status}] {name}")

    # 1. First call for a given idx seeds from the CURRENT waypoints[idx]
    #    value, not the fresh reading — no smoothing lag on a brand new
    #    target. With alpha close to 1 (tiny dt vs. tau), the very first
    #    call's output should stay very close to the ORIGINAL waypoint, not
    #    jump toward the fresh reading.
    t = _make_tracker()
    original = t.waypoints[1].copy()
    fresh = np.array([50.0, 50.0, 50.0])   # wildly different from original
    t.set_live_target(1, fresh, dt=0.001)   # tiny dt -> alpha ~= 1 (mostly old)
    check("first call seeds from the current waypoint, doesn't snap to the fresh reading",
          np.linalg.norm(t.waypoints[1] - original) < np.linalg.norm(t.waypoints[1] - fresh))

    # 2. Repeated calls at the SAME idx converge toward the fresh reading
    #    over time (ordinary EMA blending), not instantly.
    t2 = _make_tracker()
    target = np.array([10.5, 0.3, -1.1])   # close to the real waypoint
    for _ in range(500):
        t2.set_live_target(1, target, dt=0.01)
    check("sustained live updates converge close to the fresh target",
          np.linalg.norm(t2.waypoints[1] - target) < 0.01)

    # 3. Switching to a DIFFERENT idx re-seeds from THAT slot's current
    #    value (not carrying over the previous idx's filter state).
    t3 = _make_tracker()
    for _ in range(500):
        t3.set_live_target(1, np.array([10.5, 0.3, -1.1]), dt=0.01)
    original_wp2 = t3.waypoints[2].copy()
    fresh2 = np.array([99.0, 99.0, 99.0])
    t3.set_live_target(2, fresh2, dt=0.001)
    check("switching idx re-seeds from that slot's own current value",
          np.linalg.norm(t3.waypoints[2] - original_wp2) < np.linalg.norm(t3.waypoints[2] - fresh2))
    check("switching idx does not disturb the OTHER slot's already-converged value",
          np.linalg.norm(t3.waypoints[1] - np.array([10.5, 0.3, -1.1])) < 0.01)

    # 4. dt=None falls back to a sane nominal value instead of raising.
    t4 = _make_tracker()
    try:
        t4.set_live_target(1, np.array([11.0, 1.0, -1.0]))
        check("dt=None does not raise", True)
    except Exception as e:
        check(f"dt=None does not raise (raised {e!r})", False)

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
