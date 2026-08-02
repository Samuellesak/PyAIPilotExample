"""
test_gate_lock.py — regression tests for gate_lock.py's UNLOCKED/ACQUIRING/
LOCKED state machine against the exact invariants of the code it replaces.
Run: `python test_gate_lock.py`.
"""

from gate_lock import GateLock
from pose_estimate import LockState


def main():
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        status = "OK" if cond else "FAIL"
        if not cond:
            n_fail += 1
        print(f"[{status}] {name}")

    def new_lock(lock_frames=5, miss_max=30, spike_tol_m=5.0, id_tol_m=10.0, id_recheck_period_s=1.5):
        return GateLock(lock_frames=lock_frames, miss_max=miss_max, spike_tol_m=spike_tol_m,
                         id_tol_m=id_tol_m, id_recheck_period_s=id_recheck_period_s)

    # 1. Acquisition: exactly lock_frames identity-plausible frames lock it,
    #    non-consecutively (an isolated miss must not reset hit_count).
    gl = new_lock(lock_frames=5)
    t = 0.0
    for _ in range(2):
        ok = gl.on_frame(20.0, 20.0, t); t += 0.03
        check("acquiring frame accepted", ok)
    ok_miss = gl.on_frame(None, None, t); t += 0.03   # isolated miss mid-acquisition
    check("isolated miss during ACQUIRING doesn't reject", ok_miss is False)
    check("isolated miss during ACQUIRING doesn't reset hit_count", gl.hit_count == 2)
    check("no-measurement reason is distinguishable from identity/spike", gl.last_reject_reason == 'no_measurement')
    for _ in range(3):
        gl.on_frame(20.0, 20.0, t); t += 0.03
    check("5th accepted frame (non-consecutive) reaches LOCKED", gl.state == LockState.LOCKED)

    # 2. Once LOCKED, a distance spike is rejected but doesn't drop the lock
    #    by itself (folded into the ordinary miss budget).
    gl2 = new_lock(lock_frames=3, miss_max=30, spike_tol_m=5.0)
    tt = 0.0
    for _ in range(3):
        gl2.on_frame(20.0, 20.0, tt); tt += 0.03
    check("locked after 3 frames", gl2.state == LockState.LOCKED)
    ok_spike = gl2.on_frame(30.0, None, tt)  # +10m jump, way past spike_tol_m=5.0
    check("distance spike rejected", ok_spike is False)
    check("single spike doesn't drop the lock", gl2.state == LockState.LOCKED)
    check("spike rejection reason is distinguishable", gl2.last_reject_reason == 'spike')

    # 3. miss_max consecutive misses release the lock entirely, resetting
    #    hit_count too (must re-acquire from scratch afterward).
    gl3 = new_lock(lock_frames=3, miss_max=4)
    ttt = 0.0
    for _ in range(3):
        gl3.on_frame(20.0, 20.0, ttt); ttt += 0.03
    check("locked (setup for miss-budget test)", gl3.state == LockState.LOCKED)
    for _ in range(4):
        gl3.on_frame(None, None, ttt); ttt += 0.03
    check("miss_max consecutive misses releases the lock", gl3.state == LockState.UNLOCKED)
    check("release also resets hit_count", gl3.hit_count == 0)

    # 4. Gate-identity rejection while LOCKED drops the lock immediately,
    #    on the FIRST bad frame, not waiting for the miss budget.
    gl4 = new_lock(lock_frames=3, miss_max=30, id_tol_m=2.0, id_recheck_period_s=0.0)
    t4 = 0.0
    for _ in range(3):
        gl4.on_frame(20.0, 20.0, t4); t4 += 0.03
    check("locked (setup for identity test)", gl4.state == LockState.LOCKED)
    ok_id = gl4.on_frame(20.0, 50.0, t4)  # measured 20, expected 50 -> way past id_tol_m
    check("identity-implausible frame rejected", ok_id is False)
    check("identity rejection while LOCKED drops the lock immediately", gl4.state == LockState.UNLOCKED)
    check("identity rejection reason is distinguishable", gl4.last_reject_reason == 'identity')

    # 5. Identity check only reruns after id_recheck_period_s while LOCKED —
    #    a bad expected_range_m supplied before the recheck is due must NOT
    #    drop the lock (this is what the periodic-not-every-frame cadence
    #    exists to avoid: an ordinary-EKF-drift self-reinforcing lockout).
    gl5 = new_lock(lock_frames=3, miss_max=30, id_tol_m=2.0, id_recheck_period_s=1.5)
    t5 = 0.0
    for _ in range(3):
        gl5.on_frame(20.0, 20.0, t5); t5 += 0.03
    check("locked (setup for recheck-cadence test)", gl5.state == LockState.LOCKED)
    ok_early = gl5.on_frame(20.0, 50.0, t5 + 0.1)  # well within id_recheck_period_s of last check
    check("identity check not yet due -> bad expected_range_m ignored", ok_early is True)
    check("lock survives because recheck wasn't due yet", gl5.state == LockState.LOCKED)

    # 6. expected_range_m=None (no independent anchor yet) must never be
    #    treated as a pass that marks the recheck as done — it should stay
    #    "due" until a real anchor appears.
    gl6 = new_lock(lock_frames=3, miss_max=30, id_recheck_period_s=0.0)
    t6 = 0.0
    for _ in range(3):
        gl6.on_frame(20.0, None, t6); t6 += 0.03
    check("acquisition proceeds fine with no independent anchor at all", gl6.state == LockState.LOCKED)
    check("last_id_recheck_t never set when expected_range_m was always None",
          gl6.last_id_recheck_t is None)

    # 7. A single miss never resets hit_count during ACQUIRING even when
    #    interleaved repeatedly, as long as miss_max is never reached.
    gl7 = new_lock(lock_frames=4, miss_max=30)
    t7 = 0.0
    for _ in range(4):
        gl7.on_frame(20.0, None, t7); t7 += 0.03
        gl7.on_frame(None, None, t7); t7 += 0.03
    check("hit_count reached lock_frames despite interleaved misses", gl7.state == LockState.LOCKED)

    # 8. reset() zeroes everything relevant.
    gl8 = new_lock(lock_frames=2)
    gl8.on_frame(20.0, None, 0.0)
    gl8.on_frame(20.0, None, 0.03)
    check("locked before external reset", gl8.state == LockState.LOCKED)
    gl8.reset()
    check("external reset -> UNLOCKED", gl8.state == LockState.UNLOCKED)
    check("external reset zeroes hit_count", gl8.hit_count == 0)
    check("external reset zeroes miss_streak", gl8.miss_streak == 0)
    check("external reset clears last_good_range_m", gl8.last_good_range_m is None)

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
