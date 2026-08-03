"""
test_rotations.py — regression tests for rotations.py against the exact
formulas it replaces, plus round-trip/consistency checks. Run before and
after migrating any call site: `python -m tests.test_rotations` from the
repo root.
"""

import numpy as np
from flight_model import rotations as rot


def _old_rot_from_quat(q):
    """controller.py's original _rot_from_quat, byte-for-byte."""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy+qw*qz),   2*(qx*qz-qw*qy)],
        [  2*(qx*qy-qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz+qw*qx)],
        [  2*(qx*qz+qw*qy),   2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


def _old_rot_b2n(q):
    """ekf.py's original _rot_b2n, byte-for-byte."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),  2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),  2*(qy*qz + qw*qx),  1 - 2*(qx*qx + qy*qy)],
    ])


def _old_quat_to_euler(q):
    """controller.py's original quat_to_euler, byte-for-byte."""
    qw,qx,qy,qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return roll, pitch, yaw


def _old_yaw_of_quat(q):
    """ekf.py's original _yaw_of_quat, byte-for-byte."""
    qw, qx, qy, qz = q
    f = 2.0 * (qw * qz + qx * qy)
    g = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(f, g))


def _old_set_attitude_quat(phi, theta, psi):
    """ekf.py's original set_attitude rebuild step, byte-for-byte."""
    cp, sp = np.cos(phi/2),   np.sin(phi/2)
    ct, st = np.cos(theta/2), np.sin(theta/2)
    cy, sy = np.cos(psi/2),   np.sin(psi/2)
    return np.array([
        cy*ct*cp + sy*st*sp,
        cy*ct*sp - sy*st*cp,
        cy*st*cp + sy*ct*sp,
        sy*ct*cp - cy*st*sp,
    ])


def _old_R_cam2body(tilt_deg):
    """controller.py's / vision_rx.py's original _R_cam2body, byte-for-byte."""
    tilt = np.radians(float(tilt_deg))
    st, ct = np.sin(tilt), np.cos(tilt)
    return np.array([[0, st, ct], [1, 0, 0], [0, ct, -st]], dtype=float)


def _old_rot_dist(Ra, Rb):
    """vision_rx.py's original _pnp_gate continuity-fallback distance."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def main():
    rng = np.random.default_rng(42)
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        status = "OK" if cond else "FAIL"
        if not cond:
            n_fail += 1
        print(f"[{status}] {name}")

    # 1. quat_to_R_ned2body matches old _rot_from_quat exactly
    for i in range(200):
        q = random_quat(rng)
        a = rot.quat_to_R_ned2body(q)
        b = _old_rot_from_quat(q)
        if not np.allclose(a, b, atol=1e-12):
            check(f"quat_to_R_ned2body matches old (sample {i})", False)
            break
    else:
        check("quat_to_R_ned2body matches old _rot_from_quat (200 random q)", True)

    # 2. quat_to_R_body2ned matches old _rot_b2n exactly
    for i in range(200):
        q = random_quat(rng)
        a = rot.quat_to_R_body2ned(q)
        b = _old_rot_b2n(q)
        if not np.allclose(a, b, atol=1e-12):
            check(f"quat_to_R_body2ned matches old (sample {i})", False)
            break
    else:
        check("quat_to_R_body2ned matches old _rot_b2n (200 random q)", True)

    # 3. quat_to_R_body2ned(q) == quat_to_R_ned2body(q).T
    for i in range(200):
        q = random_quat(rng)
        A = rot.quat_to_R_ned2body(q)
        B = rot.quat_to_R_body2ned(q)
        if not np.allclose(A.T, B, atol=1e-12):
            check(f"body2ned == ned2body.T (sample {i})", False)
            break
    else:
        check("quat_to_R_body2ned(q) == quat_to_R_ned2body(q).T (200 random q)", True)

    # 4. quat_to_euler matches old formula exactly
    for i in range(200):
        q = random_quat(rng)
        a = rot.quat_to_euler(q)
        b = _old_quat_to_euler(q)
        if not np.allclose(a, b, atol=1e-12):
            check(f"quat_to_euler matches old (sample {i})", False)
            break
    else:
        check("quat_to_euler matches old controller.py formula (200 random q)", True)

    # 5. quat_to_yaw matches old _yaw_of_quat exactly
    for i in range(200):
        q = random_quat(rng)
        a = rot.quat_to_yaw(q)
        b = _old_yaw_of_quat(q)
        if not np.isclose(a, b, atol=1e-12):
            check(f"quat_to_yaw matches old (sample {i})", False)
            break
    else:
        check("quat_to_yaw matches old ekf.py _yaw_of_quat (200 random q)", True)
    # and matches quat_to_euler's own yaw component
    for i in range(50):
        q = random_quat(rng)
        _, _, yaw_full = rot.quat_to_euler(q)
        yaw_only = rot.quat_to_yaw(q)
        if not np.isclose(yaw_full, yaw_only, atol=1e-12):
            check(f"quat_to_yaw == quat_to_euler(q)[2] (sample {i})", False)
            break
    else:
        check("quat_to_yaw == quat_to_euler(q)[2] (50 random q)", True)

    # 6. euler_to_quat matches old set_attitude rebuild exactly
    for i in range(200):
        r, p, y = rng.uniform(-np.pi, np.pi, size=3)
        a = rot.euler_to_quat(r, p, y)
        b = _old_set_attitude_quat(r, p, y)
        if not np.allclose(a, b, atol=1e-12):
            check(f"euler_to_quat matches old (sample {i})", False)
            break
    else:
        check("euler_to_quat matches old ekf.py set_attitude (200 random angles)", True)

    # 7. euler_to_quat / quat_to_euler are exact inverses (round-trip),
    #    away from the pitch=+-90deg gimbal-lock singularity.
    max_err = 0.0
    for i in range(500):
        r = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-np.pi/2 + 0.05, np.pi/2 - 0.05)
        y = rng.uniform(-np.pi, np.pi)
        q = rot.euler_to_quat(r, p, y)
        r2, p2, y2 = rot.quat_to_euler(q)
        max_err = max(max_err, abs(r-r2), abs(p-p2), abs(((y-y2+np.pi) % (2*np.pi)) - np.pi))
    check(f"euler_to_quat/quat_to_euler round-trip (max err {max_err:.2e} rad, away from gimbal lock)",
          max_err < 1e-9)

    # 8. euler_from_R_body2ned matches quat_to_euler for R = quat_to_R_body2ned(euler_to_quat(...))
    #    -- this closes the gap flagged during the codebase inventory: _attitude_from_pnp's
    #    matrix-based extraction was claimed-but-unverified equivalent to the quaternion formula.
    max_err = 0.0
    for i in range(500):
        r = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-np.pi/2 + 0.05, np.pi/2 - 0.05)
        y = rng.uniform(-np.pi, np.pi)
        q = rot.euler_to_quat(r, p, y)
        R = rot.quat_to_R_body2ned(q)
        r2, p2, y2 = rot.euler_from_R_body2ned(R)
        max_err = max(max_err, abs(r-r2), abs(p-p2), abs(((y-y2+np.pi) % (2*np.pi)) - np.pi))
    check(f"euler_from_R_body2ned matches quat_to_euler chain (max err {max_err:.2e} rad)",
          max_err < 1e-9)

    # 8b. gt_yaw_flip: identity when gt_mode=False, pure negation (wrapped) when True,
    #     and involutive (applying it twice returns the original angle).
    for i in range(200):
        psi = rng.uniform(-np.pi, np.pi)
        if not np.isclose(rot.gt_yaw_flip(psi, False), psi, atol=1e-12):
            check(f"gt_yaw_flip identity when gt_mode=False (sample {i})", False)
            break
    else:
        check("gt_yaw_flip is identity when gt_mode=False (200 random psi)", True)
    for i in range(200):
        psi = rng.uniform(-np.pi, np.pi)
        flipped = rot.gt_yaw_flip(psi, True)
        back = rot.gt_yaw_flip(flipped, True)
        expected = ((-psi) + np.pi) % (2*np.pi) - np.pi
        if not (np.isclose(flipped, expected, atol=1e-9) and np.isclose(back, psi, atol=1e-9)):
            check(f"gt_yaw_flip negates and is involutive (sample {i})", False)
            break
    else:
        check("gt_yaw_flip negates (wrapped) and undoes itself (200 random psi)", True)

    # 8c. gt_correct_quat: identity when gt_mode=False; when True, flips ONLY yaw
    #     (roll/pitch preserved) and round-trips through quat_to_euler correctly.
    for i in range(200):
        q = random_quat(rng)
        if not np.allclose(rot.gt_correct_quat(q, False), q, atol=1e-12):
            check(f"gt_correct_quat identity when gt_mode=False (sample {i})", False)
            break
    else:
        check("gt_correct_quat is identity when gt_mode=False (200 random q)", True)
    for i in range(200):
        r = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-np.pi/2 + 0.05, np.pi/2 - 0.05)
        y = rng.uniform(-np.pi, np.pi)
        q = rot.euler_to_quat(r, p, y)
        q_fixed = rot.gt_correct_quat(q, True)
        r2, p2, y2 = rot.quat_to_euler(q_fixed)
        yaw_ok = np.isclose(((y2 - (-y) + np.pi) % (2*np.pi)) - np.pi, 0.0, atol=1e-9)
        if not (np.isclose(r2, r, atol=1e-9) and np.isclose(p2, p, atol=1e-9) and yaw_ok):
            check(f"gt_correct_quat flips only yaw (sample {i})", False)
            break
    else:
        check("gt_correct_quat flips only yaw, preserves roll/pitch (200 random angles)", True)

    # 9. cam_to_body_matrix matches old _R_cam2body exactly
    for tilt in [0.0, 5.0, 20.0, 45.0, -10.0]:
        a = rot.cam_to_body_matrix(tilt)
        b = _old_R_cam2body(tilt)
        if not np.allclose(a, b, atol=1e-12):
            check(f"cam_to_body_matrix matches old (tilt={tilt})", False)
            break
    else:
        check("cam_to_body_matrix matches old controller.py/vision_rx.py formula", True)

    # 10. rotation_angle_distance matches old _rot_dist exactly, and is 0 for identical R
    for i in range(200):
        qa, qb = random_quat(rng), random_quat(rng)
        Ra, Rb = rot.quat_to_R_body2ned(qa), rot.quat_to_R_body2ned(qb)
        a = rot.rotation_angle_distance(Ra, Rb)
        b = _old_rot_dist(Ra, Rb)
        if not np.isclose(a, b, atol=1e-9):
            check(f"rotation_angle_distance matches old (sample {i})", False)
            break
    else:
        check("rotation_angle_distance matches old vision_rx.py _rot_dist (200 random pairs)", True)
    q = random_quat(rng)
    R = rot.quat_to_R_body2ned(q)
    _d_self = rot.rotation_angle_distance(R, R)
    check(f"rotation_angle_distance(R,R) == 0 (got {_d_self!r})", np.isclose(_d_self, 0.0, atol=1e-6))

    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    return n_fail == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
