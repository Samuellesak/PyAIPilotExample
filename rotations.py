"""
rotations.py — single source of truth for quaternion / Euler / rotation-matrix
conversions used across the flight-critical path (controller, EKF, vision).

Why this exists
----------------
Before this module, the same handful of formulas were hand-copied across
controller.py, dyn.py, ekf.py, and vision_rx.py (7 separate inline copies in
vision_rx.py alone). Two of those copies were transposes of each other with
the same function name (`_quat_to_R` in dyn.py vs vision_rx.py) — a real bug
was traced to vision_rx.py's `_attitude_from_pnp` reasoning about which
matrix direction it needed. Centralizing here means a formula only exists
once, and every call site names the direction explicitly so the dyn.py-vs-
vision_rx.py mix-up can't happen again.

Conventions
-----------
Quaternions are always [qw, qx, qy, qz].

Two rotation-matrix directions are used throughout this codebase, given
distinct explicit names rather than a single ambiguous `quat_to_R`:

    quat_to_R_ned2body(q):  v_body = R @ v_ned   ("NED-to-body")
    quat_to_R_body2ned(q):  v_ned  = R @ v_body  ("body-to-NED")

quat_to_R_body2ned(q) is exactly quat_to_R_ned2body(q).T for the same q —
computed directly (not via .T) at each call site for clarity and to avoid a
runtime transpose.

Euler angles are ZYX Tait-Bryan throughout: R_body2ned = Rz(yaw) @ Ry(pitch) @ Rx(roll).
quat_to_euler / euler_to_quat are exact inverses of each other (verified by
round-trip test in test_rotations.py).
"""

import numpy as np


# ── Quaternion -> rotation matrix ───────────────────────────────────────────

def quat_to_R_ned2body(q):
    """R: v_body = R @ v_NED.  q = [qw, qx, qy, qz].

    Reference: controller.py's original _rot_from_quat / dyn.py's original
    _quat_to_R (byte-identical formulas prior to centralization).
    """
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy+qw*qz),   2*(qx*qz-qw*qy)],
        [  2*(qx*qy-qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz+qw*qx)],
        [  2*(qx*qz+qw*qy),   2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


def quat_to_R_body2ned(q):
    """R: v_NED = R @ v_body.  q = [qw, qx, qy, qz].  == quat_to_R_ned2body(q).T

    Reference: ekf.py's original _rot_b2n / vision_rx.py's original _quat_to_R
    (byte-identical formulas prior to centralization).
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),  2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),  2*(qy*qz + qw*qx),  1 - 2*(qx*qx + qy*qy)],
    ])


# ── Quaternion <-> Euler (ZYX Tait-Bryan) ───────────────────────────────────

def quat_to_euler(q):
    """q = [qw,qx,qy,qz] -> (roll, pitch, yaw) [rad], ZYX convention.

    Reference: controller.py's original quat_to_euler (byte-identical formula
    prior to centralization; also matches ekf.py's set_yaw/set_roll
    extraction and the ~9 duplicate _quat_to_euler implementations across
    imu_ekf.py/log.py/flight_sysid*.py/gate_vision_calib.py).
    """
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return roll, pitch, yaw


def quat_to_yaw(q):
    """q = [qw,qx,qy,qz] -> yaw [rad] only. Avoids computing roll/pitch for
    hot paths (e.g. ekf.py's update_yaw Jacobian) that only need yaw.

    Reference: ekf.py's original _yaw_of_quat (byte-identical formula).
    """
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))


def euler_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) [rad], ZYX Tait-Bryan -> q = [qw,qx,qy,qz].
    Exact inverse of quat_to_euler.

    Reference: ekf.py's original set_attitude (byte-identical formula; also
    matches set_yaw/set_roll's rebuild step, update_attitude's q_synth,
    lqi.py's euler_to_quat, and mavlink_rx.py's on_attitude quaternion build
    — see mavlink_rx.py for a deliberate, documented yaw-sign exception on
    the raw sim ATTITUDE message that is NOT reproduced here).
    """
    cp, sp = np.cos(roll/2),  np.sin(roll/2)
    ct, st = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    qw = cy*ct*cp + sy*st*sp
    qx = cy*ct*sp - sy*st*cp
    qy = cy*st*cp + sy*ct*sp
    qz = sy*ct*cp - cy*st*sp
    return np.array([qw, qx, qy, qz])


def euler_from_R_body2ned(R):
    """Extract (roll, pitch, yaw) [rad] directly from a body-to-NED rotation
    matrix (R with v_ned = R @ v_body), without a matrix->quaternion step.

    Used where the caller already has R_body2ned from composing other
    rotations (e.g. vision_rx.py's PnP-derived attitude chain) rather than
    from a native quaternion. Verified equivalent to
    quat_to_euler(q) for R = quat_to_R_body2ned(q) by round-trip test in
    test_rotations.py.
    """
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


# ── GT-mode yaw convention ──────────────────────────────────────────────────
#
# The sim's raw ATTITUDE message yaw is deliberately left in a different sign
# convention than everything else in this codebase (see mavlink_rx.py's
# on_attitude: pitch is corrected, yaw is not — "GT mode relies on the
# inverted sign"). So whenever ground_truth_mode is on, mav_state['quat']'s
# YAW component (and therefore anything read via quat_to_euler(quat)[2], or
# any rotation matrix built from that quaternion) is psi_gt = -psi_NED, not
# standard NED. Roll/pitch in that same quaternion ARE already standard.
#
# This bit us twice in one investigation: once comparing psi_meas against a
# standard-convention psi_ref (controller.py's outer loop — confirmed live,
# the true heading converged to a left/right mirror of the intended one
# while the logged error still read ~0), and again in vision_rx.py, which
# builds rotation matrices straight from mav_state['quat'] to rotate
# body-frame bearing/position vectors into NED (drone position from PnP, the
# IPPE position-consistency tie-break, the attitude-ambiguity disambiguation
# reference) with no GT awareness at all. Use gt_yaw_flip/gt_correct_quat at
# any such site instead of another ad hoc negation.

def gt_yaw_flip(psi, gt_mode):
    """Flip a scalar yaw angle [rad] between standard NED and the sim's
    GT-mode ATTITUDE convention (see module note above). A pure negation, so
    the same call converts either direction. Identity when gt_mode is False.
    """
    if not gt_mode:
        return psi
    return float(((-psi) + np.pi) % (2 * np.pi) - np.pi)


def gt_correct_quat(q, gt_mode):
    """Correct a mav_state['quat']-style quaternion's yaw component for GT
    mode (see module note above) before using it to build a rotation matrix
    or extract Euler angles. Roll/pitch are already standard in GT mode —
    only yaw needs flipping. Identity when gt_mode is False.
    """
    if not gt_mode:
        return q
    roll, pitch, yaw = quat_to_euler(q)
    return euler_to_quat(roll, pitch, gt_yaw_flip(yaw, True))


# ── Rotation comparison ──────────────────────────────────────────────────

def rotation_angle_distance(Ra, Rb):
    """Angle [rad] of the rotation that takes Ra to Rb (both proper 3x3
    rotation matrices, same convention). 0 = identical, pi = opposite.

    Used to disambiguate between candidate rotations by comparing each
    against a reference (e.g. the EKF's current attitude) — the same pattern
    already used in vision_rx.py's _pnp_gate continuity fallback, and now
    also by _attitude_from_pnp's corner-ambiguity fix.
    """
    cos_theta = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


# ── Camera -> body ───────────────────────────────────────────────────────

def cam_to_body_matrix(tilt_deg):
    """R: v_body = R @ v_cam.  Camera frame (Z-forward, X-right, Y-down)
    rotated into body FRD by a single upward tilt about the camera's X axis.

    Reference: controller.py's and vision_rx.py's original (byte-identical)
    _R_cam2body construction, previously two independently-maintained copies.
    """
    tilt = np.radians(float(tilt_deg))
    st, ct = np.sin(tilt), np.cos(tilt)
    return np.array([
        [0, st,  ct],
        [1, 0,   0 ],
        [0, ct, -st],
    ], dtype=float)
