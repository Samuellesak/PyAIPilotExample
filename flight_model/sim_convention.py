"""
Sim ↔ FRD body-frame adapter.

The sim's HIGHRES_IMU reports gyro with all three axes sign-flipped vs standard
FRD (Forward-Right-Down).  All conversions between sim sensor data and the FRD
convention used internally by the EKF live here so they are changed in one place.

Body frame (FRD):
    x = forward   y = right   z = down
    φ > 0 = right wing down   θ < 0 = nose down   ψ: 0 = North, CW positive

World frame: NED (North-East-Down).

Sim ATTITUDE message specifics:
    roll      — standard, no correction needed
    pitch     — sign-inverted vs standard ZYX; negated in mavlink_rx.on_attitude()
    yaw       — sign-inverted vs standard ZYX; NOT negated (see mavlink_rx note)
    pitchspeed — sign-inverted; negated in mavlink_rx.on_attitude()

Sim HIGHRES_IMU specifics:
    gyro [xgyro, ygyro, zgyro] — all three axes sign-flipped vs FRD → negate all three
    acc  [xacc,  yacc,  zacc ] — convention matches FRD specific force (no correction)
"""

import numpy as np


def sim_to_frd_gyro(gx: float, gy: float, gz: float) -> np.ndarray:
    """Convert raw sim gyro [rad/s] to FRD body-frame angular rates."""
    return np.array([-gx, -gy, -gz])
