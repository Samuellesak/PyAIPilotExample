"""
carrot_tracker.py

Path-following carrot (look-ahead) tracker.

Algorithm
---------
For the current segment  r0 → r1  (unit tangent ea):

  1.  Project drone position onto the segment:
        lambda_L = (pos - r0) · ea          [m along segment]

  2.  Place the carrot c metres ahead of the foot:
        rC = foot + c * ea  (= r0 + (lambda_L + c) * ea)

  3.  Reference velocity:  v_ned_ref = v_ref * unit(rC - pos)  (NED frame)
      Reference yaw:       psi_ref   = atan2(ea_E, ea_N)

  4.  Waypoint advance is driven exclusively by the COLLISION signal from
      the sim (controller calls self.wp += 1 when gate_passed fires).

Note: velocities and positions are all in the NED frame.
      NED z is positive downward → negative altitude.

Usage
-----
  from dyn import load_params
  from carrot_tracker import CarrotTracker

  param   = load_params("params.yaml")
  tracker = CarrotTracker(param)

  # in control loop:
  v_ned_ref, psi_ref = tracker.update(pos_ned)
  print(f"wp {tracker.wp}/{tracker.n_waypoints-1}")
"""

import numpy as np


class CarrotTracker:

    def __init__(self, param):
        self.waypoints      = param['waypoints']          # (M, 3) NED
        self.v_ref          = float(param['v_ref'])
        self.c              = float(param['lookahead_gain']) * self.v_ref
        self.n_waypoints    = len(self.waypoints)

        self.wp             = 1        # index of current target waypoint
        self.carrot_pos     = np.zeros(3)  # carrot NED position, exposed for logging
        self.finished       = False

    # ------------------------------------------------------------------

    def update(self, pos_ned):
        """
        Compute NED reference velocity and yaw for one timestep.

        Parameters
        ----------
        pos_ned : array-like (3,)  current NED position [m]

        Returns
        -------
        v_ned_ref : np.ndarray (3,)  reference velocity in NED  [m/s]
        psi_ref   : float            reference yaw [rad]  0=North, +ve CW
        """
        pos = np.asarray(pos_ned, dtype=float)

        if self.finished:
            return np.zeros(3), self._last_psi

        WP = self.wp
        r0 = self.waypoints[WP - 1]
        r1 = self.waypoints[WP]

        ea, _ = self._unit(r1 - r0)

        # --- project drone onto segment; place carrot c metres ahead ------
        lambda_L  = float(np.dot(pos - r0, ea))
        foot      = r0 + lambda_L * ea
        # Carrot is always exactly c metres ahead of the drone's projection along the
        # current segment — no clamping to segment end.  When the drone is within c of
        # the gate, the carrot sits past the gate on the old segment direction, which is
        # fine: the drone still flies through the gate toward the carrot.  When past the
        # gate (gate_pass_dist_m delay window), the carrot keeps pointing forward along
        # the old segment so vN_ref stays at -v_ref rather than collapsing to ~0.
        # Clamping is no longer needed because gate_pass_dist_m controls wp advance timing.
        rC        = foot + self.c * ea
        self.carrot_pos = rC

        # Velocity points from drone to carrot → automatic cross-track correction.
        # Yaw uses path tangent (ea) so heading stays stable even with altitude error.
        to_carrot, _ = self._unit(rC - pos)
        v_ned_ref     = self.v_ref * to_carrot
        psi_ref       = self._yaw(ea)
        self._last_psi = psi_ref

        return v_ned_ref, psi_ref

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unit(v):
        """Return (unit_vector, length). Handles near-zero gracefully."""
        length = np.linalg.norm(v)
        if length < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return v / length, float(length)

    @staticmethod
    def _yaw(ea):
        """
        Yaw angle from a direction vector in NED.
        atan2(East, North) → heading measured CW from North.
        """
        return np.arctan2(ea[1], ea[0])
