"""
carrot_tracker.py

Path-following carrot (look-ahead) tracker with smooth waypoint transitions.

Algorithm
---------
For the current segment  r0 → r1  (unit tangent ea):

  1.  Project drone position onto the segment:
        lambda_L = (pos - r0) · ea          [m along segment]

  2.  Place the carrot c metres ahead of the foot:
        rC = r0 + (lambda_L + c) * ea        (before the waypoint)

  3.  When the carrot overshoots waypoint WP by d_over metres, blend
      the current tangent ea toward the NEXT segment tangent ea_next:
        alpha    = 1 - exp(-d_over / tau)    [0 at WP, →1 far past]
        ea_blend = normalise((1-alpha)*ea + alpha*ea_next)

  4.  Reference velocity:    v_ned_ref = v_ref * ea_blend  (NED frame)
      Reference yaw:         psi_ref   = atan2(ea_blend_E, ea_blend_N)

  5.  Advance the waypoint index when the carrot passes the segment
      end (lambda_L + c >= seg_length) OR the drone comes within c
      metres of the waypoint (buffer for off-path approaches).

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
  print(f"wp {tracker.wp}/{tracker.n_waypoints-1}  "
        f"alpha={tracker.blend_alpha:.2f}")
"""

import numpy as np


class CarrotTracker:

    def __init__(self, param):
        self.waypoints      = param['waypoints']          # (M, 3) NED
        self.v_ref          = float(param['v_ref'])
        self.c              = float(param['lookahead_gain']) * self.v_ref
        self.tau            = float(param['tau'])
        self.n_waypoints    = len(self.waypoints)

        self.wp             = 1        # index of current target waypoint
        self.blend_alpha    = 0.0      # exposed for logging / debugging
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

        ea, seg_length = self._unit(r1 - r0)

        # --- project drone onto segment ------------------------------------
        lambda_L = float(np.dot(pos - r0, ea))

        # --- carrot overshoot past current waypoint -----------------------
        d_over = max(0.0, lambda_L + self.c - seg_length)

        # --- smooth directional blend with next segment -------------------
        if d_over > 0.0 and WP + 1 < self.n_waypoints:
            ea_next, _ = self._unit(self.waypoints[WP + 1] - r1)
            alpha       = 1.0 - np.exp(-d_over / self.tau)
            ea_blend, _ = self._unit((1.0 - alpha) * ea + alpha * ea_next)
        else:
            alpha    = 0.0
            ea_blend = ea

        self.blend_alpha = alpha

        # Waypoint advance is driven exclusively by the COLLISION signal from the
        # sim (controller calls self.wp += 1 when gate_passed fires).  Automatic
        # distance-based advance is disabled: the locked gate NED must remain the
        # target until the drone physically passes through it.

        # --- carrot position: foot on segment + c metres in blended direction --
        # Velocity points from the drone's current 3-D position toward the carrot,
        # which gives automatic cross-track correction when the drone drifts off the
        # path — the lateral displacement pulls the velocity reference sideways.
        foot         = r0 + lambda_L * ea
        rC           = foot + self.c * ea_blend
        self.carrot_pos = rC

        to_carrot, _ = self._unit(rC - pos)
        v_ned_ref     = self.v_ref * to_carrot
        psi_ref       = self._yaw(to_carrot)
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
