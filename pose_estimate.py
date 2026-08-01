"""
pose_estimate.py — the Layer1 (perception, vision_rx.py) -> Layer2 (reference
shaping, controller.py) data contract.

Part of the gate-transition/vision-reacquisition rewrite (see the plan doc
for the full redesign rationale). Replaces two previously-mismatched
`gate_detection` dict shapes (one from a debug-only early-return path, one
from the main path — different key sets) with one dataclass every producer
and consumer agrees on.

Deliberately dependency-free (numpy/dataclasses/enum only, no cv2, no
threading) so it can be imported and unit-tested without pulling in
vision_rx.py's live camera/model machinery.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class LockState(Enum):
    """Perception's own confidence in the current gate lock — the single
    source of truth Layer 2's mode state machine (vision_mode.py) derives
    from, replacing today's independently-re-derived vis_ctrl_mode/
    pnp_locked/_agi_matches_wp trio."""
    UNLOCKED  = "UNLOCKED"
    ACQUIRING = "ACQUIRING"
    LOCKED    = "LOCKED"


@dataclass
class PoseEstimate:
    """One tick's perception output. Pose fields (position/velocity/yaw/
    roll/pitch/gate_bearing_body_m/range_m/centre_px) hold the last-ACCEPTED
    measurement and persist (with fresh=False) across tolerated-miss ticks
    while state != UNLOCKED — this is the relocated replacement for
    controller.py's old _vis_tvec_hold cache, now owned by the layer that
    actually knows whether a value is still trustworthy. All fields are
    None when there has never been an accepted measurement (state ==
    UNLOCKED with no history, or after a reset).
    """
    state:   LockState
    fresh:   bool                          # True iff THIS tick produced a new accepted measurement

    # Diagnostics — not load-bearing for control (Layer 2's mode machine
    # needs only `state`; see vision_mode.py).
    t_capture: Optional[float] = None      # wall time the frame was CAPTURED (not when inference finished)
    frame_id:  Optional[int]   = None
    gate_id:   Optional[int]   = None      # active_gate_index this estimate was resolved against

    # Pose. position_ned/velocity_ned are perception's own absolute-position
    # belief; gate_bearing_body_m is kept as a SEPARATE field (not derived
    # back out of position_ned) because consumers that re-anchor it onto
    # this control tick's freshest EKF position/attitude (waypoint-live-
    # update, pursuit override) need the raw body-frame vector, not
    # perception's own (necessarily slightly older) position snapshot.
    position_ned:         Optional[np.ndarray] = None   # (3,) world NED
    velocity_ned:         Optional[np.ndarray] = None   # (3,) world NED
    yaw:                  Optional[float]      = None   # rad, standard NED, gt-corrected + smoothed
    roll:                 Optional[float]      = None   # rad
    pitch:                Optional[float]      = None   # rad
    gate_bearing_body_m:  Optional[np.ndarray] = None   # (3,) body-FRD drone->gate vector AT CAPTURE TIME
    range_m:              Optional[float]      = None   # camera-frame depth (tvec_cam[2]) — the
                                                          # same "range" convention used throughout
                                                          # vision_rx.py's lock/spike/gate-id logic,
                                                          # NOT norm(gate_bearing_body_m)
    centre_px:            Optional[np.ndarray] = None   # (2,) pixel centroid; set for ANY detection (PnP or centroid-only)

    @property
    def has_pnp(self) -> bool:
        return self.position_ned is not None

    @staticmethod
    def blind() -> "PoseEstimate":
        """The estimate published before any lock has ever been acquired,
        or immediately after a full reset (agi change, dropped lock)."""
        return PoseEstimate(state=LockState.UNLOCKED, fresh=False)
