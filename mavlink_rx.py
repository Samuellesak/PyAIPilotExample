import struct
import time
import threading

import numpy as np
from pymavlink import mavutil

from ekf import QuadEKF

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data, logger=None):
        self.mavlink_conn = mavlink_connection
        self.data         = data
        self.logger       = logger          # optional — may be None
        self.thread       = None
        self.is_running   = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

        # EKF — attitude and velocity estimation from HIGHRES_IMU
        self._ekf        = QuadEKF()
        self._last_imu_t = None
        # World-frame EKF position at the moment of the hover-entry reset.
        # Vision position updates (PnP) are in world NED; after the reset the EKF
        # tracks a LOCAL frame (origin = hover-entry point).  Subtract this offset
        # before every vision position update so the two frames are consistent.
        self._pos_offset_ned = np.zeros(3)

        # Gyro + accelerometer calibration during the WAIT phase (drone static on slope).
        # Gyro mean → gyro bias, injected into EKF at hover-entry reset.
        # Acc mean  → static tilt → pitch/roll, used to initialise attitude when the
        # sim does not send ATTITUDE messages (prevents the ~15° pitch spike at blip).
        self._wait_gyro_sum = np.zeros(3)
        self._wait_gyro_cnt = 0
        self._wait_acc_sum  = np.zeros(3)
        self._wait_acc_cnt  = 0
        self._hover_reset_done = False   # gates accumulation to WAIT phase only

        # Set initial EKF yaw from params.yaml (initial_yaw_deg).
        # Match this to the drone's heading in the sim (0=North, 90=East, 180=South).
        from dyn import load_params as _lp
        _p = _lp("params.yaml")
        _init_yaw_deg = float(_p.get('initial_yaw_deg', 0.0))
        self._init_yaw_rad = np.deg2rad(_init_yaw_deg)
        if _init_yaw_deg != 0.0:
            self._ekf.set_yaw(self._init_yaw_rad)
            print(f"[MAVLinkRX] EKF yaw pre-set from params: {_init_yaw_deg:.1f}°",
                  flush=True)

        self._hover_reset_t = None   # wall-clock time of hover-entry reset
        self._param         = _p     # full params dict (for blip_dur_sec)
        self._blip_frozen_q = None   # quaternion pinned during blip + motor-lag transient
        self._post_blip_att_reset_done = False  # one-shot backup attitude re-init after blip

        # Gate-collision debounce: each gate_id can only trigger gate_passed once
        # per debounce window to prevent rapid repeated COLLISION messages from
        # advancing the waypoint index multiple times on a single gate pass.
        self._gate_debounce     = {}    # gate_id -> last accepted wall time
        self._gate_debounce_sec = float(_p.get('gate_debounce_sec', 2.0))

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data, logger=None):
        rx = cls(mavlink_connection, data, logger)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon=False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        One bad handler never crashes the loop.
        """
        _msg_counts  = {}
        _diag_t      = time.time()
        _diag_period = 10.0   # print seen message types every 10 s

        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except Exception as exc:
                print(f"[MAVLinkRX] recv_match error: {type(exc).__name__}: {exc}",
                      flush=True)
                time.sleep(0.01)
                continue

            now = time.time()
            if now - _diag_t >= _diag_period:
                _diag_t = now
                top = dict(sorted(_msg_counts.items(), key=lambda kv: -kv[1])[:15])
                print(f"[MAVLinkRX] msg types seen: {top}", flush=True)

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()
            _msg_counts[msg_type] = _msg_counts.get(msg_type, 0) + 1

            if msg_type == "BAD_DATA":
                continue

            if self.logger:
                try:
                    self.logger.log_mavlink(msg)
                except Exception as exc:
                    print(f"[MAVLinkRX] log_mavlink error: {type(exc).__name__}: {exc}",
                          flush=True)

            try:
                if msg_type == "HEARTBEAT":
                    self.on_heartbeat(msg)
                elif msg_type == "TIMESYNC":
                    self.on_timesync(msg)
                elif msg_type == "HIGHRES_IMU":
                    self.on_highres_imu(msg)
                elif msg_type == "ENCAPSULATED_DATA":
                    self.on_encapsulated_data(msg)
                elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                    self.on_actuator_output_status(msg)
                elif msg_type == "COLLISION":
                    self.on_collision(msg)
                elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
                    track_data_transfer_id = msg.width
                    self.track_chunks[track_data_transfer_id] = {}
                    self.expected_num_track_chunks[track_data_transfer_id] = msg.packets
            except Exception as exc:
                print(f"[MAVLinkRX] handler error on {msg_type}: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_heartbeat(self, msg):
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

    def on_timesync(self, msg):
        # ts1 != 0 → this is a response; tc1 is our original send time [ns].
        if msg.ts1 != 0:
            rtt_ns = time.time_ns() - msg.tc1
            self.data['timesync_rtt_ms'] = rtt_ns / 1e6

    def on_highres_imu(self, msg):
        ax, ay, az = msg.xacc, msg.yacc, msg.zacc
        gx, gy, gz = msg.xgyro, msg.ygyro, msg.zgyro
        t_us        = msg.time_usec

        if self.logger:
            self.logger.log_imu(t_us, ax, ay, az, gx, gy, gz)

        # Hover-entry reset: full EKF reset with calibrated gyro bias and
        # slope attitude injected so acc→NED is correct from the first predict
        # step after lift-off.
        if self.data.get('reset_vel_flag'):
            self.data['reset_vel_flag'] = False
            self._hover_reset_done = True
            self._blip_frozen_q = None              # will be set after set_attitude below
            self._post_blip_att_reset_done = False  # arm backup reset for new launch
            # Save world-frame position (accurate because vision locked on during WAIT)
            self._pos_offset_ned = self._ekf.x[0:3].copy()
            self.data['pos_offset_ned'] = self._pos_offset_ned.copy()
            # Full clean reset — zeros pos, vel, attitude, all biases
            self._ekf.reset()
            # Re-inject calibrated gyro bias from static WAIT phase.
            # Clamp per-axis to ±0.05 rad/s (~3 deg/s): a stationary drone cannot
            # have a larger true bias.  Values above this indicate contamination
            # from motion/vibration during WAIT and must be rejected to prevent
            # the EKF from diverging immediately after the reset.
            _MAX_GYRO_BIAS = 0.05   # rad/s
            if self._wait_gyro_cnt > 50:
                _gyro_bias = self._wait_gyro_sum / self._wait_gyro_cnt
                if np.any(np.abs(_gyro_bias) > _MAX_GYRO_BIAS):
                    print(f"[MAVLinkRX] gyro bias {_gyro_bias} exceeds limit "
                          f"±{_MAX_GYRO_BIAS} rad/s -- zeroed", flush=True)
                    _gyro_bias = np.zeros(3)
                self._ekf.x[10:13] = _gyro_bias
                # Freeze all gyro bias states after WAIT calibration.
                # During forward flight the acc update mistakes linear acceleration
                # for gravity tilt and drives bgy (pitch-rate bias) wrong, causing
                # yaw to drift ~4°/s.  The WAIT-phase mean is the best estimate we
                # have; pin it by zeroing the entire bias row/column of P so no
                # subsequent measurement can perturb the calibrated values.
                self._ekf.P[10:13, :] = 0.0
                self._ekf.P[:, 10:13] = 0.0
            else:
                _gyro_bias = np.zeros(3)
            # Derive pitch/roll from the static accelerometer average collected
            # during WAIT (drone stationary on slope).  Eliminates the ~15° pitch
            # spike at blip onset that occurs when the EKF starts from identity.
            if self._wait_acc_cnt > 50:
                _acc_avg = self._wait_acc_sum / self._wait_acc_cnt
                _g       = 9.81
                _theta = float(np.arcsin(np.clip(_acc_avg[0] / _g, -1.0, 1.0)))
                _phi   = float(np.arcsin(
                    np.clip(_acc_avg[1] / (_g * max(np.cos(_theta), 0.1)), -1.0, 1.0)))
                self._ekf.set_attitude(_phi, _theta, self._init_yaw_rad)
                self._blip_frozen_q = self._ekf.x[6:10].copy()
                print(f"[MAVLinkRX] attitude from static acc: "
                      f"phi={np.rad2deg(_phi):.1f}°  theta={np.rad2deg(_theta):.1f}°  "
                      f"psi={np.rad2deg(self._init_yaw_rad):.1f}°  "
                      f"(n={self._wait_acc_cnt})", flush=True)
            else:
                self._ekf.set_yaw(self._init_yaw_rad)
                self._blip_frozen_q = self._ekf.x[6:10].copy()
            print(f"[MAVLinkRX] EKF full reset at hover entry; "
                  f"pos_offset={self._pos_offset_ned}  "
                  f"gyro_bias={_gyro_bias} (n={self._wait_gyro_cnt})", flush=True)
            self._hover_reset_t = time.time()

        # Sim reports all three gyro axes sign-flipped vs FRD convention:
        # +xgyro = roll LEFT (not roll-right), +ygyro = nose DOWN (not nose-up),
        # +zgyro = yaw LEFT (not yaw-right).  Negate all three so the EKF and
        # controller receive standard FRD rates where +r = yaw right (CW from above).
        gyro = np.array([-gx, -gy, -gz])
        acc  = np.array([ax, ay, az])

        # Accumulate gyro + accelerometer during WAIT phase.
        # Gyro mean → gyro bias; acc mean → static tilt for attitude init.
        if not self._hover_reset_done:
            self._wait_gyro_sum += gyro
            self._wait_gyro_cnt += 1
            self._wait_acc_sum  += acc
            self._wait_acc_cnt  += 1
            # Keep EKF position pinned to zero during WAIT.  Without this, position
            # integrates noise/drift from the previous flight.  At hover entry the
            # saved pos_offset_ned would then be non-zero, shifting the local
            # waypoints in the wrong direction.  Vision updates (if present) still
            # overwrite position this same tick, so we only zero when no vision
            # update is pending.
            if self.data.get('_vision_ekf_update') is None:
                self._ekf.x[0:3] = 0.0

        # Gate motor-vibration spikes from the EKF.
        # Physical max = full throttle / m = 4*57.53/5.49 ≈ 41.9 m/s² (4.3 g).
        # Samples above 42 m/s² are sim artefacts.  Clipping the magnitude but
        # keeping an arbitrary direction still corrupts velocity integration;
        # instead replace with the exact gravity vector in body frame so that
        # R_b2n @ acc_fallback + g_ned = 0 — velocity stays constant this step.
        _acc_norm = np.linalg.norm(acc)
        if _acc_norm > 42.0:
            qw, qx, qy, qz = self._ekf.x[6:10]
            _g = 9.81
            # R_n2b @ [0, 0, -g]: last column of R_n2b scaled by -g
            acc = np.array([
                2.0*(qx*qz - qw*qy) * (-_g),
                2.0*(qy*qz + qw*qx) * (-_g),
                (1.0 - 2.0*qx*qx - 2.0*qy*qy) * (-_g),
            ])

        self.data['_imu_rates'] = gyro
        self.data['imu_raw'] = {
            'ax': float(ax), 'ay': float(ay), 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz,
            't_us': t_us,
        }

        now = time.time()
        dt  = (now - self._last_imu_t) if self._last_imu_t is not None else 0.004
        dt  = float(np.clip(dt, 0.0005, 0.05))
        self._last_imu_t = now

        # EKF: predict from IMU, then correct with accelerometer and ZUPT.
        # ZUPT is only safe on the ground: a hovering drone has acc_norm ≈ g and
        # low gyro, satisfying the ZUPT gate even while airborne.  The controller
        # disables ZUPT at hover entry via the 'zupt_enabled' flag.
        self._ekf.predict(gyro, acc, dt)
        _acc_norm_upd = float(np.linalg.norm(acc))

        # Attitude freeze during blip + motor-lag transient.
        #
        # At 0.35 blip fraction the drone barely rotates physically.  The EKF
        # nevertheless accumulates ~24° of error because the motor surge drives
        # acc_norm away from g, which (a) contaminates the acc update during the
        # motor ramp and (b) causes the Kalman gain to be near-zero so post-blip
        # acc updates cannot recover the error.
        #
        # Fix: pin the quaternion to the hover-reset value (derived from static slope
        # acc in WAIT) for blip_dur + 6×tau_motor = 0.45 s.  The motor returns to
        # hover thrust after 6 time constants, so the acc update that fires after the
        # freeze sees a clean [0,0,-g] signal and can confirm (not corrupt) the attitude.
        # Position and velocity still integrate freely — only the quaternion is held.
        _freeze_active = (
            self._blip_frozen_q is not None
            and self._hover_reset_t is not None
            and now - self._hover_reset_t
                < self._param.get('blip_dur_sec', 0.15) + 0.30
        )
        if _freeze_active:
            self._ekf.x[6:10] = self._blip_frozen_q
            _qn = float(np.linalg.norm(self._ekf.x[6:10]))
            if _qn > 1e-9:
                self._ekf.x[6:10] /= _qn

        # Accelerometer attitude update.
        # Allowed during WAIT (drone stationary) and for 0.50 s after hover reset so
        # the acc can confirm the attitude once the motor returns to hover thrust.
        # Blocked during the freeze window (quaternion is pinned; wrong-magnitude acc
        # during motor ramp would perturb P to no benefit).
        # Blocked during forward flight: acc cannot distinguish tilt from linear accel.
        _blip_window = (
            not self._hover_reset_done
            or (self._hover_reset_t is not None
                and now - self._hover_reset_t
                    < self._param.get('blip_dur_sec', 0.15)
                      + self._param.get('blip_acc_extra_sec', 0.50))
        )
        if not _freeze_active and abs(_acc_norm_upd - 9.81) < 2.0 and _blip_window:
            acc_applied, acc_innov = self._ekf.update_accel(acc)
        else:
            acc_applied, acc_innov = False, 0.0
        if self.data.get('zupt_enabled', True):
            zupt_applied, zupt_innov = self._ekf.update_zupt(gyro, acc)
        else:
            zupt_applied, zupt_innov = False, 0.0

        # Backup one-shot attitude re-init once the freeze window ends.
        # The freeze (above) should have kept the EKF at the correct attitude; this
        # fires immediately after and sets theta from acc=[0,0,-g] → ~0°, which is
        # a no-op when the drone is already level and a small correction otherwise.
        if (self._hover_reset_done
                and not self._post_blip_att_reset_done
                and not _freeze_active
                and self._hover_reset_t is not None
                and now - self._hover_reset_t
                    > self._param.get('blip_dur_sec', 0.15) + 0.30):
            if abs(_acc_norm_upd - 9.81) < 1.5:
                _g      = 9.81
                _ax_pb  = float(acc[0])
                _ay_pb  = float(acc[1])
                _th_pb  = float(np.arcsin(np.clip(_ax_pb / _g, -1.0, 1.0)))
                _ph_pb  = float(np.arcsin(
                    np.clip(_ay_pb / (_g * max(np.cos(_th_pb), 0.1)), -1.0, 1.0)))
                _qw, _qx, _qy, _qz = self._ekf.x[6:10]
                _psi_pb = float(np.arctan2(
                    2.0 * (_qw * _qz + _qx * _qy),
                    1.0 - 2.0 * (_qy * _qy + _qz * _qz)))
                self._ekf.set_attitude(_ph_pb, _th_pb, _psi_pb)
                print(f"[MAVLinkRX] post-blip att reset FIRED: "
                      f"phi={np.rad2deg(_ph_pb):.1f}°  "
                      f"theta={np.rad2deg(_th_pb):.1f}°  "
                      f"acc_norm={_acc_norm_upd:.3f}  "
                      f"dt={now - self._hover_reset_t:.3f}s", flush=True)
                self._post_blip_att_reset_done = True
                self.data['post_blip_att_reset_done'] = True   # notify flight_sysid.py

        # Vision position + velocity + yaw update from PnP (written by vision_rx thread).
        vis = self.data.pop('_vision_ekf_update', None)
        if vis is not None:
            if vis.get('pos_ned') is not None:
                # PnP returns world-frame NED; the EKF tracks local frame (zeroed at
                # hover entry).  Subtract the saved world-frame offset so both agree.
                local_pos = np.asarray(vis['pos_ned']) - self._pos_offset_ned
                self._ekf.update_position(
                    local_pos, sigma_pos=vis['sigma_pos'], gate_dist=vis['gate'])
            if vis.get('vel_ned') is not None:
                self._ekf.update_velocity(
                    vis['vel_ned'], sigma_vel=vis['sigma_vel'],
                    gate_dist=vis['vel_gate'])
            if vis.get('yaw_ned') is not None:
                self._ekf.update_yaw(
                    vis['yaw_ned'], sigma_yaw=vis['sigma_yaw'],
                    gate_dist=vis.get('yaw_gate', 1.0))

        if self.logger:
            self.logger.log_ekf(
                t_us, now,
                self._ekf.x.copy(),
                np.diag(self._ekf.P).copy(),
                acc_applied, acc_innov,
                zupt_applied, zupt_innov,
                gyro, acc,
            )

        self.data['mav_state'] = {
            'pos_ned':  self._ekf.pos_ned,
            'vel_body': np.zeros(3),
            'vel_ned':  self._ekf.vel_ned,
            'quat':     self._ekf.quat,
            'rates':    gyro,
            'wall_t':   now,
            'sim_t_us': t_us,
        }

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type   = raw_payload[0]
            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        (data_type, sim_boot_time_ms, race_start_boot_time_ms,
         race_finish_time_ns, active_gate_index,
         last_gate_race_time) = struct.unpack_from("<BQqqIq", raw_payload)
        self.data['active_gate_index'] = int(active_gate_index)
        self.data['race_started']      = (race_start_boot_time_ms > 0)

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload += self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = {}
        for i in range(num_gates):
            (gate_id,
             position_ned_x, position_ned_y, position_ned_z,
             orientation_ned_w, orientation_ned_x,
             orientation_ned_y, orientation_ned_z,
             width, height) = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates[int(gate_id)] = {
                'ned':    np.array([float(position_ned_x),
                                    float(position_ned_y),
                                    float(position_ned_z)]),
                'quat':   np.array([float(orientation_ned_w), float(orientation_ned_x),
                                    float(orientation_ned_y), float(orientation_ned_z)]),
                'width':  float(width),
                'height': float(height),
            }
        self.data['track_gates_ned'] = gates
        if not getattr(self, '_track_printed', False):
            self._track_printed = True
            for gid, g in sorted(gates.items()):
                print(f"[TRACK] gate {gid}: NED={g['ned']}  "
                      f"{g['width']:.1f}x{g['height']:.1f}m", flush=True)

    def on_actuator_output_status(self, msg):
        time_boot_us      = msg.time_usec
        motor_front_left  = msg.actuator[0]
        motor_front_right = msg.actuator[1]
        motor_back_left   = msg.actuator[2]
        motor_back_right  = msg.actuator[3]
        if self.logger:
            self.logger.log_motors(
                time_boot_us,
                motor_front_left, motor_front_right,
                motor_back_left,  motor_back_right,
            )

    def on_collision(self, msg):
        gate_id = int(msg.id)
        now     = time.time()
        if now - self._gate_debounce.get(gate_id, 0.0) < self._gate_debounce_sec:
            return   # duplicate within debounce window — ignore
        self._gate_debounce[gate_id] = now
        self.data['gate_passed']  = True
        self.data['last_gate_id'] = gate_id
        print(f"[COLLISION] gate_id={gate_id} accepted", flush=True)
