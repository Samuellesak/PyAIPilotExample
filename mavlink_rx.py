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
        self._yaw_initialised = False   # True after first ATTITUDE injects yaw into EKF

        # Apply initial yaw from params if the sim does not send ATTITUDE messages.
        # Set initial_yaw_deg in params.yaml to match the drone's heading in the sim
        # (e.g. 90.0 if the drone faces East).  Ignored once ATTITUDE arrives.
        from dyn import load_params as _lp
        _p = _lp("params.yaml")
        _init_yaw_deg = float(_p.get('initial_yaw_deg', 0.0))
        if _init_yaw_deg != 0.0:
            self._ekf.set_yaw(np.deg2rad(_init_yaw_deg))
            print(f"[MAVLinkRX] EKF yaw pre-set from params: {_init_yaw_deg:.1f}°",
                  flush=True)

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
                elif msg_type == "ATTITUDE":
                    self.on_attitude(msg)
                elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                    self.on_actuator_output_status(msg)
                elif msg_type == "COLLISION":
                    self.on_collision(msg)
                elif msg_type == "LOCAL_POSITION_NED":
                    self.on_local_position_ned(msg)
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

        # EKF reset requested by controller at hover entry.
        if self.data.get('reset_attitude_flag'):
            self.data['reset_attitude_flag'] = False
            self._ekf.reset()
            print("[MAVLinkRX] EKF reset", flush=True)

        # Velocity-only reset: clears spurious velocity built up during slope
        # attitude convergence, without discarding the converged attitude estimate.
        if self.data.get('reset_vel_flag'):
            self.data['reset_vel_flag'] = False
            self._ekf.x[0:3] = 0.0          # zero position — world origin at hover entry
            self._ekf.x[3:6] = 0.0          # zero velocity built up during wait/slope
            self._ekf.set_roll(0.0)          # force phi=0 at hover entry (drone is level laterally)
            print("[MAVLinkRX] EKF position, velocity zeroed; roll forced to 0", flush=True)

        # Sim reports all three gyro axes sign-flipped vs FRD convention:
        # +xgyro = roll LEFT (not roll-right), +ygyro = nose DOWN (not nose-up),
        # +zgyro = yaw LEFT (not yaw-right).  Negate all three so the EKF and
        # controller receive standard FRD rates where +r = yaw right (CW from above).
        gyro = np.array([-gx, -gy, -gz])
        acc  = np.array([ax, ay, az])

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
            'ax': ax, 'ay': ay, 'az': az,
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
        acc_applied, acc_innov = self._ekf.update_accel(acc)
        if self.data.get('zupt_enabled', True):
            zupt_applied, zupt_innov = self._ekf.update_zupt(gyro, acc)
        else:
            zupt_applied, zupt_innov = False, 0.0

        # Vision position + velocity + yaw update from PnP (written by vision_rx thread).
        vis = self.data.pop('_vision_ekf_update', None)
        if vis is not None:
            if vis.get('pos_ned') is not None:
                self._ekf.update_position(
                    vis['pos_ned'], sigma_pos=vis['sigma_pos'], gate_dist=vis['gate'])
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

    def on_attitude(self, msg):
        """
        Receive ATTITUDE ground-truth from the sim.
        On the first message inject the true yaw into the EKF so that
        _psi_cmd at hover entry is consistent with the drone's actual heading.
        Subsequent messages keep 'attitude_yaw' in shared_data for diagnostics.
        """
        yaw   = float(msg.yaw)    # radians, range [-π, π]
        roll  = float(msg.roll)   # sim ground truth
        pitch = float(msg.pitch)  # sim ground truth
        self.data['attitude_yaw']   = yaw
        self.data['attitude_roll']  = roll
        self.data['attitude_pitch'] = pitch
        if not self._yaw_initialised:
            self._ekf.set_yaw(yaw)
            self._yaw_initialised = True
            print(f"[MAVLinkRX] EKF yaw initialised from ATTITUDE: "
                  f"{np.degrees(yaw):.1f}°", flush=True)

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

    def on_local_position_ned(self, msg):
        pos = np.array([float(msg.x),  float(msg.y),  float(msg.z)],  dtype=float)
        vel = np.array([float(msg.vx), float(msg.vy), float(msg.vz)], dtype=float)
        self.data['_sim_pos_ned'] = pos
        self.data['_sim_vel_ned'] = vel
        if self.logger:
            self.logger.log_position(msg.time_boot_ms,
                                     msg.x,  msg.y,  msg.z,
                                     msg.vx, msg.vy, msg.vz)

    def on_collision(self, msg):
        self.data['gate_passed']  = True
        self.data['last_gate_id'] = int(msg.id)
