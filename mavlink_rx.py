import struct
import time
import threading
from collections import defaultdict

import numpy as np
from pymavlink import mavutil


ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data, logger=None, monitor_enabled=True):
        self.mavlink_conn      = mavlink_connection
        self.data              = data
        self.logger            = logger
        self.thread            = None
        self.is_running        = False
        self._monitor_enabled  = monitor_enabled

        self.track_chunks              = {}
        self.expected_num_track_chunks = {}

        # Message tracking & 1 Hz rate monitor
        self.msg_count              = defaultdict(int)
        self.msg_count_last         = defaultdict(int)
        self.msg_last_wall_s        = {}
        self.msg_first_seen_printed = set()
        self._diag_period_s         = 1.0
        self._diag_last_s           = time.time()

        self.data['mavlink'] = {'latest': {}, 'rates_hz': {}, 'counts': {}}

        # Gate-collision debounce
        self._gate_debounce     = {}
        self._gate_debounce_sec = 2.0

        # Optional IMU callback — set by IMUEKFHandler.register(rx)
        self._on_imu_cb = None

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data, logger=None, monitor_enabled=True):
        rx = cls(mavlink_connection, data, logger, monitor_enabled=monitor_enabled)
        rx.thread = threading.Thread(target=rx.mavlink_receive_loop, daemon=False)
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def request_ground_truth_streams(self, rate_hz=50):
        """Request ATTITUDE (msg 30) and LOCAL_POSITION_NED (msg 32) from the sim."""
        interval_us = int(1e6 / rate_hz)
        for msg_id in (30, 32):  # ATTITUDE, LOCAL_POSITION_NED
            self.mavlink_conn.mav.command_long_send(
                self.mavlink_conn.target_system,
                self.mavlink_conn.target_component,
                511,         # MAV_CMD_SET_MESSAGE_INTERVAL
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0, 0,
            )
        print(f'[MAVLinkRX] requested ATTITUDE + LOCAL_POSITION_NED at {rate_hz} Hz',
              flush=True)

    def mavlink_receive_loop(self):
        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('[MAVLinkRX] ConnectionResetError — stopping.', flush=True)
                return
            except Exception as exc:
                print(f'[MAVLinkRX] recv_match error: {type(exc).__name__}: {exc}',
                      flush=True)
                time.sleep(0.01)
                continue

            now_s = time.time()

            # 1 Hz rate monitor
            if now_s - self._diag_last_s >= self._diag_period_s:
                dt = now_s - self._diag_last_s
                rates = {}
                for k, v in self.msg_count.items():
                    dv = v - self.msg_count_last[k]
                    rates[k] = dv / dt
                    self.msg_count_last[k] = v
                self.data['mavlink']['rates_hz'] = rates
                self.data['mavlink']['counts']   = dict(self.msg_count)
                if self._monitor_enabled:
                    print('\n[MAVLink monitor]', flush=True)
                    for k in sorted(rates):
                        print(f'  {k:32s} {rates[k]:7.1f} Hz   total={self.msg_count[k]}',
                              flush=True)
                    latest = self.data['mavlink']['latest']
                    imu = latest.get('HIGHRES_IMU')
                    if imu:
                        ax, ay, az = imu['acc_b_mps2']
                        gx, gy, gz = imu['gyro_b_radps']
                        print(f'  IMU acc  [{ax:+.3f}, {ay:+.3f}, {az:+.3f}] m/s²  '
                              f'gyro [{gx:+.3f}, {gy:+.3f}, {gz:+.3f}] rad/s', flush=True)
                    att = latest.get('ATTITUDE')
                    if att:
                        print(f'  ATTITUDE   roll={np.rad2deg(att["roll_rad"]):+7.2f}°  '
                              f'pitch={np.rad2deg(att["pitch_rad"]):+7.2f}°  '
                              f'yaw={np.rad2deg(att["yaw_rad"]):+7.2f}°', flush=True)
                    else:
                        print('  ATTITUDE   -- no data --', flush=True)
                    lpn = latest.get('LOCAL_POSITION_NED')
                    if lpn:
                        px, py, pz = lpn['pos_ned_m']
                        vx, vy, vz = lpn['vel_ned_mps']
                        print(f'  LOCAL_POS  N={px:+7.2f} E={py:+7.2f} D={pz:+7.2f} m  '
                              f'vN={vx:+6.2f} vE={vy:+6.2f} vD={vz:+6.2f} m/s', flush=True)
                    else:
                        print('  LOCAL_POS  -- no data --', flush=True)
                    act = latest.get('ACTUATOR_OUTPUT_STATUS')
                    if act:
                        m = act['actuator'][:4]
                        print(f'  actuator [{m[0]:+.3f}, {m[1]:+.3f}, '
                              f'{m[2]:+.3f}, {m[3]:+.3f}]', flush=True)
                    race = latest.get('RACE_STATUS')
                    if race:
                        print(f'  gate={race["active_gate_index"]}  '
                              f'started={race["race_started"]}  '
                              f'finished={race["race_finished"]}', flush=True)
                self._diag_last_s = now_s

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()
            self.msg_count[msg_type]      += 1
            self.msg_last_wall_s[msg_type] = now_s

            if msg_type not in self.msg_first_seen_printed:
                self.msg_first_seen_printed.add(msg_type)
                if self._monitor_enabled:
                    print(f'[MAVLink] first seen: {msg_type}', flush=True)
                    try:
                        print(msg.to_dict(), flush=True)
                    except Exception:
                        print(msg, flush=True)

            if msg_type == 'BAD_DATA':
                continue

            if self.logger:
                try:
                    self.logger.log_mavlink(msg)
                except Exception as exc:
                    print(f'[MAVLinkRX] log_mavlink error: {type(exc).__name__}: {exc}',
                          flush=True)

            try:
                if msg_type == 'HEARTBEAT':
                    self.on_heartbeat(msg)
                elif msg_type == 'TIMESYNC':
                    self.on_timesync(msg)
                elif msg_type == 'COMMAND_ACK':
                    self.on_command_ack(msg)
                elif msg_type == 'ATTITUDE':
                    self.on_attitude(msg)
                elif msg_type == 'LOCAL_POSITION_NED':
                    self.on_local_position_ned(msg)
                elif msg_type == 'ODOMETRY':
                    self.on_odometry(msg)
                elif msg_type == 'HIGHRES_IMU':
                    self.on_highres_imu(msg)
                elif msg_type == 'ENCAPSULATED_DATA':
                    self.on_encapsulated_data(msg)
                elif msg_type == 'ACTUATOR_OUTPUT_STATUS':
                    self.on_actuator_output_status(msg)
                elif msg_type == 'COLLISION':
                    self.on_collision(msg)
                elif msg_type == 'DATA_TRANSMISSION_HANDSHAKE':
                    track_data_transfer_id = msg.width
                    self.track_chunks[track_data_transfer_id] = {}
                    self.expected_num_track_chunks[track_data_transfer_id] = msg.packets
            except Exception as exc:
                print(f'[MAVLinkRX] handler error on {msg_type}: '
                      f'{type(exc).__name__}: {exc}', flush=True)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def on_heartbeat(self, msg):
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

    def on_timesync(self, msg):
        if msg.ts1 != 0:
            rtt_ns = time.time_ns() - msg.tc1
            self.data['timesync_rtt_ms'] = rtt_ns / 1e6

    def on_command_ack(self, msg):
        _CMD = {400: 'MAV_CMD_COMPONENT_ARM_DISARM', 511: 'MAV_CMD_SET_MESSAGE_INTERVAL'}
        _RES = {0: 'ACCEPTED', 1: 'TEMPORARILY_REJECTED', 2: 'DENIED',
                3: 'UNSUPPORTED', 4: 'FAILED', 5: 'IN_PROGRESS', 6: 'CANCELLED'}
        cmd_name = _CMD.get(msg.command, f'UNKNOWN_{msg.command}')
        res_name = _RES.get(msg.result,  f'UNKNOWN_{msg.result}')
        print(f'[COMMAND_ACK] {cmd_name}  result={res_name}', flush=True)
        self.data['mavlink']['latest']['COMMAND_ACK'] = {
            't_wall_s':      time.time(),
            'command':       msg.command,
            'command_name':  cmd_name,
            'result':        msg.result,
            'result_name':   res_name,
            'progress':      getattr(msg, 'progress', None),
            'result_param2': getattr(msg, 'result_param2', None),
        }

    def on_attitude(self, msg):
        # Sim ATTITUDE sign conventions (see sim_convention.py for gyro/acc):
        #   roll      — standard FRD/NED; no correction
        #   pitch     — sign-inverted vs standard ZYX (msg.pitch>0 = nose DOWN); negate
        #   yaw       — sign-inverted vs standard ZYX; NOT negated here (GT mode relies
        #               on the inverted sign; see _send_attitude_target r_des for correction)
        #   pitchspeed — sign-inverted; negate to match corrected pitch angle
        #   rollspeed, yawspeed — standard; no correction
        roll, pitch, yaw = float(msg.roll), float(msg.pitch), float(msg.yaw)
        pitch = -pitch
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        self.data['mavlink']['latest']['ATTITUDE'] = {
            't_wall_s':     time.time(),
            'time_boot_ms': msg.time_boot_ms,
            'roll_rad':     roll,
            'pitch_rad':    pitch,
            'yaw_rad':      yaw,
            'rollspeed':    float(msg.rollspeed),
            'pitchspeed':   -float(msg.pitchspeed),
            'yawspeed':     float(msg.yawspeed),
            'quat': np.array([
                cr * cp * cy + sr * sp * sy,  # qw
                sr * cp * cy - cr * sp * sy,  # qx
                cr * sp * cy + sr * cp * sy,  # qy
                cr * cp * sy - sr * sp * cy,  # qz
            ]),
        }

    def on_local_position_ned(self, msg):
        #
        #
        # PLEASE NOTE:
        # As per the configuration of the latest version of the simulator, Local Position NED telemetry has been disabled.
        #
        #
        pos_x = msg.x
        pos_y = msg.y
        pos_z = msg.z
        vel_x = msg.vx
        vel_y = msg.vy
        vel_z = msg.vz
        time_boot_ms = msg.time_boot_ms
        sample = {
            "t_wall_s": time.time(),
            "time_boot_ms": msg.time_boot_ms,
            "pos_ned_m": [msg.x, msg.y, msg.z],
            "vel_ned_mps": [msg.vx, msg.vy, msg.vz],
        }

        self.data["mavlink"]["latest"]["LOCAL_POSITION_NED"] = sample

        if not hasattr(self, "_printed_local_position_ned"):
            self._printed_local_position_ned = True
            print("[MAVLink] LOCAL_POSITION_NED RECEIVED!")
            print(sample)

    def on_odometry(self, *_):
        # Disabled in this sim configuration.
        pass

    def on_highres_imu(self, msg):
        self.data['mavlink']['latest']['HIGHRES_IMU'] = {
            't_wall_s':     time.time(),
            'time_boot_us': msg.time_usec,
            'acc_b_mps2':   [msg.xacc, msg.yacc, msg.zacc],
            'gyro_b_radps': [msg.xgyro, msg.ygyro, msg.zgyro],
        }
        if self.logger:
            try:
                self.logger.log_imu(msg.time_usec,
                                    msg.xacc, msg.yacc, msg.zacc,
                                    msg.xgyro, msg.ygyro, msg.zgyro)
            except Exception:
                pass
        if self._on_imu_cb is not None:
            self._on_imu_cb(msg)

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
         last_gate_race_time) = struct.unpack_from('<BQqqIq', raw_payload)
        row = {
            't_wall_s':                time.time(),
            'sim_boot_time_ms':        sim_boot_time_ms,
            'race_start_boot_time_ms': race_start_boot_time_ms,
            'race_finish_time_ns':     race_finish_time_ns,
            'active_gate_index':       int(active_gate_index),
            'last_gate_race_time_s':   last_gate_race_time,
            'race_started':            int(race_start_boot_time_ms >= 0),
            'race_finished':           int(race_finish_time_ns >= 0),
        }
        self.data['mavlink']['latest']['RACE_STATUS'] = row
        # Backward-compat keys read by controller and flight scripts
        self.data['active_gate_index'] = int(active_gate_index)
        self.data['race_started']      = (race_start_boot_time_ms > 0)

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        data_type, transfer_id = struct.unpack_from('<BH', raw_payload)
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
        num_gates, = struct.unpack_from('<H', payload)
        payload = payload[2:]
        gates = {}
        for _ in range(num_gates):
            (gate_id,
             px, py, pz,
             qw, qx, qy, qz,
             width, height) = struct.unpack_from('<Hfffffffff', payload)
            payload = payload[38:]
            gates[int(gate_id)] = {
                'ned':    np.array([float(px), float(py), float(pz)]),
                'quat':   np.array([float(qw), float(qx), float(qy), float(qz)]),
                'width':  float(width),
                'height': float(height),
            }
        self.data['track_gates_ned'] = gates
        if not getattr(self, '_track_printed', False):
            self._track_printed = True
            for gid, g in sorted(gates.items()):
                print(f'[TRACK] gate {gid}: NED={g["ned"]}  '
                      f'{g["width"]:.1f}x{g["height"]:.1f}m', flush=True)

    def on_actuator_output_status(self, msg):
        actuator = list(msg.actuator)
        self.data['mavlink']['latest']['ACTUATOR_OUTPUT_STATUS'] = {
            't_wall_s':     time.time(),
            'time_boot_us': msg.time_usec,
            'active':       msg.active,
            'actuator':     actuator,
        }
        if self.logger:
            try:
                self.logger.log_motors(msg.time_usec,
                                       actuator[0], actuator[1],
                                       actuator[2], actuator[3])
            except Exception:
                pass

    def on_collision(self, msg):
        gate_id = int(msg.id)
        now     = time.time()
        self.data['mavlink']['latest']['COLLISION'] = {
            't_wall_s':      now,
            'collision_id':  gate_id,
            'threat_level':  msg.threat_level,
            'impact_kg_mps': msg.horizontal_minimum_delta,
        }
        if now - self._gate_debounce.get(gate_id, 0.0) < self._gate_debounce_sec:
            return
        self._gate_debounce[gate_id] = now
        self.data['gate_passed']  = True
        self.data['last_gate_id'] = gate_id
        print(f'[COLLISION] gate_id={gate_id}  threat={msg.threat_level}  '
              f'impact={msg.horizontal_minimum_delta:.3f}', flush=True)
