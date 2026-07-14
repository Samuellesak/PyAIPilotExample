import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write to file without a display
import matplotlib.pyplot as plt


class Logger:
    """
    Thread-safe flight logger.

    Collects:
      - Every MAVLink message  ->  logs/<session>/mavlink.txt
      - HIGHRES_IMU samples    ->  logs/<session>/imu.png  (plotted on save)
      - Every 100th camera frame->  logs/<session>/frames/frame_XXXXXX.jpg
    """

    def __init__(self, log_dir="logs"):
        session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(log_dir, session)
        self.frames_dir  = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._lock        = threading.Lock()
        self._frame_count = 0
        self._closed      = False

        self._mavlink_file = open(os.path.join(self.session_dir, "mavlink.txt"), "w")

        self._imu = {
            "t":  [],
            "ax": [], "ay": [], "az": [],
            "gx": [], "gy": [], "gz": [],
        }

        self._pos = {
            "t":  [],
            "x":  [], "y":  [], "z":  [],
            "vx": [], "vy": [], "vz": [],
        }

        self._ctrl = {
            "t":  [],
            "u0": [], "u1": [], "u2": [], "u3": [],
        }

        self._carrot = {
            "t": [], "wp": [], "alpha": [],
            "vcx": [], "vcy": [], "vcz": [],
            "cx":  [], "cy":  [], "cz":  [],   # carrot NED position
        }

        # Cascade controller signal log — one row per control tick (~250 Hz)
        self._cascade = {
            "t": [],
            # Velocity: reference vs measured
            "vN_ref": [], "vE_ref": [], "vD_ref": [],
            "vN_meas": [], "vE_meas": [], "vD_meas": [],
            # Velocity error
            "eN": [], "eE": [], "eD": [],
            # Integrator state
            "xi_N": [], "xi_E": [], "xi_D": [],
            # Desired vs measured attitude (deg)
            "phi_des": [], "theta_des": [],
            "phi_meas": [], "theta_meas": [],
            # Attitude error (deg)
            "phi_err": [], "theta_err": [],
            # Yaw: reference vs measured vs error
            "psi_meas": [], "psi_ref": [], "psi_err": [],
            # Desired body rates
            "p_des": [], "q_des": [], "r_des": [],
            # Measured body rates
            "p_meas": [], "q_meas": [], "r_meas": [],
            # Collective and tilt factor
            "T_coll": [], "R22": [],
        }

        # EKF state log — one row per HIGHRES_IMU message
        self._ekf = {
            "t_us":  [], "wall_t": [],
            # Position (13-state EKF: indices 0:3)
            "pN": [], "pE": [], "pD": [],
            # Velocity (indices 3:6)
            "vN": [], "vE": [], "vD": [],
            # Quaternion (indices 6:10)
            "qw": [], "qx": [], "qy": [], "qz": [],
            # Gyro bias (indices 10:13)
            "bgx": [], "bgy": [], "bgz": [],
            # Covariance diagonal std-devs
            "sig_pN": [], "sig_pE": [], "sig_pD": [],
            "sig_vN": [], "sig_vE": [], "sig_vD": [],
            "sig_qw": [], "sig_qx": [], "sig_qy": [], "sig_qz": [],
            "sig_bgx": [], "sig_bgy": [], "sig_bgz": [],
            # Raw IMU (same data as imu log, stored here for aligned comparison)
            "ax": [], "ay": [], "az": [],
            "gx": [], "gy": [], "gz": [],
            # Innovation diagnostics
            "acc_applied":  [], "acc_innov":  [],   # gravity-alignment update
            "zupt_applied": [], "zupt_innov": [],   # zero-velocity update
        }

        self._waypoints = None   # set via set_waypoints(); used for path comparison plot

        # Motor feedback from ACTUATOR_OUTPUT_STATUS — used for inertia sysid
        self._motors = {
            "t_us": [],
            "FL": [], "FR": [], "BL": [], "BR": [],
        }

        # Vision / YOLO / PnP log — one row per processed camera frame
        self._vision = {
            "wall_t": [], "frame_id": [], "detected": [],
            # YOLO outputs
            "conf": [],
            "bb_cx": [], "bb_cy": [], "bb_w": [], "bb_h": [],
            # Keypoint pixel coordinates (4 corners: TL TR BR BL)
            "kp0x": [], "kp0y": [], "kp1x": [], "kp1y": [],
            "kp2x": [], "kp2y": [], "kp3x": [], "kp3y": [],
            # PnP output: gate in OpenCV camera frame [m]
            "pnp_ok": [],
            "tvec_x": [], "tvec_y": [], "tvec_z": [],   # z = forward distance
            # Drone NED position derived from PnP + known gate NED
            "pos_N": [], "pos_E": [], "pos_D": [],
            # Velocity derived from consecutive PnP positions
            "vel_ok": [],
            "vel_N": [], "vel_E": [], "vel_D": [], "speed_ms": [],
        }

        print(f"Logger: session directory -> {self.session_dir}")

    # ------------------------------------------------------------------
    # Public API called from other threads
    # ------------------------------------------------------------------

    def set_waypoints(self, waypoints):
        """Store waypoints (M×3 NED array) for path comparison plots."""
        self._waypoints = np.asarray(waypoints, dtype=float)

    def reset_flight_data(self):
        """Discard any pre-arm samples so plots start at t=0 from arm time."""
        with self._lock:
            for key in self._imu:
                self._imu[key].clear()
            for key in self._pos:
                self._pos[key].clear()
            for key in self._ctrl:
                self._ctrl[key].clear()
            for key in self._carrot:
                self._carrot[key].clear()
            for key in self._cascade:
                self._cascade[key].clear()
            for key in self._ekf:
                self._ekf[key].clear()
            for key in self._motors:
                self._motors[key].clear()
            for key in self._vision:
                self._vision[key].clear()

    def log_mavlink(self, msg):
        """Append one MAVLink message to the text log."""
        line = f"[{time.time():.6f}] {msg.get_type()} {msg.to_dict()}\n"
        with self._lock:
            if not self._closed:
                self._mavlink_file.write(line)

    def log_imu(self, time_us, ax, ay, az, gx, gy, gz):
        """Accumulate one HIGHRES_IMU sample for the end-of-flight plot."""
        with self._lock:
            d = self._imu
            d["t"].append(time_us)
            d["ax"].append(ax); d["ay"].append(ay); d["az"].append(az)
            d["gx"].append(gx); d["gy"].append(gy); d["gz"].append(gz)

    def log_position(self, time_boot_ms, x, y, z, vx, vy, vz):
        """Accumulate one LOCAL_POSITION_NED sample."""
        with self._lock:
            d = self._pos
            d["t"].append(time_boot_ms)
            d["x"].append(x);   d["y"].append(y);   d["z"].append(z)
            d["vx"].append(vx); d["vy"].append(vy); d["vz"].append(vz)

    def log_control(self, time_ms, u):
        """Accumulate one motor command sample. u = [T1,T2,T3,T4] in Newtons."""
        with self._lock:
            d = self._ctrl
            d["t"].append(time_ms)
            d["u0"].append(float(u[0])); d["u1"].append(float(u[1]))
            d["u2"].append(float(u[2])); d["u3"].append(float(u[3]))

    def log_carrot(self, time_ms, wp, alpha, v_cmd, carrot_pos=None):
        """Accumulate one carrot-tracker sample."""
        with self._lock:
            d = self._carrot
            d["t"].append(time_ms)
            d["wp"].append(int(wp))
            d["alpha"].append(float(alpha))
            d["vcx"].append(float(v_cmd[0]))
            d["vcy"].append(float(v_cmd[1]))
            d["vcz"].append(float(v_cmd[2]))
            cp = carrot_pos if carrot_pos is not None else [float('nan')] * 3
            d["cx"].append(float(cp[0]))
            d["cy"].append(float(cp[1]))
            d["cz"].append(float(cp[2]))

    def log_ekf(self, t_us, wall_t, x, P_diag,
                acc_applied, acc_innov,
                zupt_applied, zupt_innov,
                gyro_raw, acc_raw):
        """
        Accumulate one EKF sample (called at IMU rate, ~250 Hz).

        x       : EKF state vector [pN,pE,pD, vN,vE,vD, qw,qx,qy,qz, bgx,bgy,bgz]  (13,)
        P_diag  : diagonal of covariance matrix (same ordering as x)
        acc_applied / acc_innov   : gravity-alignment update was applied, |innovation|
        zupt_applied / zupt_innov : ZUPT update was applied, |innovation|
        gyro_raw, acc_raw         : raw IMU [rad/s], [m/s²]
        """
        with self._lock:
            d = self._ekf
            d["t_us"].append(int(t_us))
            d["wall_t"].append(float(wall_t))
            # position (indices 0:3)
            d["pN"].append(float(x[0])); d["pE"].append(float(x[1])); d["pD"].append(float(x[2]))
            # velocity (indices 3:6)
            d["vN"].append(float(x[3])); d["vE"].append(float(x[4])); d["vD"].append(float(x[5]))
            # quaternion (indices 6:10)
            d["qw"].append(float(x[6])); d["qx"].append(float(x[7]))
            d["qy"].append(float(x[8])); d["qz"].append(float(x[9]))
            # gyro bias (indices 10:13)
            d["bgx"].append(float(x[10])); d["bgy"].append(float(x[11])); d["bgz"].append(float(x[12]))
            # std devs from covariance diagonal
            sd = np.sqrt(np.maximum(P_diag, 0.0))
            d["sig_pN"].append(float(sd[0])); d["sig_pE"].append(float(sd[1])); d["sig_pD"].append(float(sd[2]))
            d["sig_vN"].append(float(sd[3])); d["sig_vE"].append(float(sd[4])); d["sig_vD"].append(float(sd[5]))
            d["sig_qw"].append(float(sd[6])); d["sig_qx"].append(float(sd[7]))
            d["sig_qy"].append(float(sd[8])); d["sig_qz"].append(float(sd[9]))
            d["sig_bgx"].append(float(sd[10])); d["sig_bgy"].append(float(sd[11])); d["sig_bgz"].append(float(sd[12]))
            # raw IMU
            d["ax"].append(float(acc_raw[0])); d["ay"].append(float(acc_raw[1])); d["az"].append(float(acc_raw[2]))
            d["gx"].append(float(gyro_raw[0])); d["gy"].append(float(gyro_raw[1])); d["gz"].append(float(gyro_raw[2]))
            # innovations
            d["acc_applied"].append(int(acc_applied));   d["acc_innov"].append(float(acc_innov))
            d["zupt_applied"].append(int(zupt_applied)); d["zupt_innov"].append(float(zupt_innov))

    def log_cascade(self, time_ms,
                    v_ned_ref, v_ned_meas,
                    phi_des_deg, theta_des_deg,
                    phi_meas_deg, theta_meas_deg,
                    psi_meas_deg, psi_ref_deg,
                    p_des, q_des, r_des,
                    rates, T_coll, R22, xi_vel):
        """Accumulate one cascade controller sample (called at control rate)."""
        with self._lock:
            d = self._cascade
            d["t"].append(float(time_ms))
            d["vN_ref"].append(float(v_ned_ref[0]));  d["vE_ref"].append(float(v_ned_ref[1]));  d["vD_ref"].append(float(v_ned_ref[2]))
            d["vN_meas"].append(float(v_ned_meas[0])); d["vE_meas"].append(float(v_ned_meas[1])); d["vD_meas"].append(float(v_ned_meas[2]))
            d["eN"].append(float(v_ned_ref[0] - v_ned_meas[0]))
            d["eE"].append(float(v_ned_ref[1] - v_ned_meas[1]))
            d["eD"].append(float(v_ned_ref[2] - v_ned_meas[2]))
            d["xi_N"].append(float(xi_vel[0])); d["xi_E"].append(float(xi_vel[1])); d["xi_D"].append(float(xi_vel[2]))
            d["phi_des"].append(float(phi_des_deg));   d["theta_des"].append(float(theta_des_deg))
            d["phi_meas"].append(float(phi_meas_deg)); d["theta_meas"].append(float(theta_meas_deg))
            d["phi_err"].append(float(phi_des_deg - phi_meas_deg))
            d["theta_err"].append(float(theta_des_deg - theta_meas_deg))
            d["psi_meas"].append(float(psi_meas_deg)); d["psi_ref"].append(float(psi_ref_deg))
            d["psi_err"].append(float(psi_ref_deg - psi_meas_deg))
            d["p_des"].append(float(p_des)); d["q_des"].append(float(q_des)); d["r_des"].append(float(r_des))
            d["p_meas"].append(float(rates[0])); d["q_meas"].append(float(rates[1])); d["r_meas"].append(float(rates[2]))
            d["T_coll"].append(float(T_coll)); d["R22"].append(float(R22))

    def log_motors(self, t_us, FL, FR, BL, BR):
        """Accumulate one ACTUATOR_OUTPUT_STATUS sample (actual motor thrusts)."""
        with self._lock:
            d = self._motors
            d["t_us"].append(int(t_us))
            d["FL"].append(float(FL)); d["FR"].append(float(FR))
            d["BL"].append(float(BL)); d["BR"].append(float(BR))

    def log_frame(self, frame_id, img, force=False, suffix=""):
        """
        Save a camera frame as JPEG.
        force=True  : always save (used for detected gates).
        suffix      : optional filename suffix, e.g. "_mask" for orange-mask images.
        Otherwise saves every 30th frame.
        """
        with self._lock:
            self._frame_count += 1
            save_this = force or (self._frame_count % 30 == 0)
        if save_this:
            path = os.path.join(self.frames_dir, f"frame_{frame_id:06d}{suffix}.jpg")
            cv2.imwrite(path, img)

    def log_vision(self, wall_t, frame_id, detected,
                   conf=0.0, bb=None, corners=None,
                   tvec=None, drone_ned=None, vel_ned=None):
        """
        Accumulate one vision/YOLO/PnP sample (called at camera frame rate).

        bb       : (cx_px, cy_px, w_px, h_px) bounding box or None
        corners  : (4, 2) ndarray of keypoint pixel coords (TL TR BR BL) or None
        tvec     : (3,) gate position in OpenCV camera frame [m] or None
        drone_ned: (3,) drone NED position estimate from PnP or None
        vel_ned  : (3,) drone NED velocity estimate from PnP differencing or None
        """
        _nan = float("nan")
        with self._lock:
            d = self._vision
            d["wall_t"].append(float(wall_t))
            d["frame_id"].append(int(frame_id))
            d["detected"].append(int(detected))
            d["conf"].append(float(conf) if detected else _nan)
            # Bounding box
            if bb is not None and detected:
                d["bb_cx"].append(float(bb[0])); d["bb_cy"].append(float(bb[1]))
                d["bb_w"].append(float(bb[2]));  d["bb_h"].append(float(bb[3]))
            else:
                d["bb_cx"].append(_nan); d["bb_cy"].append(_nan)
                d["bb_w"].append(_nan);  d["bb_h"].append(_nan)
            # Keypoints
            if corners is not None and detected:
                for i, (kx, ky) in enumerate(corners[:4]):
                    d[f"kp{i}x"].append(float(kx))
                    d[f"kp{i}y"].append(float(ky))
            else:
                for i in range(4):
                    d[f"kp{i}x"].append(_nan); d[f"kp{i}y"].append(_nan)
            # PnP
            pnp_ok = tvec is not None
            d["pnp_ok"].append(int(pnp_ok))
            if pnp_ok:
                d["tvec_x"].append(float(tvec[0])); d["tvec_y"].append(float(tvec[1]))
                d["tvec_z"].append(float(tvec[2]))
            else:
                d["tvec_x"].append(_nan); d["tvec_y"].append(_nan); d["tvec_z"].append(_nan)
            # Drone NED from PnP
            if drone_ned is not None and pnp_ok:
                d["pos_N"].append(float(drone_ned[0])); d["pos_E"].append(float(drone_ned[1]))
                d["pos_D"].append(float(drone_ned[2]))
            else:
                d["pos_N"].append(_nan); d["pos_E"].append(_nan); d["pos_D"].append(_nan)
            # Velocity
            vel_ok = vel_ned is not None
            d["vel_ok"].append(int(vel_ok))
            if vel_ok:
                d["vel_N"].append(float(vel_ned[0])); d["vel_E"].append(float(vel_ned[1]))
                d["vel_D"].append(float(vel_ned[2]))
                d["speed_ms"].append(float(np.linalg.norm(vel_ned)))
            else:
                d["vel_N"].append(_nan); d["vel_E"].append(_nan)
                d["vel_D"].append(_nan); d["speed_ms"].append(_nan)

    # ------------------------------------------------------------------
    # Call once at the end of the flight
    # ------------------------------------------------------------------

    def save(self):
        """Flush the MAVLink log and write the IMU plot to disk."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._mavlink_file.flush()
            self._mavlink_file.close()

        self._plot_imu()
        self._plot_position()
        self._plot_control()
        self._plot_carrot()
        self._plot_ekf()
        self._plot_cascade()
        self._plot_path()
        self._write_motors_sysid()
        self._write_position_csv()
        self._write_control_csv()
        self._write_carrot_csv()
        self._write_ekf_csv()
        self._write_cascade_csv()
        self._write_vision_csv()
        self._plot_vision()
        print(f"Logger: all data saved to {self.session_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_motors_sysid(self):
        """
        Write a CSV pairing actual motor thrusts with gyro rates, aligned by timestamp.
        Also estimates Ixx from tau_roll / alpha_roll using finite differences.

        The estimate is valid during periods of single-axis excitation and is printed
        as a guide for correcting m_motor in params.yaml.

        Torque convention (same as dyn.py mixer, DJI X config):
            d = L / sqrt(2) = 0.14 / 1.414 = 0.09899 m
            tau_roll  = d * (-T_BR + T_BL + T_FL - T_FR)
                      = d * (FL + BL - FR - BR)   [matches controller M_inv sign]
            tau_pitch = d * ( T_BR + T_BL - T_FL - T_FR)
                      = d * (BL + BR - FL - FR)
        Motor labels in ACTUATOR_OUTPUT_STATUS: [0]=FL,[1]=FR,[2]=BL,[3]=BR
        """
        with self._lock:
            m = {k: list(v) for k, v in self._motors.items()}
            e = {k: list(v) for k, v in self._ekf.items()}

        if not m["t_us"] or not e["t_us"]:
            return

        import yaml
        try:
            with open("params.yaml") as f:
                raw = yaml.safe_load(f)
            L = float(raw.get("L", 0.14))
        except Exception:
            L = 0.14
        d_arm = L / np.sqrt(2.0)

        # Align motor timestamps to EKF timestamps (nearest-neighbour)
        t_m   = np.array(m["t_us"], dtype=float)
        FL_a  = np.array(m["FL"]); FR_a = np.array(m["FR"])
        BL_a  = np.array(m["BL"]); BR_a = np.array(m["BR"])

        t_e   = np.array(e["t_us"], dtype=float)
        gx_e  = np.array(e["gx"])
        gy_e  = np.array(e["gy"])

        idx = np.searchsorted(t_e, t_m, side="left")
        idx = np.clip(idx, 0, len(t_e) - 1)
        # pull left neighbour when it is closer
        idx_l = np.clip(idx - 1, 0, len(t_e) - 1)
        idx   = np.where(np.abs(t_e[idx] - t_m) <= np.abs(t_e[idx_l] - t_m), idx, idx_l)
        gx_aligned = gx_e[idx]
        gy_aligned = gy_e[idx]

        # Applied torques
        tau_roll  = d_arm * (FL_a + BL_a - FR_a - BR_a)
        tau_pitch = d_arm * (BL_a + BR_a - FL_a - FR_a)

        # Write aligned CSV
        out = os.path.join(self.session_dir, "motors_sysid.csv")
        with open(out, "w") as f:
            f.write("t_us,FL_N,FR_N,BL_N,BR_N,"
                    "tau_roll_Nm,tau_pitch_Nm,gx_rads,gy_rads\n")
            for i in range(len(t_m)):
                f.write(
                    f"{int(t_m[i])},"
                    f"{FL_a[i]:.4f},{FR_a[i]:.4f},{BL_a[i]:.4f},{BR_a[i]:.4f},"
                    f"{tau_roll[i]:.5f},{tau_pitch[i]:.5f},"
                    f"{gx_aligned[i]:.5f},{gy_aligned[i]:.5f}\n"
                )
        print(f"Logger: motors sysid CSV -> {out}")

        # Ixx estimate: Ixx ≈ tau_roll / alpha_roll
        # Use only samples where |tau_roll| > 1 N·m (significant excitation)
        dt_m = np.diff(t_m) / 1e6   # µs -> s
        alpha_roll = np.diff(gx_aligned) / np.clip(dt_m, 1e-4, 0.1)
        tau_mid    = 0.5 * (tau_roll[:-1] + tau_roll[1:])
        mask       = np.abs(tau_mid) > 1.0   # at least 1 N·m excitation
        if mask.sum() >= 20:
            Ixx_est = np.median(tau_mid[mask] / np.clip(alpha_roll[mask], 0.01, None))
            print(
                f"Logger: [SYSID] Estimated Ixx ≈ {Ixx_est:.5f} kg·m²  "
                f"(from {mask.sum()} samples with |tau_roll|>1 N·m)\n"
                f"         Current params.yaml Ixx = "
                f"2*m_motor*L² + 0.25*m_frame*r_body² — adjust m_motor to match.\n"
                f"         Rule of thumb: m_motor ≈ Ixx_est / (2*L²) (ignores frame term)\n"
                f"         m_motor ≈ {Ixx_est / (2.0 * L**2):.3f} kg"
            )
        else:
            print("Logger: [SYSID] Not enough roll excitation to estimate Ixx "
                  f"(need |tau_roll|>1 N·m for ≥20 samples, got {mask.sum()}).\n"
                  "         Fly with K_att higher briefly, or run a dedicated sysid manoeuvre.")

    def _plot_imu(self):
        with self._lock:
            d = {k: list(v) for k, v in self._imu.items()}
        if not d["t"]:
            print("Logger: no IMU data collected, skipping plot.")
            return

        t = (np.array(d["t"]) - d["t"][0]) / 1e6  # µs -> seconds

        fig, (ax_acc, ax_gyr) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        ax_acc.plot(t, d["ax"], label="ax", linewidth=0.8)
        ax_acc.plot(t, d["ay"], label="ay", linewidth=0.8)
        ax_acc.plot(t, d["az"], label="az", linewidth=0.8)
        ax_acc.set_ylabel("Acceleration (m/s²)")
        ax_acc.set_title("Accelerometer")
        ax_acc.legend(loc="upper right")
        ax_acc.grid(True, alpha=0.4)

        ax_gyr.plot(t, d["gx"], label="gx", linewidth=0.8)
        ax_gyr.plot(t, d["gy"], label="gy", linewidth=0.8)
        ax_gyr.plot(t, d["gz"], label="gz", linewidth=0.8)
        ax_gyr.set_ylabel("Angular Rate (rad/s)")
        ax_gyr.set_xlabel("Time (s)")
        ax_gyr.set_title("Gyroscope")
        ax_gyr.legend(loc="upper right")
        ax_gyr.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "imu.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: IMU plot -> {out}")

    def _plot_position(self):
        with self._lock:
            d = {k: list(v) for k, v in self._pos.items()}
        if not d["t"]:
            print("Logger: no position data collected, skipping plot.")
            return

        t = (np.array(d["t"]) - d["t"][0]) / 1e3  # ms -> seconds

        fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        ax_pos.plot(t, d["x"],  label="x (N)",   linewidth=0.8)
        ax_pos.plot(t, d["y"],  label="y (E)",   linewidth=0.8)
        ax_pos.plot(t, [-z for z in d["z"]], label="alt (-z)", linewidth=0.8)
        ax_pos.set_ylabel("Position (m)")
        ax_pos.set_title("Position NED")
        ax_pos.legend(loc="upper right")
        ax_pos.grid(True, alpha=0.4)

        ax_vel.plot(t, d["vx"], label="vx (N)",  linewidth=0.8)
        ax_vel.plot(t, d["vy"], label="vy (E)",  linewidth=0.8)
        ax_vel.plot(t, d["vz"], label="vz (D)",  linewidth=0.8)
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_title("Velocity NED")
        ax_vel.legend(loc="upper right")
        ax_vel.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "position.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: position plot -> {out}")

    def _plot_control(self):
        with self._lock:
            d = {k: list(v) for k, v in self._ctrl.items()}
        if not d["t"]:
            print("Logger: no control data collected, skipping plot.")
            return

        t = (np.array(d["t"]) - d["t"][0]) / 1e3  # ms -> seconds

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(t, d["u0"], label="T1 BR", linewidth=0.8)
        ax.plot(t, d["u1"], label="T2 BL", linewidth=0.8)
        ax.plot(t, d["u2"], label="T3 FL", linewidth=0.8)
        ax.plot(t, d["u3"], label="T4 FR", linewidth=0.8)
        T_max_plot = max(d["u0"] + d["u1"] + d["u2"] + d["u3"], default=11.3)
        ax.axhline(0,          color="k", linewidth=0.5, linestyle="--")
        ax.axhline(T_max_plot, color="r", linewidth=0.5, linestyle="--", label=f"T_max≈{T_max_plot:.1f}N")
        ax.set_ylabel("Motor thrust (N)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Motor Commands [T1=BR, T2=BL, T3=FL, T4=FR]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "control.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: control plot -> {out}")

    def _plot_carrot(self):
        with self._lock:
            d = {k: list(v) for k, v in self._carrot.items()}
        if not d["t"]:
            print("Logger: no carrot data collected, skipping plot.")
            return

        t = (np.array(d["t"]) - d["t"][0]) / 1e3  # ms -> seconds

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        ax_wp, ax_v, ax_c = axes

        ax_wp.step(t, d["wp"],    label="waypoint index", where="post", linewidth=1.2)
        ax_wp.plot(t, d["alpha"], label="blend α",        linewidth=0.8, linestyle="--")
        ax_wp.set_ylabel("Waypoint / α")
        ax_wp.set_title("Carrot Tracker — Active Waypoint")
        ax_wp.legend(loc="upper left")
        ax_wp.grid(True, alpha=0.4)

        ax_v.plot(t, d["vcx"], label="vc_N", linewidth=0.8)
        ax_v.plot(t, d["vcy"], label="vc_E", linewidth=0.8)
        ax_v.plot(t, d["vcz"], label="vc_D", linewidth=0.8)
        ax_v.set_ylabel("Reference velocity (m/s)")
        ax_v.set_title("Carrot Reference Velocity (NED)")
        ax_v.legend(loc="upper left")
        ax_v.grid(True, alpha=0.4)

        ax_c.plot(t, d["cx"], label="carrot_N", linewidth=0.8)
        ax_c.plot(t, d["cy"], label="carrot_E", linewidth=0.8)
        ax_c.plot(t, d["cz"], label="carrot_D", linewidth=0.8)
        ax_c.set_ylabel("Carrot NED position (m)")
        ax_c.set_xlabel("Time (s)")
        ax_c.set_title("Carrot Position (NED)")
        ax_c.legend(loc="upper left")
        ax_c.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "carrot.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: carrot plot -> {out}")

    def _write_position_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._pos.items()}
        if not d["t"]:
            return
        t0 = d["t"][0]
        out = os.path.join(self.session_dir, "position.csv")
        with open(out, "w") as f:
            f.write("time_s,x_N_m,y_E_m,z_D_m,alt_m,vx_N_ms,vy_E_ms,vz_D_ms\n")
            for i in range(len(d["t"])):
                t_s = (d["t"][i] - t0) / 1e3
                f.write(
                    f"{t_s:.4f},"
                    f"{d['x'][i]:.4f},{d['y'][i]:.4f},{d['z'][i]:.4f},"
                    f"{-d['z'][i]:.4f},"
                    f"{d['vx'][i]:.4f},{d['vy'][i]:.4f},{d['vz'][i]:.4f}\n"
                )
        print(f"Logger: position CSV -> {out}")

    def _write_control_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._ctrl.items()}
        if not d["t"]:
            return
        t0 = d["t"][0]
        out = os.path.join(self.session_dir, "control.csv")
        with open(out, "w") as f:
            f.write("time_s,T1_BR_N,T2_BL_N,T3_FL_N,T4_FR_N,T_total_N\n")
            for i in range(len(d["t"])):
                t_s   = (d["t"][i] - t0) / 1e3
                total = d["u0"][i] + d["u1"][i] + d["u2"][i] + d["u3"][i]
                f.write(
                    f"{t_s:.4f},"
                    f"{d['u0'][i]:.4f},{d['u1'][i]:.4f},"
                    f"{d['u2'][i]:.4f},{d['u3'][i]:.4f},"
                    f"{total:.4f}\n"
                )
        print(f"Logger: control CSV -> {out}")

    def _write_carrot_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._carrot.items()}
        if not d["t"]:
            return
        t0 = d["t"][0]
        out = os.path.join(self.session_dir, "carrot.csv")
        with open(out, "w") as f:
            f.write("time_s,wp,blend_alpha,"
                    "vc_N_ms,vc_E_ms,vc_D_ms,v_cmd_ms,"
                    "carrot_N_m,carrot_E_m,carrot_D_m\n")
            for i in range(len(d["t"])):
                t_s   = (d["t"][i] - t0) / 1e3
                v_mag = (d["vcx"][i]**2 + d["vcy"][i]**2 + d["vcz"][i]**2) ** 0.5
                f.write(
                    f"{t_s:.4f},"
                    f"{d['wp'][i]},{d['alpha'][i]:.4f},"
                    f"{d['vcx'][i]:.4f},{d['vcy'][i]:.4f},{d['vcz'][i]:.4f},"
                    f"{v_mag:.4f},"
                    f"{d['cx'][i]:.4f},{d['cy'][i]:.4f},{d['cz'][i]:.4f}\n"
                )
        print(f"Logger: carrot CSV -> {out}")

    # ── EKF ───────────────────────────────────────────────────────────────

    def _euler_from_quat(self, qw, qx, qy, qz):
        """Return (phi, theta, psi) in degrees from quaternion [qw qx qy qz]."""
        phi   = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        theta = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
        psi   = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        return phi, theta, psi

    def _plot_ekf(self):
        with self._lock:
            d = {k: list(v) for k, v in self._ekf.items()}
        if not d["t_us"]:
            print("Logger: no EKF data collected, skipping plot.")
            return

        t = (np.array(d["t_us"]) - d["t_us"][0]) / 1e6   # µs -> s

        # Euler angles from quaternion
        phi_arr   = np.array([self._euler_from_quat(d["qw"][i], d["qx"][i],
                                                     d["qy"][i], d["qz"][i])[0]
                               for i in range(len(t))])
        theta_arr = np.array([self._euler_from_quat(d["qw"][i], d["qx"][i],
                                                     d["qy"][i], d["qz"][i])[1]
                               for i in range(len(t))])
        psi_arr   = np.array([self._euler_from_quat(d["qw"][i], d["qx"][i],
                                                     d["qy"][i], d["qz"][i])[2]
                               for i in range(len(t))])

        acc_norm = np.sqrt(np.array(d["ax"])**2 + np.array(d["ay"])**2 + np.array(d["az"])**2)

        # ── 5-panel EKF overview ──────────────────────────────────────────
        fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)

        # Panel 0: NED position (dead-reckoned)
        ax = axes[0]
        ax.plot(t, d["pN"], lw=0.8, label="pN (m)")
        ax.plot(t, d["pE"], lw=0.8, label="pE (m)")
        ax.plot(t, [-z for z in d["pD"]], lw=0.8, label="altitude (-pD) (m)")
        ax.fill_between(t,
                        np.array(d["pN"]) - 2*np.array(d["sig_pN"]),
                        np.array(d["pN"]) + 2*np.array(d["sig_pN"]),
                        alpha=0.12, label="pN 2-sigma")
        ax.set_ylabel("Position (m)")
        ax.set_title("EKF NED Position Estimate (dead-reckoned)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 1: Attitude
        ax = axes[1]
        ax.plot(t, phi_arr,   lw=0.8, label="phi (roll) [deg]")
        ax.plot(t, theta_arr, lw=0.8, label="theta (pitch) [deg]")
        ax.plot(t, psi_arr,   lw=0.8, label="psi (yaw) [deg]")
        ax.fill_between(t,
                        phi_arr - np.degrees(np.array(d["sig_qx"])*2),
                        phi_arr + np.degrees(np.array(d["sig_qx"])*2),
                        alpha=0.15, label="phi 2-sigma")
        ax.fill_between(t,
                        theta_arr - np.degrees(np.array(d["sig_qy"])*2),
                        theta_arr + np.degrees(np.array(d["sig_qy"])*2),
                        alpha=0.15, label="theta 2-sigma")
        ax.set_ylabel("Angle (deg)")
        ax.set_title("EKF Attitude Estimate (phi / theta / psi)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 2: NED velocity
        ax = axes[2]
        ax.plot(t, d["vN"], lw=0.8, label="vN (m/s)")
        ax.plot(t, d["vE"], lw=0.8, label="vE (m/s)")
        ax.plot(t, d["vD"], lw=0.8, label="vD (m/s)")
        ax.fill_between(t,
                        np.array(d["vN"]) - 2*np.array(d["sig_vN"]),
                        np.array(d["vN"]) + 2*np.array(d["sig_vN"]),
                        alpha=0.12, label="vN 2-sigma")
        ax.fill_between(t,
                        np.array(d["vD"]) - 2*np.array(d["sig_vD"]),
                        np.array(d["vD"]) + 2*np.array(d["sig_vD"]),
                        alpha=0.12, label="vD 2-sigma")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("EKF NED Velocity Estimate")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 3: Innovations — model accuracy vs sensor
        ax = axes[3]
        ax.plot(t, d["acc_innov"],  lw=0.8, color="tab:blue",
                label="|accel innovation| (m/s2) — 0 when update skipped")
        ax.plot(t, d["zupt_innov"], lw=0.8, color="tab:orange",
                label="|ZUPT innovation| (m/s) — 0 when update skipped")
        ax.plot(t, np.abs(np.array(acc_norm) - 9.81), lw=0.6, color="tab:green",
                linestyle="--", label="|acc_norm - g| (m/s2)")
        # Shade ZUPT active periods
        zupt_mask = np.array(d["zupt_applied"], dtype=bool)
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        where=zupt_mask, alpha=0.15, color="tab:orange",
                        label="ZUPT active")
        ax.set_ylabel("Innovation magnitude")
        ax.set_title("EKF Innovations — dyn.py model accuracy vs sensor (small = good fit)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=0)

        # Panel 4: Gyro bias estimate + raw gyro
        ax = axes[4]
        ax.plot(t, d["gx"],  lw=0.6, alpha=0.5, label="gx raw (rad/s)")
        ax.plot(t, d["gy"],  lw=0.6, alpha=0.5, label="gy raw (rad/s)")
        ax.plot(t, d["gz"],  lw=0.6, alpha=0.5, label="gz raw (rad/s)")
        ax.plot(t, d["bgx"], lw=1.2, label="bias x (rad/s)")
        ax.plot(t, d["bgy"], lw=1.2, label="bias y (rad/s)")
        ax.plot(t, d["bgz"], lw=1.2, label="bias z (rad/s)")
        ax.set_ylabel("Angular rate / bias (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("EKF Gyro Bias Estimate vs Raw Gyro")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "ekf.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: EKF plot -> {out}")

        # ── Covariance convergence (separate plot) ────────────────────────
        fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax = axes2[0]
        ax.plot(t, d["sig_vN"], lw=0.8, label="sig_vN"); ax.plot(t, d["sig_vE"], lw=0.8, label="sig_vE")
        ax.plot(t, d["sig_vD"], lw=0.8, label="sig_vD")
        ax.set_ylabel("Std dev (m/s)"); ax.set_title("EKF Velocity Uncertainty (1-sigma)")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4); ax.set_ylim(bottom=0)

        ax = axes2[1]
        ax.plot(t, np.degrees(np.array(d["sig_qx"])), lw=0.8, label="sig_phi")
        ax.plot(t, np.degrees(np.array(d["sig_qy"])), lw=0.8, label="sig_theta")
        ax.plot(t, np.degrees(np.array(d["sig_qz"])), lw=0.8, label="sig_psi")
        ax.set_ylabel("Std dev (deg)"); ax.set_title("EKF Attitude Uncertainty (1-sigma)")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4); ax.set_ylim(bottom=0)

        ax = axes2[2]
        ax.plot(t, np.degrees(np.array(d["sig_bgx"])), lw=0.8, label="sig_bgx")
        ax.plot(t, np.degrees(np.array(d["sig_bgy"])), lw=0.8, label="sig_bgy")
        ax.plot(t, np.degrees(np.array(d["sig_bgz"])), lw=0.8, label="sig_bgz")
        ax.set_ylabel("Std dev (deg/s)"); ax.set_xlabel("Time (s)")
        ax.set_title("EKF Gyro Bias Uncertainty (1-sigma)")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4); ax.set_ylim(bottom=0)

        plt.tight_layout()
        out2 = os.path.join(self.session_dir, "ekf_covariance.png")
        plt.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f"Logger: EKF covariance plot -> {out2}")

    def _write_ekf_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._ekf.items()}
        if not d["t_us"]:
            return
        t0_us = d["t_us"][0]
        out = os.path.join(self.session_dir, "ekf.csv")
        with open(out, "w") as f:
            f.write(
                "time_s,t_us,"
                "pN,pE,pD,"
                "vN,vE,vD,"
                "qw,qx,qy,qz,"
                "phi_deg,theta_deg,psi_deg,"
                "bgx,bgy,bgz,"
                "sig_pN,sig_pE,sig_pD,"
                "sig_vN,sig_vE,sig_vD,"
                "sig_phi_deg,sig_theta_deg,sig_psi_deg,"
                "sig_bgx,sig_bgy,sig_bgz,"
                "ax,ay,az,gx,gy,gz,"
                "acc_applied,acc_innov,"
                "zupt_applied,zupt_innov\n"
            )
            for i in range(len(d["t_us"])):
                t_s = (d["t_us"][i] - t0_us) / 1e6
                phi, theta, psi = self._euler_from_quat(
                    d["qw"][i], d["qx"][i], d["qy"][i], d["qz"][i])
                f.write(
                    f"{t_s:.6f},{d['t_us'][i]},"
                    f"{d['pN'][i]:.4f},{d['pE'][i]:.4f},{d['pD'][i]:.4f},"
                    f"{d['vN'][i]:.5f},{d['vE'][i]:.5f},{d['vD'][i]:.5f},"
                    f"{d['qw'][i]:.6f},{d['qx'][i]:.6f},{d['qy'][i]:.6f},{d['qz'][i]:.6f},"
                    f"{phi:.4f},{theta:.4f},{psi:.4f},"
                    f"{d['bgx'][i]:.6f},{d['bgy'][i]:.6f},{d['bgz'][i]:.6f},"
                    f"{d['sig_pN'][i]:.4f},{d['sig_pE'][i]:.4f},{d['sig_pD'][i]:.4f},"
                    f"{d['sig_vN'][i]:.5f},{d['sig_vE'][i]:.5f},{d['sig_vD'][i]:.5f},"
                    f"{float(np.degrees(d['sig_qx'][i])):.4f},"
                    f"{float(np.degrees(d['sig_qy'][i])):.4f},"
                    f"{float(np.degrees(d['sig_qz'][i])):.4f},"
                    f"{d['sig_bgx'][i]:.6f},{d['sig_bgy'][i]:.6f},{d['sig_bgz'][i]:.6f},"
                    f"{d['ax'][i]:.5f},{d['ay'][i]:.5f},{d['az'][i]:.5f},"
                    f"{d['gx'][i]:.5f},{d['gy'][i]:.5f},{d['gz'][i]:.5f},"
                    f"{d['acc_applied'][i]},{d['acc_innov'][i]:.5f},"
                    f"{d['zupt_applied'][i]},{d['zupt_innov'][i]:.5f}\n"
                )
        print(f"Logger: EKF CSV -> {out}")

    # ── Path comparison ───────────────────────────────────────────────────

    def _plot_path(self):
        """
        Plot EKF dead-reckoned trajectory vs planned waypoint path.

        Panel 1: North–East plan view (top-down map)
          - EKF path coloured by time
          - Waypoint markers numbered and connected by dashed line
        Panel 2: Altitude vs time
          - EKF altitude = −pD (m above NED origin)
          - Horizontal lines at each waypoint altitude
        Panel 3: 3-D perspective (optional)
        """
        with self._lock:
            d  = {k: list(v) for k, v in self._ekf.items()}
            dc = {k: list(v) for k, v in self._carrot.items()}
        if not d["t_us"]:
            print("Logger: no EKF data for path plot, skipping.")
            return

        t  = (np.array(d["t_us"]) - d["t_us"][0]) / 1e6   # seconds
        pN = np.array(d["pN"])
        pE = np.array(d["pE"])
        pD = np.array(d["pD"])
        alt = -pD   # altitude above NED origin (positive up)

        # The EKF position is reset to [0,0,0] at hover entry (end of WAIT phase).
        # Before that reset the dead-reckoned drift is meaningless — substitute zeros
        # so the WAIT phase does not pollute the path plot.
        # Detection: find the first near-zero crossing after the initial WAIT drift.
        pos_norm = np.abs(pN) + np.abs(pE) + np.abs(pD)
        reset_idx = 0
        drifted = np.where(pos_norm > 0.05)[0]          # first sample with >5 cm drift
        if len(drifted) > 0:
            after_drift = int(drifted[0])
            near_zero = np.where(pos_norm[after_drift:] < 0.01)[0]   # returns to <1 cm
            if len(near_zero) > 0:
                reset_idx = after_drift + int(near_zero[0])
        if reset_idx > 0:
            pN[:reset_idx]  = 0.0
            pE[:reset_idx]  = 0.0
            pD[:reset_idx]  = 0.0
            alt[:reset_idx] = 0.0

        wp = self._waypoints   # (M, 3) NED or None

        # ── Figure with 2 panels ─────────────────────────────────────────
        fig, (ax_map, ax_alt) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("EKF Dead-Reckoned Path vs Planned Waypoints", fontsize=13)

        # ── Panel 1: North–East plan view ────────────────────────────────
        sc = ax_map.scatter(pE, pN, c=t, cmap="plasma", s=1, zorder=2)
        plt.colorbar(sc, ax=ax_map, label="Time (s)")

        if wp is not None:
            wp_N = wp[:, 0]; wp_E = wp[:, 1]
            ax_map.plot(wp_E, wp_N, "k--", lw=1.2, zorder=3, label="Planned path")
            for i, (wn, we) in enumerate(zip(wp_N, wp_E)):
                ax_map.plot(we, wn, "ko", ms=8, zorder=4)
                ax_map.annotate(f"WP{i}", (we, wn),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=8, color="black", zorder=5)

        # Mark start and end of EKF path
        ax_map.plot(pE[0],  pN[0],  "g^", ms=10, zorder=6, label="Start")
        ax_map.plot(pE[-1], pN[-1], "rs", ms=10, zorder=6, label="End")

        # ── Carrot quivers ───────────────────────────────────────────────
        # Show where the carrot point is and which direction it is commanding,
        # subsampled to ~40 arrows so the map stays readable.
        if dc["t"] and len(dc["t"]) >= 2:
            _n   = len(dc["t"])
            _step = max(1, _n // 40)
            _cE  = np.array(dc["cy"])[::_step]    # carrot East  position
            _cN  = np.array(dc["cx"])[::_step]    # carrot North position
            _vE  = np.array(dc["vcy"])[::_step]   # velocity East  component
            _vN  = np.array(dc["vcx"])[::_step]   # velocity North component
            # Normalise arrow length so scale is independent of v_ref value
            _spd = np.hypot(_vN, _vE)
            _mask = _spd > 0.01
            if _mask.any():
                ax_map.quiver(
                    _cE[_mask], _cN[_mask],
                    _vE[_mask] / _spd[_mask], _vN[_mask] / _spd[_mask],
                    color="darkorange", alpha=0.75,
                    scale=25, scale_units="width", width=0.004,
                    zorder=7, label="Carrot direction",
                )

        ax_map.set_xlabel("East (m)")
        ax_map.set_ylabel("North (m)")
        ax_map.set_title("Plan View (top-down)")
        ax_map.legend(loc="best", fontsize=8)
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.grid(True, alpha=0.4)

        # ── Panel 2: Altitude vs time ─────────────────────────────────────
        ax_alt.plot(t, alt, lw=0.8, color="tab:blue", label="EKF altitude (m)")

        if wp is not None:
            colours = plt.cm.tab10(np.linspace(0, 1, len(wp)))
            for i, wpt in enumerate(wp):
                wp_alt = -wpt[2]   # NED z → altitude
                ax_alt.axhline(wp_alt, color=colours[i], lw=0.8,
                               linestyle="--", alpha=0.7, label=f"WP{i} alt={wp_alt:.1f}m")

        ax_alt.set_xlabel("Time (s)")
        ax_alt.set_ylabel("Altitude above origin (m)")
        ax_alt.set_title("Altitude Profile vs Waypoint Targets")
        ax_alt.legend(loc="best", fontsize=8)
        ax_alt.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "path.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: path plot -> {out}")

        # ── 3-D perspective ───────────────────────────────────────────────
        try:
            from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
            fig3 = plt.figure(figsize=(10, 8))
            ax3  = fig3.add_subplot(111, projection="3d")

            ax3.plot(pE, pN, alt, lw=0.8, color="tab:blue", label="EKF path")
            ax3.scatter([pE[0]], [pN[0]], [alt[0]], c="g", s=60, zorder=5, label="Start")
            ax3.scatter([pE[-1]], [pN[-1]], [alt[-1]], c="r", s=60, marker="s",
                        zorder=5, label="End")

            if wp is not None:
                ax3.plot(wp[:, 1], wp[:, 0], -wp[:, 2],
                         "k--o", lw=1.2, ms=6, label="Planned path")
                for i, wpt in enumerate(wp):
                    ax3.text(wpt[1], wpt[0], -wpt[2], f" WP{i}", fontsize=8)

            ax3.set_xlabel("East (m)")
            ax3.set_ylabel("North (m)")
            ax3.set_zlabel("Altitude (m)")
            ax3.set_title("3-D EKF Path vs Planned Waypoints")
            ax3.legend(fontsize=8)
            # Match the 2D top-down convention: East=right, North=up.
            # azim=90 places the camera north of the scene looking south:
            # near objects (south) appear at the bottom, far objects (north)
            # at the top — same as the 2D scatter where positive pN is up.
            # Camera-right aligns with +X (East), so East also goes right.
            ax3.view_init(elev=30, azim=90)

            out3 = os.path.join(self.session_dir, "path_3d.png")
            plt.savefig(out3, dpi=150)
            plt.close(fig3)
            print(f"Logger: 3-D path plot -> {out3}")
        except Exception as exc:
            print(f"Logger: 3-D plot skipped ({exc})")

    # ── Cascade controller ────────────────────────────────────────────────

    def _plot_cascade(self):
        with self._lock:
            d = {k: list(v) for k, v in self._cascade.items()}
        if not d["t"]:
            print("Logger: no cascade data collected, skipping plot.")
            return

        # Trim all series to the shortest one.  A KeyboardInterrupt between
        # two semicolon-separated appends in log_cascade can leave one list
        # one element longer than the others.
        min_n = min(len(v) for v in d.values())
        if min_n < len(d["t"]):
            print(f"Logger: cascade length mismatch — trimming to {min_n} samples.")
            d = {k: v[:min_n] for k, v in d.items()}

        t = (np.array(d["t"]) - d["t"][0]) / 1e3   # ms -> s

        fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

        # Panel 1: NED velocity — reference vs measured
        ax = axes[0]
        ax.plot(t, d["vN_meas"], lw=0.8, label="vN meas"); ax.plot(t, d["vN_ref"],  lw=1.0, ls="--", label="vN ref")
        ax.plot(t, d["vE_meas"], lw=0.8, label="vE meas"); ax.plot(t, d["vE_ref"],  lw=1.0, ls="--", label="vE ref")
        ax.plot(t, d["vD_meas"], lw=0.8, label="vD meas"); ax.plot(t, d["vD_ref"],  lw=1.0, ls="--", label="vD ref")
        ax.set_ylabel("Velocity (m/s)"); ax.set_title("Outer loop: NED velocity ref vs measured")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4)

        # Panel 2: Attitude — desired vs measured (roll, pitch, yaw)
        ax = axes[1]
        ax.plot(t, d["phi_meas"],  lw=0.8, label="phi meas");   ax.plot(t, d["phi_des"],   lw=1.0, ls="--", label="phi des")
        ax.plot(t, d["theta_meas"],lw=0.8, label="theta meas"); ax.plot(t, d["theta_des"], lw=1.0, ls="--", label="theta des")
        ax.plot(t, d["psi_meas"],  lw=0.8, label="psi meas");   ax.plot(t, d["psi_ref"],   lw=1.0, ls="--", label="psi ref")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_ylabel("Angle (deg)"); ax.set_title("Middle loop: attitude desired vs measured (phi/theta/psi)")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4)

        # Panel 3: Body rate — desired vs measured
        ax = axes[2]
        ax.plot(t, d["p_meas"], lw=0.8, label="p meas"); ax.plot(t, d["p_des"], lw=1.0, ls="--", label="p des")
        ax.plot(t, d["q_meas"], lw=0.8, label="q meas"); ax.plot(t, d["q_des"], lw=1.0, ls="--", label="q des")
        ax.plot(t, d["r_meas"], lw=0.8, label="r meas"); ax.plot(t, d["r_des"], lw=1.0, ls="--", label="r des")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_ylabel("Rate (rad/s)"); ax.set_title("Inner loop: rate desired vs measured")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.4)

        # Panel 4: Integrators and collective
        ax = axes[3]
        ax.plot(t, d["xi_N"], lw=0.8, label="xi_N"); ax.plot(t, d["xi_E"], lw=0.8, label="xi_E"); ax.plot(t, d["xi_D"], lw=0.8, label="xi_D")
        ax2 = ax.twinx()
        ax2.plot(t, d["T_coll"], lw=0.8, color="tab:red", label="T_coll (N)")
        ax2.plot(t, d["R22"],    lw=0.8, color="tab:purple", ls="--", label="R22")
        ax2.set_ylabel("T_coll (N) / R22", color="tab:red")
        ax.set_ylabel("Integrator (m)"); ax.set_xlabel("Time (s)")
        ax.set_title("Velocity integrators and collective")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "cascade.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: cascade plot -> {out}")

    def _write_cascade_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._cascade.items()}
        if not d["t"]:
            return
        min_n = min(len(v) for v in d.values())
        if min_n < len(d["t"]):
            d = {k: v[:min_n] for k, v in d.items()}
        t0 = d["t"][0]
        out = os.path.join(self.session_dir, "cascade.csv")
        with open(out, "w") as f:
            f.write(
                "time_s,"
                "vN_ref,vE_ref,vD_ref,vN_meas,vE_meas,vD_meas,"
                "eN,eE,eD,xi_N,xi_E,xi_D,"
                "phi_des_deg,theta_des_deg,phi_meas_deg,theta_meas_deg,"
                "phi_err_deg,theta_err_deg,"
                "psi_meas_deg,psi_ref_deg,psi_err_deg,"
                "p_des,q_des,r_des,p_meas,q_meas,r_meas,"
                "T_coll,R22\n"
            )
            for i in range(len(d["t"])):
                t_s = (d["t"][i] - t0) / 1e3
                f.write(
                    f"{t_s:.4f},"
                    f"{d['vN_ref'][i]:.4f},{d['vE_ref'][i]:.4f},{d['vD_ref'][i]:.4f},"
                    f"{d['vN_meas'][i]:.4f},{d['vE_meas'][i]:.4f},{d['vD_meas'][i]:.4f},"
                    f"{d['eN'][i]:.4f},{d['eE'][i]:.4f},{d['eD'][i]:.4f},"
                    f"{d['xi_N'][i]:.4f},{d['xi_E'][i]:.4f},{d['xi_D'][i]:.4f},"
                    f"{d['phi_des'][i]:.3f},{d['theta_des'][i]:.3f},"
                    f"{d['phi_meas'][i]:.3f},{d['theta_meas'][i]:.3f},"
                    f"{d['phi_err'][i]:.3f},{d['theta_err'][i]:.3f},"
                    f"{d['psi_meas'][i]:.3f},{d['psi_ref'][i]:.3f},{d['psi_err'][i]:.3f},"
                    f"{d['p_des'][i]:.4f},{d['q_des'][i]:.4f},{d['r_des'][i]:.4f},"
                    f"{d['p_meas'][i]:.4f},{d['q_meas'][i]:.4f},{d['r_meas'][i]:.4f},"
                    f"{d['T_coll'][i]:.3f},{d['R22'][i]:.4f}\n"
                )
        print(f"Logger: cascade CSV -> {out}")

    # ── Vision / YOLO / PnP ──────────────────────────────────────────────

    def _write_vision_csv(self):
        with self._lock:
            d = {k: list(v) for k, v in self._vision.items()}
        if not d["wall_t"]:
            print("Logger: no vision data collected, skipping CSV.")
            return
        t0 = d["wall_t"][0]
        out = os.path.join(self.session_dir, "vision.csv")
        with open(out, "w") as f:
            f.write(
                "time_s,frame_id,detected,conf,"
                "bb_cx,bb_cy,bb_w,bb_h,"
                "kp0x,kp0y,kp1x,kp1y,kp2x,kp2y,kp3x,kp3y,"
                "pnp_ok,tvec_x,tvec_y,tvec_z,"
                "pos_N,pos_E,pos_D,"
                "vel_ok,vel_N,vel_E,vel_D,speed_ms\n"
            )
            for i in range(len(d["wall_t"])):
                t_s = d["wall_t"][i] - t0

                def _f(v):
                    return f"{v:.4f}" if v == v else "nan"   # nan-safe formatter

                f.write(
                    f"{t_s:.4f},{d['frame_id'][i]},{d['detected'][i]},{_f(d['conf'][i])},"
                    f"{_f(d['bb_cx'][i])},{_f(d['bb_cy'][i])},"
                    f"{_f(d['bb_w'][i])},{_f(d['bb_h'][i])},"
                    f"{_f(d['kp0x'][i])},{_f(d['kp0y'][i])},"
                    f"{_f(d['kp1x'][i])},{_f(d['kp1y'][i])},"
                    f"{_f(d['kp2x'][i])},{_f(d['kp2y'][i])},"
                    f"{_f(d['kp3x'][i])},{_f(d['kp3y'][i])},"
                    f"{d['pnp_ok'][i]},{_f(d['tvec_x'][i])},{_f(d['tvec_y'][i])},{_f(d['tvec_z'][i])},"
                    f"{_f(d['pos_N'][i])},{_f(d['pos_E'][i])},{_f(d['pos_D'][i])},"
                    f"{d['vel_ok'][i]},{_f(d['vel_N'][i])},{_f(d['vel_E'][i])},{_f(d['vel_D'][i])},{_f(d['speed_ms'][i])}\n"
                )
        print(f"Logger: vision CSV -> {out}")

    def _plot_vision(self):
        with self._lock:
            d = {k: list(v) for k, v in self._vision.items()}
        if not d["wall_t"] or not any(d["detected"]):
            print("Logger: no gate detections in vision log, skipping plot.")
            return

        t0  = d["wall_t"][0]
        t   = np.array(d["wall_t"]) - t0
        det = np.array(d["detected"], dtype=bool)

        # NaN-safe arrays for detected-only metrics
        conf    = np.array(d["conf"],    dtype=float)
        dist    = np.array(d["tvec_z"], dtype=float)   # forward distance [m]
        pnp_ok  = np.array(d["pnp_ok"], dtype=bool)
        vel_ok  = np.array(d["vel_ok"], dtype=bool)
        pos_N   = np.array(d["pos_N"],  dtype=float)
        pos_E   = np.array(d["pos_E"],  dtype=float)
        pos_D   = np.array(d["pos_D"],  dtype=float)
        speed   = np.array(d["speed_ms"], dtype=float)

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # Panel 0: Detection flag + confidence
        ax = axes[0]
        ax.fill_between(t, 0, det.astype(float), step="post",
                        alpha=0.3, color="tab:green", label="detected")
        ax2 = ax.twinx()
        ax2.plot(t[det], conf[det], ".", ms=3, color="tab:blue", label="confidence")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Confidence")
        ax.set_ylabel("Detected (0/1)")
        ax.set_title("YOLO Gate Detection — rate and confidence")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 1: Gate forward distance from PnP (tvec_z)
        ax = axes[1]
        ax.plot(t[pnp_ok], dist[pnp_ok], ".", ms=2, color="tab:orange", label="distance (m)")
        ax.set_ylabel("Gate distance [m]")
        ax.set_title("PnP — forward distance to gate (tvec_z)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 2: Drone NED position estimate from PnP
        ax = axes[2]
        ax.plot(t[pnp_ok], pos_N[pnp_ok], ".", ms=2, label="pos_N (m)")
        ax.plot(t[pnp_ok], pos_E[pnp_ok], ".", ms=2, label="pos_E (m)")
        ax.plot(t[pnp_ok], -pos_D[pnp_ok], ".", ms=2, label="altitude -pos_D (m)")
        ax.set_ylabel("Position (m)")
        ax.set_title("PnP Drone NED Position Estimate")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        # Panel 3: Speed from PnP velocity (consecutive frames only)
        ax = axes[3]
        ax.plot(t[vel_ok], speed[vel_ok], ".", ms=2, color="tab:red", label="speed (m/s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("PnP Velocity Estimate — speed (consecutive frames only)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        out = os.path.join(self.session_dir, "vision.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Logger: vision plot -> {out}")
