# -*- coding: utf-8 -*-
"""
ned_calibration.py -- NED heading and coordinate-system diagnostics.

Reads the latest log's ekf.csv and carrot.csv and runs five self-contained
tests that verify the NED frame is configured correctly.  No ground-truth
signals are required (LOCAL_POSITION_NED / ATTITUDE / ODOMETRY / GATE_INFO are
all absent in this sim configuration).

Usage:
    python ned_calibration.py [log_dir]

If log_dir is omitted the most-recent subdirectory of logs/ is used.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _latest_log(logs_root="logs"):
    dirs = sorted(glob.glob(os.path.join(logs_root, "*")))
    if not dirs:
        raise FileNotFoundError("No log directories found under '%s'" % logs_root)
    return dirs[-1]

def _pass(msg): print("  [PASS]  " + msg)
def _fail(msg): print("  [FAIL]  " + msg)
def _info(msg): print("          " + msg)
def _warn(msg): print("  [WARN]  " + msg)

PASS = True
FAIL = False


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_initial_heading(ekf, expected_yaw_deg, tol_deg=10.0, pos_reset_thresh=10.0):
    """
    T1: EKF yaw immediately after hover-entry reset should match initial_yaw_deg.

    The hover-entry reset zeroes the EKF position.  The log may include a
    pre-reset WAIT phase (with drifted or diverged psi), so T1 finds the reset
    moment rather than reading the first 50 rows blindly.

    Detection: the reset row is the first row where ALL of |pN|, |pE|, |pD| are
    below pos_reset_thresh AND at least one of the preceding rows had a larger
    position (avoids false-triggering when the log starts after a clean reset).
    Falls back to the first 50 rows if no reset transition is detected.
    """
    print("\n[T1] Initial heading check (post-hover-reset)")
    pos_mag = (ekf["pN"].abs() + ekf["pE"].abs() + ekf["pD"].abs()).values
    psi_arr = ekf["psi_deg"].values

    # Detect hover-reset: look for the first drop of pos_mag from above the
    # threshold to below it.  BUT if the log already starts near the origin
    # (pos_mag[0] < threshold), that means the EKF was reset before logging
    # began -- use the first 50 rows directly.  Searching for a later crossing
    # would falsely trigger on the drone returning to the origin mid-flight.
    reset_idx = None
    if pos_mag[0] < pos_reset_thresh:
        reset_idx = 0
        _info("Log starts near origin -- T1 uses first 50 rows")
    else:
        for i in range(1, len(pos_mag)):
            if pos_mag[i] < pos_reset_thresh and pos_mag[i - 1] >= pos_reset_thresh:
                reset_idx = i
                break
        if reset_idx is None:
            _warn("Cannot locate hover-reset transition -- T1 may be unreliable")
            reset_idx = 0

    window = psi_arr[reset_idx : reset_idx + 50]
    psi_early = float(np.mean(window))
    err = abs(((psi_early - expected_yaw_deg) + 180) % 360 - 180)
    _info("Hover-reset detected at EKF row %d (pos_mag %.1f -> %.1f m)" %
          (reset_idx, pos_mag[reset_idx - 1] if reset_idx > 0 else 0.0, pos_mag[reset_idx]))
    _info("EKF psi after hover-reset (mean of 50 rows): %.1f deg" % psi_early)
    _info("Expected (initial_yaw_deg): %.1f deg  |error|: %.1f deg" % (expected_yaw_deg, err))
    if err < tol_deg:
        _pass("|psi_err| = %.1f deg < %.1f deg" % (err, tol_deg))
        return PASS
    else:
        _fail("|psi_err| = %.1f deg >= %.1f deg" % (err, tol_deg))
        suggestion = round(expected_yaw_deg - (psi_early - expected_yaw_deg), 0)
        _info("Hint: try initial_yaw_deg: %.0f in params.yaml" % suggestion)
        return FAIL


def test_velocity_heading(ekf, expected_yaw_deg, tol_deg=25.0, speed_thresh=0.5):
    """T2: First sustained velocity should point in the expected direction."""
    print("\n[T2] Velocity heading alignment")
    vh = np.sqrt(ekf["vN"] ** 2 + ekf["vE"] ** 2)
    fast = ekf[vh > speed_thresh]
    if fast.empty:
        _warn("Drone never reached %.1f m/s horizontal speed -- skip T2" % speed_thresh)
        return None
    window = fast.iloc[:20]
    vN_mean = window["vN"].mean()
    vE_mean = window["vE"].mean()
    vel_heading = np.rad2deg(np.arctan2(vE_mean, vN_mean))
    err = abs(((vel_heading - expected_yaw_deg) + 180) % 360 - 180)
    _info("Mean velocity at first fast window: vN=%.2f  vE=%.2f m/s" % (vN_mean, vE_mean))
    _info("Velocity heading: %.1f deg  Expected: %.1f deg  |error|: %.1f deg" %
          (vel_heading, expected_yaw_deg, err))
    if err < tol_deg:
        _pass("|heading_err| = %.1f deg < %.1f deg" % (err, tol_deg))
        return PASS
    else:
        _fail("|heading_err| = %.1f deg >= %.1f deg" % (err, tol_deg))
        _info("Hint: initial_yaw_deg or gyro sign convention may be wrong")
        return FAIL


def test_lambda_L_progression(ekf, waypoints, skip_sec=3.5, min_progress_m=2.0):
    """
    T3: Progress along the path (lambda_L) should increase after the blip.
    lambda_L = (pos - r0) . ea  where ea is the first segment unit tangent.
    """
    print("\n[T3] Path-progress (lambda_L) test")
    wps = [np.array(w, dtype=float) for w in waypoints]
    if len(wps) < 2:
        _warn("Need at least 2 waypoints -- skip T3")
        return None
    r0  = wps[0]
    r1  = wps[1]
    seg = r1 - r0
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        _warn("First segment has zero length -- skip T3")
        return None
    ea = seg / seg_len   # unit path tangent (world NED = local NED after reset)

    pos_local = ekf[["pN", "pE", "pD"]].values   # (N,3) local NED
    lambda_L  = pos_local @ ea                     # scalar progress [m]

    t     = ekf["time_s"].values
    after = t > skip_sec
    if not np.any(after):
        _warn("No samples after t=%.1fs -- skip T3" % skip_sec)
        return None

    lam_after = lambda_L[after]
    t_after   = t[after]
    progress  = lam_after[-1] - lam_after[0]
    duration  = t_after[-1] - t_after[0]

    _info("Path tangent ea = [%.3f, %.3f, %.3f]" % (ea[0], ea[1], ea[2]))
    _info("lambda_L after blip: %.2f -> %.2f m  (progress: %+.2f m over %.1fs)" %
          (lam_after[0], lam_after[-1], progress, duration))

    if progress >= min_progress_m:
        _pass("lambda_L increased by %.2f m >= %.2f m -- correct NED direction" %
              (progress, min_progress_m))
        return PASS
    elif progress > 0:
        _warn("lambda_L increased by only %.2f m (expected >= %.2f m)" %
              (progress, min_progress_m))
        return FAIL
    else:
        _fail("lambda_L DECREASED by %.2f m" % abs(progress))
        _info("If T1+T2 passed: EKF diverged mid-flight (not a NED direction error)")
        _info("If T1+T2 failed: negate initial_yaw_deg or flip the waypoint signs")
        return FAIL


def test_psi_stability(ekf, expected_yaw_deg, skip_sec=3.5,
                       max_std_deg=15.0, max_drift_deg=30.0):
    """T4: After the blip, yaw should stay near initial_yaw_deg with low variance."""
    print("\n[T4] Yaw stability test")
    t     = ekf["time_s"].values
    after = t > skip_sec
    if not np.any(after):
        _warn("No samples after t=%.1fs -- skip T4" % skip_sec)
        return None

    psi     = ekf["psi_deg"].values[after]
    psi_uw  = np.rad2deg(np.unwrap(np.deg2rad(psi)))
    std_psi = float(np.std(psi_uw))
    drift   = float(abs(np.mean(psi_uw[-10:]) - np.mean(psi_uw[:10])))
    mean_psi = float(np.mean(psi_uw))
    bias     = abs(((mean_psi - expected_yaw_deg) + 180) % 360 - 180)

    _info("psi_meas after blip: mean=%.1f deg  std=%.1f deg  drift=%.1f deg  bias=%.1f deg" %
          (mean_psi, std_psi, drift, bias))

    ok = True
    if std_psi > max_std_deg:
        _fail("std(psi) = %.1f deg > %.1f deg -- yaw is oscillating / drifting" %
              (std_psi, max_std_deg))
        ok = False
    if drift > max_drift_deg:
        _fail("Yaw drifted %.1f deg > %.1f deg -- yaw runaway detected" %
              (drift, max_drift_deg))
        ok = False
    if ok:
        _pass("std=%.1f deg  drift=%.1f deg -- yaw is stable" % (std_psi, drift))
    return PASS if ok else FAIL


def test_carrot_proximity(carrot, ekf, skip_sec=3.5, max_dist_m=20.0):
    """
    T5: Carrot should stay within max_dist_m of the drone.
    Uses drone_N_m columns if present; falls back to interpolated EKF position.
    """
    print("\n[T5] Carrot proximity test")
    carrot_after = carrot[carrot["time_s"] > skip_sec]
    if carrot_after.empty:
        _warn("No carrot samples after t=%.1fs -- skip T5" % skip_sec)
        return None

    has_drone_cols = all(c in carrot.columns
                         for c in ["drone_N_m", "drone_E_m", "drone_D_m"])

    if has_drone_cols:
        drone_N = carrot_after["drone_N_m"].values
        drone_E = carrot_after["drone_E_m"].values
        drone_D = carrot_after["drone_D_m"].values
        _info("Using drone_N_m/E_m/D_m columns from carrot.csv")
    else:
        ekf_t   = ekf["time_s"].values
        t_c     = carrot_after["time_s"].values
        drone_N = np.interp(t_c, ekf_t, ekf["pN"].values)
        drone_E = np.interp(t_c, ekf_t, ekf["pE"].values)
        drone_D = np.interp(t_c, ekf_t, ekf["pD"].values)
        _warn("drone_N_m columns absent -- using interpolated EKF position")

    carr_N = carrot_after["carrot_N_m"].values
    carr_E = carrot_after["carrot_E_m"].values
    carr_D = carrot_after["carrot_D_m"].values

    dist   = np.sqrt((carr_N - drone_N)**2 + (carr_E - drone_E)**2 + (carr_D - drone_D)**2)
    max_d  = float(dist.max())
    mean_d = float(dist.mean())

    _info("Carrot-drone distance: mean=%.2f m  max=%.2f m" % (mean_d, max_d))
    if max_d <= max_dist_m:
        _pass("Max distance %.2f m <= %.2f m -- carrot tracking OK" % (max_d, max_dist_m))
        return PASS
    else:
        _fail("Max distance %.2f m > %.2f m -- carrot diverged from drone" %
              (max_d, max_dist_m))
        _info("Hint: may indicate wrong path direction or NED coordinate mismatch")
        return FAIL


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else _latest_log()
    print("Log directory: " + log_dir)

    ekf_path    = os.path.join(log_dir, "ekf.csv")
    carrot_path = os.path.join(log_dir, "carrot.csv")

    if not os.path.exists(ekf_path):
        print("ERROR: " + ekf_path + " not found"); sys.exit(1)
    if not os.path.exists(carrot_path):
        print("ERROR: " + carrot_path + " not found"); sys.exit(1)

    ekf    = pd.read_csv(ekf_path)
    carrot = pd.read_csv(carrot_path)

    params    = _load_params()
    yaw_deg   = float(params.get("initial_yaw_deg", 0.0))
    waypoints = params.get("waypoints", [])

    print("Params: initial_yaw_deg=%.1f deg  n_waypoints=%d" % (yaw_deg, len(waypoints)))
    print("EKF rows: %d  Carrot rows: %d" % (len(ekf), len(carrot)))
    print("EKF time span: %.1fs to %.1fs" % (ekf["time_s"].min(), ekf["time_s"].max()))

    results = {}
    results["T1_heading"]          = test_initial_heading(ekf, yaw_deg)
    results["T2_vel_heading"]      = test_velocity_heading(ekf, yaw_deg)
    results["T3_lambda_L"]         = test_lambda_L_progression(ekf, waypoints)
    results["T4_psi_stability"]    = test_psi_stability(ekf, yaw_deg)
    results["T5_carrot_proximity"] = test_carrot_proximity(carrot, ekf)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        if result is True:
            print("  PASS  " + name)
        elif result is False:
            print("  FAIL  " + name)
            all_pass = False
        else:
            print("  SKIP  " + name)

    if all_pass:
        print("\nAll tests passed -- NED frame looks correct.")
    else:
        print("\nOne or more tests failed.  Check the hints above.")

    # EKF path summary
    print("\n-- EKF path summary (" + log_dir + ") --")
    for ax, col in [("North", "pN"), ("East", "pE"), ("Down->alt", "pD")]:
        v = ekf[col]
        print("  %-12s: %+8.2f to %+8.2f m  (net: %+.2f m)" %
              (ax, v.min(), v.max(), v.iloc[-1] - v.iloc[0]))


if __name__ == "__main__":
    main()
