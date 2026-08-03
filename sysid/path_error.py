"""
path_error.py — per-segment reference-path-following metrics for one flight log.

    py -m sysid.path_error logs/<session>            # from repo root
    py -m sysid.path_error logs/<a> logs/<b>         # baseline vs after

Read-only: reads the session CSVs and params.yaml, writes nothing.

WHY THIS EXISTS
---------------
"Does the drone follow the straight line between gates?" was being answered with
ad-hoc shell one-liners, and two independent attempts at it disagreed on the
per-segment numbers purely because of windowing choices. This module pins the
metric definition so successive flights are actually comparable, and so a
regression gate can be stated in numbers rather than in impressions.

Three separate frame/timebase traps are handled explicitly, because every one of
them has already produced a wrong conclusion at least once in this project:

1. carrot.csv's time_s is NOT ekf.csv's time_s. carrot.csv zeroes at the first
   log_carrot() call (once _carrot_active begins, i.e. after WAIT/hover-entry/
   leveling); ekf.csv zeroes at the first IMU message, seconds earlier. Comparing
   them directly as if they shared an axis produced a phantom "EKF diverged 60 m"
   and a phantom collision report. Here the offset is RECOVERED by aligning the
   two logs' shared EKF position trace, then asserted to fit within a few cm.

2. params.yaml waypoints are WORLD NED; ekf.csv/ekf_gt_error_shadow.csv positions
   are LOCAL (zeroed at hover entry). controller.py subtracts a LATERAL-only
   offset (N,E; D deliberately left alone) from the waypoints at carrot
   activation — mirrored here. The offset is estimated from the data rather than
   assumed, since it is not logged directly.

3. ekf_gt_error_shadow.csv's yaw_err_deg column is a MIRRORED ARTIFACT and is
   deliberately not used. mavlink_rx.on_attitude negates sim pitch but leaves sim
   yaw sign-inverted on purpose (GT mode depends on it), and the shadow writer
   compares EKF yaw against that un-flipped value — so the column reports
   yaw_ekf - (-yaw_true). On this ~180 deg-heading track that read 25-38 deg when
   the true error was under 10. This module recomputes yaw error with the flip
   applied unconditionally.
"""

import csv
import os
import sys

import numpy as np
import yaml

# Along-track fraction defining the "mid-segment" window. The turn-in at each end
# of a segment is legitimately off-line (corner-cut is intentional, see
# carrot_tracker.py), so the bow in the MIDDLE — where the drone should simply be
# flying the line — is reported separately rather than being averaged in with it.
MID_LO, MID_HI = 0.20, 0.80

# Max lateral world->local offset we will silently accept when reconstructing the
# waypoint frame. Measured ~0.12 m on a healthy log; a metre-scale value means the
# offset estimate itself is untrustworthy and every cross-track number below would
# inherit that error, so we refuse rather than report something plausible-looking.
MAX_TRUSTED_OFFSET_M = 2.0

# Max acceptable residual when fitting carrot.csv's time axis onto ekf.csv's.
# The two logs record the same EKF position, so a good fit is centimetres; a large
# residual means the alignment landed on the wrong shift and segment membership
# (hence every per-segment number) would be silently misattributed.
MAX_TIME_ALIGN_RESID_M = 0.30


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _col(rows, *names, dtype=float):
    return np.array([[dtype(r[n]) for n in names] for r in rows])


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _interp_rows(t_query, t_ref, v_ref):
    """Column-wise linear interpolation of v_ref(t_ref) at t_query."""
    return np.column_stack([np.interp(t_query, t_ref, v_ref[:, i])
                            for i in range(v_ref.shape[1])])


def _cross_track(pts, r0, ea):
    """Perpendicular distance of each row of pts from the infinite line through
    r0 with unit tangent ea, plus the along-track coordinate. 2-D (N,E) only:
    altitude is a separate control axis with its own reference and mixing it in
    would let a vertical profile difference masquerade as path deviation."""
    rel = pts - r0
    along = rel @ ea
    perp = rel - np.outer(along, ea)
    return np.linalg.norm(perp, axis=1), along


def _estimate_pos_offset(session):
    """Recover controller.py's world->local lateral shift.

    Not logged anywhere, so it is measured: vision_fix.csv publishes the PnP
    drone position in WORLD NED, while ekf.csv publishes the same instant in
    LOCAL NED, and imu_ekf.py bridges them with exactly
    `local = world - pos_offset_ned`. The median over every accepted fix rejects
    the per-fix PnP error (~1 m, and skewed) far better than a mean would.
    """
    ekf = _read_csv(os.path.join(session, 'ekf.csv'))
    vfx = _read_csv(os.path.join(session, 'vision_fix.csv'))
    e_t = _col(ekf, 'wall_t')[:, 0]
    e_p = _col(ekf, 'pN', 'pE', 'pD')

    have = [r for r in vfx if r['has_pos'] in ('1', 'True')]
    if not have:
        raise SystemExit(f'{session}: vision_fix.csv has no accepted position fixes '
                         '— cannot reconstruct the waypoint frame.')
    v_t = _col(have, 'wall_t')[:, 0]
    v_p = _col(have, 'pos_N', 'pos_E', 'pos_D')

    diff = v_p - _interp_rows(v_t, e_t, e_p)
    off = np.median(diff, axis=0)
    spread = np.percentile(diff, 75, axis=0) - np.percentile(diff, 25, axis=0)

    if np.linalg.norm(off[:2]) > MAX_TRUSTED_OFFSET_M:
        raise SystemExit(
            f'{session}: world->local lateral offset estimated at {off[:2]} m '
            f'(|.|={np.linalg.norm(off[:2]):.2f} > {MAX_TRUSTED_OFFSET_M} m). '
            'The waypoint-frame reconstruction is not trustworthy, so every '
            'cross-track number would be wrong by that amount. Refusing to report.')
    return off, spread, len(have)


def _align_carrot_clock(session):
    """Recover the constant that maps carrot.csv time_s onto wall-clock.

    carrot.csv stores wall-clock ms but the writer subtracts its own first
    sample, so the zero point is unrecoverable from the file alone (see this
    module's docstring, trap 1). Both logs record the same EKF position though,
    so the shift is found by scanning for the one that makes carrot's
    drone_N/E/D coincide with ekf.csv's pN/pE/pD, coarse-to-fine.
    """
    ekf = _read_csv(os.path.join(session, 'ekf.csv'))
    car = _read_csv(os.path.join(session, 'carrot.csv'))
    e_t = _col(ekf, 'wall_t')[:, 0]
    e_p = _col(ekf, 'pN', 'pE', 'pD')
    c_t = _col(car, 'time_s')[:, 0]
    c_p = _col(car, 'drone_N_m', 'drone_E_m', 'drone_D_m')

    # Subsample: the fit is over-determined by thousands of rows and this keeps
    # the scan fast without changing the optimum.
    idx = np.arange(0, len(c_t), max(1, len(c_t) // 400))
    ct, cp = c_t[idx], c_p[idx]

    def resid(shift):
        t = ct + shift
        if t[0] < e_t[0] or t[-1] > e_t[-1]:
            return np.inf
        return float(np.median(np.linalg.norm(cp - _interp_rows(t, e_t, e_p), axis=1)))

    lo, hi = e_t[0] - c_t[-1], e_t[-1] - c_t[-1]
    best, step = None, max((hi - lo) / 500.0, 1e-3)
    for shift in np.arange(lo, hi + step, step):
        r = resid(shift)
        if best is None or r < best[1]:
            best = (shift, r)
    for _ in range(6):                       # refine around the coarse optimum
        step /= 6.0
        for shift in np.arange(best[0] - 3 * step, best[0] + 3 * step, step):
            r = resid(shift)
            if r < best[1]:
                best = (shift, r)

    if best[1] > MAX_TIME_ALIGN_RESID_M:
        raise SystemExit(
            f'{session}: could not align carrot.csv onto the wall clock '
            f'(best residual {best[1]:.2f} m > {MAX_TIME_ALIGN_RESID_M} m). '
            'Segment membership would be misattributed. Refusing to report.')
    return best[0], best[1]


def _yaw_error(shadow, yaw, yaw_gt_raw):
    """Yaw error in degrees, correct for both the pre- and post-fix log formats.

    Returns (abs_error_per_sample, best_available_true_gt_yaw).

    The two formats are told apart exactly, not heuristically. The OLD writer
    computed yaw_err_deg = wrap(yaw_deg - yaw_gt_deg) verbatim, so an old log
    reproduces that identity to floating point. The FIXED writer subtracts a
    separately yaw-flipped GT instead, so the identity no longer holds (except
    in the degenerate yaw_gt == 0 case, where both agree anyway and the choice
    does not matter).

    Preferring the stored column on fixed logs is not just tidiness: the writer
    applies the flip BEFORE composing with the GT frame-alignment rotation,
    which is the frame-exact order. Recomputing here from the composed output
    can only flip afterwards, and the two differ by twice the alignment angle.
    That angle is ~0 on this track, but relying on it is the kind of silent
    frame assumption that has already produced wrong conclusions in this
    project, so the exact value is used whenever the log carries one.
    """
    stored = _col(shadow, 'yaw_err_deg')[:, 0]
    naive = _wrap180(yaw - yaw_gt_raw)
    # Tolerance is set by the CSV's 3-decimal precision (all three columns are
    # rounded independently), not by float epsilon. It is nowhere near tight
    # enough to matter: the two formats differ by twice the GT yaw, i.e. tens to
    # hundreds of degrees on any real flight, so anything from 1e-3 to ~1 degree
    # separates them identically.
    is_old_format = float(np.median(np.abs(_wrap180(stored - naive)))) < 0.01

    if is_old_format:
        # Pre-fix log: the stored column is yaw_ekf - (-yaw_true). Recover the
        # real error by flipping the GT yaw ourselves (see module docstring).
        true_gt = -yaw_gt_raw
        return np.abs(_wrap180(yaw - true_gt)), true_gt
    return np.abs(stored), -yaw_gt_raw


def analyse(session, params_path='params.yaml'):
    params = yaml.safe_load(open(params_path))
    wps_world = np.array(params['waypoints'], dtype=float)

    offset, off_spread, n_fix = _estimate_pos_offset(session)
    t_shift, t_resid = _align_carrot_clock(session)

    # Mirror controller.py's activation-time shift exactly: N and E only, D left
    # alone (the launch slope sits several metres above the track floor, so
    # applying the D offset would command a dive).
    wps = wps_world.copy()
    wps[:, 0] -= offset[0]
    wps[:, 1] -= offset[1]

    car = _read_csv(os.path.join(session, 'carrot.csv'))
    c_t = _col(car, 'time_s')[:, 0] + t_shift
    c_wp = _col(car, 'wp', dtype=int)[:, 0]
    c_car = _col(car, 'carrot_N_m', 'carrot_E_m')

    shadow = _read_csv(os.path.join(session, 'ekf_gt_error_shadow.csv'))
    s_t = _col(shadow, 't_wall_s')[:, 0]
    s_gt = _col(shadow, 'gt_pN', 'gt_pE')
    s_ekf = _col(shadow, 'pN', 'pE')
    s_gtv = _col(shadow, 'gt_vN', 'gt_vE')
    s_yaw = _col(shadow, 'yaw_deg')[:, 0]
    s_yaw_gt_raw = _col(shadow, 'yaw_gt_deg')[:, 0]
    s_yaw_err, s_yaw_true = _yaw_error(shadow, s_yaw, s_yaw_gt_raw)

    # Segment membership comes from carrot.csv's own wp counter (the authority on
    # which segment the controller believed it was flying), carried onto the
    # shadow samples through the recovered clock.
    seg_of_shadow = np.interp(s_t, c_t, c_wp, left=np.nan, right=np.nan)
    seg_of_shadow = np.where(np.isfinite(seg_of_shadow),
                             np.round(seg_of_shadow), np.nan)

    print(f'\n{"=" * 78}\n{session}\n{"=" * 78}')
    print(f'world->local lateral offset : N{offset[0]:+.3f} E{offset[1]:+.3f} m '
          f'(from {n_fix} vision fixes, IQR N{off_spread[0]:.2f}/E{off_spread[1]:.2f})')
    print(f'carrot.csv clock alignment  : residual {t_resid:.3f} m')

    rows = []
    for wp in range(1, len(wps)):
        m = seg_of_shadow == wp
        if m.sum() < 3:
            continue
        r0, r1 = wps[wp - 1, :2], wps[wp, :2]
        ea = (r1 - r0) / np.linalg.norm(r1 - r0)
        seg_len = float(np.linalg.norm(r1 - r0))

        gt_xt, gt_along = _cross_track(s_gt[m], r0, ea)
        ekf_xt, _ = _cross_track(s_ekf[m], r0, ea)
        lat_err = np.abs(gt_xt - ekf_xt)

        yaw_err = s_yaw_err[m]
        spd = np.linalg.norm(s_gtv[m], axis=1)
        moving = spd > 4.0
        crab = (np.abs(_wrap180(np.degrees(np.arctan2(s_gtv[m][:, 1], s_gtv[m][:, 0]))
                                - s_yaw_true[m]))[moving]
                if moving.any() else np.array([0.0]))

        frac = gt_along / max(seg_len, 1e-9)
        mid = (frac >= MID_LO) & (frac <= MID_HI)

        cm = (c_wp == wp)
        car_xt = _cross_track(c_car[cm], r0, ea)[0] if cm.sum() else np.array([np.nan])

        rows.append(dict(
            wp=wp, n=int(m.sum()),
            gt_max=gt_xt.max(), gt_mean=gt_xt.mean(),
            gt_mid=gt_xt[mid].max() if mid.any() else np.nan,
            ekf_max=ekf_xt.max(),
            optimism=gt_xt.max() / ekf_xt.max() if ekf_xt.max() > 1e-6 else np.nan,
            lat_max=lat_err.max(), yaw_max=yaw_err.max(), yaw_med=np.median(yaw_err),
            crab_max=crab.max(), carrot_max=np.nanmax(car_xt),
        ))

    print('\n  PRIMARY - cross-track vs surveyed line (m), whole segment unless noted')
    print(f'  {"seg":>4} {"n":>5} {"GT max":>7} {"GT mean":>8} {"GT mid":>7} '
          f'{"EKF max":>8} {"optimism":>9}')
    for r in rows:
        print(f'  {r["wp"]:>4} {r["n"]:>5} {r["gt_max"]:>7.2f} {r["gt_mean"]:>8.2f} '
              f'{r["gt_mid"]:>7.2f} {r["ekf_max"]:>8.2f} {r["optimism"]:>9.2f}')

    print('\n  SECONDARY')
    print(f'  {"seg":>4} {"EKF lat err":>12} {"yaw err max":>12} {"yaw err med":>12} '
          f'{"crab max":>9} {"carrot off":>11}')
    for r in rows:
        print(f'  {r["wp"]:>4} {r["lat_max"]:>12.2f} {r["yaw_max"]:>12.2f} '
              f'{r["yaw_med"]:>12.2f} {r["crab_max"]:>9.1f} {r["carrot_max"]:>11.2f}')

    _gate_clearance(session, wps, s_t, s_gt, c_t, c_wp)
    return rows


def _gate_clearance(session, wps, s_t, s_gt, c_t, c_wp):
    """Closest approach to each gate centre — the miss-safety check.

    Any clamp added to the live-target path trades accuracy for safety margin,
    and this is the number that says whether the trade went too far: the gates
    are ~2.7 m wide, so a pass more than ~1.35 m off centre strikes the frame.
    SMALL IS GOOD here — this is distance from the middle of the opening, not
    clearance from an obstacle.

    A gate the drone never reached (flight ended early, or it advanced past
    without ever closing) is reported as such rather than as a large miss: an
    unreached gate says nothing about tracking accuracy and would otherwise
    read as the worst result in the table.
    """
    print('\n  GATE CLEARANCE (closest approach to surveyed centre - SMALL IS GOOD, '
          '>1.35 m = frame strike)')
    for wp in range(1, len(wps)):
        m = c_wp == wp
        if not m.any():
            continue
        t0, t1 = c_t[m][0], c_t[m][-1]
        sm = (s_t >= t0) & (s_t <= t1)
        if not sm.any():
            continue
        pts = s_gt[sm]
        d = np.linalg.norm(pts - wps[wp, :2], axis=1)

        # Did the drone actually cross the gate plane? Project onto the segment
        # tangent and check the along-track coordinate passed the gate itself.
        r0 = wps[wp - 1, :2]
        ea = (wps[wp, :2] - r0) / np.linalg.norm(wps[wp, :2] - r0)
        reached = float(np.max((pts - r0) @ ea)) >= float((wps[wp, :2] - r0) @ ea)

        if not reached:
            print(f'    gate {wp}: not reached (closest {d.min():.2f} m) '
                  '— excluded from miss check')
        else:
            flag = '  <-- FRAME STRIKE' if d.min() > 1.35 else ''
            print(f'    gate {wp}: {d.min():.2f} m{flag}')


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for session in argv[1:]:
        if not os.path.isdir(session):
            print(f'not a directory: {session}')
            return 1
        analyse(session)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
