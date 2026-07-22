from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from imu_ekf import IMUEKFHandler
from controller import Controller
from log import Logger
from dyn import load_params

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Vehicle parameters
    # -------------------------------
    param = load_params()

    # -------------------------------
    # Logger  (None when logging=0)
    # -------------------------------
    logger = Logger() if param.get('logging', 1) else None
    if logger is not None:
        logger.set_waypoints(param['waypoints'])

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    monitor_enabled = bool(param.get('mavlink_monitor', True))
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data, logger,
                                             monitor_enabled=monitor_enabled)

    # -------------------------------
    # IMU + EKF handler
    # -------------------------------
    print("Setting up IMU EKF handler...", flush=True)
    ekf_handler = IMUEKFHandler(shared_data, param, logger=logger)
    ekf_handler.register(mavlink_rx)

    # Request LOCAL_POSITION_NED stream (re-requested after sim reset in main.py)
    mavlink_rx.request_ground_truth_streams(rate_hz=50)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data, logger)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms, param, logger=logger)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ekf_handler': ekf_handler,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'logger': logger,
    }