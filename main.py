#
# Sample Python client for the AI GP controller
#

import os
import time
import msvcrt
import traceback

from setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1" 
SIM_SERVER_UDP_PORT = 14550 

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller  = components['controller']
ts_loop     = components['ts_loop']
mavlink_rx  = components['mavlink_rx']
vision_rx   = components['vision_rx']
logger      = components['logger']
ekf_handler = components['ekf_handler']

print("Press 's' to arm and start...", flush=True)
while True:
    if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
        break
    time.sleep(0.05)

print("Resetting sim...", flush=True)
controller.send_sim_reset_command()
time.sleep(2.0)   # wait for sim to settle into clean initial state

# Re-request ground truth streams after reset (sim may clear message intervals on reset)
mavlink_rx.request_ground_truth_streams(rate_hz=50)

print("Arming drone...", flush=True)
controller.arm()
# Wait long enough for all buffered pre-reset MAVLink messages (large time_boot_ms)
# to arrive in the background thread, then purge them.  The control loop's 3 s
# wait-phase means no flight data is lost by clearing here.
time.sleep(1.0)
if logger is not None:
    logger.reset_flight_data()

FLIGHT_TIMEOUT_S = 150.0

print("Starting control loop...", flush=True)
_flight_start = time.time()
try:
    while True:
        # An uncaught exception here previously just unwound straight to
        # the bottom of this try (still saving logs via finally, still
        # printing a traceback via Python's default handling) — but with
        # nothing recorded IN the session's own log directory, there was no
        # way to correlate a later flight-log analysis back to the exact
        # failure. Confirmed in a flight log: carrot.csv/cascade.csv (both
        # logged from inside controller.update()) stopped mid-flight while
        # ekf.csv/vision.csv (logged from independent MAVLink/vision
        # threads that don't call into controller.update() at all) kept
        # running for tens of seconds after — consistent with
        # controller.update() throwing here, silently ending active
        # control (hence the drone "losing control") while the background
        # threads kept logging whatever the now-uncontrolled drone did
        # next. Persisting the traceback alongside the rest of that
        # flight's logs turns the next occurrence into a direct answer
        # instead of another round of inference from position/attitude
        # traces.
        try:
            gate_passed = bool(shared_data.get('gate_passed', False))
            if gate_passed:
                gate_id = shared_data.get('last_gate_id')
                ekf_pos = shared_data.get('mav_state', {}).get('pos_ned')
                if ekf_pos is not None:
                    pos_str = f"[{ekf_pos[0]:.6f}, {ekf_pos[1]:.6f}, {ekf_pos[2]:.6f}]"
                else:
                    pos_str = 'unknown'
                print(f"[main] gate passed gate_id={gate_id} ekf_pos={pos_str}", flush=True)
            controller.update()
            if gate_passed:
                shared_data['gate_passed'] = False
        except Exception:
            _tb = traceback.format_exc()
            print(_tb, flush=True)
            if logger is not None:
                _tb_path = os.path.join(logger.session_dir, 'crash_traceback.txt')
                with open(_tb_path, 'w') as _f:
                    _f.write(_tb)
                print(f"[main] crash traceback saved -> {_tb_path}", flush=True)
            raise
        if time.time() - _flight_start >= FLIGHT_TIMEOUT_S:
            print(f"\nFlight timeout ({FLIGHT_TIMEOUT_S:.0f}s), saving logs...", flush=True)
            break
except KeyboardInterrupt:
    print("\nCtrl+C received, saving logs...", flush=True)
finally:
    if logger is not None:
        logger.save()
    ekf_handler.save()
    for _c in [ts_loop, mavlink_rx, vision_rx]:
        _t = _c.get_thread_for_join()
        if _t is not None:
            _t.join(timeout=1.0)

print("Client exited!", flush=True)
