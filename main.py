#
# Sample Python client for the AI GP controller
#

import time
import msvcrt

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

FLIGHT_TIMEOUT_S = 50.0

print("Starting control loop...", flush=True)
_flight_start = time.time()
try:
    while True:
        controller.update()
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
