"""
msg_probe.py
============
Connect to the sim, listen for MSG_DURATION seconds, and print every unique
MAVLink message type along with its field names and a sample value snapshot.

Usage:
    python msg_probe.py

Run this while the sim is already open (before arming).  The script is read-only
and sends no commands.
"""

import time
from pymavlink import mavutil

SIM_IP       = "127.0.0.1"
SIM_PORT     = 14550
MSG_DURATION = 10.0    # seconds to listen

print(f"Connecting to {SIM_IP}:{SIM_PORT}…", flush=True)
conn = mavutil.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
conn.wait_heartbeat()
print(f"Connected (sys {conn.target_system}). Listening for {MSG_DURATION:.0f} s…\n",
      flush=True)

seen   = {}   # msg_type → {count, sample_fields}
counts = {}

t0 = time.time()
while time.time() - t0 < MSG_DURATION:
    msg = conn.recv_match(blocking=True, timeout=0.1)
    if msg is None:
        continue
    mt = msg.get_type()
    if mt == "BAD_DATA":
        continue
    counts[mt] = counts.get(mt, 0) + 1
    if mt not in seen:
        # Capture field names and a snapshot of values
        try:
            fields = {k: getattr(msg, k) for k in msg.fieldnames}
        except AttributeError:
            fields = {}
        seen[mt] = fields

elapsed  = time.time() - t0
hz_total = sum(counts.values()) / elapsed

# ── Targeted check ────────────────────────────────────────────────────────────
TARGETS = ["ATTITUDE", "LOCAL_POSITION_NED"]
print(f"{'='*60}")
print("Targeted check:")
for t_msg in TARGETS:
    if t_msg in counts:
        print(f"  [FOUND]   {t_msg}  ({counts[t_msg]} msgs, "
              f"{counts[t_msg]/elapsed:.1f} Hz)")
        for k, v in seen[t_msg].items():
            print(f"    {k:<30s} = {v}")
    else:
        print(f"  [ABSENT]  {t_msg}")
print()

# ── Full message list ─────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"All messages seen over {elapsed:.1f} s  (total {hz_total:.0f} msg/s)")
print(f"{'='*60}\n")

for mt in sorted(seen, key=lambda k: -counts[k]):
    hz = counts[mt] / elapsed
    print(f"  {mt:<40s}  {counts[mt]:>6d} msgs  ({hz:>6.1f} Hz)")
    fields = seen[mt]
    for k, v in fields.items():
        try:
            vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        except Exception:
            vs = repr(v)
        if len(vs) > 60:
            vs = vs[:57] + "…"
        print(f"      {k:<30s} = {vs}")
    print()
