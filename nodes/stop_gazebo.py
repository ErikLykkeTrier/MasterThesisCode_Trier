#!/usr/bin/env python3
"""
gazebo_fast_shutdown.py   (ROS 1)

On normal ROS shutdown this node waits a short grace period.
If gzserver/gzclient are still running afterwards, it sends them SIGTERM,
cutting Gazebo’s exit time from ~10 s to ~1–2 s.

Usage:
    rosrun <your_pkg> gazebo_fast_shutdown.py
"""

import os
import signal
import subprocess
import time

import rospy

# --- settings ---------------------------------------------------------------
WAIT_SECS = 0.0            # grace period before escalation
SIG = signal.SIGTERM       # use SIGKILL if SIGTERM is not enough
TARGETS = ["gzserver", "gzclient"]
# ---------------------------------------------------------------------------


def _pids_of(pattern: str):
    """Return PIDs of processes whose cmdline contains *pattern* (excluding me)."""
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        return [int(p) for p in out.strip().splitlines() if int(p) != os.getpid()]
    except subprocess.CalledProcessError:
        return []  # none found


def terminate_gazebo():
    time.sleep(WAIT_SECS)   # let Gazebo finish gracefully first
    for proc in TARGETS:
        for pid in _pids_of(proc):
            try:
                os.kill(pid, SIG)
                rospy.logwarn(f"[fast_shutdown] {SIG.name} sent to {proc} (PID {pid})")
            except ProcessLookupError:
                pass  # it died while we were iterating
    rospy.loginfo("[fast_shutdown] done.")


def main():
    rospy.init_node("gazebo_fast_shutdown", anonymous=True)
    rospy.loginfo("Gazebo fast-shutdown armed "
                  f"(waiting {WAIT_SECS}s, then {SIG.name}).")
    rospy.on_shutdown(terminate_gazebo)
    rospy.signal_shutdown()
    rospy.spin()


if __name__ == "__main__":
    main()
