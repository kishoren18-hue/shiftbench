"""LeRobot plugin: `--robot.type=so101_watchdog` (an SO-101 follower with a ShiftBench alarm)."""
from .so101_watchdog import SO101Watchdog, SO101WatchdogConfig, Monitor

__all__ = ["SO101Watchdog", "SO101WatchdogConfig", "Monitor"]
