"""ShiftBench: do robot failure alarms keep their promise after the camera moves?"""
from .adaptive import AdaptiveAlarm, run_adaptive
from .calibrate import conformal_cutoff, recalibrated_cutoff
from .data import Episode, load_cache, read_labels, write_label
from .detector import Detector, DetectorConfig, OnlineScorer

__version__ = "0.1.0"
__all__ = ["AdaptiveAlarm", "run_adaptive", "conformal_cutoff", "recalibrated_cutoff", "Episode",
           "load_cache", "read_labels", "write_label", "Detector", "DetectorConfig", "OnlineScorer"]
