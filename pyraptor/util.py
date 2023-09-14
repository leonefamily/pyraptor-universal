"""Utility functions"""
import os
import random
import numpy as np


TRANSFER_COST = 2 * 60  # Default transfer time is 2 minutes
LARGE_NUMBER = 2147483647  # Earliest arrival time at start of algorithm
TRANSFER_TRIP = None
ROUNDS = 5
START_TIME = "08:00:00"
END_TIME = "08:30:00"
ORIGIN_STOP = "__random__"


def mkdir_if_not_exists(name: str) -> None:
    """Create directory if not exists"""
    if not os.path.exists(name):
        os.makedirs(name)


def str2sec(time_str: str) -> int:
    """
    Convert hh:mm:ss to seconds since midnight
    :param time_str: String in format hh:mm:ss
    """
    split_time = time_str.strip().split(":")
    if len(split_time) == 3:
        # Has seconds
        hours, minutes, seconds = split_time
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    hour, minutes = split_time
    return int(hour) * 3600 + int(minutes) * 60


def sec2str(scnds: int, show_sec: bool = False) -> str:
    """
    Convert hh:mm:ss to seconds since midnight

    :param show_sec: only show :ss if True
    :param scnds: Seconds to translate to hh:mm:ss
    """
    scnds = np.round(scnds)
    hours = int(scnds / 3600)
    minutes = int((scnds % 3600) / 60)
    seconds = int(scnds % 60)
    return (
        "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        if show_sec
        else "{:02d}:{:02d}".format(hours, minutes)
    )


def pick_random_station(timetable) -> str:
    return random.choice(list(timetable.stations.set_idx.keys()))
