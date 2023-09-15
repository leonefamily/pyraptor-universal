from pyraptor_universal.query_range_mcraptor import main as query
from pyraptor_universal.dao import read_timetable
from pyraptor_universal.util import str2sec
from loguru import logger
import pandas as pd
from datetime import datetime as dt, timedelta as td
from itertools import product
import logging

# logger.disable('pyraptor_universal')
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(message)s"
)

INPUT_FOLDER = "/home/leonefamily/PycharmProjects/raptor/raptor_gtfs"
ROUNDS = 3


def get_journey_with_earliest_arrival(
        journeys
):
    return min(journeys, key=lambda x: x.arr())


timetable = read_timetable(INPUT_FOLDER)

journeys = query(
    'Ústřední hřbitov', 'Janáčkovo divadlo',
    "16:00", "16:10",
    ROUNDS,
    cached_timetable=timetable
)

print(journeys)