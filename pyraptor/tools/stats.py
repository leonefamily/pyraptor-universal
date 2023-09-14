# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:15:44 2022

@author: DGrishchuk
"""

from pyraptor.query_range_mcraptor import main as query
from pyraptor.dao import read_timetable
from pyraptor.util import str2sec
from loguru import logger

import pandas as pd

from datetime import datetime as dt, timedelta as td

from itertools import product
import logging

logger.disable('pyraptor')
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(message)s")

INPUT_FOLDER = r"raptor_data"
ROUNDS = 3
DEPARTURE_TIME1 = "07:00:00"
DEPARTURE_TIME2 = "07:30:00"

FROMTO = {
    "N":  ["Řečkovice", "Technologický park", "Semilasso", "Mácova", "Ořešín"],
    "NE": ["Moskalykova", "Královopolská strojírna", "Klarisky", "Haškova",
           "Štefánikova čtvrť", "Obřanský most", "Tomkovo náměstí"],
    "E":  ["Jírova", "Novolíšeňská", "Pálavské náměstí"],
    "SE": ["Areál Slatina", "Těžební", "Slatina, sídliště",
           "Letiště Tuřany - terminál"],
    "S":  ["Hanácká", "Přízřenice, smyčka", "Modřice, Olympia",
           "Modřice, Tyršova"],
    "SW": ["Dunajská", "Nemocnice Bohunice", "Osová", "Oblá"],
    "W":  ["Bellova", "Křivánkovo náměstí", "Jundrovský most"],
    "NW": ["Ečerova", "Kamechy", "Zoologická zahrada", "U Luhu", "Černého"],
    "C":  ["Česká", "Hlavní nádraží", "Úzká",
           "Mendlovo náměstí", "Stará osada"]
    }

STATSCOLS = ['fastest_travel_time',
             'fastest_transfer_time',
             'fastest_transfer_count',
             'mean_travel_time',
             'mean_transfer_time',
             'mean_transfer_count',
             'max_travel_time',
             'max_transfer_time',
             'max_transfer_count',
             'departures_count',
             'from_dir',
             'to_dir',
             'from_stop',
             'to_stop']


def get_directions_stops_combinations(fromto_dict=FROMTO):
    relations = product(fromto_dict, fromto_dict)
    combs = []
    for from_dir, to_dir in relations:
        if from_dir == to_dir:
            continue
        for from_stop, to_stop in product(fromto_dict[from_dir],
                                          fromto_dict[to_dir]):
            combs.append(
                (from_dir, to_dir, from_stop, to_stop)
                )
    return combs


def is_within_times(stats, tdtime1, tdtime2):
    return (stats['board_times'][0] > tdtime1 and
            stats['board_times'][0] < tdtime2)


def get_fastest_journey(journeys):
    return min(journeys, key=lambda x: x.travel_time())


def get_df_stats(fdf, within_count, statscols=STATSCOLS):
    fdf['transfer_count'] = fdf['transfer_times'].apply(lambda x: len(x))

    statser = pd.Series(dtype=object, index=statscols)
    fastest = fdf.loc[fdf['travel_time_since_departure'].idxmin()]
    descr = fdf.describe()
    means = descr.loc['mean']
    maxes = descr.loc['max']

    for label, ser in zip(['fastest', 'mean', 'max'], [fastest, means, maxes]):
        statser[[f'{label}_travel_time',
                 f'{label}_transfer_time',
                 f'{label}_transfer_count']] = ser[
                     ['travel_time_since_departure',
                      'wait_transfer_time',
                      'transfer_count']].tolist()
    statser['departures_count'] = within_count
    return statser


def filter_times(journeys, time1: str = None, time2: str = None):
    return list(
        filter(
            lambda x: (x.dep() >= str2sec(time1))
            and (x.dep() <= str2sec(time2)),
            journeys))


def reformat_stats(journeys):
    fstats = [j.get_statistics() for j in journeys]
    within_count = len(fstats)
    fdf = pd.DataFrame(fstats)
    statser = get_df_stats(fdf, within_count, STATSCOLS)
    return statser


timetable = read_timetable('raptor_gtfs')
combs = get_directions_stops_combinations(FROMTO)
results = []
txt_fastest = ''

for i, (from_dir, to_dir, from_stop, to_stop) in enumerate(combs):
    logging.info(f"CONNECTION : {from_stop} -> {to_stop}")
    # (origin_station,
    # destination_station,
    # departure_start_time,
    # departure_end_time,
    # rounds,
    # input_folder,
    # cached_timetable) = ('Technologický park', 'Jírova',
    #                      DEPARTURE_TIME1, DEPARTURE_TIME2,
    #                      ROUNDS, None, timetable)
    journeys = query(from_stop, None,  # to_stop
                     DEPARTURE_TIME1, DEPARTURE_TIME2,
                     ROUNDS, cached_timetable=timetable)
    journeys = filter_times(journeys, DEPARTURE_TIME1, DEPARTURE_TIME2)
    if journeys:
        msg = get_fastest_journey(journeys).print()
        logging.info(msg)
        statser = reformat_stats(journeys)
    else:
        statser = pd.Series(dtype=object, index=STATSCOLS)
        msg = str([from_stop, to_stop]) + ' failed \n'
    statser['from_dir', 'to_dir',
            'from_stop', 'to_stop'] = from_dir, to_dir, from_stop, to_stop
    statser.name = from_stop, to_stop
    results.append(statser)
    txt_fastest += (f'\n{dt.now()} ' + msg)
    logging.info(f'Progress: {round(i * 100 / (len(combs) - 1), 2)} %')

result_df = pd.concat(results, axis=1).transpose()
result_df.convert_dtypes().to_csv(
    r'D:\disser\data\travelling_stats\current\stats.csv',
    index=False, sep=';', decimal=',', encoding='utf-8-sig')
with open(r'D:\disser\data\travelling_stats\current\log.txt',
          mode='w', encoding='utf-8') as f:
    f.write(txt_fastest)
