"""Parse timetable from GTFS files"""
import os
import sys
import argparse
from typing import List
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime as dt

import pandas as pd
from loguru import logger

from pyraptor.dao import write_timetable
from pyraptor.util import mkdir_if_not_exists, str2sec, TRANSFER_COST
from pyraptor.model.structures import (
    Timetable,
    Stop,
    Stops,
    Trip,
    Trips,
    TripStopTime,
    TripStopTimes,
    Station,
    Stations,
    Routes,
    Transfer,
    Transfers,
)

WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
WEEKEND = ['saturday', 'sunday']


@dataclass
class GtfsTimetable:
    """Gtfs Timetable data"""

    routes = None
    trips = None
    calendar = None
    stop_times = None
    stops = None


def parse_arguments(args_from: list = sys.argv[1:]):
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/input/NL-gtfs",
        help="Input directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Input directory",
    )
    parser.add_argument(
        "-d", "--date", type=str, default="20221105", help="Departure date (yyyymmdd)"
    )

    arguments = parser.parse_args(args_from)
    return arguments


def get_trips(
    input_folder: str,
    departure_date: str = None,
) -> pd.DataFrame:

    date_dt = None if departure_date is None else dt.strptime(departure_date, '%Y%m%d')

    trips = pd.read_csv(input_folder + '/trips.txt')
    calendar = pd.read_csv(input_folder + '/calendar.txt')
    calendar.start_date = pd.to_datetime(calendar.start_date, format='%Y%m%d')
    calendar.end_date = pd.to_datetime(calendar.end_date, format='%Y%m%d')

    cdpath = input_folder + '/calendar_dates.txt'
    cdexist = os.path.exists(cdpath)
    if cdexist:
        cdates = pd.read_csv(cdpath)
        cdates.date = pd.to_datetime(cdates.date, format='%Y%m%d').tolist()

    mostday = defaultdict(int)
    servday = defaultdict(list)
    for service_id, service_df in trips.groupby('service_id'):
        cal_df = calendar[calendar['service_id'] == service_id]
        sd, ed = cal_df[['start_date', 'end_date']].iloc[0]
        allowed_wdays = cal_df[WEEKDAYS + WEEKEND].iloc[0].to_dict()
        
        if date_dt is not None:
            sdd = pd.to_datetime(max(sd, date_dt))
            edd = pd.to_datetime(min(ed, date_dt))
            drange = pd.date_range(sdd, edd)
        else:
            drange = pd.date_range(sd, ed)
        allowed_days = []

        for awd in drange:
            d_name = awd.day_name().lower()
            if cdexist:
                excluded = bool(
                    len(cdates[(cdates['date'].dt.date == awd.date()) &
                               (cdates['service_id'] == service_id) &
                               (cdates['exception_type'] == 2)])
                    )
            else:
                excluded = False
            if allowed_wdays[d_name] and not excluded:
                allowed_days.append(awd)

        for ald in allowed_days:
            mostday[str(ald.date())] += len(service_df)
            servday[str(ald.date())].append(service_id)
        # print(service_id)

    mk = max(mostday, key=mostday.get)
    trips = trips[trips['service_id'].isin(servday[mk])]
    return trips


def main(
    input_folder: str,
    output_folder: str,
    departure_date: str,
):
    """Main function"""

    logger.info("Parse timetable from GTFS files")
    mkdir_if_not_exists(output_folder)

    gtfs_timetable = read_gtfs_timetable(input_folder, departure_date)
    timetable = gtfs_to_pyraptor2_timetable(gtfs_timetable)
    write_timetable(output_folder, timetable)


def read_gtfs_timetable(
    input_folder: str, departure_date: str
) -> GtfsTimetable:
    """Extract operators from GTFS data"""

    logger.info("Read GTFS data")

    # Read routes
    logger.debug("Read Routes")

    routes = pd.read_csv(os.path.join(input_folder, "routes.txt"))
    routes = routes[
        ["route_id", "route_short_name", "route_long_name", "route_type"]
    ]

    # Read trips
    logger.debug("Read Trips")

    # trips = get_trips(input_folder, departure_date)
    trips = pd.read_csv(os.path.join(input_folder, "trips.txt"))
    trips = trips.merge(routes[['route_id', 'route_short_name', 'route_type']])
    trips = trips[
        [
            "route_id",
            "service_id",
            "trip_id",
            "route_short_name",
            "route_type"
            # "trip_short_name",
            # "trip_long_name",
        ]
    ]
    # trips["trip_short_name"] = trips["trip_short_name"].astype("Int64")

    calendar = pd.read_csv(
        os.path.join(input_folder, "calendar_dates.txt"), dtype={"date": str}
    )
    calendar = calendar[calendar.service_id.isin(trips.service_id.values)]
    trips = trips.merge(calendar[["service_id", "date"]], on="service_id")
    trips = trips[trips.date == departure_date]

    # Read stop times
    logger.debug("Read Stop Times")

    stop_times = pd.read_csv(
        os.path.join(input_folder, "stop_times.txt"), dtype={"stop_id": str}
    )
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id.values)]
    stop_times = stop_times[
        [
            "trip_id",
            "stop_sequence",
            "stop_id",
            "arrival_time",
            "departure_time",
        ]
    ]
    # Convert times to seconds
    stop_times["arrival_time"] = stop_times["arrival_time"].apply(str2sec)
    stop_times["departure_time"] = stop_times["departure_time"].apply(str2sec)

    # Read stops (platforms)
    logger.debug("Read Stops")

    stops_full = pd.read_csv(
        os.path.join(input_folder, "stops.txt"), dtype={"stop_id": str}
    )

    if 'platform_code' not in stops_full.columns:
        stops_full['platform_code'] = '?'

    stops = stops_full.loc[
        stops_full["stop_id"].isin(stop_times.stop_id.unique())
    ].copy()

    # Read stopareas, i.e. stations
    stopareas = stops["parent_station"].unique()
    stops = pd.concat([stops, stops_full.loc[stops_full["stop_id"].isin(stopareas)]])

    stops = stops[
        [
            "stop_id",
            "stop_name",
            "parent_station",
            "platform_code",
        ]
    ]

    # Filter out the general station codes
    stops = stops.loc[~stops.parent_station.isna()]
    
    logger.debug(f"There are {len(stops)} stops, "
                 f"{len(trips)} trips and "
                 f"{len(stop_times)} stop times")

    gtfs_timetable = GtfsTimetable()
    gtfs_timetable.routes = routes
    gtfs_timetable.trips = trips
    gtfs_timetable.stop_times = stop_times
    gtfs_timetable.stops = stops

    return gtfs_timetable


def gtfs_to_pyraptor2_timetable(
    gtfs_timetable: GtfsTimetable
) -> Timetable:
    """
    Convert timetable for usage in Raptor algorithm.
    """
    logger.info("Convert GTFS timetable to timetable for PyRaptor algorithm")

    # Stations and stops, i.e. platforms
    logger.debug("Add stations and stops")

    stations = Stations()
    stops = Stops()

    gtfs_timetable.stops.platform_code = gtfs_timetable.stops.platform_code.fillna("?")

    for s in gtfs_timetable.stops.itertuples():
        station = Station(s.stop_name, s.stop_name)
        station = stations.add(station)

        stop_name = f"{s.stop_name}-{s.stop_id}"
        stop = Stop(s.stop_id, stop_name, station, s.platform_code)

        station.add_stop(stop)
        stops.add(stop)

    # Stop Times
    stop_times = defaultdict(list)
    for stop_time in gtfs_timetable.stop_times.itertuples():
        stop_times[stop_time.trip_id].append(stop_time)

    # Trips and Trip Stop Times
    logger.debug("Add trips and trip stop times")

    trips = Trips()
    trip_stop_times = TripStopTimes()

    for trip_row in gtfs_timetable.trips.itertuples():
        trip = Trip()
        trip.hint = trip_row.route_short_name
        trip.route_type = trip_row.route_type
        # trip.long_name = trip_row.trip_long_name  # e.g., Sprinter

        # Iterate over stops
        sort_stop_times = sorted(
            stop_times[trip_row.trip_id], key=lambda s: int(s.stop_sequence)
        )
        for stopidx, stop_time in enumerate(sort_stop_times):
            # Timestamps
            dts_arr = stop_time.arrival_time
            dts_dep = stop_time.departure_time

            # Trip Stop Times
            stop = stops.get(stop_time.stop_id)

            # GTFS files do not contain ICD supplement fare, so hard-coded here

            trip_stop_time = TripStopTime(trip, stopidx, stop, dts_arr, dts_dep)

            trip_stop_times.add(trip_stop_time)
            trip.add_stop_time(trip_stop_time)

        # Add trip
        if trip:
            trips.add(trip)

    # Routes
    logger.debug("Add routes")

    routes = Routes()
    for trip in trips:
        routes.add(trip)

    # Transfers
    logger.debug("Add transfers")

    transfers = Transfers()
    for station in stations:
        station_stops = station.stops
        station_transfers = [
            Transfer(from_stop=stop_i, to_stop=stop_j, layovertime=TRANSFER_COST)
            for stop_i in station_stops
            for stop_j in station_stops
            if stop_i != stop_j
        ]
        for st in station_transfers:
            transfers.add(st)

    # Timetable
    timetable = Timetable(
        stations=stations,
        stops=stops,
        trips=trips,
        trip_stop_times=trip_stop_times,
        routes=routes,
        transfers=transfers,
    )
    timetable.counts()

    return timetable


if __name__ == "__main__":
    args = parse_arguments()
    # import shlex
    # args = parse_arguments(
    #     shlex.split(r'-i "D:\disser\raptor_timetables\November_2022_2" '
    #                 r'-o "D:\disser\raptor_timetables\November_2022_2" '
    #                 r'-d "20221104"')
    # )
    main(args.input, args.output, args.date)
