"""Run query with RAPTOR algorithm"""
import argparse
from typing import Dict
import sys
from loguru import logger

from pyraptor_universal.dao.timetable import read_timetable
from pyraptor_universal.model.structures import Journey, Station, Timetable
from pyraptor_universal.model.raptor import (
    RaptorAlgorithm,
    reconstruct_journey,
    best_stop_at_target_station,
)
from pyraptor_universal.util import str2sec, pick_random_station


def parse_arguments(args_from: list = sys.argv[1:]):
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Input directory",
    )
    parser.add_argument(
        "-or",
        "--origin",
        type=str,
        default="__random__",
        help="Origin station of the journey",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="__random__",
        help="Destination station of the journey",
    )
    parser.add_argument(
        "-t", "--time", type=str, default="08:35:00", help="Departure time (hh:mm:ss)"
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds to execute the RAPTOR algorithm",
    )
    arguments = parser.parse_args(args_from)
    return arguments


def main(
    origin_station,
    destination_station,
    departure_time,
    rounds,
    input_folder: str = None,
    cached_timetable: Timetable = None
):
    """Run RAPTOR algorithm"""

    logger.debug("Departure time      : {}", departure_time)
    logger.debug("Rounds              : {}", str(rounds))

    if cached_timetable is None:
        logger.debug("Input directory     : {}", input_folder)
        timetable = read_timetable(input_folder)
    else:
        timetable = cached_timetable

    if origin_station == '__random__':
        origin_station = pick_random_station(timetable)
    logger.debug("Origin station      : {}", origin_station)
    if destination_station == '__random__':
        destination_station = pick_random_station(timetable)
    logger.debug("Destination station : {}", destination_station)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time       : " + departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations
    journey_to_destinations = run_raptor(
        timetable,
        origin_station,
        dep_secs,
        rounds,
    )

    # Print journey to destination
    journey = journey_to_destinations[destination_station]
    journey.print(dep_secs=dep_secs)
    return journey


def run_raptor(
    timetable: Timetable,
    origin_station: str,
    dep_secs: int,
    rounds: int,
) -> Dict[Station, Journey]:
    """
    Run the Raptor algorithm.

    :param timetable: timetable
    :param origin_station: Name of origin station
    :param dep_secs: Time of departure in seconds
    :param rounds: Number of iterations to perform
    """

    # Get stops for origin and all destinations
    from_stops = timetable.stations.get(origin_station).stops
    destination_stops = {
        st.name: timetable.stations.get_stops(st.name) for st in timetable.stations
    }
    destination_stops.pop(origin_station, None)

    # Run Round-Based Algorithm
    raptor = RaptorAlgorithm(timetable)
    bag_round_stop = raptor.run(from_stops, dep_secs, rounds)
    best_labels = bag_round_stop[rounds]

    # Determine the best journey to all possible destination stations
    journey_to_destinations = dict()
    for destination_station_name, to_stops in destination_stops.items():
        dest_stop = best_stop_at_target_station(to_stops, best_labels)
        if dest_stop != 0:
            journey = reconstruct_journey(dest_stop, best_labels)
            journey_to_destinations[destination_station_name] = journey

    return journey_to_destinations


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.origin,
        args.destination,
        args.time,
        args.rounds,
        args.input,
    )
