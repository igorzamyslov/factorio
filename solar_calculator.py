#!/bin/python3
""" Module for calculation the number solar panels / accumulators """
import math

from typing import Dict, Union

Number = Union[int, float]
MULTIPLIER: Dict[str, Number] = {
    "y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12,
    "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "c": 1e-2, "d": 0.1,
    "h": 100, "k": 1000, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15,
    "E": 1e18, "Z": 1e21, "Y": 1e24
}


def convert_str_to_int(string: str) -> Union[float, int]:
    """
    Converts a string with multiplier to a number
    Example: 100k -> 100000
    """
    if string[-1] in MULTIPLIER:
        multiplier = MULTIPLIER[string[-1]]
        string = string[:-1]
    else:
        multiplier = 1

    try:
        return int(string) * multiplier
    except (TypeError, ValueError):
        return float(string) * multiplier


def calculate_solar_panels(power: Number, solar_output: Number,
                           solar_efficiency: Number) -> int:
    """
    Calculates the number of required solar panels
    power (W) - required power to be sustained by the solar panels
    solar_output (W) - max. output of the solar panels
    solar_efficiency (%) - efficiency of the solar panels on the given planet
    """
    return math.ceil(power / (solar_output * solar_efficiency / 100 * 42.8 / 60))


def calculate_accumulators(power: Number, accumulator_capacity: int,
                           day_night_cycle: Number) -> int:
    """
    Calculates the number of required accumulators
    power (W) - required power to be sustained by the accumulators
    accumulator_capacity (J) - capacity of one accumulator
    day_night_cycle (min.) - day/night cycle in minutes
    """
    return math.ceil(power / accumulator_capacity * day_night_cycle * 17.2)


def main():
    """ Main entrypoint """
    import argparse

    # Init argument parser
    parser = argparse.ArgumentParser(
        description=("Calculate the number of required solar panels/accumulators.\n"
                     "NOTE: Some of the parameters support multipliers, e.g. 100k, 42M, etc."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("power", type=str,
                        help="Power (W) required to be sustained. Supports multipliers.")
    parser.add_argument("solar_output", type=str,
                        help="Max. output (W) of the used solar panels. Supports multipliers.")
    parser.add_argument("solar_efficiency", type=float,
                        help="Solar efficiency (%%) on the given planet.")
    parser.add_argument("accumulator_capacity", type=str,
                        help="Capacity of one accumulator (J). Supports multipliers.")
    parser.add_argument("day_night_cycle", type=float,
                        help="Day/night cycle (minutes).")
    args = parser.parse_args()

    # Calculate
    power = convert_str_to_int(args.power)
    solar_output = convert_str_to_int(args.solar_output)
    accumulator_capacity = convert_str_to_int(args.accumulator_capacity)
    solar_panels = calculate_solar_panels(power, solar_output, args.solar_efficiency)
    accumulators = calculate_accumulators(power, accumulator_capacity,
                                          args.day_night_cycle)

    # Output
    print(f"Required number of solar panels: {solar_panels}")
    print(f"Required number of accumulators: {accumulators}")


if __name__ == "__main__":
    main()
