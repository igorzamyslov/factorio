#!/bin/python3
def calculate_solar_panels(power: int, solar_output: int, 
                           solar_efficiency: int):
    """
    Calculates the number of required solar panels
    power (W) - required power to be sustained by the solar panels
    solar_output (W) - max. output of the solar panels
    solar_efficiency (%) - efficiency of the solar panels on the given planet
    """
    return power / (solar_output * solar_efficiency / 100 * 42.8 / 60) 


def calculate_accumulators(power: int, accumulator_capacity: int,
                           day_night_cycle: float):
    """
    Calculates the number of required accumulators
    power (W) - required power to be sustained by the accumulators
    accumulator_capacity (J) - capacity of one accumulator
    day_night_cycle (min.) - day/night cycle in minutes
    """
    return power / accumulator_capacity * day_night_cycle * 17.2


def main():
    """ Main entrypoint """
    import argparse

    # Init argument parser
    parser = argparse.ArgumentParser(
        description=("Calculate the number of required solar panels/accumulators"))
    parser.add_argument("power", type=int, 
                        help="Power (W) required to be sustained.")
    parser.add_argument("solar_output", type=int,
                        help="Max. output (W) of the used solar panels.")
    parser.add_argument("solar_efficiency", type=int, 
                        help="Solar efficiency (%%) on the given planet")
    parser.add_argument("accumulator_capacity", type=int,
                        help="Capacity of one accumulator (J)")
    parser.add_argument("day_night_cycle", type=float,
                        help="Day/night cycle (minutes)")
    args = parser.parse_args()

    # Calculate
    solar_panels = calculate_solar_panels(args.power, args.solar_output, args.solar_efficiency)
    accumulators = calculate_accumulators(args.power, args.accumulator_capacity, 
                                          args.day_night_cycle)

    # Output
    print(f"Required number of solar panels: {solar_panels}")
    print(f"Required number of accumulators: {accumulators}")


if __name__ == "__main__":
    main()
