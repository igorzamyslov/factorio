#!/usr/bin/env python3
""" Script to calculate the perfect/near-perfect ratios for different recipes """
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

# TODO:
# [x] Move input to separate file
# [x] Populate the input file automatically
# [ ] Handling items with multiple recipes
#     (right now first recipe is used or the one that has "priority: true" set)
# [ ] Update algorithm to be able to
#     [ ] minimise either error OR number of assemblers
#     [ ] provide all possible combination of assemblers
#     [ ] use better algorithm to find equation solution
# [ ] Add docstrings + check with pylint
# [ ] Run the script via terminal with input
# [ ] Handling items with probability (right now skipped)
# [ ] Draw directed graph of inputs (visjs or python library)


InputMatrixType = List[Tuple[str, float, float, int]]
RatioMatrixType = List[Tuple[str, int]]
ProductionInfoType = Tuple[float, int]  # (time to produce, quantity)
InputsInfoType = List[Tuple[str, int]]  # [(input, required quantity), ...]


PRODUCTION_DATA: Dict[str, ProductionInfoType] = {}
INPUT_DATA: Dict[str, InputsInfoType] = {}


def parse_recipes(filename: str = "aai-se-recipes.json"):
    """
    Parses the recipes file and initialises
    PRODUCTION_DATA and INPUT_DATA dictionaries.

    Disclaimer:
        1. Skips the recipes with multiple outputs
           or outputs with probability other than 0.
        2. Updates global dictionaries.
    """

    with open(os.path.join("recipes", filename), "r") as f:
        data = json.load(f)

    for component_info in data.values():
        if (len(component_info["products"]) != 1
                or component_info["products"][0]["probability"] != 1):
            continue
        component = component_info["products"][0]["name"]
        if component in PRODUCTION_DATA:
            if not component_info.get("priority", False):
                continue
        PRODUCTION_DATA[component] = (component_info["energy"],
                                      component_info["products"][0]["amount"])
        INPUT_DATA[component] = [(i["name"], i["amount"])
                                 for i in component_info["ingredients"]]


def find_perfect_ratio(input_matrix: InputMatrixType,
                       max_multiplier: int,
                       max_precision_error: float = 0) -> RatioMatrixType:
    # find float number of required assemblers
    ratio_matrix = [(name, number / output * time)
                    for name, number, time, output in input_matrix]
    # find minimum integer numbers of required assemblers
    for i in range(1, max_multiplier + 1):
        accumulated_error = 0
        for _, ratio in ratio_matrix:
            value = ratio * i
            if value % 1 == 0:
                continue
            if value < 1:
                break
            accumulated_error += 1 - value % 1
            if accumulated_error > max_precision_error:
                break
        else:
            return [(n, math.ceil(r * i)) for n, r in ratio_matrix]
    raise ValueError(f"Not found for precision {max_precision_error}")


def find_best_possible_perfect_ratio(input_matrix: InputMatrixType,
                                     max_multiplier: int = 100) -> Tuple[RatioMatrixType, float]:
    """
    Find best possible perfect ratio matrix
    by incrementally increasing the allowed precision error
    Returns the matrix and the precision error
    """
    precision_error = 0
    while True:
        try:
            perfect_ratio_matrix = find_perfect_ratio(
                input_matrix,
                max_precision_error=precision_error,
                max_multiplier=max_multiplier)
        except ValueError:
            precision_error += 0.01
            continue
        else:
            return perfect_ratio_matrix, precision_error


def find_required_ratio(end_product: str, ratio_matrix: RatioMatrixType,
                        required_output: float, assembler_speed: float,
                        efficiency: float):
    current_output = get_product_output(end_product, ratio_matrix,
                                        assembler_speed, efficiency)
    multiplier = round(required_output / current_output, 6)
    return [(n, math.ceil(r * multiplier)) for n, r in ratio_matrix]


def get_product_output(end_product: str, ratio_matrix: RatioMatrixType,
                       assembler_speed: float, efficiency: float) -> float:
    """
    Returns output per second of the end product with the given assembler speed
    Disclaimer: Only works with the perfect ratio matrix
    """
    ep_time, ep_output = PRODUCTION_DATA[end_product]
    ep_assemblers = next(a for n, a in ratio_matrix if n == end_product)
    output_per_second = ep_assemblers * ep_output / ep_time * assembler_speed
    # return output per second considering the efficiency
    return output_per_second + output_per_second * efficiency / 100


def create_input_matrix(end_product: str,
                        efficiency: float,
                        base_components: Iterable[str] = None,
                        ) -> InputMatrixType:
    # Init base components
    base_components = set(base_components) if base_components else set()
    # Create temporary matrix
    efficiency_ratio = 1 + efficiency / 100
    add_to_matrix = [(end_product, 1)]
    temp_matrix_dict = {}
    while add_to_matrix:
        component, quantity = add_to_matrix.pop()
        temp_matrix_dict.setdefault(component, 0)
        temp_matrix_dict[component] += quantity
        if component not in INPUT_DATA:
            base_components.add(component)
        if component in base_components:
            continue
        _, produced = PRODUCTION_DATA[component]
        add_to_matrix.extend((c, q * quantity / produced / efficiency_ratio)
                             for c, q in INPUT_DATA[component])
    # Print base components
    print(f"Base components per unit:")
    for component in temp_matrix_dict:
        if component in base_components:
            print(f"- {temp_matrix_dict[component]} {component}(s)")

    # Print requirements
    print(f"\nInputs:")
    for i, component in enumerate(sorted(temp_matrix_dict.keys(),
                                         key=lambda x: x not in base_components)):
        if component == end_product:
            continue
        prefix = "*" if component in base_components else ""
        print(f"  {i+1:2d}. {prefix}{component}:")
        for built_component in temp_matrix_dict:
            if built_component in base_components:
                continue
            if component in (c for c, _ in INPUT_DATA[built_component]):
                print(f"      -> {built_component}")

    # Generate input matrix
    output = []
    for component, quantity in temp_matrix_dict.items():
        if component in base_components:
            continue
        time_to_produce, quantity_produced = PRODUCTION_DATA[component]
        output.append((component, quantity, time_to_produce, quantity_produced))
    return output


def print_ratio_matrix(matrix: RatioMatrixType, prefix: str = ""):
    if prefix:
        print(f"\n{prefix} ratio:")
    else:
        print("\nRatio:")

    for entry in matrix:
        print("- {}: {}".format(*entry))
    print(f"Total number of assemblers: {sum(e[1] for e in matrix)}")


def main():
    parse_recipes()

    # Input
    end_product = "electronic-circuit"
    base_components = [
        # fluid
        "water", "sulfuric-acid",
        # chips
        # "electronic-circuit",
        # "advanced-circuit",
        # "processing-unit",
        # smelted
        "iron-plate", "stone-brick", "copper-plate",
        # "steel-plate", "iron-plate", "stone-brick", "copper-plate",
        # other
        "sulfur", "plastic-bar", "concrete",
    ]
    assembler_speed = 3.5
    efficiency = 48  # %
    required_output = 45 / 1  # per second

    # Create input matrix
    print("\n\n==", end_product, "==")
    input_matrix = create_input_matrix(end_product, efficiency,
                                       base_components=base_components)
    perfect_ratio_matrix, precision_error = find_best_possible_perfect_ratio(input_matrix)
    # Find perfect ratio
    print_ratio_matrix(perfect_ratio_matrix, prefix="Perfect")
    print(f"Precision error: {precision_error:.2f}")
    output = get_product_output(end_product, perfect_ratio_matrix, assembler_speed, 0)
    print(f"Output: {output}/s ({output * 60}/m)")
    output = get_product_output(end_product, perfect_ratio_matrix, assembler_speed, efficiency)
    print(f"Output (+efficiency): {output}/s ({output * 60}/m)")

    # Find required ratio
    required_ratio_matrix = find_required_ratio(end_product, perfect_ratio_matrix,
                                                required_output, assembler_speed,
                                                efficiency)
    print_ratio_matrix(required_ratio_matrix, prefix="Required")
    print(f"Output: ~{required_output}/s (~{required_output * 60}/m)")
    print(f"Output (-efficiency): ~{required_output / (1 + efficiency / 100)}/s")


if __name__ == "__main__":
    main()
