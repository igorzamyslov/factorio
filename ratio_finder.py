#!/usr/bin/env python3
""" Script to calculate the perfect/near-perfect ratios for different recipes """
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# TODO:
# [x] Move input to separate file
# [x] Populate the input file automatically
# [x] Handling items with multiple recipes (asking user to choose)
# [ ] Make a better user-interface (run the script via user input)
# [ ] Update algorithm to be able to
#     [ ] minimise either error OR number of assemblers
#     [ ] provide all possible combinations of assemblers
#     [ ] use better algorithm to find equation solution
# [ ] Add docstrings + check with pylint
# [ ] Handling items with probability (right now skipped)
# [ ] Draw directed graph of inputs (visjs or python library)


InputMatrixType = List[Tuple[str, float, float, int]]
RatioMatrixType = List[Tuple[str, int]]


@dataclass
class Input:
    """ Information about the input required by a recipe """
    component: str
    quantity: int


@dataclass
class Recipe:
    """
    Information about the recipe, containing:
    - component name
    - time to produce
    - quantity produced
    - required inputs with their required quantities
    """
    name: str  # name of the recipe
    component: str  # name of the output component
    time: float  # seconds
    quantity: int
    inputs: List[Input]


RECIPES: Dict[str, List[Recipe]] = {}  # Recipes grouped by their name
CHOSEN_RECIPES: Dict[str, Recipe] = {}  # Recipes that were already chosen by the user


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

    for recipe_data in data.values():
        # skip multi-products recipes
        if len(recipe_data["products"]) != 1:
            continue
        # skip <1 probability recipes
        [component_data] = recipe_data["products"]
        if component_data["probability"] != 1:
            continue

        recipe = Recipe(name=recipe_data["name"],
                        time=recipe_data["energy"],
                        component=component_data["name"],
                        quantity=component_data["amount"],
                        inputs=[Input(component=i["name"], quantity=i["amount"])
                                for i in recipe_data["ingredients"]])
        RECIPES.setdefault(recipe.component, []).append(recipe)


def choose_recipe(component: str) -> Recipe:
    """ Propose user to select a recipe for the given component """
    # Check if was already chosen
    if component in CHOSEN_RECIPES:
        return CHOSEN_RECIPES[component]

    component_recipes = RECIPES[component]
    # Check if there's only one recept
    if len(component_recipes) == 1:
        [chosen_recipe] = component_recipes
        CHOSEN_RECIPES[component] = chosen_recipe
        return chosen_recipe

    # Ask the user otherwise
    print(f"\nChoose which recipe to use to craft \"{component}\":")
    for i, recipe in enumerate(component_recipes):
        print(f"{i + 1}. {recipe.name}")
    while True:
        try:
            recipe_to_use = int(input("Recipe to use: "))
            if recipe_to_use < 1 or recipe_to_use > len(component_recipes):
                raise ValueError
        except (TypeError, ValueError):
            print(f"Please enter a number between 1 and {len(component_recipes)}")
            continue
        else:
            chosen_recipe = component_recipes[recipe_to_use - 1]
            CHOSEN_RECIPES[component] = chosen_recipe
            return chosen_recipe


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


def get_product_output(component: str, ratio_matrix: RatioMatrixType,
                       assembler_speed: float, efficiency: float) -> float:
    """
    Returns output per second of the end product with the given assembler speed
    Disclaimer: Only works with the perfect ratio matrix
    """
    recipe = choose_recipe(component)
    ep_assemblers = next(a for n, a in ratio_matrix if n == recipe.component)
    output_per_second = ep_assemblers * recipe.quantity / recipe.time * assembler_speed
    # return output per second considering the efficiency
    return output_per_second + output_per_second * efficiency / 100


def create_input_matrix(end_product: str,
                        efficiency: float,
                        base_components: Iterable[str] = None,
                        ) -> InputMatrixType:
    """ Create input matrix for 1 unit of end product """
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
        if component not in RECIPES:
            base_components.add(component)
        if component in base_components:
            continue
        recipe = choose_recipe(component)
        add_to_matrix.extend((r_input.component,
                              r_input.quantity * quantity / recipe.quantity / efficiency_ratio)
                             for r_input in recipe.inputs)
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
        print(f"  {i + 1:2d}. {prefix}{component}:")
        for built_component in temp_matrix_dict:
            if built_component in base_components:
                continue
            recipe = choose_recipe(built_component)
            if any(i.component == component for i in recipe.inputs):
                print(f"      -> {built_component}")

    # Generate input matrix
    output = []
    for component, quantity in temp_matrix_dict.items():
        if component in base_components:
            continue
        recipe = choose_recipe(component)
        output.append((component, quantity, recipe.time, recipe.quantity))
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
        # "iron-plate", "stone-brick", "copper-plate",
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
    print(f"Output: {output:.1f}/s ({output * 60:.1f}/m)")
    output = get_product_output(end_product, perfect_ratio_matrix, assembler_speed, efficiency)
    print(f"Output (+efficiency): {output:.1f}/s ({output * 60:.1f}/m)")

    # Find required ratio
    required_ratio_matrix = find_required_ratio(end_product, perfect_ratio_matrix,
                                                required_output, assembler_speed,
                                                efficiency)
    print_ratio_matrix(required_ratio_matrix, prefix="Required")
    print(f"Output: ~{required_output:.1f}/s (~{required_output * 60:.1f}/m)")
    print(f"Output (-efficiency): ~{required_output / (1 + efficiency / 100):.1f}/s")


if __name__ == "__main__":
    main()
