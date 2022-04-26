#!/usr/bin/env python3
""" Script to calculate the perfect/near-perfect ratios for different recipes """
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import Dict, List, Optional, Set

# TODO:
# [x] Move input to separate file
# [x] Populate the input file automatically
# [x] Handling items with multiple recipes (asking user to choose)
# [ ] Make a better user-interface (run the script via user input)
# [ ] Add docstrings + check with pylint
# [ ] Handle different assemblers with their own efficiency / speed
# [ ] Update algorithm to be able to
#     [ ] minimise either error OR number of assemblers
#     [ ] provide all possible combinations of assemblers
#     [ ] use better algorithm to find equation solution
# [ ] Handling items with probability (right now skipped)
# [ ] Draw directed graph of inputs (visjs or python library)


# Globally available data
_INPUT_COMPONENTS: Set[Component] = set()  # Components that are given as input
_RECIPES: Dict[Component, List[Recipe]] = {}  # Recipes grouped by their name


@dataclass(frozen=True)
class Component:
    """ Simple class for a single component, containing useful methods """
    name: str

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    @property
    def recipe(self) -> Optional[Recipe]:
        """ Component's recipe """
        return self._choose_recipe()

    @property
    def is_base_input(self) -> bool:
        """
        Checks whether a component is given as input:
        - Either it is a base component (which doesn't have its own recipe)
        - Or the component is marked as input-component
        """
        return self in _INPUT_COMPONENTS or self.recipe is None

    @lru_cache(maxsize=None)
    def _choose_recipe(self) -> Optional[Recipe]:
        """ Propose user to select a recipe for the given component """
        # Check if recipe for the component exists
        if self not in _RECIPES:
            return None

        component_recipes = _RECIPES[self]
        # Check if there's only one recipe
        if len(component_recipes) == 1:
            [chosen_recipe] = component_recipes
            return chosen_recipe

        # Ask the user otherwise
        print(f"\nChoose which recipe to use to craft \"{self}\":")
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
                return component_recipes[recipe_to_use - 1]

    @classmethod
    @lru_cache(maxsize=None)
    def from_name(cls, name: str) -> Component:
        """ Create singleton-instance from the name """
        return cls(name=name)


@dataclass
class Input:
    """ Information about the input required by a recipe """
    component: Component
    quantity: float


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
    component: Component  # name of the output component
    time: float  # seconds
    quantity: int
    inputs: List[Input]

    @staticmethod
    def parse_recipes_data(filename: str = "aai-se-recipes.json"):
        """
        Parses the recipes file and initialises the RECIPES and BASE_COMPONENTS dictionaries.

        Disclaimer:
            1. Skips the recipes with multiple outputs or outputs with probability other than 0.
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
                            component=Component.from_name(component_data["name"]),
                            quantity=component_data["amount"],
                            inputs=[Input(component=Component.from_name(i["name"]),
                                          quantity=i["amount"])
                                    for i in recipe_data["ingredients"]])
            _RECIPES.setdefault(recipe.component, []).append(recipe)


AssemblersRatios = Dict[Component, float]


@dataclass(unsafe_hash=True)
class Pipeline:
    """ Build pipeline for the selected component,  """
    component: Component
    assemblers_speed: float # speed of assemblers in the pipeline
    efficiency: float  # efficiency of the assemblers in the pipeline (%)
    required_output: float  # required output of the pipeline (component per second)
    # TODO: input_components: List[Component]  # components that are provided as inputs

    _max_multiplier: int = 100
    _precision_error: Optional[float] = None

    @property
    def precision_error(self):
        """ Precision error of the defined best assemblers ratio """
        if self._precision_error is None:
            raise RuntimeError("Best assemblers ratio was not defined for the pipeline yet")
        return self._precision_error

    @property
    @lru_cache()
    def inputs(self) -> List[Input]:
        """
        Calculates intermediate inputs for 1 unit of end product, taking efficiency into account
        Disclaimer: doesn't take assemblers into account
                    and assumes the same efficiency for every production
        """
        efficiency_ratio = (100 + self.efficiency) / 100
        queue = [Input(component=self.component, quantity=1 / efficiency_ratio)]
        inputs = []
        while queue:
            input_ = queue.pop()
            inputs.append(input_)

            # Extend queue with sub-inputs of the given input
            main_component = input_.component
            if not main_component.is_base_input:
                for sub_input in main_component.recipe.inputs:
                    quantity = (input_.quantity / main_component.recipe.quantity
                                * sub_input.quantity / efficiency_ratio)
                    queue.append(Input(component=sub_input.component, quantity=quantity))

        # group and sum all inputs
        def group_func(element: Input) -> Component:
            """ Grouping function (group by component) """
            return element.component

        grouped_inputs = []
        for component, group in groupby(sorted(inputs, key=group_func), key=group_func):
            grouped_inputs.append(Input(component=component,
                                        quantity=sum(i.quantity for i in group)))
        return grouped_inputs

    @property
    @lru_cache()
    def craftable_inputs(self):
        return [i for i in self.inputs if not i.component.is_base_input]

    def _find_best_assemblers_ratio(self, max_precision_error: float):
        """
        Find a best ratio for assemblers,
        provided that a multiplier shouldn't exceed <max_multiplier>
        and the accumulated error shouldn't exceed <max_precision_error>
        """
        # find float number of required assemblers
        float_assemblers_ratios = {}
        for input_ in self.craftable_inputs:
            if input_.component in float_assemblers_ratios:
                raise RuntimeError("Inputs are not grouped by component")
            recipe = input_.component.recipe
            float_assemblers_ratios[input_.component] = input_.quantity / recipe.quantity * recipe.time

        # find minimum integer numbers of required assemblers
        for i in range(1, self._max_multiplier + 1):
            accumulated_error = 0
            for ratio in float_assemblers_ratios.values():
                value = ratio * i
                if value % 1 == 0:
                    continue
                if value < 1:
                    break
                accumulated_error += 1 - value % 1
                if accumulated_error > max_precision_error:
                    break
            else:
                return {c: math.ceil(r * i) for c, r in float_assemblers_ratios.items()}
        raise ValueError(f"Not found for precision {max_precision_error}")

    @property
    @lru_cache()
    def best_assemblers_ratio(self) -> AssemblersRatios:
        """
        Find best possible perfect ratio matrix
        by incrementally increasing the allowed precision error
        Returns the matrix and the precision error
        """
        self._precision_error = 0.0
        while True:
            try:
                return self._find_best_assemblers_ratio(self._precision_error)
            except ValueError:
                self._precision_error += 0.01
                continue

    @property
    @lru_cache()
    def required_assemblers_ratio(self) -> AssemblersRatios:
        """
        Calculates the required assemblers ratio,
        based on the best assemblers ratio and required output
        """
        current_output = self.get_product_output(self.component, self.best_assemblers_ratio)
        multiplier = round(self.required_output / current_output, 6)
        return {c: math.ceil(r * multiplier) for c, r in self.best_assemblers_ratio.items()}

    def get_product_output(self, component: Component,
                           assemblers_ratios: AssemblersRatios,
                           efficiency: float = None) -> float:
        """
        Returns output per second of the given component
        from the given assemblers assembly
        """
        if efficiency is None:
            efficiency = self.efficiency
        n_assemblers = next(a for c, a in assemblers_ratios.items() if c == component)
        recipe = component.recipe
        output_per_second = n_assemblers * recipe.quantity / recipe.time * self.assemblers_speed
        # return output per second considering the efficiency
        return output_per_second + output_per_second * efficiency / 100


def print_inputs(inputs: List[Input]):
    """ Print inputs """
    # Print base components
    print(f"\nBase components per unit:")
    for input_ in inputs:
        if input_.component.is_base_input:
            print(f"- {input_.quantity:.2f} {input_.component}")

    # Print inputs
    print(f"\nInputs:")
    for i, input_ in enumerate(sorted(inputs, key=lambda i: not i.component.is_base_input)):
        prefix = "*" if input_.component.is_base_input else ""
        print(f"  {i + 1:2d}. {prefix}{input_.component}:")
        for built_input in inputs:
            if built_input.component.is_base_input:
                continue
            if any(i.component == input_.component for i in built_input.component.recipe.inputs):
                print(f"      -> {built_input.component}")


def print_ratios(assemblers_ratios: AssemblersRatios, prefix: str = ""):
    if prefix:
        print(f"\n{prefix} ratio:")
    else:
        print("\nRatio:")

    for component, ratio in assemblers_ratios.items():
        print("- {}: {}".format(component, ratio))
    print(f"Total number of assemblers: {sum(assemblers_ratios.values())}")


def main():
    Recipe.parse_recipes_data()

    _INPUT_COMPONENTS.update(
        Component.from_name(c) for c in [
            # fluid
            "water",  # "sulfuric-acid",
            # chips
            # "electronic-circuit",
            # "advanced-circuit",
            "processing-unit",
            # smelted
            "iron-plate", "stone-brick", "copper-plate",
            "steel-plate", "iron-plate", "stone-brick", "copper-plate",
            # other
            "sulfur", "plastic-bar", "concrete",
        ])
    pipeline = Pipeline(component=Component.from_name("industrial-furnace"),
                        assemblers_speed=3.5,
                        efficiency=48,
                        required_output=45)
    # Calculate intermediate inputs
    print("\n\n==", pipeline.component, "==")
    print_inputs(pipeline.inputs)
    print_ratios(pipeline.best_assemblers_ratio, prefix="Best")
    print(f"Precision error: {pipeline.precision_error:.2f}")
    # Report best assemblers ratio
    output = pipeline.get_product_output(pipeline.component,
                                         pipeline.best_assemblers_ratio,
                                         efficiency=0)
    print(f"Output: {output:.1f}/s ({output * 60:.1f}/m)")
    output = pipeline.get_product_output(pipeline.component,
                                         pipeline.best_assemblers_ratio)
    print(f"Output (+efficiency): {output:.1f}/s ({output * 60:.1f}/m)")

    # Report required assemblers ratio
    print_ratios(pipeline.required_assemblers_ratio, prefix="Required")
    output = pipeline.get_product_output(pipeline.component,
                                         pipeline.required_assemblers_ratio,
                                         efficiency=0)
    print(f"Output: {output:.1f}/s ({output * 60:.1f}/m)")
    output = pipeline.get_product_output(pipeline.component,
                                         pipeline.required_assemblers_ratio)
    print(f"Output (+efficiency): {output:.1f}/s ({output * 60:.1f}/m)")


if __name__ == "__main__":
    main()
