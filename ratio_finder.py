#!/usr/bin/env python3
""" Script to calculate the perfect/near-perfect ratios for different recipes """
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import Dict, List, Optional, Set, Tuple

# TODO:
# [x] Move input to separate file
# [x] Populate the input file automatically
# [x] Handling items with multiple recipes (asking user to choose)
# [x] Handling items with probability (right now skipped)
# [ ] Make a better user-interface (run the script via user input)
# [ ] Add docstrings + check with pylint
# [ ] Handle different assemblers with their own efficiency / speed
# [ ] Handle recipes with recursion (e.g. a component is present in both inputs and outputs)
# [ ] Update algorithm to be able to
#     [ ] minimise either error OR number of assemblers
#     [ ] provide all possible combinations of assemblers
#     [ ] use better algorithm to find equation solution
# [ ] Better handling of recipes with multiple products (right now no report about artifacts)
# [ ] Draw directed graph of inputs (visjs or python library)


# Globally available data
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

    @lru_cache(maxsize=None)
    def _choose_recipe(self) -> Optional[Recipe]:
        """ Propose user to select a recipe for the given component """
        # Check if recipe for the component exists
        if self not in _RECIPES:
            return None

        # Ask the user which recipe to use / or mark the component as input
        component_recipes: List[Optional[Recipe]] = _RECIPES[self].copy()
        component_recipes.insert(0, None)
        print(f"\nChoose which recipe to use to craft \"{self}\":")
        for i, recipe in enumerate(component_recipes):
            print(f"{i}. {recipe.name if recipe else 'None: mark as input component'}")
        while True:
            try:
                recipe_to_use = int(input("Recipe to use: "))
                if recipe_to_use < 0 or recipe_to_use > len(component_recipes) - 1:
                    raise ValueError
            except (TypeError, ValueError):
                print(f"Please enter a number between 0 and {len(component_recipes) - 1}")
                continue
            else:
                return component_recipes[recipe_to_use]

    def choose_all_relevant_recipes(self) -> Set[Component]:
        """
        Go through the components recursively and for every one of them:
        either choose a recipe or mark as a base component.
        Return a list of components marked as base.
        """
        base_components = set()
        if self.recipe is None:
            base_components.add(self)
        else:
            for input_ in self.recipe.inputs:
                base_components.update(input_.component.choose_all_relevant_recipes())
        return base_components

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
    quantity: float
    probability: float  # [0;1]
    inputs: List[Input]

    @property
    def average_quantity(self):
        """ Quantity which takes probability into account """
        return self.quantity * self.probability

    @staticmethod
    def parse_recipes_data(filename: str):
        """ Parses the recipes file and initialises the global _RECIPES dictionary. """
        with open(os.path.join("recipes", filename), "r") as f:
            data = json.load(f)

        for recipe_data in data.values():
            for product_data in recipe_data["products"]:
                if "amount" not in product_data:
                    # calculate amount based on the average between max and min
                    product_data["amount"] = \
                        (product_data["amount_max"] + product_data["amount_min"]) / 2
                recipe = Recipe(name=recipe_data["name"],
                                time=recipe_data["energy"],
                                component=Component.from_name(product_data["name"]),
                                quantity=product_data["amount"],
                                probability=product_data["probability"],
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
    input_components: Tuple[Component, ...]  # components that are provided as inputs

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
            if not self.is_base_input(main_component):
                for sub_input in main_component.recipe.inputs:
                    quantity = (input_.quantity / main_component.recipe.average_quantity
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

    def is_base_input(self, component: Component) -> bool:
        """
        Checks whether a component is given as input:
        - Either it is a base component (which doesn't have its own recipe)
        - Or the component is marked as input-component
        """
        return component in self.input_components or component.recipe is None

    @property
    @lru_cache()
    def craftable_inputs(self):
        return [i for i in self.inputs if not self.is_base_input(i.component)]

    @property
    @lru_cache()
    def base_inputs(self):
        return [i for i in self.inputs if self.is_base_input(i.component)]

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
            if len(self.craftable_inputs) == 1:
                # special case when there's no need to balance anything,
                # there's just one craftable
                ratio = 1
            else:
                ratio = input_.quantity / recipe.average_quantity * recipe.time
            float_assemblers_ratios[input_.component] = ratio

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
        output_per_second = (n_assemblers * recipe.average_quantity
                             / recipe.time * self.assemblers_speed)
        # return output per second considering the efficiency
        return output_per_second + output_per_second * efficiency / 100

    def print_inputs(self):
        """ Print inputs """
        # Print base components
        print(f"\nBase components per unit:")
        for input_ in self.base_inputs:
            print(f"- {input_.quantity:.2f} {input_.component}")

        # Print inputs
        print(f"\nInputs:")
        prefixed_inputs = []
        prefixed_inputs.extend(("*", i) for i in self.base_inputs)
        prefixed_inputs.extend(("", i) for i in self.craftable_inputs)
        for i, (prefix, input_) in enumerate(prefixed_inputs):
            if input_.component == self.component:
                continue
            print(f"  {i + 1:2d}. {prefix}{input_.component}:")
            for craftable_input in self.craftable_inputs:
                if any(i.component == input_.component
                       for i in craftable_input.component.recipe.inputs):
                    print(f"      -> {craftable_input.component}")


def print_ratios(assemblers_ratios: AssemblersRatios, prefix: str = ""):
    if prefix:
        print(f"\n{prefix} ratio:")
    else:
        print("\nRatio:")

    for component, ratio in assemblers_ratios.items():
        print("- {}: {}".format(component, ratio))
    print(f"Total number of assemblers: {sum(assemblers_ratios.values())}")


def main():
    Recipe.parse_recipes_data("aai-se-recipes.json")
    end_product = Component.from_name("petroleum-gas")
    input_components = end_product.choose_all_relevant_recipes()

    pipeline = Pipeline(component=end_product,
                        assemblers_speed=3.5,
                        efficiency=48,
                        required_output=45,
                        input_components=tuple(input_components))
    # Generate report
    print("\n\n==", pipeline.component, "==")
    pipeline.print_inputs()
    # Report best assemblers ratio
    print_ratios(pipeline.best_assemblers_ratio, prefix="Best")
    print(f"Precision error: {pipeline.precision_error:.2f}")
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
