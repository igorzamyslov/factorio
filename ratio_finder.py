from collections import defaultdict
from typing import List, Tuple, Dict, Iterable
import math


InputMatrixType = List[Tuple[str, float, float, int]]
RatioMatrixType = List[Tuple[str, int]]


# Provides the following info per craftable:
# - time to produce
# - number of items produces
PRODUCTION_DATA: Dict[str, Tuple[float, int]] = {
    "Advanced circuit": (6, 1),
    "Assembling machine 1": (0.5, 1),
    "Assembling machine 2": (0.5, 1),
    "Burner assembling machine": (0.5, 1),
    "Concrete": (10, 10),
    "Copper cable": (0.5, 2),
    "Electric motor": (0.8, 1),
    "Electronic circuit": (0.5, 1),
    "Engine unit": (10, 1),
    "Iron gear wheel": (0.5, 1),
    "Iron stick": (0.5, 2),
    "Motor": (0.6, 1),
    "Pipe": (0.5, 1),
    "Roboport": (5, 1),
    "Sand": (0.5, 2),
    "Stone tablet": (0.5, 4),
} 

# Provides the list of inputs per craftable, containing:
# - Name of the input
# - Required quantity
INPUT_DATA: Dict[str, List[Tuple[str, int]]] = {
    "Advanced circuit": [("Copper cable", 4), ("Electronic circuit", 2), ("Plastic bar", 2)],
    "Assembling machine 1": [("Iron gear wheel", 4), ("Electric motor", 1), ("Burner assembling machine", 1)],
    "Assembling machine 2": [("Electronic circuit", 2), ("Electric motor", 2), ("Assembling machine 1", 1), ("Steel plate", 2)],
    "Burner assembling machine": [("Motor", 1), ("Stone brick", 4), ("Iron plate", 8)],
    "Concrete": [("Water", 100), ("Sand", 10), ("Stone brick", 5), ("Iron stick", 2)],
    "Copper cable": [("Copper plate", 1)],
    "Electric motor": [("Copper cable", 6), ("Motor", 1)],
    "Electronic circuit": [("Copper cable", 3), ("Stone tablet", 1)],
    "Engine unit": [("Iron gear wheel", 2), ("Motor", 1), ("Pipe", 2), ("Steel plate", 2)],
    "Iron gear wheel": [("Iron plate", 2)],
    "Iron stick": [("Iron plate", 1)],
    "Motor": [("Iron gear wheel", 1), ("Iron plate", 1)],
    "Pipe": [("Iron plate", 1)],
    "Roboport": [("Advanced circuit", 50), ("Electric motor", 50), ("Concrete", 50), ("Steel plate", 50)],
    "Sand": [("Stone", 1)],
    "Stone tablet": [("Stone brick", 1)],
}

assert INPUT_DATA.keys() == PRODUCTION_DATA.keys()


def find_perfect_ratio(input_matrix: InputMatrixType, 
                       max_multiplier: int = 100, 
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


def find_required_ratio(end_product: str, ratio_matrix: RatioMatrixType, 
                        required_output: float, assembler_speed: float):
    current_output = get_product_output(end_product, ratio_matrix, assembler_speed)
    multiplier = round(required_output / current_output, 6)
    return [(n, math.ceil(r * multiplier)) for n, r in ratio_matrix]


def get_product_output(end_product: str, ratio_matrix: RatioMatrixType, 
                       assembler_speed: float) -> float:
    """ 
    Returns output per second of the end product with the given assembler speed 
    Disclaimer: Only works with the perfect ratio matrix
    """
    ep_time, ep_output = PRODUCTION_DATA[end_product]
    ep_assemblers = next(a for n, a in ratio_matrix if n == end_product)
    return ep_assemblers * ep_output / ep_time * assembler_speed


def create_input_matrix(end_product: str, 
                        base_components: Iterable[str] = None) -> InputMatrixType:
    # Init base components
    base_components = set(base_components) if base_components else set()
    # Create temporary matrix
    add_to_matrix = [(end_product, 1)]
    temp_matrix_dict = {}
    while add_to_matrix:
        component, quantity = add_to_matrix.pop()
        temp_matrix_dict.setdefault(component, 0)
        temp_matrix_dict[component] += quantity
        if component in base_components: 
            continue
        if component not in INPUT_DATA:
            base_components.add(component)
            continue
        _, produced = PRODUCTION_DATA[component]
        add_to_matrix.extend((c, q * quantity / produced) 
                             for c, q in INPUT_DATA[component])
    # Print base components 
    print(f"Base components per unit:")
    for component in base_components:
        print(f"- {temp_matrix_dict[component]} {component}(s)")
    # Print requirements
    print(f"\nInputs:")
    for component in temp_matrix_dict:
        if component in base_components:
            continue
        print(f"- {component}: ", end="")
        print(", ".join(f"*{c}" if c in base_components else c
                        for c, _ in INPUT_DATA[component]))

    # Generate input matrix
    return [(c, q, *PRODUCTION_DATA[c])
            for c, q in temp_matrix_dict.items()
            if c not in base_components]


def print_ratio_matrix(matrix: RatioMatrixType, prefix: str = ""):
    if prefix:
        print(f"\n{prefix} ratio:")
    else:
        print("\nRatio:")

    for entry in matrix:
        print("- {}: {}".format(*entry))
    print(f"Total number of assemblers: {sum(e[1] for e in matrix)}")
    

def main():
    # Input
    end_product = "Roboport"
    base_components = ["Advanced circuit", "Concrete"]
    assembler_speed = 0.75
    required_output = 1 / 120  # per second 

    # Create input matrix
    print("\n\n==", end_product, "==")
    input_matrix = create_input_matrix(end_product, base_components=base_components)

    # Find perfect ratio
    precision_error = 0
    while True:
        try:
            perfect_ratio_matrix = find_perfect_ratio(
                input_matrix, max_precision_error=precision_error)
        except ValueError:
            precision_error += 0.01
            continue
        break
    print_ratio_matrix(perfect_ratio_matrix, prefix="Perfect")
    print(f"Precision error: {precision_error:.2f}")
    print(f"Output: {get_product_output(end_product, perfect_ratio_matrix, assembler_speed)}/s")

    # Find required ratio
    required_ratio_matrix = find_required_ratio(end_product, perfect_ratio_matrix, 
                                                required_output, assembler_speed)
    print_ratio_matrix(required_ratio_matrix, prefix="Required")


if __name__ == "__main__":
    main()