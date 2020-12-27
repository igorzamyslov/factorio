from typing import List, Tuple, Dict, Set
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
                precision_error: float = 0) -> RatioMatrixType:
    # find float number of required assemblers
    ratio_matrix = [(name, number / output * time)
                    for name, number, time, output in input_matrix]
    # find minimum integer numbers of required assemblers
    for i in range(1, max_multiplier + 1):
        # TODO: accumulate error
        for _, ratio in ratio_matrix:
            value = ratio * i
            if value % 1 == 0:
                continue
            if value < 1 or 1 - value % 1 > precision_error:
                break
        else:
            return [(n, math.ceil(r * i)) for n, r in ratio_matrix]
    raise ValueError("Not found")


def find_required_ratio(end_product: str, ratio_matrix: RatioMatrixType, 
                         required_output: float, assembler_speed: float):
    ep_time, ep_output = PRODUCTION_DATA[end_product]
    _, ep_assemblers = next(e for e in ratio_matrix if e[0] == end_product)
    multiplier = round(1 / (ep_assemblers * ep_output / ep_time / required_output) / assembler_speed, 6)
    return [(n, math.ceil(r * multiplier)) for n, r in ratio_matrix]


def create_input_matrix(end_product: str, base_components: Set[str] = None) -> InputMatrixType:
    add_to_matrix = [(end_product, 1)]
    if base_components is None:
        base_components = set()
    temp_matrix = {}
    while add_to_matrix:
        component, quantity = add_to_matrix.pop()
        temp_matrix.setdefault(component, 0)
        temp_matrix[component] += quantity
        if component in base_components: 
            continue
        if component not in INPUT_DATA:
            base_components.add(component)
            continue
        _, produced = PRODUCTION_DATA[component]
        add_to_matrix.extend((c, q * quantity / produced) 
                                for c, q in INPUT_DATA[component])
    # Print base components 
    print(f"Base components:")
    for component in base_components:
        print(f"- {temp_matrix[component]} {component}(s)")
    # Generate input matrix
    return [(component, quantity, *PRODUCTION_DATA[component])
            for component, quantity in temp_matrix.items()
            if component not in base_components]


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
    base_components = set(["Advanced circuit"])
    print("\n\n==", end_product, "==")

    # Create input matrix
    input_matrix = create_input_matrix(end_product, base_components=base_components)

    # Find perfect ratio
    precision_error = 0
    while True:
        try:
            perfect_ratio_matrix = find_perfect_ratio(
                input_matrix, precision_error=precision_error)
        except ValueError:
            precision_error += 0.01
            continue
        break
    print_ratio_matrix(perfect_ratio_matrix, prefix="Perfect")
    print(f"Precision error: {precision_error:.2f}")

    # Find required ratio
    required_output = 1 / 30  # per second 
    assembler_speed = 0.75
    required_ratio_matrix = find_required_ratio(end_product, perfect_ratio_matrix, 
                                                required_output, assembler_speed)
    print_ratio_matrix(required_ratio_matrix, prefix="Required")


if __name__ == "__main__":
    main()