import math
import random

import graph


def two_opt_neighbor(route: list[int], i: int, j: int) -> list[int]:
    n = len(route)
    if i < 0 or j < 0 or i >= n or j >= n:
        raise IndexError("i and j must be valid indices of route.")
    if i >= j:
        raise ValueError("Require i < j for 2-opt.")

    new_route = route[:]
    new_route[i : j + 1] = reversed(new_route[i : j + 1])
    return new_route


def accept_move(current_cost: float, new_cost: float, temperature: float) -> bool:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    if new_cost <= current_cost:
        return True
    else:
        probability = math.exp((current_cost - new_cost) / temperature)
        return random.random() < probability
    

print(accept_move(10, 8, 100))
print(accept_move(10, 10, 100))
print(accept_move(10, 12, 0.0001))
print(accept_move(10, 12, 100))
try:
    print(accept_move(10, 12, 0))
except ValueError as e:
    print("Expected error:", e)