import math
import random

from src.graph import Graph


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


def sa_iteration(
    graph: Graph,
    current_route: list[int],
    current_cost: float,
    best_route: list[int],
    best_cost: float,
    temperature: float,
) -> tuple[list[int], float, list[int], float]:
    
    n = len(current_route)
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
    
    new_route = two_opt_neighbor(current_route, i, j)
    new_cost = graph.cycle_length(new_route)

    if accept_move(current_cost, new_cost, temperature):
        current_route = new_route
        current_cost = new_cost

    if current_cost < best_cost:
        best_route = current_route[:]
        best_cost = current_cost

    return current_route, current_cost, best_route, best_cost

