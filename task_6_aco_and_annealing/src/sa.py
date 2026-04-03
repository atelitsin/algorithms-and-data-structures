from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.graph import Graph


@dataclass(frozen=True)
class AnnealingConfig:
    initial_temperature: float = 1000.0
    final_temperature: float = 0.1
    cooling_rate: float = 0.995
    iterations_per_temperature: int = 100
    seed: int | None = None


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
    i, j = sorted(random.sample(range(n), 2))

    candidate_route = two_opt_neighbor(current_route, i, j)
    candidate_cost = graph.cycle_length(candidate_route)

    if accept_move(current_cost, candidate_cost, temperature):
        current_route = candidate_route
        current_cost = candidate_cost

    if current_cost < best_cost:
        best_route = current_route[:]
        best_cost = current_cost

    return current_route, current_cost, best_route, best_cost


def _random_route(graph: Graph, rng: random.Random) -> list[int]:
    route = list(range(len(graph.nodes)))
    rng.shuffle(route)
    return route


def simulated_annealing(
    graph: Graph,
    config: AnnealingConfig | None = None,
    initial_route: list[int] | None = None,
) -> tuple[list[int], float, list[float]]:
    if graph.nodes == []:
        raise ValueError("Graph must contain at least one node.")

    config = config or AnnealingConfig()
    rng = random.Random(config.seed)

    current_route = initial_route[:] if initial_route is not None else _random_route(graph, rng)
    current_cost = graph.cycle_length(current_route)
    best_route = current_route[:]
    best_cost = current_cost

    history = [best_cost]
    temperature = config.initial_temperature

    while temperature > config.final_temperature:
        for _ in range(config.iterations_per_temperature):
            i, j = sorted(rng.sample(range(len(current_route)), 2))
            candidate_route = two_opt_neighbor(current_route, i, j)
            candidate_cost = graph.cycle_length(candidate_route)

            if candidate_cost <= current_cost:
                current_route = candidate_route
                current_cost = candidate_cost
            else:
                delta = candidate_cost - current_cost
                probability = math.exp(-delta / temperature)
                if rng.random() < probability:
                    current_route = candidate_route
                    current_cost = candidate_cost

            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost

        history.append(best_cost)
        temperature *= config.cooling_rate

    return best_route, best_cost, history
