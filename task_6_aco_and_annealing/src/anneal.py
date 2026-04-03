from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


INF = float("inf")


@dataclass(frozen=True)
class AnnealingConfig:
	initial_temp: float = 1000.0
	final_temp: float = 0.1
	cooling_rate: float = 0.995
	iterations_per_temp: int = 2000
	max_no_improve_levels: int = 250
	seed: int | None = None


@dataclass
class WeightedGraph:
	nodes: List[str]
	weights: List[List[float]]

	def __post_init__(self) -> None:
		if len(self.weights) != len(self.nodes):
			raise ValueError("Weights matrix size must match number of nodes")
		for row in self.weights:
			if len(row) != len(self.nodes):
				raise ValueError("Weights matrix must be square")

	@property
	def size(self) -> int:
		return len(self.nodes)

	@staticmethod
	def from_adjacency(adjacency: Dict[str, Dict[str, float]]) -> "WeightedGraph":
		nodes = sorted(adjacency.keys())
		index = {node: i for i, node in enumerate(nodes)}
		n = len(nodes)
		weights = [[INF for _ in range(n)] for _ in range(n)]

		for i in range(n):
			weights[i][i] = 0.0

		for src, neighbours in adjacency.items():
			if src not in index:
				continue
			src_i = index[src]
			for dst, weight in neighbours.items():
				if dst not in index:
					raise ValueError(f"Unknown node '{dst}' in adjacency list")
				weights[src_i][index[dst]] = float(weight)

		return WeightedGraph(nodes=nodes, weights=weights)


def parse_stp_file(file_path: str | Path) -> WeightedGraph:
	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"STP file not found: {path}")

	node_count: int | None = None
	edges: List[Tuple[int, int, float]] = []

	with path.open("r", encoding="utf-8") as file:
		for raw_line in file:
			line = raw_line.strip()
			if not line:
				continue

			parts = line.split()
			if parts[0] == "Nodes" and len(parts) >= 2:
				node_count = int(parts[1])
			elif parts[0] == "E" and len(parts) >= 4:
				u = int(parts[1]) - 1
				v = int(parts[2]) - 1
				w = float(parts[3])
				edges.append((u, v, w))

	if node_count is None:
		raise ValueError(f"Could not read 'Nodes' section in: {path}")

	nodes = [str(i + 1) for i in range(node_count)]
	weights = [[INF for _ in range(node_count)] for _ in range(node_count)]
	for i in range(node_count):
		weights[i][i] = 0.0

	# STP instances in this task are undirected; store symmetric weights.
	for u, v, w in edges:
		weights[u][v] = w
		weights[v][u] = w

	return WeightedGraph(nodes=nodes, weights=weights)


def cycle_length(graph: WeightedGraph, route: Sequence[int]) -> float:
	if not route:
		return INF

	total = 0.0
	n = len(route)
	for i in range(n):
		src = route[i]
		dst = route[(i + 1) % n]
		edge_weight = graph.weights[src][dst]
		if math.isinf(edge_weight):
			return INF
		total += edge_weight
	return total


def route_to_nodes(graph: WeightedGraph, route: Sequence[int]) -> List[str]:
	return [graph.nodes[i] for i in route]


class SimulatedAnnealingTSP:
	def __init__(self, graph: WeightedGraph, config: AnnealingConfig | None = None) -> None:
		if graph.size < 3:
			raise ValueError("At least 3 nodes are required to form a Hamiltonian cycle")
		self.graph = graph
		self.config = config or AnnealingConfig()
		self.rng = random.Random(self.config.seed)

	def _random_route(self) -> List[int]:
		route = list(range(self.graph.size))
		self.rng.shuffle(route)
		return route

	def _greedy_route(self, start: int) -> List[int] | None:
		n = self.graph.size
		route = [start]
		unvisited = set(range(n))
		unvisited.remove(start)
		current = start

		while unvisited:
			candidates = [node for node in unvisited if not math.isinf(self.graph.weights[current][node])]
			if not candidates:
				return None
			next_node = min(candidates, key=lambda node: self.graph.weights[current][node])
			route.append(next_node)
			unvisited.remove(next_node)
			current = next_node

		if math.isinf(self.graph.weights[route[-1]][route[0]]):
			return None
		return route

	def _initial_route(self) -> List[int]:
		# First, try deterministic greedy starts.
		best_greedy: List[int] | None = None
		best_greedy_cost = INF
		for start in range(self.graph.size):
			route = self._greedy_route(start)
			if route is None:
				continue
			cost = cycle_length(self.graph, route)
			if cost < best_greedy_cost:
				best_greedy = route
				best_greedy_cost = cost

		if best_greedy is not None:
			return best_greedy

		# Fallback to random attempts.
		attempts = max(1000, self.graph.size * 100)
		for _ in range(attempts):
			route = self._random_route()
			if not math.isinf(cycle_length(self.graph, route)):
				return route

		raise ValueError("Could not build a valid initial Hamiltonian cycle")

	def _two_opt_neighbor(self, route: Sequence[int]) -> List[int]:
		n = len(route)
		i, j = sorted(self.rng.sample(range(n), 2))
		if i == 0 and j == n - 1:
			# This reversal produces the same cycle with opposite orientation.
			i = 1
		new_route = list(route)
		new_route[i : j + 1] = reversed(new_route[i : j + 1])
		return new_route

	def solve(self, return_history: bool = False) -> Tuple[List[int], float] | Tuple[List[int], float, List[float]]:
		current_route = self._initial_route()
		current_cost = cycle_length(self.graph, current_route)

		if math.isinf(current_cost):
			raise ValueError("Initial route is invalid due to missing edges")

		best_route = current_route[:]
		best_cost = current_cost

		temperature = self.config.initial_temp
		no_improve_levels = 0
		best_cost_history: List[float] = [best_cost]

		while temperature > self.config.final_temp and no_improve_levels < self.config.max_no_improve_levels:
			improved_on_level = False

			for _ in range(self.config.iterations_per_temp):
				candidate_route = self._two_opt_neighbor(current_route)
				candidate_cost = cycle_length(self.graph, candidate_route)
				if math.isinf(candidate_cost):
					continue

				delta = candidate_cost - current_cost
				if delta <= 0 or self.rng.random() < math.exp(-delta / temperature):
					current_route = candidate_route
					current_cost = candidate_cost

				if current_cost < best_cost:
					best_cost = current_cost
					best_route = current_route[:]
					improved_on_level = True

			if improved_on_level:
				no_improve_levels = 0
			else:
				no_improve_levels += 1

			best_cost_history.append(best_cost)
			temperature *= self.config.cooling_rate

		if return_history:
			return best_route, best_cost, best_cost_history
		return best_route, best_cost


def solve_tsp_with_annealing(
	graph: WeightedGraph,
	config: AnnealingConfig | None = None,
) -> Tuple[List[str], float]:
	solver = SimulatedAnnealingTSP(graph=graph, config=config)
	best_route_idx, best_cost = solver.solve()
	return route_to_nodes(graph, best_route_idx), best_cost


def solve_tsp_with_annealing_history(
	graph: WeightedGraph,
	config: AnnealingConfig | None = None,
) -> Tuple[List[str], float, List[float]]:
	solver = SimulatedAnnealingTSP(graph=graph, config=config)
	best_route_idx, best_cost, history = solver.solve(return_history=True)
	return route_to_nodes(graph, best_route_idx), best_cost, history

