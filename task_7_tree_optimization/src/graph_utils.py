from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    weight: int


@dataclass
class WeightedGraph:
    n_vertices: int
    edges: list[Edge]
    adjacency: list[list[tuple[int, int]]]
    positions: list[tuple[float, float]]

    @staticmethod
    def max_edges_for_vertices(n_vertices: int) -> int:
        return n_vertices * (n_vertices - 1) // 2

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        max_edges = self.max_edges_for_vertices(self.n_vertices)
        return 0.0 if max_edges == 0 else self.edge_count / max_edges


def _make_positions(n_vertices: int, seed: int | None) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    if n_vertices == 1:
        return [(0.0, 0.0)]

    positions: list[tuple[float, float]] = []
    for idx in range(n_vertices):
        angle = 2.0 * math.pi * idx / n_vertices
        radius = 1.0 + rng.uniform(-0.12, 0.12)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append((x, y))
    return positions


def _build_adjacency(n_vertices: int, edges: Iterable[Edge]) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_vertices)]
    for edge in edges:
        adjacency[edge.u].append((edge.v, edge.weight))
        adjacency[edge.v].append((edge.u, edge.weight))
    return adjacency


def generate_connected_random_graph(
    n_vertices: int,
    density: float,
    max_weight: int = 100,
    seed: int | None = None,
) -> WeightedGraph:
    if n_vertices < 2:
        raise ValueError("n_vertices must be >= 2")
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")
    if max_weight < 1:
        raise ValueError("max_weight must be >= 1")

    rng = random.Random(seed)
    max_edges = WeightedGraph.max_edges_for_vertices(n_vertices)
    target_edges = max(n_vertices - 1, min(max_edges, int(round(max_edges * density))))

    edge_keys: set[tuple[int, int]] = set()
    edges: list[Edge] = []

    # Start with a random spanning tree to guarantee connectivity.
    for vertex in range(1, n_vertices):
        parent = rng.randrange(0, vertex)
        u, v = sorted((vertex, parent))
        edge_keys.add((u, v))
        edges.append(Edge(u=u, v=v, weight=rng.randint(1, max_weight)))

    while len(edges) < target_edges:
        u = rng.randrange(0, n_vertices)
        v = rng.randrange(0, n_vertices)
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) in edge_keys:
            continue
        edge_keys.add((a, b))
        edges.append(Edge(u=a, v=b, weight=rng.randint(1, max_weight)))

    adjacency = _build_adjacency(n_vertices, edges)
    positions = _make_positions(n_vertices, seed)
    return WeightedGraph(
        n_vertices=n_vertices,
        edges=edges,
        adjacency=adjacency,
        positions=positions,
    )
