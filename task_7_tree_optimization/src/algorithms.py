from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from time import perf_counter
from typing import Callable

from .graph_utils import Edge, WeightedGraph


@dataclass
class MSTResult:
    algorithm: str
    total_weight: int
    mst_edges: list[Edge]
    elapsed_ms: float
    operations: int


class DisjointSetUnion:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


class FibNode:
    def __init__(self, key: float, value: int):
        self.key = key
        self.value = value
        self.degree = 0
        self.mark = False
        self.parent: FibNode | None = None
        self.child: FibNode | None = None
        self.left: FibNode = self
        self.right: FibNode = self
        self.in_heap = True


class FibonacciHeap:
    def __init__(self):
        self.min_node: FibNode | None = None
        self.total_nodes = 0

    def insert(self, key: float, value: int) -> FibNode:
        node = FibNode(key, value)
        self.min_node = self._merge_lists(self.min_node, node)
        self.total_nodes += 1
        return node

    def extract_min(self) -> tuple[float, int]:
        z = self.min_node
        if z is None:
            raise IndexError("extract_min from empty heap")

        if z.child is not None:
            children = list(self._iterate(z.child))
            for child in children:
                child.parent = None
                child.mark = False
            self.min_node = self._merge_lists(self.min_node, z.child)

        if z.right == z:
            self.min_node = None
        else:
            z.left.right = z.right
            z.right.left = z.left
            self.min_node = z.right
            self._consolidate()

        self.total_nodes -= 1
        z.in_heap = False
        z.left = z
        z.right = z
        return z.key, z.value

    def decrease_key(self, node: FibNode, new_key: float) -> None:
        if new_key > node.key:
            raise ValueError("new key is greater than current key")
        node.key = new_key
        parent = node.parent

        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)

        if self.min_node is None or node.key < self.min_node.key:
            self.min_node = node

    def _consolidate(self) -> None:
        degree_table: dict[int, FibNode] = {}
        root_nodes = list(self._iterate(self.min_node)) if self.min_node else []

        for node in root_nodes:
            x = node
            d = x.degree
            while d in degree_table:
                y = degree_table.pop(d)
                if y.key < x.key:
                    x, y = y, x
                self._link(y, x)
                d = x.degree
            degree_table[d] = x

        self.min_node = None
        for node in degree_table.values():
            node.left = node
            node.right = node
            self.min_node = self._merge_lists(self.min_node, node)

    def _link(self, child: FibNode, parent: FibNode) -> None:
        child.left.right = child.right
        child.right.left = child.left

        child.parent = parent
        child.mark = False
        child.left = child
        child.right = child
        parent.child = self._merge_lists(parent.child, child)
        parent.degree += 1

    def _cut(self, node: FibNode, parent: FibNode) -> None:
        if node.right == node:
            parent.child = None
        else:
            node.right.left = node.left
            node.left.right = node.right
            if parent.child == node:
                parent.child = node.right

        parent.degree -= 1
        node.parent = None
        node.mark = False
        node.left = node
        node.right = node
        self.min_node = self._merge_lists(self.min_node, node)

    def _cascading_cut(self, node: FibNode) -> None:
        parent = node.parent
        if parent is None:
            return
        if not node.mark:
            node.mark = True
            return
        self._cut(node, parent)
        self._cascading_cut(parent)

    def _merge_lists(self, a: FibNode | None, b: FibNode | None) -> FibNode | None:
        if a is None:
            return b
        if b is None:
            return a

        a_right = a.right
        b_left = b.left

        a.right = b
        b.left = a
        a_right.left = b_left
        b_left.right = a_right

        return a if a.key <= b.key else b

    def _iterate(self, start: FibNode):
        node = stop = start
        first = True
        while first or node != stop:
            first = False
            yield node
            node = node.right


def _build_result(
    algorithm: str,
    mst_edges: list[Edge],
    elapsed_start: float,
    operations: int,
) -> MSTResult:
    total_weight = sum(edge.weight for edge in mst_edges)
    elapsed_ms = (perf_counter() - elapsed_start) * 1000.0
    return MSTResult(
        algorithm=algorithm,
        total_weight=total_weight,
        mst_edges=mst_edges,
        elapsed_ms=elapsed_ms,
        operations=operations,
    )


def _ensure_tree(graph: WeightedGraph, mst_edges: list[Edge]) -> None:
    if len(mst_edges) != graph.n_vertices - 1:
        raise ValueError("Graph appears disconnected; MST cannot be formed")


def kruskal_mst(graph: WeightedGraph) -> MSTResult:
    start = perf_counter()
    dsu = DisjointSetUnion(graph.n_vertices)
    mst_edges: list[Edge] = []
    operations = 0

    for edge in sorted(graph.edges, key=lambda e: e.weight):
        operations += 1
        if dsu.union(edge.u, edge.v):
            mst_edges.append(edge)
            if len(mst_edges) == graph.n_vertices - 1:
                break

    _ensure_tree(graph, mst_edges)
    return _build_result("Kruskal", mst_edges, start, operations)


def prim_binary_heap_mst(graph: WeightedGraph) -> MSTResult:
    start = perf_counter()
    n = graph.n_vertices
    in_mst = [False] * n
    best_weight = [inf] * n
    parent = [-1] * n
    heap: list[tuple[float, int]] = [(0.0, 0)]
    best_weight[0] = 0.0
    operations = 0

    while heap:
        current_weight, vertex = heappop(heap)
        if in_mst[vertex]:
            continue
        in_mst[vertex] = True

        for neighbor, weight in graph.adjacency[vertex]:
            operations += 1
            if in_mst[neighbor] or weight >= best_weight[neighbor]:
                continue
            best_weight[neighbor] = weight
            parent[neighbor] = vertex
            heappush(heap, (weight, neighbor))

    mst_edges: list[Edge] = []
    for vertex in range(1, n):
        if parent[vertex] == -1:
            continue
        mst_edges.append(Edge(parent[vertex], vertex, int(best_weight[vertex])))

    _ensure_tree(graph, mst_edges)
    return _build_result("Prim (binary heap)", mst_edges, start, operations)


def prim_fibonacci_heap_mst(graph: WeightedGraph) -> MSTResult:
    start = perf_counter()
    n = graph.n_vertices
    in_mst = [False] * n
    best_weight = [inf] * n
    parent = [-1] * n
    operations = 0

    heap = FibonacciHeap()
    node_handles: dict[int, FibNode] = {}

    best_weight[0] = 0.0
    node_handles[0] = heap.insert(0.0, 0)

    while heap.total_nodes > 0:
        _, vertex = heap.extract_min()
        if in_mst[vertex]:
            continue
        in_mst[vertex] = True

        for neighbor, weight in graph.adjacency[vertex]:
            operations += 1
            if in_mst[neighbor] or weight >= best_weight[neighbor]:
                continue
            best_weight[neighbor] = float(weight)
            parent[neighbor] = vertex
            handle = node_handles.get(neighbor)
            if handle is None or not handle.in_heap:
                node_handles[neighbor] = heap.insert(float(weight), neighbor)
            else:
                heap.decrease_key(handle, float(weight))

    mst_edges: list[Edge] = []
    for vertex in range(1, n):
        if parent[vertex] == -1:
            continue
        mst_edges.append(Edge(parent[vertex], vertex, int(best_weight[vertex])))

    _ensure_tree(graph, mst_edges)
    return _build_result("Prim (Fibonacci heap)", mst_edges, start, operations)


def boruvka_mst(graph: WeightedGraph) -> MSTResult:
    start = perf_counter()
    dsu = DisjointSetUnion(graph.n_vertices)
    components = graph.n_vertices
    mst_edges: list[Edge] = []
    operations = 0

    while components > 1:
        cheapest: list[Edge | None] = [None] * graph.n_vertices

        for edge in graph.edges:
            operations += 1
            root_u = dsu.find(edge.u)
            root_v = dsu.find(edge.v)
            if root_u == root_v:
                continue

            current_u = cheapest[root_u]
            if current_u is None or edge.weight < current_u.weight:
                cheapest[root_u] = edge

            current_v = cheapest[root_v]
            if current_v is None or edge.weight < current_v.weight:
                cheapest[root_v] = edge

        merges_this_round = 0
        for edge in cheapest:
            if edge is None:
                continue
            if dsu.union(edge.u, edge.v):
                mst_edges.append(edge)
                components -= 1
                merges_this_round += 1
                if len(mst_edges) == graph.n_vertices - 1:
                    break

        if merges_this_round == 0:
            break

    _ensure_tree(graph, mst_edges)
    return _build_result("Boruvka", mst_edges, start, operations)


def run_mst_algorithm(algorithm_code: str, graph: WeightedGraph) -> MSTResult:
    runners: dict[str, Callable[[WeightedGraph], MSTResult]] = {
        "kruskal": kruskal_mst,
        "prim_binary": prim_binary_heap_mst,
        "prim_fibonacci": prim_fibonacci_heap_mst,
        "boruvka": boruvka_mst,
    }
    if algorithm_code not in runners:
        raise ValueError(f"Unknown algorithm code: {algorithm_code}")
    return runners[algorithm_code](graph)
