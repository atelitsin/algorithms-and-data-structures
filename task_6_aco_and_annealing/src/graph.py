class Graph:
    def __init__(self, nodes: list[str], weights: list[list[float]]):
        n = len(nodes)
        if len(weights) != n:
            raise ValueError("Matrix row count must match number of nodes.")
        for row in weights:
            if len(row) != n:
                raise ValueError("Weight matrix must be square (n x n).")
        self.nodes = nodes[:]
        self.weights = [row[:] for row in weights]

    def get_weight(self, u: int, v: int) -> float:
        return self.weights[u][v]


nodes = ["a", "b", "c"]
inf = float("inf")
weights = [
    [0,   3,   inf],
    [3,   0,   5  ],
    [inf, 5,   0  ],
]
graph = Graph(nodes, weights)
print(graph.get_weight(0, 1))  # Output: 3
print(graph.get_weight(0, 2))  # Output: inf
print(graph.get_weight(1, 2))  # Output: 5