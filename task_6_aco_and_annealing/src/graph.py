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

    def weight(self, u: int, v: int) -> float:
        return self.weights[u][v]
    
    def cycle_length(self, route: list[int]) -> float:
        if route == []:
            return float("inf")

        if len(route) != len(self.nodes):
            raise ValueError("Route must include all nodes exactly once.")

        length = 0.0
        for i in range(len(route) - 1):
            w = self.weight(route[i], route[i + 1])
            if w == float("inf"):
                return float("inf")
            length += w

        last_w = self.weight(route[-1], route[0])
        if last_w == float("inf"):
            return float("inf")
        length += last_w
        return length