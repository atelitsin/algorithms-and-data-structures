from src.graph import Graph
from src.sa import AnnealingConfig, simulated_annealing


def build_demo_graph() -> Graph:
    nodes = ["a", "b", "c", "d", "f", "g"]
    inf = float("inf")
    weights = [
        [0, 3, inf, inf, 3, inf],
        [3, 0, 3, inf, inf, 3],
        [inf, 3, 0, 1, inf, 1],
        [inf, inf, 8, 0, 3, inf],
        [1, inf, inf, 1, 0, inf],
        [3, 3, 3, 5, 4, 0],
    ]
    return Graph(nodes, weights)


def route_to_names(graph: Graph, route: list[int]) -> list[str]:
    return [graph.nodes[i] for i in route]


def main() -> None:
    graph = build_demo_graph()
    config = AnnealingConfig(
        initial_temperature=100.0,
        final_temperature=0.1,
        cooling_rate=0.99,
        iterations_per_temperature=50,
        seed=42,
    )

    best_route, best_cost, history = simulated_annealing(graph, config)

    print("Best route:", route_to_names(graph, best_route))
    print("Best cost:", best_cost)
    print("History points:", len(history))


if __name__ == "__main__":
    main()
