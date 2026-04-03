from __future__ import annotations

import argparse

from src.graph import Graph
from src.gui_qt import launch_gui
from src.stp_parser import parse_stp_file
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


def run_instance(name: str, graph: Graph) -> None:
    config = AnnealingConfig(
        initial_temperature=100.0,
        final_temperature=0.1,
        cooling_rate=0.99,
        iterations_per_temperature=50,
        seed=42,
    )

    best_route, best_cost, history = simulated_annealing(graph, config)

    print(f"[{name}] Best route:", route_to_names(graph, best_route))
    print(f"[{name}] Best cost:", best_cost)
    print(f"[{name}] History points:", len(history))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated annealing TSP demo")
    parser.add_argument("--gui", action="store_true", help="Launch the Qt GUI")
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    run_instance("demo", build_demo_graph())
    run_instance("berlin52.stp", parse_stp_file("berlin52.stp"))
    run_instance("world666.stp", parse_stp_file("world666.stp"))


if __name__ == "__main__":
    main()
