import json
import os
import sys
import time
from statistics import mean, stdev

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graph_utils import Graph
from annealing import SimulatedAnnealing, BoltzmannAnnealing
from aco import AntColonyOptimization, AntColonyOptimizationInitial


def run_with_repeats(factory, repeats=3, seed_base=42):
    distances = []
    times = []
    iterations = []
    for i in range(repeats):
        np.random.seed(seed_base + i)
        algo = factory()
        _, dist, metrics = algo.solve()
        distances.append(float(dist))
        times.append(float(metrics["time"]))
        iterations.append(int(metrics["iterations"]))

    result = {
        "distance_mean": mean(distances),
        "distance_std": stdev(distances) if len(distances) > 1 else 0.0,
        "time_mean": mean(times),
        "time_std": stdev(times) if len(times) > 1 else 0.0,
        "iterations_mean": mean(iterations),
        "best_distance": min(distances),
    }
    return result


def graph_settings(graph_name):
    if graph_name == "berlin52":
        return {
            "sa_repeats": 2,
            "boltzmann_repeats": 2,
            "aco_repeats": 2,
            "sa_base_iterations": 2000,
            "boltzmann_base_iterations": 2000,
            "aco_base_n_ants": 12,
            "aco_base_n_iterations": 12,
        }

    return {
        "sa_repeats": 1,
        "boltzmann_repeats": 1,
        "aco_repeats": 1,
        "sa_base_iterations": 1200,
        "boltzmann_base_iterations": 1200,
        "aco_base_n_ants": 8,
        "aco_base_n_iterations": 8,
    }


def analyze_sa(graph, graph_name):
    settings = graph_settings(graph_name)
    base = {
        "initial_temp": 1000,
        "cooling_rate": 0.95,
        "max_iterations": settings["sa_base_iterations"],
    }

    results = {
        "initial_temp": {},
        "cooling_rate": {},
        "max_iterations": {},
    }

    for t0 in [100, 500, 1000, 1500]:
        results["initial_temp"][str(t0)] = run_with_repeats(
            lambda: SimulatedAnnealing(
                graph,
                initial_temp=t0,
                cooling_rate=base["cooling_rate"],
                max_iterations=base["max_iterations"],
            ),
            repeats=settings["sa_repeats"],
        )

    for alpha in [0.90, 0.95, 0.99]:
        results["cooling_rate"][str(alpha)] = run_with_repeats(
            lambda: SimulatedAnnealing(
                graph,
                initial_temp=base["initial_temp"],
                cooling_rate=alpha,
                max_iterations=base["max_iterations"],
            ),
            repeats=settings["sa_repeats"],
        )

    for kmax in [300, 600, 1200, 2000]:
        results["max_iterations"][str(kmax)] = run_with_repeats(
            lambda: SimulatedAnnealing(
                graph,
                initial_temp=base["initial_temp"],
                cooling_rate=base["cooling_rate"],
                max_iterations=kmax,
            ),
            repeats=settings["sa_repeats"],
        )

    return results


def analyze_boltzmann(graph, graph_name):
    settings = graph_settings(graph_name)
    base = {
        "initial_temp": 1000,
        "max_iterations": settings["boltzmann_base_iterations"],
    }

    results = {
        "initial_temp": {},
        "max_iterations": {},
    }

    for t0 in [100, 500, 1000, 1500]:
        results["initial_temp"][str(t0)] = run_with_repeats(
            lambda: BoltzmannAnnealing(
                graph,
                initial_temp=t0,
                max_iterations=base["max_iterations"],
            ),
            repeats=settings["boltzmann_repeats"],
        )

    for kmax in [300, 600, 1200, 2000]:
        results["max_iterations"][str(kmax)] = run_with_repeats(
            lambda: BoltzmannAnnealing(
                graph,
                initial_temp=base["initial_temp"],
                max_iterations=kmax,
            ),
            repeats=settings["boltzmann_repeats"],
        )

    return results


def analyze_aco(graph, graph_name, initial_variant=False):
    cls = AntColonyOptimizationInitial if initial_variant else AntColonyOptimization

    settings = graph_settings(graph_name)
    base = {
        "n_ants": settings["aco_base_n_ants"],
        "n_iterations": settings["aco_base_n_iterations"],
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "Q": 100,
    }

    results = {
        "n_ants": {},
        "n_iterations": {},
        "alpha": {},
        "beta": {},
        "rho": {},
        "Q": {},
    }

    for ants in [5, 8, 12]:
        results["n_ants"][str(ants)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=ants,
                n_iterations=base["n_iterations"],
                alpha=base["alpha"],
                beta=base["beta"],
                rho=base["rho"],
                Q=base["Q"],
            ),
            repeats=settings["aco_repeats"],
        )

    for iters in [5, 8, 12]:
        results["n_iterations"][str(iters)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=base["n_ants"],
                n_iterations=iters,
                alpha=base["alpha"],
                beta=base["beta"],
                rho=base["rho"],
                Q=base["Q"],
            ),
            repeats=settings["aco_repeats"],
        )

    for alpha in [0.5, 1.0, 1.5]:
        results["alpha"][str(alpha)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=base["n_ants"],
                n_iterations=base["n_iterations"],
                alpha=alpha,
                beta=base["beta"],
                rho=base["rho"],
                Q=base["Q"],
            ),
            repeats=settings["aco_repeats"],
        )

    for beta in [1.0, 2.0, 3.0]:
        results["beta"][str(beta)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=base["n_ants"],
                n_iterations=base["n_iterations"],
                alpha=base["alpha"],
                beta=beta,
                rho=base["rho"],
                Q=base["Q"],
            ),
            repeats=settings["aco_repeats"],
        )

    for rho in [0.05, 0.1, 0.2]:
        results["rho"][str(rho)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=base["n_ants"],
                n_iterations=base["n_iterations"],
                alpha=base["alpha"],
                beta=base["beta"],
                rho=rho,
                Q=base["Q"],
            ),
            repeats=settings["aco_repeats"],
        )

    for q in [25, 50, 100]:
        results["Q"][str(q)] = run_with_repeats(
            lambda: cls(
                graph,
                n_ants=base["n_ants"],
                n_iterations=base["n_iterations"],
                alpha=base["alpha"],
                beta=base["beta"],
                rho=base["rho"],
                Q=q,
            ),
            repeats=settings["aco_repeats"],
        )

    return results


def main():
    start = time.time()

    graphs = {
        "berlin52": Graph.load_from_stp("data/berlin52.stp"),
        "world666": Graph.load_from_stp("data/world666.stp"),
    }

    all_results = {}

    for graph_name, graph in graphs.items():
        print(f"Running sweeps for {graph_name}...")
        all_results[graph_name] = {
            "sa_basic": analyze_sa(graph, graph_name),
            "sa_boltzmann": analyze_boltzmann(graph, graph_name),
            "aco_basic": analyze_aco(graph, graph_name, initial_variant=False),
            "aco_initial": analyze_aco(graph, graph_name, initial_variant=True),
        }

    os.makedirs("analysis", exist_ok=True)
    out_path = os.path.join("analysis", "parameter_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(f"Total time: {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()
