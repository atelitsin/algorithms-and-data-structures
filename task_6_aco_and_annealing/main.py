from __future__ import annotations

from pathlib import Path

from src.anneal import AnnealingConfig, WeightedGraph, parse_stp_file, solve_tsp_with_annealing_history
from src.visualization import save_convergence_plot, save_cycle_plot


def build_demo_graph() -> WeightedGraph:
	# Directed weighted graph from the assignment image.
	adjacency = {
		"a": {"b": 3, "f": 3},
		"b": {"a": 3, "c": 3, "g": 3},
		"c": {"b": 8, "d": 1, "g": 1},
		"d": {"c": 8, "f": 3},
		"f": {"a": 1, "d": 1},
		"g": {"a": 3, "b": 3, "c": 3, "d": 5, "f": 4},
	}
	return WeightedGraph.from_adjacency(adjacency)


def format_cycle(route: list[str]) -> str:
	if not route:
		return "<empty>"
	return " -> ".join(route + [route[0]])


def run_demo() -> None:
	base_dir = Path(__file__).resolve().parent
	out_dir = base_dir / "visualizations"
	graph = build_demo_graph()
	config = AnnealingConfig(
		initial_temp=120.0,
		final_temp=0.01,
		cooling_rate=0.995,
		iterations_per_temp=800,
		max_no_improve_levels=120,
		seed=42,
	)
	route, length, history = solve_tsp_with_annealing_history(graph, config)
	print("[demo] best length:", length)
	print("[demo] best cycle:", format_cycle(route))

	save_cycle_plot(graph, route, out_dir / "demo_cycle.png", "Demo Graph: Hamiltonian Cycle")
	save_convergence_plot(history, out_dir / "demo_convergence.png", "Demo Graph: SA Convergence")
	print(f"[demo] visualization saved to: {out_dir}")


def run_stp(file_name: str, seed: int) -> None:
	base_dir = Path(__file__).resolve().parent
	out_dir = base_dir / "visualizations"
	graph = parse_stp_file(base_dir / file_name)

	# For larger instances use fewer temperature levels but more moves per level.
	config = AnnealingConfig(
		initial_temp=5000.0,
		final_temp=0.5,
		cooling_rate=0.998,
		iterations_per_temp=max(3000, graph.size * 10),
		max_no_improve_levels=180,
		seed=seed,
	)
	route, length, history = solve_tsp_with_annealing_history(graph, config)

	print(f"[{file_name}] nodes: {graph.size}")
	print(f"[{file_name}] best length: {length}")
	preview = route[:12]
	if len(route) > 12:
		print(f"[{file_name}] cycle preview: {' -> '.join(preview)} -> ... -> {preview[0]}")
	else:
		print(f"[{file_name}] best cycle: {format_cycle(route)}")

	stem = Path(file_name).stem
	save_cycle_plot(
		graph,
		route,
		out_dir / f"{stem}_cycle.png",
		f"{file_name}: Hamiltonian Cycle (Scale-Preserving MDS)",
		preserve_scale=True,
	)
	save_convergence_plot(history, out_dir / f"{stem}_convergence.png", f"{file_name}: SA Convergence")
	print(f"[{file_name}] visualization saved to: {out_dir}")


if __name__ == "__main__":
	run_demo()
	run_stp("berlin52.stp", seed=42)
	# run_stp("world666.stp", seed=42)

