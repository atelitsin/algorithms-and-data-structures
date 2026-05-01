from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .algorithms import MSTResult, run_mst_algorithm
from .graph_utils import generate_connected_random_graph


ALGORITHMS: list[tuple[str, str]] = [
    ("kruskal", "Kruskal"),
    ("prim_binary", "Prim (binary heap)"),
    ("prim_fibonacci", "Prim (Fibonacci heap)"),
    ("boruvka", "Boruvka"),
]

DENSITY_PROFILES: list[tuple[str, float]] = [
    ("0.05", 0.05),
    ("0.20", 0.20),
    ("0.60", 0.60),
    ("0.9", 0.90),
]

DEFAULT_SIZES: list[int] = [10, 100, 250, 500, 1000, 2000]


@dataclass
class BenchmarkRow:
    algorithm_code: str
    algorithm_name: str
    n_vertices: int
    profile_name: str
    density: float
    run_id: int
    mst_weight: int
    elapsed_ms: float
    operations: int


@dataclass
class BenchmarkAggregate:
    algorithm_name: str
    n_vertices: int
    profile_name: str
    density: float
    avg_elapsed_ms: float
    avg_operations: float


def _run_all_algorithms_once(n_vertices: int, density: float, seed: int, max_weight: int) -> list[MSTResult]:
    graph = generate_connected_random_graph(
        n_vertices=n_vertices,
        density=density,
        max_weight=max_weight,
        seed=seed,
    )
    return [run_mst_algorithm(code, graph) for code, _ in ALGORITHMS]


def run_benchmarks(
    sizes: Iterable[int] = DEFAULT_SIZES,
    density_profiles: Iterable[tuple[str, float]] = DENSITY_PROFILES,
    repeats: int = 3,
    max_weight: int = 100,
    seed_base: int = 42,
) -> tuple[list[BenchmarkRow], list[BenchmarkAggregate]]:
    rows: list[BenchmarkRow] = []

    for n_vertices in sizes:
        for profile_name, density in density_profiles:
            for run_id in range(repeats):
                seed = seed_base + n_vertices * 1000 + int(density * 1000) * 10 + run_id
                results = _run_all_algorithms_once(
                    n_vertices=n_vertices,
                    density=density,
                    seed=seed,
                    max_weight=max_weight,
                )
                for result in results:
                    rows.append(
                        BenchmarkRow(
                            algorithm_code=_find_algorithm_code(result.algorithm),
                            algorithm_name=result.algorithm,
                            n_vertices=n_vertices,
                            profile_name=profile_name,
                            density=density,
                            run_id=run_id,
                            mst_weight=result.total_weight,
                            elapsed_ms=result.elapsed_ms,
                            operations=result.operations,
                        )
                    )

    aggregates = aggregate_rows(rows)
    return rows, aggregates


def _find_algorithm_code(algorithm_name: str) -> str:
    for code, name in ALGORITHMS:
        if name == algorithm_name:
            return code
    return algorithm_name.lower().replace(" ", "_")


def aggregate_rows(rows: list[BenchmarkRow]) -> list[BenchmarkAggregate]:
    grouped: dict[tuple[str, int, str, float], list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        key = (row.algorithm_name, row.n_vertices, row.profile_name, row.density)
        grouped[key].append(row)

    aggregated: list[BenchmarkAggregate] = []
    for (algorithm_name, n_vertices, profile_name, density), values in grouped.items():
        aggregated.append(
            BenchmarkAggregate(
                algorithm_name=algorithm_name,
                n_vertices=n_vertices,
                profile_name=profile_name,
                density=density,
                avg_elapsed_ms=mean(value.elapsed_ms for value in values),
                avg_operations=mean(value.operations for value in values),
            )
        )

    aggregated.sort(key=lambda item: (item.n_vertices, item.profile_name, item.algorithm_name))
    return aggregated


def save_rows_to_csv(rows: list[BenchmarkRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "algorithm_code",
                "algorithm_name",
                "n_vertices",
                "profile_name",
                "density",
                "run_id",
                "mst_weight",
                "elapsed_ms",
                "operations",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.algorithm_code,
                    row.algorithm_name,
                    row.n_vertices,
                    row.profile_name,
                    row.density,
                    row.run_id,
                    row.mst_weight,
                    f"{row.elapsed_ms:.6f}",
                    row.operations,
                ]
            )


def save_aggregates_to_csv(rows: list[BenchmarkAggregate], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "algorithm_name",
                "n_vertices",
                "profile_name",
                "density",
                "avg_elapsed_ms",
                "avg_operations",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.algorithm_name,
                    row.n_vertices,
                    row.profile_name,
                    row.density,
                    f"{row.avg_elapsed_ms:.6f}",
                    f"{row.avg_operations:.2f}",
                ]
            )


def save_plots(aggregates: list[BenchmarkAggregate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_profile: dict[str, list[BenchmarkAggregate]] = defaultdict(list)
    for row in aggregates:
        by_profile[row.profile_name].append(row)

    for profile_name, rows in by_profile.items():
        file_safe_profile_name = profile_name.replace(".", "_")
        plt.figure(figsize=(9, 5))
        for _, algorithm_name in ALGORITHMS:
            filtered = [item for item in rows if item.algorithm_name == algorithm_name]
            filtered.sort(key=lambda item: item.n_vertices)
            if not filtered:
                continue
            xs = [item.n_vertices for item in filtered]
            ys = [item.avg_elapsed_ms for item in filtered]
            plt.plot(xs, ys, marker="o", linewidth=2, label=algorithm_name)

        plt.title(f"MST runtime by size ({profile_name})")
        plt.xlabel("Vertices")
        plt.ylabel("Average runtime, ms")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"runtime_{file_safe_profile_name}.png", dpi=160)
        plt.close()


def run_and_save_default_benchmarks() -> None:
    rows, aggregates = run_benchmarks()
    out_dir = Path("data") / "benchmarks"
    save_rows_to_csv(rows, out_dir / "raw_runs.csv")
    save_aggregates_to_csv(aggregates, out_dir / "aggregated.csv")
    save_plots(aggregates, out_dir)

    print("Saved benchmark data:")
    print(f"- {out_dir / 'raw_runs.csv'}")
    print(f"- {out_dir / 'aggregated.csv'}")
    for profile_name, _ in DENSITY_PROFILES:
        print(f"- {out_dir / f'runtime_{profile_name.replace('.', '_')}.png'}")
