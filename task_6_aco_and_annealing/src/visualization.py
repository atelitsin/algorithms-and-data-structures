from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from src.anneal import WeightedGraph


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc


def _circular_layout(nodes: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    n = len(nodes)
    coords: Dict[str, Tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2.0 * math.pi * i / n
        coords[node] = (math.cos(angle), math.sin(angle))
    return coords


def _distance_preserving_layout(graph: WeightedGraph) -> Dict[str, Tuple[float, float]]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for scale-preserving visualization. Install it with: pip install numpy"
        ) from exc

    n = graph.size
    dist = np.array(graph.weights, dtype=float)

    # For directed/sparse cases, build a finite symmetric surrogate distance matrix.
    dist = np.minimum(dist, dist.T)
    finite_mask = np.isfinite(dist) & (dist > 0)
    max_finite = float(np.max(dist[finite_mask])) if np.any(finite_mask) else 1.0
    dist[~np.isfinite(dist)] = max_finite * 1.25
    np.fill_diagonal(dist, 0.0)

    # Classical MDS via eigen-decomposition of double-centered squared distances.
    d2 = dist**2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    lam1 = max(float(eigenvalues[0]), 1e-12)
    lam2 = max(float(eigenvalues[1]) if n > 1 else 0.0, 1e-12)
    x = eigenvectors[:, 0] * math.sqrt(lam1)
    y = eigenvectors[:, 1] * math.sqrt(lam2)

    return {graph.nodes[i]: (float(x[i]), float(y[i])) for i in range(n)}


def save_cycle_plot(
    graph: WeightedGraph,
    route: Sequence[str],
    output_path: str | Path,
    title: str,
    preserve_scale: bool = False,
) -> None:
    plt = _require_matplotlib()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    coords = _distance_preserving_layout(graph) if preserve_scale else _circular_layout(graph.nodes)
    node_set = set(graph.nodes)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title(title)

    for node in graph.nodes:
        x, y = coords[node]
        ax.scatter(x, y, s=120, c="#f5f5f5", edgecolor="#1f2937", linewidth=1.2, zorder=3)

    # Skip labels for very large graphs to keep image readable.
    if graph.size <= 100:
        for node in graph.nodes:
            x, y = coords[node]
            ax.text(x, y + 0.04, node, fontsize=8, ha="center", va="center", color="#111827")

    if route:
        cycle = list(route) + [route[0]]
        for i in range(len(cycle) - 1):
            src = cycle[i]
            dst = cycle[i + 1]
            if src not in node_set or dst not in node_set:
                continue

            x1, y1 = coords[src]
            x2, y2 = coords[dst]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "color": "#2563eb", "lw": 1.1, "alpha": 0.8},
                zorder=2,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_convergence_plot(history: Iterable[float], output_path: str | Path, title: str) -> None:
    plt = _require_matplotlib()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    values = list(history)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(values, color="#dc2626", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Temperature level")
    ax.set_ylabel("Best cycle length")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
