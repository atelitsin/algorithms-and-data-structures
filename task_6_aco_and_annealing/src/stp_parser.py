from __future__ import annotations

from pathlib import Path

from src.graph import Graph


def parse_stp_file(file_path: str | Path) -> Graph:
    path = Path(file_path)
    if not path.exists() and not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        for candidate in (
            project_root / path,
            project_root / "data" / path,
        ):
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"STP file not found: {path}")

    node_count: int | None = None
    edges: list[tuple[int, int, float]] = []

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == "Nodes" and len(parts) >= 2:
                node_count = int(parts[1])
            elif parts[0] == "E" and len(parts) >= 4:
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                w = float(parts[3])
                edges.append((u, v, w))

    if node_count is None:
        raise ValueError(f"Could not read node count from STP file: {file_path}")

    nodes = [str(i + 1) for i in range(node_count)]
    inf = float("inf")
    weights = [[inf for _ in range(node_count)] for _ in range(node_count)]

    for i in range(node_count):
        weights[i][i] = 0.0

    for u, v, w in edges:
        weights[u][v] = w
        weights[v][u] = w

    return Graph(nodes, weights)
