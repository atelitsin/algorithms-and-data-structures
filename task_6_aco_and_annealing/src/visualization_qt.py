from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication

from src.graph import Graph


def _finite_symmetric_distances(graph: Graph) -> np.ndarray:
    n = len(graph.nodes)
    distances = np.array(graph.weights, dtype=float)

    # Use the smallest finite distance in either direction for asymmetric graphs.
    reverse = distances.T
    combined = np.minimum(distances, reverse)

    finite_mask = np.isfinite(combined) & (combined > 0)
    if np.any(finite_mask):
        max_finite = float(np.max(combined[finite_mask]))
    else:
        max_finite = 1.0

    combined[~np.isfinite(combined)] = max_finite * 1.25
    np.fill_diagonal(combined, 0.0)
    return combined


def classical_mds_coordinates(graph: Graph) -> np.ndarray:
    distances = _finite_symmetric_distances(graph)
    n = distances.shape[0]

    d2 = distances**2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    lambda_1 = max(float(eigenvalues[0]), 1e-12)
    lambda_2 = max(float(eigenvalues[1]) if n > 1 else 0.0, 1e-12)

    x = eigenvectors[:, 0] * math.sqrt(lambda_1)
    y = eigenvectors[:, 1] * math.sqrt(lambda_2)
    return np.column_stack((x, y))


def _normalize_points(points: np.ndarray, width: int, height: int, margin: int) -> list[QPointF]:
    xs = points[:, 0]
    ys = points[:, 1]

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    drawable_w = max(width - 2 * margin, 1)
    drawable_h = max(height - 2 * margin, 1)
    scale = min(drawable_w / span_x, drawable_h / span_y)

    offset_x = margin + (drawable_w - span_x * scale) / 2
    offset_y = margin + (drawable_h - span_y * scale) / 2

    mapped: list[QPointF] = []
    for x, y in points:
        px = offset_x + (float(x) - min_x) * scale
        py = offset_y + (max_y - float(y)) * scale
        mapped.append(QPointF(px, py))
    return mapped


def render_graph_image(
    graph: Graph,
    route: Iterable[int] | None = None,
    width: int = 1100,
    height: int = 800,
    margin: int = 60,
) -> QImage:
    points = classical_mds_coordinates(graph)
    mapped_points = _normalize_points(points, width, height, margin)

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    route_list = list(route) if route is not None else []
    route_edges = set()
    if route_list:
        for i in range(len(route_list)):
            u = route_list[i]
            v = route_list[(i + 1) % len(route_list)]
            route_edges.add((u, v))
            route_edges.add((v, u))

    # Draw edges first.
    edge_pen = QPen(QColor("#cbd5e1"))
    edge_pen.setWidthF(1.0)
    route_pen = QPen(QColor("#2563eb"))
    route_pen.setWidthF(2.6)
    draw_all_edges = len(graph.nodes) <= 80

    n = len(graph.nodes)
    if draw_all_edges:
        for i in range(n):
            for j in range(i + 1, n):
                w_ij = graph.weights[i][j]
                w_ji = graph.weights[j][i]
                if math.isinf(w_ij) and math.isinf(w_ji):
                    continue

                pen = route_pen if (i, j) in route_edges else edge_pen
                painter.setPen(pen)
                painter.drawLine(mapped_points[i], mapped_points[j])
    else:
        for i in range(len(route_list)):
            u = route_list[i]
            v = route_list[(i + 1) % len(route_list)]
            painter.setPen(route_pen)
            painter.drawLine(mapped_points[u], mapped_points[v])

    # Draw vertices.
    if len(graph.nodes) <= 50:
        label_size = 9
    elif len(graph.nodes) <= 200:
        label_size = 6
    else:
        label_size = 5

    painter.setFont(QFont("Arial", label_size))
    for index, point in enumerate(mapped_points):
        painter.setPen(QPen(QColor("#1f2937"), 1.5))
        painter.setBrush(QColor("#f8fafc"))
        painter.drawEllipse(point, 10, 10)
        painter.drawText(QPointF(point.x() + 12, point.y() - 12), graph.nodes[index])

    painter.end()
    return image


def save_graph_image(graph: Graph, output_path: str | Path, route: Iterable[int] | None = None) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_graph_image(graph, route=route)
    image.save(str(output))
    return output


def graph_pixmap(graph: Graph, route: Iterable[int] | None = None) -> QPixmap:
    return QPixmap.fromImage(render_graph_image(graph, route=route))
