from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from PyQt6.QtCore import QPointF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .algorithms import MSTResult, run_mst_algorithm
from .benchmarking import (
    ALGORITHMS,
    DENSITY_PROFILES,
    DEFAULT_SIZES,
    BenchmarkAggregate,
    run_benchmarks,
    save_aggregates_to_csv,
    save_plots,
    save_rows_to_csv,
)
from .graph_utils import Edge, WeightedGraph, generate_connected_random_graph


ALGORITHM_NAME_BY_CODE = {code: name for code, name in ALGORITHMS}
DENSITY_BY_NAME = {name: density for name, density in DENSITY_PROFILES}


@dataclass
class CompareResult:
    graph: WeightedGraph
    results: list[MSTResult]


class AlgorithmWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, graph: WeightedGraph, algorithm_codes: list[str]):
        super().__init__()
        self.graph = graph
        self.algorithm_codes = algorithm_codes

    def run(self) -> None:
        try:
            results = [run_mst_algorithm(code, self.graph) for code in self.algorithm_codes]
            self.finished.emit(CompareResult(graph=self.graph, results=results))
        except Exception as exc:
            self.failed.emit(str(exc))


class BenchmarkWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, max_weight: int, seed_base: int):
        super().__init__()
        self.max_weight = max_weight
        self.seed_base = seed_base

    def run(self) -> None:
        try:
            rows, aggregates = run_benchmarks(
                sizes=DEFAULT_SIZES,
                density_profiles=DENSITY_PROFILES,
                repeats=3,
                max_weight=self.max_weight,
                seed_base=self.seed_base,
            )
            out_dir = Path("data") / "benchmarks"
            save_rows_to_csv(rows, out_dir / "raw_runs.csv")
            save_aggregates_to_csv(aggregates, out_dir / "aggregated.csv")
            save_plots(aggregates, out_dir)
            self.finished.emit((rows, aggregates, out_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


class GraphCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph: WeightedGraph | None = None
        self.mst_edges: list[Edge] = []
        self.setMinimumHeight(340)
        self.setStyleSheet("border: 1px solid #c7c7c7; background: #ffffff;")

    def set_graph(self, graph: WeightedGraph, mst_edges: list[Edge] | None = None) -> None:
        self.graph = graph
        self.mst_edges = list(mst_edges or [])
        self.update()

    def clear(self) -> None:
        self.graph = None
        self.mst_edges = []
        self.update()

    def _to_screen(self, rect, x_value: float, y_value: float) -> QPointF:
        x = rect.left() + (x_value + 1.25) / 2.5 * rect.width()
        y = rect.top() + (y_value + 1.25) / 2.5 * rect.height()
        return QPointF(x, y)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(24, 40, -24, -24)
        painter.setPen(QPen(QColor("#1f2937"), 1))
        painter.setFont(QFont("Helvetica Neue", 11))
        painter.drawText(14, 22, "Граф и выделенное минимальное остовное дерево")

        if self.graph is None:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Сгенерируйте граф")
            return

        points = [self._to_screen(rect, x, y) for x, y in self.graph.positions]

        painter.setPen(QPen(QColor("#d1d5db"), 1))
        for edge in self.graph.edges:
            painter.drawLine(points[edge.u], points[edge.v])

        mst_set = {(min(edge.u, edge.v), max(edge.u, edge.v), edge.weight) for edge in self.mst_edges}
        if mst_set:
            painter.setPen(QPen(QColor("#dc2626"), 2))
            for edge in self.graph.edges:
                key = (min(edge.u, edge.v), max(edge.u, edge.v), edge.weight)
                if key in mst_set:
                    painter.drawLine(points[edge.u], points[edge.v])

        font = QFont("Helvetica Neue", 8)
        painter.setFont(font)
        show_labels = self.graph.n_vertices <= 120
        for idx, point in enumerate(points):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#111827"))
            painter.drawEllipse(point, 3.8, 3.8)
            if show_labels:
                painter.setPen(QColor("#111827"))
                painter.drawText(QPointF(point.x() + 5, point.y() - 5), str(idx))


class BenchPlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aggregates: list[BenchmarkAggregate] = []
        self.profile_name = "sparse"
        self.setMinimumHeight(260)
        self.setStyleSheet("border: 1px solid #c7c7c7; background: #ffffff;")

    def set_data(self, aggregates: list[BenchmarkAggregate], profile_name: str) -> None:
        self.aggregates = list(aggregates)
        self.profile_name = profile_name
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#1f2937"), 1))
        painter.setFont(QFont("Helvetica Neue", 11))
        painter.drawText(14, 22, f"Среднее время по размерам ({self.profile_name})")

        rect = self.rect().adjusted(48, 36, -24, -34)
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        painter.drawRect(rect)

        rows = [row for row in self.aggregates if row.profile_name == self.profile_name]
        if not rows:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Нет данных бенчмарка")
            return

        xs = sorted({row.n_vertices for row in rows})
        ys = [row.avg_elapsed_ms for row in rows]
        y_min, y_max = min(ys), max(ys)
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0

        palette = [QColor("#1d4ed8"), QColor("#dc2626"), QColor("#059669"), QColor("#7c3aed")]

        for algorithm_index, (_, algorithm_name) in enumerate(ALGORITHMS):
            line_rows = [row for row in rows if row.algorithm_name == algorithm_name]
            line_rows.sort(key=lambda item: item.n_vertices)
            if not line_rows:
                continue

            color = palette[algorithm_index % len(palette)]
            painter.setPen(QPen(color, 2))

            last_point: QPointF | None = None
            for row in line_rows:
                x_ratio = 0.0 if len(xs) == 1 else xs.index(row.n_vertices) / (len(xs) - 1)
                y_ratio = (row.avg_elapsed_ms - y_min) / (y_max - y_min)
                x = rect.left() + x_ratio * rect.width()
                y = rect.bottom() - y_ratio * rect.height()
                point = QPointF(x, y)
                if last_point is not None:
                    painter.drawLine(last_point, point)
                painter.setBrush(color)
                painter.drawEllipse(point, 2.8, 2.8)
                last_point = point

            painter.setPen(color)
            painter.drawText(rect.right() - 190, rect.top() + 16 + 16 * algorithm_index, algorithm_name)

        painter.setPen(QColor("#374151"))
        painter.setFont(QFont("Helvetica Neue", 9))
        painter.drawText(rect.left(), rect.top() - 6, f"min={y_min:.2f} ms")
        painter.drawText(rect.right() - 120, rect.top() - 6, f"max={y_max:.2f} ms")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimization")
        self.resize(1280, 780)

        self.current_graph: WeightedGraph | None = None
        self.current_aggregates: list[BenchmarkAggregate] = []
        self.algorithm_worker: AlgorithmWorker | None = None
        self.benchmark_worker: BenchmarkWorker | None = None

        self.n_vertices_spin = QSpinBox()
        # Allow large graphs by default; remove previous 500-vertex cap.
        self.n_vertices_spin.setRange(1, 1_000_000)
        self.n_vertices_spin.setSingleStep(10)
        self.n_vertices_spin.setValue(100)

        self.density_profile_combo = QComboBox()
        self.density_profile_combo.addItems([name for name, _ in DENSITY_PROFILES] + ["custom"])

        self.custom_density_spin = QDoubleSpinBox()
        self.custom_density_spin.setRange(0.01, 1.00)
        self.custom_density_spin.setDecimals(2)
        self.custom_density_spin.setSingleStep(0.01)
        self.custom_density_spin.setValue(0.20)

        self.max_weight_spin = QSpinBox()
        self.max_weight_spin.setRange(10, 1000)
        self.max_weight_spin.setValue(100)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([f"{name} [{code}]" for code, name in ALGORITHMS])

        self.profile_plot_combo = QComboBox()
        self.profile_plot_combo.addItems([name for name, _ in DENSITY_PROFILES])
        self.profile_plot_combo.currentTextChanged.connect(self._refresh_plot)

        self.generate_button = QPushButton("Сгенерировать граф")
        self.generate_button.clicked.connect(self.generate_graph)

        self.run_button = QPushButton("Запустить выбранный алгоритм")
        self.run_button.clicked.connect(self.run_selected_algorithm)

        self.compare_button = QPushButton("Сравнить все алгоритмы на текущем графе")
        self.compare_button.clicked.connect(self.run_all_algorithms)

        self.benchmark_button = QPushButton("Запустить полный бенчмарк")
        self.benchmark_button.clicked.connect(self.run_full_benchmark)

        self.graph_canvas = GraphCanvas()
        self.bench_canvas = BenchPlotCanvas()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self._build_layout()
        self._append_log("Интерфейс готов. Сгенерируйте граф и запустите алгоритм.")

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        controls_box = QGroupBox("Параметры")
        controls_form = QFormLayout(controls_box)
        controls_form.addRow("Количество вершин", self.n_vertices_spin)
        controls_form.addRow("Плотность графа", self.density_profile_combo)
        controls_form.addRow("Плотность вручную", self.custom_density_spin)
        controls_form.addRow("Максимальный вес ребра", self.max_weight_spin)
        controls_form.addRow("Seed", self.seed_spin)
        controls_form.addRow("Алгоритм", self.algorithm_combo)

        buttons_box = QGroupBox("Действия")
        buttons_layout = QVBoxLayout(buttons_box)
        buttons_layout.addWidget(self.generate_button)
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.compare_button)
        buttons_layout.addWidget(self.benchmark_button)

        left_column = QVBoxLayout()
        left_column.addWidget(controls_box)
        left_column.addWidget(buttons_box)
        left_column.addWidget(QLabel("Логи"))
        left_column.addWidget(self.log)

        right_column = QVBoxLayout()
        right_column.addWidget(self.graph_canvas, 2)

        bench_header = QHBoxLayout()
        bench_header.addWidget(QLabel("Плотность для графика"))
        bench_header.addWidget(self.profile_plot_combo)
        bench_header.addStretch(1)
        right_column.addLayout(bench_header)
        right_column.addWidget(self.bench_canvas, 1)

        top_layout = QGridLayout(root)
        top_layout.addLayout(left_column, 0, 0)
        top_layout.addLayout(right_column, 0, 1)
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 2)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _resolve_density(self) -> float:
        profile = self.density_profile_combo.currentText()
        if profile == "custom":
            return float(self.custom_density_spin.value())
        return DENSITY_BY_NAME[profile]

    def _selected_algorithm_code(self) -> str:
        text = self.algorithm_combo.currentText()
        start = text.rfind("[")
        end = text.rfind("]")
        return text[start + 1 : end] if start != -1 and end != -1 else ALGORITHMS[0][0]

    def _set_busy(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.compare_button.setEnabled(not busy)
        self.benchmark_button.setEnabled(not busy)

    def generate_graph(self) -> None:
        n_vertices = int(self.n_vertices_spin.value())
        density = self._resolve_density()
        max_weight = int(self.max_weight_spin.value())
        seed = int(self.seed_spin.value())

        try:
            self.current_graph = generate_connected_random_graph(
                n_vertices=n_vertices,
                density=density,
                max_weight=max_weight,
                seed=seed,
            )
            self.graph_canvas.set_graph(self.current_graph)
            self._append_log(
                f"Сгенерирован граф: |V|={n_vertices}, |E|={self.current_graph.edge_count}, density={self.current_graph.density:.3f}, seed={seed}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка генерации", str(exc))

    def _ensure_graph(self) -> WeightedGraph | None:
        if self.current_graph is None:
            QMessageBox.information(self, "Нет графа", "Сначала сгенерируйте граф.")
            return None
        return self.current_graph

    def run_selected_algorithm(self) -> None:
        graph = self._ensure_graph()
        if graph is None:
            return

        code = self._selected_algorithm_code()
        self._run_algorithms([code])

    def run_all_algorithms(self) -> None:
        graph = self._ensure_graph()
        if graph is None:
            return
        self._run_algorithms([code for code, _ in ALGORITHMS])

    def _run_algorithms(self, codes: list[str]) -> None:
        if self.current_graph is None:
            return

        self._set_busy(True)
        self._append_log("Запуск алгоритмов...")

        self.algorithm_worker = AlgorithmWorker(self.current_graph, codes)
        self.algorithm_worker.finished.connect(self._on_algorithms_finished)
        self.algorithm_worker.failed.connect(self._on_worker_failed)
        self.algorithm_worker.start()

    def _on_algorithms_finished(self, payload: CompareResult) -> None:
        self._set_busy(False)
        if not payload.results:
            self._append_log("Алгоритмы не вернули результаты.")
            return

        baseline_weight = payload.results[0].total_weight
        for result in payload.results:
            same_weight = "OK" if result.total_weight == baseline_weight else "MISMATCH"
            self._append_log(
                f"{result.algorithm}: weight={result.total_weight}, edges={len(result.mst_edges)}, time={result.elapsed_ms:.3f} ms, ops={result.operations}, check={same_weight}"
            )

        best = min(payload.results, key=lambda item: item.elapsed_ms)
        self._append_log(f"Быстрейший алгоритм на текущем графе: {best.algorithm} ({best.elapsed_ms:.3f} ms)")
        self.graph_canvas.set_graph(payload.graph, best.mst_edges)

    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ошибка", message)

    def run_full_benchmark(self) -> None:
        self._set_busy(True)
        self._append_log("Запуск полного бенчмарка. Это может занять время...")

        max_weight = int(self.max_weight_spin.value())
        seed_base = int(self.seed_spin.value())
        self.benchmark_worker = BenchmarkWorker(max_weight=max_weight, seed_base=seed_base)
        self.benchmark_worker.finished.connect(self._on_benchmark_finished)
        self.benchmark_worker.failed.connect(self._on_worker_failed)
        self.benchmark_worker.start()

    def _on_benchmark_finished(self, payload) -> None:
        self._set_busy(False)
        _, aggregates, out_dir = payload
        self.current_aggregates = aggregates
        self._append_log("Бенчмарк завершен. Артефакты:")
        self._append_log(f"- {out_dir / 'raw_runs.csv'}")
        self._append_log(f"- {out_dir / 'aggregated.csv'}")
        for profile, _ in DENSITY_PROFILES:
            self._append_log(f"- {out_dir / f'runtime_{profile}.png'}")
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        profile = self.profile_plot_combo.currentText()
        self.bench_canvas.set_data(self.current_aggregates, profile)


def run_app() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
