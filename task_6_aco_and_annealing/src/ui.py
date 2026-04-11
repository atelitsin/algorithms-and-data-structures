import os
import math
import sys
import time
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .aco import AntColonyOptimization, AntColonyOptimizationInitial
from .annealing import BoltzmannAnnealing, SimulatedAnnealing
from .graph_utils import Graph


@dataclass
class RunResult:
    algorithm: str
    graph_name: str
    n_cities: int
    layout_points: list[tuple[float, float]]
    best_distance: float
    iterations: int
    elapsed: float
    best_tour: list[int]
    history: list[float]


class RouteVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_cities = 0
        self._layout_points: list[tuple[float, float]] = []
        self._best_tour: list[int] = []
        self._graph_name = ""
        self._best_distance = 0.0
        self.setMinimumHeight(300)
        self.setStyleSheet("border: 1px solid #c7c7c7; background: white;")

    def clear(self):
        self._n_cities = 0
        self._layout_points = []
        self._best_tour = []
        self._graph_name = ""
        self._best_distance = 0.0
        self.update()

    def set_route(self, graph_name: str, n_cities: int, layout_points: list[tuple[float, float]], best_tour: list[int], best_distance: float):
        self._graph_name = graph_name
        self._n_cities = n_cities
        self._layout_points = list(layout_points)
        self._best_tour = list(best_tour)
        self._best_distance = best_distance
        self.update()

    def _build_positions(self, rect):
        if self._n_cities <= 0 or not self._layout_points:
            return []

        xs = [point[0] for point in self._layout_points]
        ys = [point[1] for point in self._layout_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)

        available_width = max(rect.width(), 1)
        available_height = max(rect.height(), 1)
        scale = min(available_width / span_x, available_height / span_y) * 0.92

        scaled_width = span_x * scale
        scaled_height = span_y * scale
        offset_x = rect.left() + (available_width - scaled_width) / 2.0
        offset_y = rect.top() + (available_height - scaled_height) / 2.0

        positions = []
        for x_value, y_value in self._layout_points:
            x = offset_x + (x_value - min_x) * scale
            y = offset_y + (y_value - min_y) * scale
            positions.append(QPointF(x, y))

        return positions

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(24, 56, -24, -24)

        painter.setPen(QPen(QColor("#1f2937"), 1))
        painter.setFont(QFont("Helvetica Neue", 11))
        title = self._graph_name if self._graph_name else "Маршрут не загружен"
        header = f"{title}  |  длина: {self._best_distance:.2f}" if self._best_tour else title
        painter.drawText(16, 24, header)

        if self._n_cities <= 0:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Запустите алгоритм, чтобы увидеть маршрут")
            return

        positions = self._build_positions(rect)
        if not positions:
            return

        if self._best_tour and len(self._best_tour) == self._n_cities:
            painter.setPen(QPen(QColor("#2563eb"), 2))
            for index, current_city in enumerate(self._best_tour):
                next_city = self._best_tour[(index + 1) % len(self._best_tour)]
                painter.drawLine(positions[current_city], positions[next_city])

        painter.setFont(QFont("Helvetica Neue", 9))
        label_needed = self._n_cities <= 120

        for index, point in enumerate(positions):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#111827")))
            painter.drawEllipse(point, 4.5, 4.5)

            painter.setPen(QColor("#111827"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if label_needed:
                painter.drawText(QPointF(point.x() + 6, point.y() - 6), str(index + 1))


class HistoryPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[float] = []
        self._graph_name = ""
        self.setMinimumHeight(240)
        self.setStyleSheet("border: 1px solid #c7c7c7; background: white;")

    def clear(self):
        self._history = []
        self._graph_name = ""
        self.update()

    def set_history(self, graph_name: str, history: list[float]):
        self._graph_name = graph_name
        self._history = list(history)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#1f2937"), 1))
        painter.setFont(QFont("Helvetica Neue", 11))
        title = self._graph_name if self._graph_name else "История не загружена"
        painter.drawText(16, 24, "История лучшего решения: " + title)

        rect = self.rect().adjusted(48, 44, -20, -28)
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        painter.drawRect(rect)

        if not self._history:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Запустите алгоритм, чтобы увидеть график")
            return

        if len(self._history) == 1:
            painter.setPen(QPen(QColor("#2563eb"), 3))
            center_point = QPointF(rect.center())
            painter.drawEllipse(center_point, 3.5, 3.5)
            return

        min_value = min(self._history)
        max_value = max(self._history)
        if max_value - min_value < 1e-9:
            max_value = min_value + 1.0

        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        width = max(rect.width(), 1)
        height = max(rect.height(), 1)

        def map_point(index: int, value: float) -> QPointF:
            x_ratio = index / (len(self._history) - 1)
            y_ratio = (value - min_value) / (max_value - min_value)
            x = left + x_ratio * width
            y = bottom - y_ratio * height
            return QPointF(x, y)

        painter.setPen(QPen(QColor("#9ca3af"), 1))
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, top, left, bottom)

        painter.setPen(QPen(QColor("#2563eb"), 2))
        previous_point = map_point(0, self._history[0])
        for index, value in enumerate(self._history[1:], start=1):
            current_point = map_point(index, value)
            painter.drawLine(previous_point, current_point)
            previous_point = current_point

        painter.setBrush(QBrush(QColor("#2563eb")))
        painter.setPen(Qt.PenStyle.NoPen)
        for index, value in enumerate(self._history):
            point = map_point(index, value)
            painter.drawEllipse(point, 2.5, 2.5)

        painter.setPen(QColor("#374151"))
        painter.setFont(QFont("Helvetica Neue", 9))
        painter.drawText(rect.left(), rect.top() - 6, f"min: {min_value:.2f}")
        painter.drawText(rect.right() - 140, rect.top() - 6, f"max: {max_value:.2f}")


class AlgorithmWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, graph_name: str, algorithm_name: str, params: dict):
        super().__init__()
        self.graph_name = graph_name
        self.algorithm_name = algorithm_name
        self.params = params

    def _load_graph(self) -> Graph:
        if self.graph_name == "small":
            return Graph.create_small_graph()
        if self.graph_name == "berlin52":
            return Graph.load_from_stp("data/berlin52.stp")
        return Graph.load_from_stp("data/world666.stp")

    def run(self):
        try:
            graph = self._load_graph()
            start_time = time.time()

            if self.algorithm_name == "sa_basic":
                algo = SimulatedAnnealing(
                    graph,
                    initial_temp=self.params["initial_temp"],
                    cooling_rate=self.params["cooling_rate"],
                    max_iterations=self.params["max_iterations"],
                )
            elif self.algorithm_name == "sa_boltzmann":
                algo = BoltzmannAnnealing(
                    graph,
                    initial_temp=self.params["initial_temp"],
                    max_iterations=self.params["max_iterations"],
                )
            elif self.algorithm_name == "aco_basic":
                algo = AntColonyOptimization(
                    graph,
                    n_ants=self.params["n_ants"],
                    n_iterations=self.params["n_iterations"],
                    alpha=self.params["alpha"],
                    beta=self.params["beta"],
                    rho=self.params["rho"],
                    Q=self.params["Q"],
                )
            else:
                algo = AntColonyOptimizationInitial(
                    graph,
                    n_ants=self.params["n_ants"],
                    n_iterations=self.params["n_iterations"],
                    alpha=self.params["alpha"],
                    beta=self.params["beta"],
                    rho=self.params["rho"],
                    Q=self.params["Q"],
                )

            best_tour, best_distance, metrics = algo.solve()
            elapsed = time.time() - start_time

            result = RunResult(
                algorithm=metrics["algorithm"],
                graph_name=self.graph_name,
                n_cities=graph.n_cities,
                layout_points=[(float(x), float(y)) for x, y in graph.get_visualization_layout()],
                best_distance=float(best_distance),
                iterations=int(metrics["iterations"]),
                elapsed=float(elapsed),
                best_tour=list(best_tour),
                history=[float(value) for value in getattr(algo, "history", [])],
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class AlgorithmPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.graph_combo = QComboBox()
        self.graph_combo.addItems(["small", "berlin52", "world666"])

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["sa_basic", "sa_boltzmann", "aco_basic", "aco_initial"])

        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run_algorithm)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.route_view = RouteVisualizationWidget()
        self.history_view = HistoryPlotWidget()

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._build_sa_page())
        self.param_stack.addWidget(self._build_sa_boltzmann_page())
        self.param_stack.addWidget(self._build_aco_page())
        self.param_stack.addWidget(self._build_aco_page(initial=True))

        self.algorithm_combo.currentIndexChanged.connect(self.param_stack.setCurrentIndex)

        top_box = QGroupBox("Параметры запуска")
        top_layout = QGridLayout(top_box)
        top_layout.addWidget(QLabel("Граф"), 0, 0)
        top_layout.addWidget(self.graph_combo, 0, 1)
        top_layout.addWidget(QLabel("Алгоритм"), 1, 0)
        top_layout.addWidget(self.algorithm_combo, 1, 1)
        top_layout.addWidget(self.run_button, 2, 0, 1, 2)

        main_layout = QHBoxLayout(self)

        left_column = QVBoxLayout()
        left_column.addWidget(top_box)
        left_column.addWidget(self.param_stack)
        left_column.addWidget(QLabel("Результаты"))
        left_column.addWidget(self.log)

        right_column = QVBoxLayout()
        right_column.addWidget(self.route_view)
        right_column.addWidget(self.history_view)

        main_layout.addLayout(left_column, 2)
        main_layout.addLayout(right_column, 1)

    def _build_sa_page(self):
        page = QWidget()
        form = QFormLayout(page)

        self.sa_initial_temp = QDoubleSpinBox()
        self.sa_initial_temp.setRange(1.0, 100000.0)
        self.sa_initial_temp.setValue(1000.0)

        self.sa_cooling_rate = QDoubleSpinBox()
        self.sa_cooling_rate.setRange(0.01, 0.9999)
        self.sa_cooling_rate.setSingleStep(0.01)
        self.sa_cooling_rate.setDecimals(4)
        self.sa_cooling_rate.setValue(0.95)

        self.sa_max_iterations = QSpinBox()
        self.sa_max_iterations.setRange(1, 200000)
        self.sa_max_iterations.setValue(5000)

        form.addRow("Начальная температура", self.sa_initial_temp)
        form.addRow("Коэффициент охлаждения", self.sa_cooling_rate)
        form.addRow("Макс. итераций", self.sa_max_iterations)
        return page

    def _build_sa_boltzmann_page(self):
        page = QWidget()
        form = QFormLayout(page)

        self.b_initial_temp = QDoubleSpinBox()
        self.b_initial_temp.setRange(1.0, 100000.0)
        self.b_initial_temp.setValue(1000.0)

        self.b_max_iterations = QSpinBox()
        self.b_max_iterations.setRange(1, 200000)
        self.b_max_iterations.setValue(5000)

        form.addRow("Начальная температура", self.b_initial_temp)
        form.addRow("Макс. итераций", self.b_max_iterations)
        return page

    def _build_aco_page(self, initial=False):
        page = QWidget()
        form = QFormLayout(page)

        ants = QSpinBox()
        ants.setRange(1, 1000)
        ants.setValue(20)

        iterations = QSpinBox()
        iterations.setRange(1, 10000)
        iterations.setValue(50)

        alpha = QDoubleSpinBox()
        alpha.setRange(0.1, 10.0)
        alpha.setSingleStep(0.1)
        alpha.setValue(1.0)

        beta = QDoubleSpinBox()
        beta.setRange(0.1, 10.0)
        beta.setSingleStep(0.1)
        beta.setValue(2.0)

        rho = QDoubleSpinBox()
        rho.setRange(0.01, 0.99)
        rho.setSingleStep(0.01)
        rho.setDecimals(2)
        rho.setValue(0.1)

        q_value = QDoubleSpinBox()
        q_value.setRange(1.0, 10000.0)
        q_value.setValue(100.0)

        form.addRow("Количество муравьев", ants)
        form.addRow("Количество итераций", iterations)
        form.addRow("alpha", alpha)
        form.addRow("beta", beta)
        form.addRow("rho", rho)
        form.addRow("Q", q_value)

        if initial:
            self.aco_initial_ants = ants
            self.aco_initial_iterations = iterations
            self.aco_initial_alpha = alpha
            self.aco_initial_beta = beta
            self.aco_initial_rho = rho
            self.aco_initial_q = q_value
        else:
            self.aco_basic_ants = ants
            self.aco_basic_iterations = iterations
            self.aco_basic_alpha = alpha
            self.aco_basic_beta = beta
            self.aco_basic_rho = rho
            self.aco_basic_q = q_value

        return page

    def _get_params(self):
        algo = self.algorithm_combo.currentText()
        if algo == "sa_basic":
            return {
                "initial_temp": self.sa_initial_temp.value(),
                "cooling_rate": self.sa_cooling_rate.value(),
                "max_iterations": self.sa_max_iterations.value(),
            }
        if algo == "sa_boltzmann":
            return {
                "initial_temp": self.b_initial_temp.value(),
                "max_iterations": self.b_max_iterations.value(),
            }
        if algo == "aco_basic":
            return {
                "n_ants": self.aco_basic_ants.value(),
                "n_iterations": self.aco_basic_iterations.value(),
                "alpha": self.aco_basic_alpha.value(),
                "beta": self.aco_basic_beta.value(),
                "rho": self.aco_basic_rho.value(),
                "Q": self.aco_basic_q.value(),
            }
        return {
            "n_ants": self.aco_initial_ants.value(),
            "n_iterations": self.aco_initial_iterations.value(),
            "alpha": self.aco_initial_alpha.value(),
            "beta": self.aco_initial_beta.value(),
            "rho": self.aco_initial_rho.value(),
            "Q": self.aco_initial_q.value(),
        }

    def run_algorithm(self):
        self.run_button.setEnabled(False)
        self.route_view.clear()
        self.history_view.clear()
        self.log.appendPlainText("Запуск...\n")

        self.worker = AlgorithmWorker(
            self.graph_combo.currentText(),
            self.algorithm_combo.currentText(),
            self._get_params(),
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, result):
        self.run_button.setEnabled(True)
        self.route_view.set_route(result.graph_name, result.n_cities, result.layout_points, result.best_tour, result.best_distance)
        self.history_view.set_history(result.graph_name, result.history)
        self.log.appendPlainText(
            f"Алгоритм: {result.algorithm}\n"
            f"Граф: {result.graph_name}\n"
            f"Лучшее расстояние: {result.best_distance}\n"
            f"Итерации: {result.iterations}\n"
            f"Время: {result.elapsed:.4f} сек\n"
            f"Маршрут: {result.best_tour}\n"
            "----------------------------------------\n"
        )

    def _on_failed(self, message):
        self.run_button.setEnabled(True)
        self.route_view.clear()
        self.history_view.clear()
        QMessageBox.critical(self, "Ошибка", message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP: SA и ACO")
        self.resize(1400, 800)
        self.setCentralWidget(AlgorithmPage(self))


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())