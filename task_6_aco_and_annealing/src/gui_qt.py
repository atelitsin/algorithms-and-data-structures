from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QFrame,
    QSizePolicy,
)

from src.graph import Graph
from src.sa import AnnealingConfig, simulated_annealing
from src.stp_parser import parse_stp_file
from src.visualization_qt import graph_pixmap, save_graph_image


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


class AnnealingWorker(QObject):
    finished = pyqtSignal(list, float, int)
    failed = pyqtSignal(str)

    def __init__(self, graph: Graph, config: AnnealingConfig) -> None:
        super().__init__()
        self.graph = graph
        self.config = config

    def run(self) -> None:
        try:
            best_route, best_cost, history = simulated_annealing(self.graph, self.config)
            self.finished.emit(best_route, best_cost, len(history))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnnealingWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simulated Annealing TSP")
        self.resize(980, 720)

        self.worker_thread: QThread | None = None
        self.worker: AnnealingWorker | None = None

        self.instance_combo = QComboBox()
        self.instance_combo.addItems(["demo", "berlin52.stp", "world666.stp", "custom"])
        self.instance_combo.currentTextChanged.connect(self._toggle_custom_path)

        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setPlaceholderText("Path to custom .stp file")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_file)

        self.initial_temp_spin = self._make_double_spin(0.1, 1_000_000.0, 100.0, 1)
        self.final_temp_spin = self._make_double_spin(0.0001, 1_000_000.0, 0.1, 4)
        self.cooling_spin = self._make_double_spin(0.5, 0.999999, 0.99, 3)
        self.iterations_spin = self._make_int_spin(1, 1_000_000, 50)
        self.seed_spin = self._make_int_spin(-1, 1_000_000_000, 42)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setMinimum(-1)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)

        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self._on_visualize)

        self.save_image_button = QPushButton("Save image")
        self.save_image_button.clicked.connect(self._on_save_image)

        self.clear_button = QPushButton("Clear output")
        self.clear_button.clicked.connect(self._clear_output)

        self.status_label = QLabel("Ready")
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)

        self.preview_label = QLabel("Visualization preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setStyleSheet("background: white; border: 1px solid #d1d5db;")

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_label)

        self.current_graph: Graph | None = None
        self.current_route: list[int] | None = None
        self.current_pixmap: QPixmap | None = None

        self._build_ui()
        self._toggle_custom_path(self.instance_combo.currentText())

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(12)

        header = QLabel("Simulated Annealing TSP")
        header.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(header)

        top = QGridLayout()
        root.addLayout(top)

        instance_box = QGroupBox("Instance")
        instance_layout = QVBoxLayout(instance_box)
        instance_layout.addWidget(self.instance_combo)

        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_path_edit, 1)
        custom_row.addWidget(self.browse_button)
        instance_layout.addLayout(custom_row)

        params_box = QGroupBox("Parameters")
        form = QFormLayout(params_box)
        form.addRow("Initial temperature", self.initial_temp_spin)
        form.addRow("Final temperature", self.final_temp_spin)
        form.addRow("Cooling rate", self.cooling_spin)
        form.addRow("Iterations / temperature", self.iterations_spin)
        form.addRow("Seed", self.seed_spin)

        top.addWidget(instance_box, 0, 0)
        top.addWidget(params_box, 0, 1)

        actions = QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.visualize_button)
        actions.addWidget(self.save_image_button)
        actions.addWidget(self.clear_button)
        actions.addStretch(1)
        actions.addWidget(self.status_label)
        root.addLayout(actions)

        root.addWidget(self.preview_scroll, 2)
        root.addWidget(self.output, 1)
        self.setCentralWidget(central)

    def _make_double_spin(self, minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(0.01)
        return spin

    def _make_int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _toggle_custom_path(self, choice: str) -> None:
        is_custom = choice == "custom"
        self.custom_path_edit.setEnabled(is_custom)
        self.browse_button.setEnabled(is_custom)

    def _browse_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select STP file",
            str(Path.cwd()),
            "STP files (*.stp);;All files (*)",
        )
        if file_name:
            self.custom_path_edit.setText(file_name)
            self.instance_combo.setCurrentText("custom")

    def _clear_output(self) -> None:
        self.output.clear()

    def _update_preview(self) -> None:
        if self.current_graph is None:
            return

        self.current_pixmap = graph_pixmap(self.current_graph, route=self.current_route)
        self.preview_label.setPixmap(
            self.current_pixmap.scaled(
                self.preview_scroll.viewport().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_visualize(self) -> None:
        try:
            self.current_graph = self._resolve_graph()
            self.current_route = None
            self._update_preview()
            self.status_label.setText("Preview ready")
            self._append_output("Visualization preview generated.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Visualization error", str(exc))

    def _on_save_image(self) -> None:
        if self.current_graph is None:
            try:
                self.current_graph = self._resolve_graph()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Save image", str(exc))
                return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save graph image",
            str(Path.cwd() / "graph_visualization.png"),
            "PNG files (*.png);;All files (*)",
        )
        if not output_path:
            return

        save_graph_image(self.current_graph, output_path, route=self.current_route)
        self._append_output(f"Saved image to: {output_path}")
        self.status_label.setText("Image saved")

    def _append_output(self, text: str) -> None:
        self.output.appendPlainText(text)

    def _resolve_graph(self) -> Graph:
        choice = self.instance_combo.currentText()
        if choice == "demo":
            return build_demo_graph()
        if choice == "custom":
            custom_path = self.custom_path_edit.text().strip()
            if not custom_path:
                raise ValueError("Choose a custom STP file path.")
            return parse_stp_file(custom_path)
        return parse_stp_file(choice)

    def _read_config(self) -> AnnealingConfig:
        seed_value = self.seed_spin.value()
        return AnnealingConfig(
            initial_temperature=float(self.initial_temp_spin.value()),
            final_temperature=float(self.final_temp_spin.value()),
            cooling_rate=float(self.cooling_spin.value()),
            iterations_per_temperature=int(self.iterations_spin.value()),
            seed=None if seed_value < 0 else seed_value,
        )

    def _on_run(self) -> None:
        try:
            graph = self._resolve_graph()
            config = self._read_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.run_button.setEnabled(False)
        self.visualize_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.status_label.setText("Running...")
        self._append_output("--- Run started ---")
        self._append_output(f"Config: {asdict(config)}")

        self.worker_thread = QThread(self)
        self.worker = AnnealingWorker(graph, config)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_run_finished)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        self.run_button.setEnabled(True)
        self.visualize_button.setEnabled(True)
        self.save_image_button.setEnabled(True)

    def _on_run_finished(self, route: list[int], best_cost: float, history_len: int) -> None:
        if self.worker is not None:
            self.current_graph = self.worker.graph
        self.current_route = route
        self._update_preview()

        graph = self.worker.graph if self.worker is not None else None
        if graph is not None:
            route_names = route_to_names(graph, route)
        else:
            route_names = [str(index) for index in route]
        self._append_output(f"Best route: {route_names}")
        self._append_output(f"Best cost: {best_cost}")
        self._append_output(f"History points: {history_len}")
        self._append_output("--- Run finished ---")
        self.status_label.setText("Done")
        self.run_button.setEnabled(True)
        self.visualize_button.setEnabled(True)
        self.save_image_button.setEnabled(True)

    def _on_run_failed(self, error_text: str) -> None:
        self._append_output(f"Error: {error_text}")
        self._append_output("--- Run failed ---")
        self.status_label.setText("Failed")
        self.run_button.setEnabled(True)


def launch_gui() -> None:
    app = QApplication([])
    window = AnnealingWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_gui()
