import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path


class Graph:
    """Класс для представления взвешенного полного графа."""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Args:
            distance_matrix: Матрица расстояний размером (n, n)
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
    
    @staticmethod
    def create_small_graph() -> 'Graph':
        """
        Создаёт малый контрольный граф с 6 вершинами.
        Расстояния между городами симметричны.
        """
        # Матрица расстояний между 6 городами (симметричная)
        distances = np.array([
            [0, 10, 15, 20, 25, 30],
            [10, 0, 35, 25, 15, 20],
            [15, 35, 0, 30, 20, 10],
            [20, 25, 30, 0, 14, 16],
            [25, 15, 20, 14, 0, 18],
            [30, 20, 10, 16, 18, 0]
        ])
        return Graph(distances)
    
    @staticmethod
    def load_from_stp(filepath: str) -> 'Graph':
        """
        Загружает граф из STP файла.
        
        Args:
            filepath: Путь до STP файла
            
        Returns:
            Graph объект с матрицей расстояний
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Парсим заголовок для получения количества узлов
        n_cities = None
        in_graph_section = False
        edges = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Section Graph'):
                in_graph_section = True
                continue
            
            if not in_graph_section:
                continue
            
            if line.startswith('Nodes'):
                n_cities = int(line.split()[1])
                continue
            
            if line.startswith('E '):
                parts = line.split()
                if len(parts) >= 4:
                    node1 = int(parts[1]) - 1  # Индексация с 0
                    node2 = int(parts[2]) - 1
                    distance = float(parts[3])
                    edges.append((node1, node2, distance))
        
        # Создаём матрицу расстояний
        distance_matrix = np.zeros((n_cities, n_cities))
        for node1, node2, distance in edges:
            distance_matrix[node1][node2] = distance
            distance_matrix[node2][node1] = distance  # Граф неориентированный
        
        return Graph(distance_matrix)
    
    def get_distance(self, city1: int, city2: int) -> float:
        """Возвращает расстояние между двумя городами."""
        return self.distance_matrix[city1][city2]
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """
        Вычисляет общую длину маршрута.
        
        Args:
            tour: Список индексов городов в порядке посещения
            
        Returns:
            Общая длина маршрута (включая возврат в начальный город)
        """
        length = 0
        for i in range(len(tour)):
            current = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            length += self.get_distance(current, next_city)
        return length
