import numpy as np
from typing import List, Tuple, Dict
import time


class AntColonyOptimization:
    """
    Базовый муравьиный алгоритм для задачи коммивояжера.
    
    Формулы:
    - Вероятность выбора города j из города i: P_ij = τ_ij^α * η_ij^β / Σ(τ_ik^α * η_ik^β)
    - Обновление феромона: τ_ij = (1 - ρ) * τ_ij + Σ(Δτ_ij)
    - Увеличение феромона: Δτ_ij = Q / L_k (для муравья k с маршрутом L_k)
    
    где:
    τ_ij - интенсивность феромона на ребре (i,j)
    η_ij = 1 / d_ij - видимость (привлекательность) ребра
    α - вес феромона, β - вес видимости
    ρ - коэффициент испарения феромона
    Q - количество феромона, откладываемого муравьём
    """
    
    def __init__(self, graph, n_ants: int = 30, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, 
                 rho: float = 0.1, Q: float = 100):
        """
        Args:
            graph: Graph объект
            n_ants: Количество муравьёв в колонии
            n_iterations: Количество итераций
            alpha: Вес феромона (важность синтез информации)
            beta: Вес видимости (важность эвристической информации)
            rho: Коэффициент испарения феромона (0 < rho < 1)
            Q: Количество феромона, откладываемого муравьём
        """
        self.graph = graph
        self.n_cities = graph.n_cities
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Инициализация феромона
        initial_pheromone = 1.0 / (self.n_cities * 10)
        self.pheromone = np.full((self.n_cities, self.n_cities), initial_pheromone)
        np.fill_diagonal(self.pheromone, 0)  # Нет самопетель
        
        # История
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
    
    def _calculate_heuristic(self) -> np.ndarray:
        """
        Вычисляет матрицу видимости η_ij = 1 / d_ij
        """
        heuristic = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distance = self.graph.get_distance(i, j)
                    if distance > 0:
                        heuristic[i][j] = 1.0 / distance
        return heuristic
    
    def _select_next_city(self, current_city: int, unvisited: set, 
                         pheromone: np.ndarray, heuristic: np.ndarray) -> int:
        """
        Выбирает следующий город на основе вероятностей.
        
        P_ij = τ_ij^α * η_ij^β / Σ(τ_ik^α * η_ik^β)
        """
        probabilities = np.zeros(self.n_cities)
        
        for city in unvisited:
            tau = self.pheromone[current_city][city] ** self.alpha
            eta = heuristic[current_city][city] ** self.beta
            probabilities[city] = tau * eta
        
        # Нормализуем вероятности
        prob_sum = probabilities.sum()
        if prob_sum == 0:
            # Если все вероятности нулевые, выбираем случайно
            return np.random.choice(list(unvisited))
        
        probabilities = probabilities / prob_sum
        return np.random.choice(self.n_cities, p=probabilities)
    
    def _construct_tour(self, heuristic: np.ndarray) -> List[int]:
        """Строит маршрут одного муравья."""
        start_city = np.random.randint(0, self.n_cities)
        tour = [start_city]
        unvisited = set(range(self.n_cities)) - {start_city}
        
        while unvisited:
            next_city = self._select_next_city(tour[-1], unvisited, 
                                               self.pheromone, heuristic)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour
    
    def _update_pheromone(self, tours: List[List[int]], distances: List[float]):
        """
        Обновляет матрицу феромона.
        τ_ij = (1 - ρ) * τ_ij + Σ(Δτ_ij)
        """
        # Испарение
        self.pheromone *= (1 - self.rho)
        
        # Откладка феромона муравьями
        for tour, distance in zip(tours, distances):
            delta_pheromone = self.Q / distance
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone[from_city][to_city] += delta_pheromone
                self.pheromone[to_city][from_city] += delta_pheromone
    
    def solve(self) -> Tuple[List[int], float, Dict]:
        """
        Решает задачу коммивояжера муравьиным алгоритмом.
        
        Returns:
            (best_tour, best_distance, metrics)
        """
        start_time = time.time()
        
        heuristic = self._calculate_heuristic()
        iterations = 0
        
        for iteration in range(self.n_iterations):
            # Каждый муравей строит маршрут
            tours = []
            distances = []
            
            for _ in range(self.n_ants):
                tour = self._construct_tour(heuristic)
                distance = self.graph.calculate_tour_length(tour)
                tours.append(tour)
                distances.append(distance)
                
                # Обновляем лучший найденный маршрут
                if distance < self.best_distance:
                    self.best_tour = tour.copy()
                    self.best_distance = distance
            
            # Обновляем феромон
            self._update_pheromone(tours, distances)
            
            # Сохраняем в историю
            self.history.append(self.best_distance)
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'algorithm': 'Ant Colony Optimization (Basic)',
            'best_distance': self.best_distance,
            'iterations': iterations,
            'time': elapsed_time,
            'function_evaluations': iterations * self.n_ants,
            'n_ants': self.n_ants,
        }
        
        return self.best_tour, self.best_distance, metrics


class AntColonyOptimizationInitial(AntColonyOptimization):
    """
    Модификация муравьиного алгоритма - "Начальное расположение".
    
    Отличие: На первой итерации все муравьи стартуют из всех вершин 
    поочередно для лучшего изучения графа. Это обеспечивает более 
    равномерное распределение муравьёв в начале работы алгоритма.
    """
    
    def __init__(self, graph, n_ants: int = 30, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, 
                 rho: float = 0.1, Q: float = 100):
        """
        Args:
            graph: Graph объект
            n_ants: Количество муравьёв в колонии
            n_iterations: Количество итераций
            alpha: Вес феромона
            beta: Вес видимости
            rho: Коэффициент испарения феромона
            Q: Количество феромона, откладываемого муравьём
        """
        super().__init__(graph, n_ants, n_iterations, alpha, beta, rho, Q)
        self.initial_phase = True
    
    def _construct_tour_from_city(self, start_city: int, heuristic: np.ndarray) -> List[int]:
        """Строит маршрут муравья, начиная с конкретного города."""
        tour = [start_city]
        unvisited = set(range(self.n_cities)) - {start_city}
        
        while unvisited:
            next_city = self._select_next_city(tour[-1], unvisited, 
                                               self.pheromone, heuristic)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour
    
    def solve(self) -> Tuple[List[int], float, Dict]:
        """
        Решает задачу коммивояжера с модификацией начального расположения.
        
        На первой итерации муравьи стартуют из разных вершин для лучшего
        исследования графа.
        
        Returns:
            (best_tour, best_distance, metrics)
        """
        start_time = time.time()
        
        heuristic = self._calculate_heuristic()
        iterations = 0
        
        for iteration in range(self.n_iterations):
            tours = []
            distances = []
            
            if iteration == 0:
                # Первая итерация: муравьи стартуют из разных вершин
                for ant_id in range(self.n_ants):
                    start_city = ant_id % self.n_cities
                    tour = self._construct_tour_from_city(start_city, heuristic)
                    distance = self.graph.calculate_tour_length(tour)
                    tours.append(tour)
                    distances.append(distance)
                    
                    if distance < self.best_distance:
                        self.best_tour = tour.copy()
                        self.best_distance = distance
            else:
                # Остальные итерации: обычный режим
                for _ in range(self.n_ants):
                    tour = self._construct_tour(heuristic)
                    distance = self.graph.calculate_tour_length(tour)
                    tours.append(tour)
                    distances.append(distance)
                    
                    if distance < self.best_distance:
                        self.best_tour = tour.copy()
                        self.best_distance = distance
            
            # Обновляем феромон
            self._update_pheromone(tours, distances)
            
            # Сохраняем в историю
            self.history.append(self.best_distance)
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'algorithm': 'Ant Colony Optimization (Initial Placement)',
            'best_distance': self.best_distance,
            'iterations': iterations,
            'time': elapsed_time,
            'function_evaluations': iterations * self.n_ants,
            'n_ants': self.n_ants,
        }
        
        return self.best_tour, self.best_distance, metrics
