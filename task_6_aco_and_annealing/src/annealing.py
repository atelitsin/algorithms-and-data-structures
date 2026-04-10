import numpy as np
from typing import List, Tuple, Dict
import time
import math


class SimulatedAnnealing:
    """
    Базовый алгоритм имитации отжига для задачи коммивояжера.
    
    Формула вероятности принятия худшего решения:
    P(delta) = exp(-delta / T), где delta = E_new - E_old, T - температура
    """
    
    def __init__(self, graph, initial_temp: float = 1000, 
                 cooling_rate: float = 0.95, max_iterations: int = 10000):
        """
        Args:
            graph: Graph объект
            initial_temp: Начальная температура
            cooling_rate: Коэффициент охлаждения (0 < rate < 1)
            max_iterations: Максимальное количество итераций
        """
        self.graph = graph
        self.n_cities = graph.n_cities
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        
        # История для метрик
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
    
    def _random_tour(self) -> List[int]:
        """Генерирует случайный маршрут."""
        tour = list(range(self.n_cities))
        np.random.shuffle(tour)
        return tour
    
    def _get_neighbor(self, tour: List[int]) -> List[int]:
        """
        Генерирует соседнее решение 2-opt свопом.
        2-opt: случайно выбираются два города и маршрут между ними разворачивается.
        """
        neighbor = tour.copy()
        i, j = sorted(np.random.choice(self.n_cities, 2, replace=False))
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        return neighbor
    
    def solve(self) -> Tuple[List[int], float, Dict]:
        """
        Решает задачу коммивояжера алгоритмом имитации отжига.
        
        Returns:
            (best_tour, best_distance, metrics)
        """
        start_time = time.time()
        
        # Инициализация
        current_tour = self._random_tour()
        current_distance = self.graph.calculate_tour_length(current_tour)
        
        self.best_tour = current_tour.copy()
        self.best_distance = current_distance
        
        temperature = self.initial_temp
        iterations = 0
        
        while temperature > 0.01 and iterations < self.max_iterations:
            # Генерируем соседнее решение
            neighbor_tour = self._get_neighbor(current_tour)
            neighbor_distance = self.graph.calculate_tour_length(neighbor_tour)
            
            # Вычисляем дельту энергии
            delta = neighbor_distance - current_distance
            
            # Решаем, принять ли новое решение
            if delta < 0 or np.random.rand() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_distance = neighbor_distance
            
            # Обновляем лучшее найденное решение
            if current_distance < self.best_distance:
                self.best_tour = current_tour.copy()
                self.best_distance = current_distance
            
            # Сохраняем в историю
            self.history.append(self.best_distance)
            
            # Охлаждение
            temperature *= self.cooling_rate
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'algorithm': 'Simulated Annealing (Basic)',
            'best_distance': self.best_distance,
            'iterations': iterations,
            'time': elapsed_time,
            'final_temperature': temperature,
            'function_evaluations': iterations,
        }
        
        return self.best_tour, self.best_distance, metrics


class BoltzmannAnnealing(SimulatedAnnealing):
    """
    Модификация имитации отжига - Больцмановский отжиг.
    
    Отличие: используется другой закон охлаждения температуры.
    Формула охлаждения: T(k) = T_0 / ln(1 + k), где k - номер итерации
    Это обеспечивает более медленное охлаждение и теоретически гарантирует сходимость.
    """
    
    def __init__(self, graph, initial_temp: float = 1000, max_iterations: int = 10000):
        """
        Args:
            graph: Graph объект
            initial_temp: Начальная температура
            max_iterations: Максимальное количество итераций
        """
        super().__init__(graph, initial_temp, cooling_rate=1.0, max_iterations=max_iterations)
        self.initial_temp = initial_temp
    
    def solve(self) -> Tuple[List[int], float, Dict]:
        """
        Решает задачу коммивояжера Больцмановским отжигом.
        
        Основное отличие: температура охлаждается по закону T(k) = T_0 / ln(1 + k)
        
        Returns:
            (best_tour, best_distance, metrics)
        """
        start_time = time.time()
        
        # Инициализация
        current_tour = self._random_tour()
        current_distance = self.graph.calculate_tour_length(current_tour)
        
        self.best_tour = current_tour.copy()
        self.best_distance = current_distance
        
        iterations = 0
        
        while iterations < self.max_iterations:
            # Больцмановское охлаждение: T(k) = T_0 / ln(1 + k)
            temperature = self.initial_temp / math.log(1 + iterations + 1)
            
            if temperature < 0.01:
                break
            
            # Генерируем соседнее решение
            neighbor_tour = self._get_neighbor(current_tour)
            neighbor_distance = self.graph.calculate_tour_length(neighbor_tour)
            
            # Вычисляем дельту энергии
            delta = neighbor_distance - current_distance
            
            # Решаем, принять ли новое решение
            if delta < 0 or np.random.rand() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_distance = neighbor_distance
            
            # Обновляем лучшее найденное решение
            if current_distance < self.best_distance:
                self.best_tour = current_tour.copy()
                self.best_distance = current_distance
            
            # Сохраняем в историю
            self.history.append(self.best_distance)
            
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'algorithm': 'Simulated Annealing (Boltzmann)',
            'best_distance': self.best_distance,
            'iterations': iterations,
            'time': elapsed_time,
            'final_temperature': temperature,
            'function_evaluations': iterations,
        }
        
        return self.best_tour, self.best_distance, metrics
