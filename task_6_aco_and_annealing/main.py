import sys
import os
import numpy as np
from pathlib import Path
import json

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graph_utils import Graph
from annealing import SimulatedAnnealing, BoltzmannAnnealing
from aco import AntColonyOptimization, AntColonyOptimizationInitial


def make_json_serializable(obj):
    """Преобразует объекты в JSON-совместимый формат."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isinf(obj):
            return str(obj)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_experiments():
    """
    Запускает эксперименты со всеми алгоритмами на всех графах.
    """
    
    # Подготовка графов
    graphs = {
        'small': Graph.create_small_graph(),
        'berlin52': Graph.load_from_stp('data/berlin52.stp'),
        'world666': Graph.load_from_stp('data/world666.stp'),
    }
    
    print("=" * 80)
    print("ИССЛЕДОВАНИЕ АЛГОРИТМОВ РЕШЕНИЯ ЗАДАЧИ КОММИВОЯЖЕРА")
    print("=" * 80)
    
    results = {}
    
    for graph_name, graph in graphs.items():
        print(f"\n{'=' * 80}")
        print(f"ГРАФ: {graph_name.upper()} ({graph.n_cities} вершин)")
        print(f"{'=' * 80}")
        
        results[graph_name] = {}
        
        # Настройка параметров в зависимости от размера графа
        if graph_name == 'small':
            sa_iterations = 5000
            boltzmann_iterations = 5000
            aco_iterations = 50
            aco_ants = 20
        elif graph_name == 'berlin52':
            sa_iterations = 10000
            boltzmann_iterations = 10000
            aco_iterations = 100
            aco_ants = 30
        else:  # world666
            sa_iterations = 2000
            boltzmann_iterations = 2000
            aco_iterations = 20
            aco_ants = 15
        
        # 1. Базовая имитация отжига
        print("\n[1] Базовая имитация отжига...")
        sa = SimulatedAnnealing(graph, initial_temp=1000, cooling_rate=0.95, 
                               max_iterations=sa_iterations)
        tour, distance, metrics = sa.solve()
        results[graph_name]['SA_Basic'] = metrics
        print(f"    ✓ Расстояние: {distance:.2f}")
        print(f"    ✓ Итерации: {metrics['iterations']}")
        print(f"    ✓ Время: {metrics['time']:.4f} сек")
        print(f"    ✓ Маршрут: {tour}")
        
        # 2. Большцмановский отжиг (модификация)
        print("\n[2] Имитация отжига (модификация Больцмановский)...")
        boltzmann = BoltzmannAnnealing(graph, initial_temp=1000, 
                                      max_iterations=boltzmann_iterations)
        tour, distance, metrics = boltzmann.solve()
        results[graph_name]['SA_Boltzmann'] = metrics
        print(f"    ✓ Расстояние: {distance:.2f}")
        print(f"    ✓ Итерации: {metrics['iterations']}")
        print(f"    ✓ Время: {metrics['time']:.4f} сек")
        print(f"    ✓ Маршрут: {tour}")
        
        # 3. Базовый муравьиный алгоритм
        print("\n[3] Базовый муравьиный алгоритм...")
        aco = AntColonyOptimization(graph, n_ants=aco_ants, 
                                    n_iterations=aco_iterations,
                                    alpha=1.0, beta=2.0, rho=0.1, Q=100)
        tour, distance, metrics = aco.solve()
        results[graph_name]['ACO_Basic'] = metrics
        print(f"    ✓ Расстояние: {distance:.2f}")
        print(f"    ✓ Итерации: {metrics['iterations']}")
        print(f"    ✓ Время: {metrics['time']:.4f} сек")
        print(f"    ✓ Маршрут: {tour}")
        
        # 4. Муравьиный алгоритм с начальным расположением (модификация)
        print("\n[4] Муравьиный алгоритм (модификация - Начальное расположение)...")
        aco_initial = AntColonyOptimizationInitial(graph, n_ants=aco_ants, 
                                                   n_iterations=aco_iterations,
                                                   alpha=1.0, beta=2.0, rho=0.1, Q=100)
        tour, distance, metrics = aco_initial.solve()
        results[graph_name]['ACO_Initial'] = metrics
        print(f"    ✓ Расстояние: {distance:.2f}")
        print(f"    ✓ Итерации: {metrics['iterations']}")
        print(f"    ✓ Время: {metrics['time']:.4f} сек")
        print(f"    ✓ Маршрут: {tour}")
    
    # Сохранение результатов в JSON
    output_file = 'results.json'
    results_serializable = make_json_serializable(results)
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n\n✅ Результаты сохранены в {output_file}")
    
    # Вывод сравнительной таблицы
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    for graph_name in graphs.keys():
        print(f"\n{graph_name.upper()}:")
        print(f"{'Алгоритм':<40} {'Расстояние':<15} {'Время (сек)':<15}")
        print("-" * 70)
        for alg_name, metrics in results[graph_name].items():
            print(f"{alg_name:<40} {metrics['best_distance']:<15.2f} {metrics['time']:<15.4f}")


if __name__ == '__main__':
    run_experiments()
