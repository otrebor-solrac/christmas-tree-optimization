import random
import copy
from decimal import Decimal

from .gravity import GravityCompactor
from .config import GeneticConfig

class GeneticOptimizer:
    def __init__(self, scale_factor, config: GeneticConfig = None):
        self.config = config if config else GeneticConfig()
        # Usamos tu compactador existente como "Lamarckian learning" (mejora local)
        self.compactor = GravityCompactor(scale_factor)
        # Necesitamos el compactor configurado con pocos pasos para el bucle
        self.compactor.config.steps = self.config.gravity_steps_per_gen
        self.compactor.config.visualize = False

    def optimize(self, initial_trees, manager=None, target_n=None):
        # 1. Inicializar Población
        print(f"🧬 Iniciando Algoritmo Genético con {self.config.population_size} individuos...")
        population = self._init_population(initial_trees, self.config.population_size)
        
        best_overall_score = float('inf')
        best_overall_solution = None

        for gen in range(self.config.generations):
            # A. Evaluar y Aplicar Gravedad (Local Search) a cada individuo
            scored_population = []
            
            for idx, individual in enumerate(population):
                # Aplicamos un poco de gravedad para que el individuo "madure"
                # Esto es clave: evaluamos el potencial del layout, no solo el layout random
                # Nota: compact modifica in-place, pero aquí trabajamos con copias de la población
                optimized_trees = self.compactor.compact(individual)
                
                # VERIFICACIÓN DE COLISIONES
                # Si hay colisiones, el score es infinito (inválido)
                has_collisions, _ = self.compactor.solver.detect_collisions(optimized_trees, early_stop=True)
                if has_collisions:
                    score = float('inf')
                else:
                    score = self.compactor.solver.evaluator.calculate_kaggle_score(optimized_trees)
                
                scored_population.append((score, optimized_trees))
            
            # B. Ordenar por Score (Menor es mejor)
            scored_population.sort(key=lambda x: x[0])
            
            # Guardar el mejor histórico
            current_best_score = scored_population[0][0]
            if current_best_score < best_overall_score:
                best_overall_score = current_best_score
                best_overall_solution = copy.deepcopy(scored_population[0][1])
                print(f"✨ Gen {gen}: Nuevo Récord Global -> {best_overall_score:.6f}")
                
                if manager and target_n is not None:
                    manager.try_save_improvement(self.compactor.solver, best_overall_solution, target_n, f"GeneticGen{gen}")
            else:
                print(f"   Gen {gen}: Mejor de gen {current_best_score:.6f} (Global: {best_overall_score:.6f})")

            # C. Selección (Elitismo)
            next_generation = []
            # Pasamos los elites directamente
            for i in range(self.config.elite_size):
                next_generation.append(copy.deepcopy(scored_population[i][1]))
            
            # D. Crossover y Mutación para llenar el resto
            while len(next_generation) < self.config.population_size:
                # Torneo simple para elegir padres
                parent1 = self._tournament_selection(scored_population)
                parent2 = self._tournament_selection(scored_population)
                
                # Cruzar
                child_trees = self._crossover(parent1, parent2)
                
                # Mutar
                if random.random() < self.config.mutation_rate:
                    self._mutate(child_trees)
                
                next_generation.append(child_trees)
            
            population = next_generation

        print("\n🏆 Ejecutando Compactación Final Extrema en el Mejor Individuo...")
        # Al final, tomamos al ganador y le damos MUCHOS pasos de gravedad y fine-tuning
        self.compactor.config.steps = self.config.gravity_final_steps
        # Usamos la estrategia de nucleación para el final (closest_first)
        # Nota: compact() acepta strategy como argumento en la implementación actual de GravityCompactor
        final_solution = self.compactor.compact(best_overall_solution, strategy='closest_first')
        
        return final_solution

    def _init_population(self, base_trees, size):
        """Crea variaciones aleatorias de los árboles iniciales"""
        population = []
        # El primero es el original (por si acaso ya era bueno)
        population.append(copy.deepcopy(base_trees))
        
        # El resto son mutaciones fuertes
        for _ in range(size - 1):
            ind = copy.deepcopy(base_trees)
            # Randomizar posiciones fuertemente para explorar el espacio global
            for tree in ind:
                tree.center_x +=  Decimal(random.uniform(-5, 5))
                tree.center_y += Decimal(random.uniform(-5, 5))
                tree.angle = Decimal(random.uniform(0, 360))
                tree.update_polygon()
            population.append(ind)
        return population

    def _tournament_selection(self, scored_pop, k=3):
        """Elige el mejor de k individuos aleatorios"""
        candidates = random.sample(scored_pop, k)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] # Retorna los árboles

    def _crossover(self, parent1, parent2):
        """
        Uniform Crossover: Para cada árbol, elegimos si viene de Papá o Mamá.
        Mantiene la "coherencia" relativa de cada árbol, pero mezcla topologías.
        """
        child = []
        for t1, t2 in zip(parent1, parent2):
            if random.random() > 0.5:
                child.append(copy.deepcopy(t1))
            else:
                child.append(copy.deepcopy(t2))
        return child

    def _mutate(self, trees):
        """Mueve un árbol aleatorio a una posición aleatoria (Salto grande)"""
        idx = random.randint(0, len(trees)-1)
        tree = trees[idx]
        
        # Mutación agresiva: Cambia de lugar radicalmente
        tree.center_x += Decimal(random.uniform(-self.config.mutation_strength, self.config.mutation_strength))
        tree.center_y += Decimal(random.uniform(-self.config.mutation_strength, self.config.mutation_strength))
        tree.angle += Decimal(random.uniform(-45, 45))
        tree.update_polygon()
