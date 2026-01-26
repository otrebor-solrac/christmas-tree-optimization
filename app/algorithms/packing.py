"""
Packing algorithm module using simulated annealing.
"""
import copy
import math
import random
from decimal import Decimal

from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

from ..models.tree import ChristmasTree
from ..evaluation.cost_evaluator import CostEvaluator
from ..utils.io import load_solution
from .config import AnnealingConfig


class PackingAlgorithm:
    def __init__(self, scale_factor, config=None):
        self.scale_factor = scale_factor
        self.config = config if config is not None else AnnealingConfig()
        self.evaluator = CostEvaluator(scale_factor)
        # best_score guardará el ÁREA (Kaggle Score)
        self.best_score = float('inf')
        # Referencias para visualización reutilizable
        self._viz_fig = None
        self._viz_ax = None
        self._viz_initialized = False        
        # Estado interno
        self.trees = []
        self.cage_bounds = None  # (min_x, min_y, max_x, max_y)
        # Para guardar frames del video
        self._frame_counter = 0
        self._frames_dir = None

    
    def _init_visualization(self, visualizer, save_frames=False):
        """
        Inicializa la figura de visualización una sola vez.
        
        Args:
            visualizer: Instancia de TreeVisualizer
            save_frames: Si True, guarda cada frame para crear un video
        """
        if visualizer is None or self._viz_initialized:
            return
        
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            
            plt.ion()  # Modo interactivo
            self._viz_fig, self._viz_ax = plt.subplots(figsize=visualizer.figsize)
            plt.show(block=False)
            self._viz_initialized = True
            
            # Crear carpeta para frames si se solicita
            if save_frames:
                self._frame_counter = 0
                self._frames_dir = Path("video_frames")
                self._frames_dir.mkdir(exist_ok=True)
                # Limpiar frames anteriores si existen
                for old_frame in self._frames_dir.glob("frame_*.png"):
                    old_frame.unlink()
        
        except Exception as e:
            print(f"   [Visualización] Error al inicializar: {e}")
            self._viz_initialized = False


    def _update_visualization(self, visualizer, step, iterations, temp, has_collisions=False, collision_pairs=None, context_info=None):
        try:
            # Limpiar el eje primero
            self._viz_ax.clear()
            
            # Dibujar árboles
            for i, tree in enumerate(self.trees):
                visualizer._dibujar_arbol_individual(self._viz_ax, tree, i)
            
            # Dibujar caja delimitadora
            side = visualizer._dibujar_bounding_box(self._viz_ax, self.trees, adjust_limits=False)
            
            # Calcular métricas
            n = len(self.trees)
            area = side * side
            n_eff = n if n > 0 else 1
            current_score = area / n_eff
            
            # Título
            title = f"Optimización: {step}/{iterations} | Temp: {temp:.3f}\n"
            title += f"Área: {area:.4f} | Métrica (Area/N): {current_score:.4f}"
            if context_info:
                title += f"\n{context_info}"
            if has_collisions:
                title += f"\n⚠️ COLISIÓN! Pares: {collision_pairs}"
                self._viz_ax.set_title(title, fontsize=10, color='red', weight='bold')
            else:
                self._viz_ax.set_title(title, fontsize=10)
            
            # Estética
            self._viz_ax.set_aspect('equal', adjustable='box')
            self._viz_ax.grid(True, alpha=0.3, linestyle=':')
            
            # Auto-escalar
            self._viz_ax.relim()
            self._viz_ax.autoscale_view()
            
            # Actualizar figura
            self._viz_fig.canvas.draw()
            self._viz_fig.canvas.flush_events()
            
            # Guardar frame si está habilitado
            if self._frames_dir is not None:
                frame_path = self._frames_dir / f"frame_{self._frame_counter:05d}.png"
                self._viz_fig.savefig(frame_path, dpi=100, bbox_inches='tight')
                self._frame_counter += 1
            
            plt.pause(0.001)
        except Exception as e:
            pass

    def set_initial_state(self, existing_trees):
        if existing_trees:
            self.trees = copy.deepcopy(existing_trees)
            self.refresh_current_score()
        else:
            self.trees = []
            self.best_score = float('inf')

    def detect_collisions(self, trees, early_stop=False):
        # Detecta colisiones entre árboles. Método público reutilizable.
        # Args: trees: Lista de árboles a verificar
        #       early_stop: Si True, retorna inmediatamente al encontrar la primera colisión.
        # Returns: tuple: (has_collisions: bool, collision_pairs: list)

        if not trees or len(trees) < 2:
            return False, []
        
        polys = [t.polygon for t in trees]
        tree_index = STRtree(polys)
        tolerance_area = (float(self.scale_factor) ** 2) * 1e-9
        collision_pairs = []
        
        for i, t in enumerate(trees):
            candidates = tree_index.query(t.polygon)
            for c_idx in candidates:
                # Optimización: Solo verificar pares una vez (i < c_idx)
                # y evitar auto-intersección (i == c_idx)
                if c_idx <= i:
                    continue
                
                # Verificar intersección real
                # intersects() es mucho más rápido que intersection().area
                if t.polygon.intersects(polys[c_idx]):
                    if t.polygon.intersection(polys[c_idx]).area > tolerance_area:
                        if early_stop:
                            return True, [(trees[i].id, trees[c_idx].id)]
                        collision_pairs.append((trees[i].id, trees[c_idx].id))
        
        return len(collision_pairs) > 0, collision_pairs

    def refresh_current_score(self):
        """Valida el estado actual y actualiza el score."""
        if not self.trees:
            self.best_score = 0.0
            return

        # Reutilizar método de detección de colisiones
        collision_detected, collision_pairs = self.detect_collisions(self.trees)

        # Actualizar Score
        if collision_detected:
            self.best_score = float('inf')
        else:
            area = self.evaluator.calculate_kaggle_score(self.trees)
            self.best_score = area
            n = len(self.trees)
            print(f"   [Estado Válido] Trees: {n} | Area Total: {area:.4f} | Score (Area/n): {(area/n):.4f}")

    def _check_strict_collision(self, new_tree, static_trees):
        """Verifica colisión con tolerancia numérica."""
        if not static_trees: return False
        
        # Optimización: Iteración directa con prepared geometry
        # Evita el costo O(N log N) de construir STRtree para una sola consulta
        new_poly = new_tree.polygon
        prepared_new = prep(new_poly)
        tolerance_area = (float(self.scale_factor) ** 2) * 1e-9

        for t in static_trees:
            if prepared_new.intersects(t.polygon):
                if new_poly.intersection(t.polygon).area > tolerance_area:
                    return True
        return False

    def add_configured_tree(self, tree_id, x, y, angle, fixed=True):
        """Agrega árbol manual."""
        new_tree = ChristmasTree(tree_id, x, y, angle, fixed=fixed)
        self.trees.append(new_tree)
        print(f"-> Manual: Árbol {tree_id} en ({x}, {y}) [Fixed={fixed}]")
        self.refresh_current_score()
    
    def _strategy_linear_zipper(self, tree_id):
        """ESTRATEGIA 1: Tira infinita (Cremallera). Bueno para n pequeño."""
        if not self.trees: return ChristmasTree(tree_id, 0.0, 0.0, 0, fixed=False)
        
        last = self.trees[-1]
        stride_x = 0.42
        
        new_x = float(last.center_x) + stride_x
        new_angle = (float(last.angle) + 180) % 360
        new_y = 0.5 if new_angle == 180 else 0.0
        
        return self._add_jitter(ChristmasTree(tree_id, new_x, new_y, new_angle, fixed=False))

    def _strategy_square_packing(self, tree_id):
        """
        ESTRATEGIA 3: Grid Matemático Estricto.
        Calcula fila y columna basado en el índice para forzar un cuadrado.
        """
        # Si es el primer árbol, al origen
        if not self.trees: 
            return ChristmasTree(tree_id, 0.0, 0.0, 0, fixed=False)

        # 1. ¿Cuántos árboles tendremos en total al final de este paso?
        #    (Asumimos que tree_id es secuencial 1, 2, 3...)
        #    Si tree_id no es secuencial, usar len(self.trees) + 1
        current_count = len(self.trees) + 1
        
        # 2. Calcular dimensiones de la rejilla (Grid)
        #    Para hacer un cuadrado, el número de columnas es la raíz cuadrada redondeada hacia arriba.
        #    Ej: 10 árboles -> sqrt(10)=3.16 -> 4 columnas. (Grilla 4x3)
        cols = math.ceil(math.sqrt(current_count))
        
        # Evitar columnas de 1 (torres) al principio
        if cols < 2: cols = 2 

        # 3. Calcular posición en la rejilla (Base 0)
        #    El índice actual es len(self.trees)
        my_index = len(self.trees)
        
        row = my_index // cols  # División entera: Fila
        col = my_index % cols   # Resto: Columna
        
        # 4. Parámetros de la "Célula" (Ajustados para compresión máxima)
        stride_x = self.config.stride_x       # Ancho efectivo
        row_height = self.config.row_height     # Alto efectivo (Nesting vertical)
        
        # --- CÁLCULO DE COORDENADAS ---
        
        # Posición X base
        new_x = col * stride_x
        
        # Posición Y base
        new_y = row * row_height
        
        # --- LOGICA DE CREMALLERA (Zipper) ---
        
        # Angulo: Alternar 0 y 180 en cada columna para encajar
        # Si col es par: 0°, Si col es impar: 180°
        if col % 2 == 0:
            new_angle = 0
            # Ajuste Y: Los de 0° van en el "suelo" de la fila
        else:
            new_angle = 180
            # Ajuste Y: Los de 180° van un poco más arriba (+0.5)
            new_y += 0.5
            
        # --- LOGICA DE LADRILLO (Brick Layering) ---
        # Desplazar las filas impares para que encajen en los huecos de las pares
        if row % 2 == 1:
            new_x += (stride_x / 2) # Desplazar medio paso a la derecha

        return self._add_jitter(ChristmasTree(tree_id, new_x, new_y, new_angle, fixed=False))

    def _strategy_radial_spiral(self, tree_id):
        """
        ESTRATEGIA RADIAL: Espiral desde el centro.
        Coloca árboles en círculos concéntricos para minimizar espacio en esquinas.
        """
        # Primer árbol en el centro
        if not self.trees:
            return ChristmasTree(tree_id, 0.0, 0.0, 0, fixed=False)
        
        # Parámetros de la espiral
        trees_per_ring = 6  # Árboles por anillo (hexagonal)
        radial_spacing = 0.45  # Distancia entre anillos
        
        # Calcular en qué anillo estamos
        n = len(self.trees)  # Número de árboles ya colocados
        
        # Determinar anillo y posición dentro del anillo
        # Anillo 0: 1 árbol (centro)
        # Anillo 1: 6 árboles
        # Anillo 2: 12 árboles
        # Anillo k: 6*k árboles
        
        if n == 0:
            ring = 0
            pos_in_ring = 0
        else:
            # Calcular anillo actual
            cumulative = 1  # Centro
            ring = 1
            while cumulative + (trees_per_ring * ring) <= n:
                cumulative += trees_per_ring * ring
                ring += 1
            pos_in_ring = n - cumulative
        
        # Calcular radio y ángulo
        if ring == 0:
            radius = 0
            angle_deg = 0
        else:
            radius = ring * radial_spacing
            trees_in_this_ring = trees_per_ring * ring
            angle_deg = (pos_in_ring / trees_in_this_ring) * 360
        
        # Convertir a coordenadas cartesianas
        angle_rad = math.radians(angle_deg)
        new_x = radius * math.cos(angle_rad)
        new_y = radius * math.sin(angle_rad)
        
        # Orientación del árbol: usar múltiples estrategias para mayor variedad
        # Combinamos: posición angular, número de anillo, y posición en anillo
        base_orientation = angle_deg  # Orientación radial base
        
        # Estrategia de orientación basada en posición
        orientation_strategy = (ring + pos_in_ring) % 4
        
        if orientation_strategy == 0:
            tree_angle = base_orientation  # Apunta radialmente hacia afuera (0°-360°)
        elif orientation_strategy == 1:
            tree_angle = (base_orientation + 180) % 360  # Apunta hacia el centro
        elif orientation_strategy == 2:
            tree_angle = (base_orientation + 90) % 360  # Perpendicular (sentido horario)
        else:  # orientation_strategy == 3
            tree_angle = (base_orientation + 270) % 360  # Perpendicular (sentido antihorario)
        
        return self._add_jitter(ChristmasTree(tree_id, new_x, new_y, tree_angle, fixed=False))

    def _strategy_random_fallback(self, tree_id):
        """ESTRATEGIA 3: Fallback aleatorio (Radial)."""
        # Si es el primer árbol, colocarlo en el origen
        if not self.trees:
            return ChristmasTree(tree_id, 0.0, 0.0, random.uniform(0, 360), fixed=False)
        
        # En lugar de ir a la esquina (diagonal), buscar un hueco radial aleatorio
        # Calculamos el radio aproximado actual
        total_area = self._calculate_total_tree_area()
        radius = math.sqrt(total_area / math.pi) * 1.5 # Un poco más afuera
        
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, radius)
        
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        
        return ChristmasTree(tree_id, x, y, random.uniform(0, 360), fixed=False)

    def _add_jitter(self, tree):
        """Añade ruido minúsculo para que el Annealing tenga gradiente."""
        j = 0.05
        tree.center_x = Decimal(float(tree.center_x) + random.uniform(-j, j))
        tree.center_y = Decimal(float(tree.center_y) + random.uniform(-j, j))
        tree.angle = Decimal(float(tree.angle) + random.uniform(-5, 5))
        tree.update_polygon()
        return tree


    def add_trees(self, n_new, strategy="square", visualizer=None):
        """
        Agrega árboles usando la estrategia seleccionada.
        strategies: 'linear', 'square', 'random'
        """
        if n_new == 0:
            print("-> Verificando estado actual...")
            # Si hay árboles no fijos, optimizarlos
            mutable_indices = [i for i, t in enumerate(self.trees) if not t.fixed]
            
            if mutable_indices:
                print(f"-> Optimizando {len(mutable_indices)} árbol(es) no fijo(s)...")
                # Usar mayor magnitud para poder mover árboles lejanos
                # No permitir crecimiento de área (solo optimizar árboles existentes)
                found = self.run_annealing(
                    iterations=self.config.mutable_opt_iters, 
                    initial_temp=self.config.mutable_opt_temp, 
                    mag_factor=self.config.mutable_opt_mag, 
                    visualizer=visualizer, 
                    allow_area_growth=False
                )

                if found:
                    print("   ¡Optimización exitosa!")
                else:
                    print("   Advertencia: No se encontró una solución válida.")
            
            self.refresh_current_score()
            return self.trees

        start_id = max([t.id for t in self.trees] + [0]) + 1
        
        # Seleccionar la función generadora dinámicamente
        if strategy == "linear": generator = self._strategy_linear_zipper
        elif strategy == "square": generator = self._strategy_square_packing
        elif strategy == "radial": generator = self._strategy_radial_spiral
        else: generator = self._strategy_random_fallback

        print(f"\n>>> INICIANDO INSERCIÓN (Estrategia: {strategy}) <<<")

        # Inicializar visualización si se proporciona (guardar frames para video)
        if visualizer is not None:
            self._init_visualization(visualizer, save_frames=True)

        for i in range(start_id, start_id + n_new):
            print(f"\n   -> Insertando Árbol {i}...")
            success = False
            attempts = 0
            max_attempts = self.config.max_insertion_attempts
            safe_state = copy.deepcopy(self.trees)
            
            # Visualizar estado inicial antes de insertar
            if visualizer is not None:
                self._update_visualization(visualizer, 0, 1, 0, context_info=f"Insertando Árbol {i}...")

            # 2. Configurar Jaula Automática para evitar explosión
            # Calculamos el área esperada y damos un margen (ej: ratio 2.5 para empezar holgado pero acotado)
            # Esto evita que el annealing explore el infinito
            self.set_cage_from_ratio(ratio=2.5)
            
            best_valid_state = None  # Inicializar variable para búsqueda exhaustiva

            while not success and attempts < max_attempts:
                attempts += 1
                
                if self.config.exhaustive_search:
                    # Si es exhaustivo, probamos los intentos 1 por 1
                    # En la iteración 1 es generator(i), luego random
                    if attempts == 1:
                        new_tree = generator(i)
                        
                        # Annealing SUAVE (Ajuste fino)
                        temp = self.config.attempt_1_temp
                        mag = self.config.attempt_1_mag
                        iters = self.config.attempt_1_iters
                    else:
                        new_tree = self._strategy_random_fallback(i)
                        # Annealing FUERTE (Búsqueda global)
                        temp = self.config.attempt_retry_temp
                        mag = self.config.attempt_retry_mag
                        iters = self.config.attempt_retry_iters
                else:
                    # Lógica original (no exhaustiva)
                    if attempts == 1:
                        new_tree = generator(i)
                        
                        # Annealing SUAVE (Ajuste fino)
                        temp = self.config.attempt_1_temp
                        mag = self.config.attempt_1_mag
                        iters = self.config.attempt_1_iters
                    else:
                        new_tree = self._strategy_random_fallback(i)
                        # Annealing FUERTE (Búsqueda global)
                        temp = self.config.attempt_retry_temp
                        mag = self.config.attempt_retry_mag
                        iters = self.config.attempt_retry_iters

                self.trees.append(new_tree)

                # 2. Optimizar (permitir crecimiento de área al insertar nuevo árbol)
                found = self.run_annealing(iterations=iters, initial_temp=temp, mag_factor=mag, 
                                          visualizer=visualizer, allow_area_growth=True)
                
                # 3. Validación Final
                if found:
                    # Doble check estricto final
                    others = self.trees[:-1]
                    if not self._check_strict_collision(self.trees[-1], others):
                        success = True
                        # Mostrar métrica actual
                        area = self.best_score
                        print(f"      ¡Éxito! Leaderboard Contrib: {(area/len(self.trees)):.4f}")
                        # Visualizar resultado exitoso
                        if visualizer is not None:
                            self._update_visualization(visualizer, iters, iters, temp, 
                                                      context_info=f"Árbol {i} insertado exitosamente")
                        
                        # Si es exhaustivo, guardamos el mejor estado válido encontrado
                        if self.config.exhaustive_search:
                            if best_valid_state is None or self.best_score < self.evaluator.calculate_kaggle_score(best_valid_state):
                                best_valid_state = copy.deepcopy(self.trees)
                                print(f"         -> Mejor estado válido para Árbol {i} guardado (Área: {self.best_score:.4f})")
                        else:
                            # Si no es exhaustivo, el primer éxito es suficiente
                            break # Salir del bucle de intentos

                # SIEMPRE restaurar para probar otro intento, salvo que sea el último (o encontremos uno y no sea exhaustivo)
                # Pero si encontramos uno válido, ya tenemos 'success = True' y 'best_valid_state'
                self.trees = copy.deepcopy(safe_state)
            
                # Si NO es exhaustivo y ya tuvimos y success, salimos
                if success and not self.config.exhaustive_search:
                     break

            # Al final de los intentos (o break), si hubo éxito, usar el mejor
            if success:
                if best_valid_state is not None:
                     self.trees = best_valid_state
                self.refresh_current_score()
            else:
                # Si no hubo éxito en ningún intento
                print(f"ERROR: No se pudo insertar Árbol {i}.")
                self.refresh_current_score()
                break

        return self.trees

    def _calculate_center_of_mass(self, tree_indices=None):
        """
        Calcula el centro de masa de los árboles especificados.
        Si tree_indices es None, usa todos los árboles.
        """
        if tree_indices is None:
            trees_to_use = self.trees
        else:
            trees_to_use = [self.trees[i] for i in tree_indices]
        
        if not trees_to_use:
            return 0.0, 0.0
        
        total_x = sum(float(t.center_x) for t in trees_to_use)
        total_y = sum(float(t.center_y) for t in trees_to_use)
        n = len(trees_to_use)
        return total_x / n, total_y / n

    def _calculate_total_tree_area(self):
        """
        Calcula el área total de todos los árboles (suma de áreas de polígonos).
        """
        if not self.trees:
            return 0.0

        sf = float(self.scale_factor)
        total_area = sum(t.polygon.area for t in self.trees)
        
        # Desescalar el área
        return total_area / (sf * sf)

    def set_cage_from_ratio(self, ratio=1.5):
        """
        Establece una 'jaula' cuadrada centrada en el origen basada en el área total de los árboles.
        
        Args:
            ratio: Relación entre el área de la jaula y el área total de los árboles.
                   Ej: 1.5 significa que la jaula es 50% más grande que la suma de las áreas.
        """
        total_tree_area = self._calculate_total_tree_area()
        if total_tree_area <= 0:
            self.cage_bounds = None
            return
            
        target_area = total_tree_area * ratio
        side_length = math.sqrt(target_area)
        half_side = side_length / 2.0
        
        # Definir límites centrados en (0,0)
        # (min_x, min_y, max_x, max_y)
        self.cage_bounds = (-half_side, -half_side, half_side, half_side)
        print(f"   [JAULA] Activada: Lado={side_length:.2f}, Área={target_area:.2f} (Ratio={ratio})")

    def _update_compaction_mode(self):
        """
        Actualiza el estado del modo de compactación basado en la relación área_cuadrado/área_pinos.
        
        Regla:
        - Si área_cuadrado >= 2.0 * área_pinos: ACTIVAR modo compactación
        - Si área_cuadrado <= 1.5 * área_pinos: DESACTIVAR modo compactación
        - Entre 1.5 y 2.0: mantener el estado actual
        
        Returns:
            bool: True si el modo compactación está activo
        """
        if not self.trees:
            self._compaction_mode_active = False
            self._max_allowed_area = float('inf')
            return False
        
        # Calculate square area
        square_area = self.evaluator.calculate_kaggle_score(self.trees)
        
        # Calculate total tree area
    def _select_farthest_tree(self, mutable_indices, reference_point=(0.0, 0.0)):
        """
        Selecciona el árbol más alejado del punto de referencia (centro de masa o origen).
        Usa probabilidad ponderada para dar más chances a los más alejados.
        """
        if not mutable_indices:
            return None
        
        distances = []
        for idx in mutable_indices:
            tree = self.trees[idx]
            dx = float(tree.center_x) - reference_point[0]
            dy = float(tree.center_y) - reference_point[1]
            dist = math.sqrt(dx * dx + dy * dy)
            distances.append((idx, dist))
        
        # Ordenar por distancia (más lejano primero)
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Usar selección ponderada: los más lejanos tienen más probabilidad
        # Pero también permitir algo de aleatoriedad
        if random.random() < 0.7:  # 70% de las veces elegir uno de los más lejanos
            # Elegir entre el 25% más lejano
            top_n = max(1, len(distances) // 4)
            candidates = distances[:top_n]
            weights = [dist ** 2 for _, dist in candidates]  # Peso cuadrático por distancia
            total_weight = sum(weights)
            r = random.uniform(0, total_weight)
            cumsum = 0
            for (idx, dist), weight in zip(candidates, weights):
                cumsum += weight
                if r <= cumsum:
                    return idx
        else:
            # 30% aleatorio para exploración
            return random.choice(mutable_indices)
        
        # Fallback
        return distances[0][0]

    def run_annealing(self, iterations, initial_temp, mag_factor, visualizer=None, allow_area_growth=False):
        """
        Annealing parametrizable.
        mag_factor: 0.2 para ajuste fino, 3.0 para búsqueda caos.
        visualizer: Opcional, para visualizar el progreso durante la optimización.
        allow_area_growth: Si True, permite que el área crezca (útil al insertar nuevos árboles).
        """
        mutable_indices = [i for i, t in enumerate(self.trees) if not t.fixed]
        if not mutable_indices: return True
        
        # Calcular centro de masa de todos los árboles
        com_x, com_y = self._calculate_center_of_mass()
        reference_point = (com_x, com_y)

        temp = initial_temp
        alpha = self.config.alpha
        min_temp = self.config.min_temp
        current_penalty_weight = self.config.initial_penalty_weight
        
        # Buscamos minimizar Area (sin colisiones)
        local_best_layout = None
        local_best_score = float('inf') 
        
        # Mejor área encontrada (puede tener colisiones menores)
        best_area_layout = None
        best_area_score = float('inf') 
        
        # Compute energy
        current_energy = self.evaluator.get_total_energy(
            self.trees, 
            current_penalty_weight
        )
        initial_energy = current_energy
        
        # Initialize with the current area
        initial_area = self.evaluator.calculate_kaggle_score(self.trees)
        best_area_score = initial_area
        best_area_layout = copy.deepcopy(self.trees)
        
        # Update compaction mode
        self._update_compaction_mode()

        # Initialize local_best_layout with the current state if valid
        initial_overlap = self.evaluator.calculate_soft_overlap(self.trees)
        
        # Verify if the initial state is valid (no strict collisions)
        # Nota: _check_strict_collision verifica un árbol contra otros. Aquí asumimos validez global o chequeamos.
        # Para simplificar, si initial_overlap es bajo, lo consideramos candidato.
        if initial_overlap < 1e-4:
             local_best_score = initial_area
             local_best_layout = copy.deepcopy(self.trees)

        # Configure periodic visualization
        if self.config.viz_frames > 0:
            viz_interval = max(1, iterations // self.config.viz_frames)  
        else:
            viz_interval = iterations + 1

        if visualizer is not None:
            self._init_visualization(visualizer, save_frames=True)

        for step in range(iterations):
            progress = step / iterations
            
            # Seleccionar árbol a perturbar (Ponderado por distancia al cuadrado)
            # Calcular distancias para los árboles mutables
            candidates = []
            weights = []
            for idx in mutable_indices:
                t = self.trees[idx]
                # Distancia al punto de referencia (CoM)
                dx = float(t.center_x) - reference_point[0]
                dy = float(t.center_y) - reference_point[1]
                d2 = dx*dx + dy*dy  # Distancia al cuadrado directamente
                candidates.append(idx)
                weights.append(d2) # Peso = distancia^2
            
            # Selección ponderada
            if weights and sum(weights) > 0:
                idx = random.choices(candidates, weights=weights, k=1)[0]
            else:
                idx = random.choice(mutable_indices)
            
            target = self.trees[idx]
            old_params = (target.center_x, target.center_y, target.angle)
            
            # Calcular distancia del árbol al centro de masa para ajustar magnitud
            dx = float(target.center_x) - reference_point[0]
            dy = float(target.center_y) - reference_point[1]
            distance_to_com = math.sqrt(dx * dx + dy * dy)
            
            # Aumentar magnitud si el árbol está muy lejos
            adjusted_mag_factor = mag_factor
            if distance_to_com > 2.0:  # Si está a más de 2 unidades
                adjusted_mag_factor = mag_factor * (1.0 + distance_to_com * 0.3)
            
            # --- PERTURBACIÓN ---
            self._perturb_tree(target, temp, initial_temp, adjusted_mag_factor, 
                             pull_towards=reference_point if distance_to_com > 1.0 else None)
            
            # --- EVALUACIÓN ---
            new_overlap = self.evaluator.calculate_soft_overlap(self.trees)
            new_side = self.evaluator.calculate_bounds_score(self.trees)
            new_square_area = new_side * new_side
            
            # Check estricto perezoso (solo si necesario)
            is_strict = False
            if new_overlap > 0 or progress > 0.6:
                others = self.trees[:idx] + self.trees[idx+1:]
                if self._check_strict_collision(target, others):
                    is_strict = True

            # Check estricto perezoso (solo si necesario)
            is_strict = False
            if new_overlap > 0 or progress > 0.6:
                others = self.trees[:idx] + self.trees[idx+1:]
                if self._check_strict_collision(target, others):
                    is_strict = True

            # Penalización dinámica
            pen_weight = self.config.initial_penalty_weight * (1 + progress * 200)
            strict_pen = self.config.strict_collision_penalty if is_strict else 0
            
            base_energy = self.evaluator.get_total_energy(self.trees, pen_weight)
            new_energy = base_energy + strict_pen
            delta = new_energy - current_energy
            
            # Aceptación
            accept = False
            if progress > 0.85 and is_strict: accept = False # Veto
            elif delta < 0: accept = True
            elif random.random() < math.exp(-delta / temp): accept = True
            
            if accept:
                current_energy = new_energy
                # Guardar el mejor estado basado en área (kaggle_score)
                # Solo guardar si no tiene colisiones estrictas
                if not is_strict:
                    kaggle_score = new_side * new_side
                    # Guardar el mejor área encontrada (puede tener overlap menor)
                    if kaggle_score < best_area_score:
                        best_area_score = kaggle_score
                        best_area_layout = copy.deepcopy(self.trees)
                    
                    # Guardar el mejor layout sin colisiones
                    # Usamos is_strict como fuente de verdad para validez
                    if kaggle_score < local_best_score:
                        local_best_score = kaggle_score
                        local_best_layout = copy.deepcopy(self.trees)
            else:
                target.center_x, target.center_y, target.angle = old_params
                target.update_polygon()
            
            temp *= alpha
            if temp < min_temp: break
            
            # Visualización periódica (llamada limpia al método separado)
            if visualizer is not None and (step % viz_interval == 0 or step == iterations - 1):
                self._update_visualization(visualizer, step, iterations, temp)

        # Priorizar layout sin colisiones
        if local_best_layout:
            self.trees = local_best_layout
            self.best_score = local_best_score
            return True
        elif best_area_layout and best_area_score < initial_area:
            # Actualizar con el mejor estado encontrado (aunque tenga colisiones menores)
            self.trees = best_area_layout
            # Recalcular score
            self._refresh_current_score()
            return True

        # Si fallamos, restaurar estado inicial para no dejar basura
        # (Aunque si era una inserción nueva, el caller manejará el fallo)
        # Pero si era optimización de existentes, mejor volver al inicio que dejar un estado random.
        if not allow_area_growth: # Si era optimización pura
             # Restaurar layout inicial si no encontramos nada mejor
             # (best_area_layout se inicializó con initial_layout)
             self.trees = best_area_layout
             self.refresh_current_score()
             
        return False

    def _perturb_tree(self, tree, temp, max_temp, mag_factor, pull_towards=None):
        """
        Movimiento controlado por mag_factor.
        Si pull_towards está especificado, tira el árbol hacia ese punto.
        """
        base_mag = 0.5 * mag_factor
        mag = base_mag * (temp / max_temp)
        if mag < 0.005: mag = 0.005
        
        if pull_towards is not None:
            # Calcular dirección hacia el punto de atracción
            dx_to_target = pull_towards[0] - float(tree.center_x)
            dy_to_target = pull_towards[1] - float(tree.center_y)
            dist_to_target = math.sqrt(dx_to_target * dx_to_target + dy_to_target * dy_to_target)
            
            if dist_to_target > 0.001:  # Evitar división por cero
                # Normalizar dirección
                dir_x = dx_to_target / dist_to_target
                dir_y = dy_to_target / dist_to_target
                
                # Calcular probabilidad de moverse hacia el objetivo (mayor si está más lejos)
                # Si está muy lejos (>5 unidades), 90% hacia objetivo
                # Si está cerca (<1 unidad), 50% hacia objetivo
                prob_towards = 0.5 + min(0.4, (dist_to_target / 5.0) * 0.4)
                
                if random.random() < prob_towards:
                    # Movimiento proporcional a la distancia: más lejos = movimiento más grande
                    # Factor proporcional: base_mag * (1 + distancia * factor)
                    # Limitar a máximo 50% de la distancia para evitar saltos excesivos
                    distance_factor = min(dist_to_target * 0.3, 2.0)  # Factor máximo de 2.0
                    pull_strength = mag * (1.0 + distance_factor)
                    # Limitar el movimiento máximo a 50% de la distancia
                    max_move = dist_to_target * 0.5
                    pull_strength = min(pull_strength, max_move)
                    
                    # Movimiento hacia el objetivo con aleatoriedad reducida
                    noise_factor = 0.2 / (1.0 + dist_to_target * 0.1)  # Menos ruido si está lejos
                    dx = dir_x * pull_strength + random.uniform(-mag * noise_factor, mag * noise_factor)
                    dy = dir_y * pull_strength + random.uniform(-mag * noise_factor, mag * noise_factor)
                else:
                    # Movimiento aleatorio para exploración (menos frecuente si está lejos)
                    dx = random.uniform(-mag, mag)
                    dy = random.uniform(-mag, mag)
            else:
                # Ya está cerca, movimiento aleatorio pequeño
                dx = random.uniform(-mag, mag)
                dy = random.uniform(-mag, mag)
        else:
            # Comportamiento por defecto: Gravedad hacia el centro (0,0)
            # Siempre intentamos movernos un poco hacia el centro para compactar
            dx_center = 0.0 - float(tree.center_x)
            dy_center = 0.0 - float(tree.center_y)
            dist_center = math.sqrt(dx_center*dx_center + dy_center*dy_center)
            
            if dist_center > 0.001:
                # Componente de atracción (10% de la magnitud hacia el centro)
                pull_x = (dx_center / dist_center) * mag * 0.1
                pull_y = (dy_center / dist_center) * mag * 0.1
            else:
                pull_x, pull_y = 0, 0
                
            # Movimiento aleatorio + atracción
            dx = random.uniform(-mag, mag) + pull_x
            dy = random.uniform(-mag, mag) + pull_y

            
        
        # Aplicar movimiento
        new_x = Decimal(float(tree.center_x) + dx)
        new_y = Decimal(float(tree.center_y) + dy)
        
        # Verificar límites de la jaula (si está activa)
        if self.cage_bounds:
            min_x, min_y, max_x, max_y = self.cage_bounds
            
            # Si el movimiento saca al árbol de la jaula, revertir o ajustar
            # Ajuste simple: clampear al borde
            if float(new_x) < min_x: new_x = Decimal(min_x)
            if float(new_x) > max_x: new_x = Decimal(max_x)
            if float(new_y) < min_y: new_y = Decimal(min_y)
            if float(new_y) > max_y: new_y = Decimal(max_y)
            
        tree.center_x = new_x
        tree.center_y = new_y
        tree.angle = Decimal(float(tree.angle) + random.uniform(-self.config.angle_noise_range, self.config.angle_noise_range))
        
        tree.update_polygon()
