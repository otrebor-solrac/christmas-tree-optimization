"""
Packing algorithm module using simulated annealing.
"""
import copy
import math
import random
from decimal import Decimal

from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..models.tree import ChristmasTree
from ..evaluation.cost_evaluator import CostEvaluator


class PackingAlgorithm:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.trees = []
        self.evaluator = CostEvaluator(scale_factor)
        # best_score guardará el ÁREA (Kaggle Score)
        self.best_score = float('inf')
        # Referencias para visualización reutilizable
        self._viz_fig = None
        self._viz_ax = None
        self._viz_initialized = False
        # Estado de compactación
        self._compaction_mode_active = False
        self._max_allowed_area = float('inf')
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

    def _update_visualization(self, visualizer, step, iterations, temp, context_info=""):
        """
        Actualiza la visualización en tiempo real.
        
        Args:
            visualizer: Instancia de TreeVisualizer
            step: Iteración actual
            iterations: Total de iteraciones
            temp: Temperatura actual
            context_info: Información adicional para el título (opcional)
        """
        if visualizer is None or not self._viz_initialized or self._viz_fig is None or self._viz_ax is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Calcular score actual
            current_score = self.evaluator.calculate_kaggle_score(self.trees)
            
            # Limpiar el eje para redibujar
            self._viz_ax.clear()
            
            # Dibujar árboles
            for i, tree in enumerate(self.trees):
                x, y = visualizer._desescalar_coords(tree.polygon)
                color = visualizer.cmap(i % 10)
                self._viz_ax.fill(x, y, alpha=0.8, fc=color, ec='black', lw=0.5)
            
            # Dibujar bounding box (sin ajustar límites automáticamente)
            side = visualizer._dibujar_bounding_box(self._viz_ax, self.trees, adjust_limits=False)
            
            # Fijar límites de coordenadas: -1 a 8 en ambos ejes
            self._viz_ax.set_xlim(-1, 8)
            self._viz_ax.set_ylim(-1, 8)
            
            # Verificar colisiones
            has_collisions, collision_pairs = self.detect_collisions(self.trees)
            
            # Título
            n = len(self.trees)
            area = side * side
            kaggle_metric = area / n if n > 0 else 0
            title = f"Optimización: {step}/{iterations} | Temp: {temp:.3f}\n"
            title += f"Área: {current_score:.4f} | Métrica: {kaggle_metric:.4f}"
            if context_info:
                title += f"\n{context_info}"
            if has_collisions:
                title += f"\n⚠️ COLISIÓN! Pares: {collision_pairs}"
                self._viz_ax.set_title(title, fontsize=10, color='red', weight='bold')
            else:
                self._viz_ax.set_title(title, fontsize=10)
            
            self._viz_ax.set_aspect('equal')
            self._viz_ax.grid(True, alpha=0.3, linestyle=':')
            
            # Actualizar la figura existente
            self._viz_fig.canvas.draw()
            self._viz_fig.canvas.flush_events()
            
            # Guardar frame si está habilitado
            if self._frames_dir is not None:
                frame_path = self._frames_dir / f"frame_{self._frame_counter:05d}.png"
                self._viz_fig.savefig(frame_path, dpi=100, bbox_inches='tight')
                self._frame_counter += 1
            
            plt.pause(0.01)  # Pausa muy corta para actualizar
        except Exception as e:
            # Si hay error en visualización, continuar sin ella
            print(f"   [Visualización] Error: {e}")

    def set_initial_state(self, existing_trees):
        if existing_trees:
            self.trees = copy.deepcopy(existing_trees)
            self._refresh_current_score()
        else:
            self.trees = []
            self.best_score = float('inf')

    def detect_collisions(self, trees):
        """
        Detecta colisiones entre árboles. Método público reutilizable.
        
        Args:
            trees: Lista de árboles a verificar
            
        Returns:
            tuple: (has_collisions: bool, collision_pairs: list)
                   collision_pairs contiene tuplas (tree_id1, tree_id2) de árboles en colisión
        """
        if not trees or len(trees) < 2:
            return False, []
        
        polys = [t.polygon for t in trees]
        tree_index = STRtree(polys)
        tolerance_area = (float(self.scale_factor) ** 2) * 1e-9
        collision_pairs = []
        checked_pairs = set()
        
        for i, t in enumerate(trees):
            candidates = tree_index.query(t.polygon)
            for c_idx in candidates:
                if c_idx == i:
                    continue
                
                # Evitar duplicados (i, j) y (j, i)
                pair = tuple(sorted([i, c_idx]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Verificar intersección real
                inter = t.polygon.intersection(polys[c_idx])
                if inter.area > tolerance_area:
                    collision_pairs.append((trees[i].id, trees[c_idx].id))
        
        return len(collision_pairs) > 0, collision_pairs

    def _refresh_current_score(self):
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
        static_polys = [t.polygon for t in static_trees]
        tree_index = STRtree(static_polys)
        candidate_indices = tree_index.query(new_tree.polygon)
        tolerance_area = (float(self.scale_factor) ** 2) * 1e-9

        for i in candidate_indices:
            if new_tree.polygon.intersection(static_polys[i]).area > tolerance_area:
                return True
        return False

    def add_configured_tree(self, tree_id, x, y, angle, fixed=True):
        """Agrega árbol manual."""
        new_tree = ChristmasTree(tree_id, x, y, angle, fixed=fixed)
        self.trees.append(new_tree)
        print(f"-> Manual: Árbol {tree_id} en ({x}, {y}) [Fixed={fixed}]")
        self._refresh_current_score()
    
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
        stride_x = 0.42       # Ancho efectivo
        row_height = 0.85     # Alto efectivo (Nesting vertical)
        
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

    def _strategy_random_fallback(self, tree_id):
        """ESTRATEGIA 3: Fallback aleatorio si todo falla."""
        polys = [t.polygon for t in self.trees]
        minx, miny, maxx, maxy = unary_union(polys).bounds
        sf = float(self.scale_factor)
        # Intentar ponerlo arriba a la derecha
        return ChristmasTree(tree_id, maxx/sf, maxy/sf, random.uniform(0, 360), fixed=False)

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
                found = self._run_annealing(iterations=2000, initial_temp=5.0, mag_factor=1.0, 
                                          visualizer=visualizer, allow_area_growth=False)
                if found:
                    print("   ¡Optimización exitosa!")
                else:
                    print("   Advertencia: No se encontró una solución válida.")
            
            self._refresh_current_score()
            return self.trees

        start_id = max([t.id for t in self.trees] + [0]) + 1
        
        # Seleccionar la función generadora dinámicamente
        if strategy == "linear": generator = self._strategy_linear_zipper
        elif strategy == "square": generator = self._strategy_square_packing
        else: generator = self._strategy_random_fallback

        print(f"\n>>> INICIANDO INSERCIÓN (Estrategia: {strategy}) <<<")

        # Inicializar visualización si se proporciona (guardar frames para video)
        if visualizer is not None:
            self._init_visualization(visualizer, save_frames=True)

        for i in range(start_id, start_id + n_new):
            print(f"\n   -> Insertando Árbol {i}...")
            success = False
            attempts = 0
            max_attempts = 5
            safe_state = copy.deepcopy(self.trees)
            
            # Visualizar estado inicial antes de insertar
            if visualizer is not None:
                self._update_visualization(visualizer, 0, 1, 0, context_info=f"Insertando Árbol {i}...")

            while not success and attempts < max_attempts:
                attempts += 1
                
                # 1. Generar Posición (Intento 1: Estrategia, Intento >1: Random)
                if attempts == 1:
                    new_tree = generator(i)
                    # Annealing SUAVE (Ajuste fino)
                    temp, mag = 2.0, 0.1
                    iters = 500
                else:
                    new_tree = self._strategy_random_fallback(i)
                    # Annealing FUERTE (Búsqueda global)
                    temp, mag = 80.0, 3.0
                    iters = 2000

                self.trees.append(new_tree)

                # 2. Optimizar (permitir crecimiento de área al insertar nuevo árbol)
                found = self._run_annealing(iterations=iters, initial_temp=temp, mag_factor=mag, 
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
                
                if not success:
                    self.trees = copy.deepcopy(safe_state)
                    # Visualizar fallo
                    if visualizer is not None:
                        self._update_visualization(visualizer, iters, iters, temp, 
                                                  context_info=f"Intento {attempts} fallido para Árbol {i}")
            
            if not success:
                print(f"ERROR: No se pudo insertar Árbol {i}.")
                self._refresh_current_score()
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
        
        # Calcular área del cuadrado bounding box
        square_area = self.evaluator.calculate_kaggle_score(self.trees)
        
        # Calcular área total de los árboles
        trees_area = self._calculate_total_tree_area()
        
        if trees_area <= 0:
            self._compaction_mode_active = False
            self._max_allowed_area = float('inf')
            return False
        
        ratio = square_area / trees_area
        
        # Lógica de activación/desactivación
        if ratio >= 2.0:
            # Activar modo compactación
            if not self._compaction_mode_active:
                # Primera vez que se activa: establecer el área máxima actual
                self._compaction_mode_active = True
                self._max_allowed_area = square_area
                print(f"   [Compactación] ACTIVADO: Ratio={ratio:.3f}, Área máx={self._max_allowed_area:.4f}")
        elif ratio <= 1.5:
            # Desactivar modo compactación
            if self._compaction_mode_active:
                self._compaction_mode_active = False
                self._max_allowed_area = float('inf')
                print(f"   [Compactación] DESACTIVADO: Ratio={ratio:.3f}")
        # Entre 1.5 y 2.0: mantener el estado actual (no hacer nada)
        
        return self._compaction_mode_active

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

    def _run_annealing(self, iterations, initial_temp, mag_factor, visualizer=None, allow_area_growth=False):
        """
        Annealing parametrizable.
        mag_factor: 0.2 para ajuste fino, 3.0 para búsqueda caos.
        visualizer: Opcional, para visualizar el progreso durante la optimización.
        allow_area_growth: Si True, permite que el área crezca (útil al insertar nuevos árboles).
        """
        mutable_indices = [i for i, t in enumerate(self.trees) if not t.fixed]
        if not mutable_indices: return True
        
        # Calcular centro de masa de árboles fijos (si hay) o de todos
        fixed_indices = [i for i, t in enumerate(self.trees) if t.fixed]
        if fixed_indices:
            # Si hay árboles fijos, usar su centro de masa como referencia
            com_x, com_y = self._calculate_center_of_mass(fixed_indices)
        else:
            # Si no hay fijos, usar el centro de masa de todos
            com_x, com_y = self._calculate_center_of_mass()
        
        reference_point = (com_x, com_y)

        temp = initial_temp
        alpha = 0.95
        min_temp = 0.005
        current_penalty_weight = 50.0
        
        local_best_layout = None
        local_best_score = float('inf') # Buscamos minimizar Area (sin colisiones)
        best_area_layout = None
        best_area_score = float('inf') # Mejor área encontrada (puede tener colisiones menores)
        
        current_energy = self.evaluator.get_total_energy(self.trees, current_penalty_weight)
        initial_energy = current_energy
        # Inicializar con el área actual
        initial_area = self.evaluator.calculate_kaggle_score(self.trees)
        best_area_score = initial_area
        best_area_layout = copy.deepcopy(self.trees)
        
        # Actualizar estado de compactación
        self._update_compaction_mode()

        # Configurar visualización periódica
        viz_interval = max(1, iterations // 20)  # Mostrar ~20 frames durante el proceso
        if visualizer is not None:
            self._init_visualization(visualizer, save_frames=True)

        for step in range(iterations):
            progress = step / iterations
            
            # Seleccionar árbol más alejado del centro de masa (en lugar de aleatorio)
            idx = self._select_farthest_tree(mutable_indices, reference_point)
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

            # Actualizar modo de compactación dinámicamente
            compaction_mode = self._update_compaction_mode()
            
            # Si estamos en modo compactación, actualizar el área máxima permitida (solo reducir, nunca aumentar)
            # PERO solo si no estamos permitiendo crecimiento (allow_area_growth=False)
            if compaction_mode and not allow_area_growth:
                self._max_allowed_area = min(self._max_allowed_area, new_square_area)
            elif allow_area_growth:
                # Si estamos permitiendo crecimiento, actualizar el límite para permitir el nuevo tamaño
                if compaction_mode:
                    self._max_allowed_area = max(self._max_allowed_area, new_square_area)

            # Penalización por violar el límite de compactación
            compaction_penalty = 0.0
            if compaction_mode and not allow_area_growth and new_square_area > self._max_allowed_area:
                # Penalización muy grande si se excede el área máxima permitida
                excess_area = new_square_area - self._max_allowed_area
                compaction_penalty = 1000000.0 * (1.0 + excess_area * 10.0)

            # Penalización dinámica
            pen_weight = 50 * (1 + progress * 200)
            strict_pen = 100000.0 if is_strict else 0
            
            base_energy = self.evaluator.get_total_energy(self.trees, pen_weight)
            new_energy = base_energy + strict_pen + compaction_penalty
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
                    
                    # Guardar el mejor layout sin colisiones (overlap muy bajo)
                    if new_overlap < 1e-4 and kaggle_score < local_best_score:
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

        # Priorizar layout sin colisiones, pero si no existe, usar el mejor encontrado
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
            # Comportamiento original: gravedad hacia (0,0)
            if random.random() < 0.6:
                dx = random.uniform(-mag, mag * 0.2)
                dy = random.uniform(-mag, mag * 0.2)
            else:
                dx = random.uniform(-mag, mag)
                dy = random.uniform(-mag, mag)
            
        tree.center_x = Decimal(float(tree.center_x) + dx)
        tree.center_y = Decimal(float(tree.center_y) + dy)
        tree.angle = Decimal(float(tree.angle) + random.uniform(-5, 5))
        
        if tree.center_x < 0: tree.center_x = Decimal('0')
        if tree.center_y < 0: tree.center_y = Decimal('0')
        tree.update_polygon()