"""
Gradual compaction tuning - moves trees slowly toward center of mass.
Usage: python tune_gradual.py <solution_id> [steps]
Example: python tune_gradual.py 10 1000
"""
import sys
import math
import copy
import random
from pathlib import Path
from decimal import Decimal
from shapely.geometry import box
from shapely.ops import unary_union

root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import SCALE_FACTOR
from app.visualization import TreeVisualizer
from app.algorithms import PackingAlgorithm
from app.utils.io import load_solution, save_solution

def gradual_compaction(solution_id, steps=1000, step_size=0.01, visualize=True):
    """
    Compacta gradualmente moviendo árboles hacia el centro de masa.
    
    Args:
        solution_id: ID de la solución
        steps: Número de pasos de compactación
        step_size: Tamaño del paso (distancia a mover por iteración)
        visualize: Si mostrar visualización
    """
    solution_file = f"solutions/T{solution_id}.csv"
    
    print(f"\n{'='*60}")
    print(f"COMPACTACIÓN GRADUAL: T{solution_id}")
    print(f"{'='*60}\n")
    
    # Cargar solución
    print(f"-> Cargando {solution_file}...")
    try:
        trees = load_solution(solution_file)
        print(f"-> Cargados {len(trees)} árboles")
    except FileNotFoundError:
        print(f"✗ Error: No se encontró {solution_file}")
        return
    
    # Inicializar solver
    solver = PackingAlgorithm(SCALE_FACTOR)
    solver.set_initial_state(trees)
    
    initial_score = solver.evaluator.calculate_kaggle_score(trees)
    print(f"-> Score inicial: {initial_score:.6f}")
    
    # Calcular centro de masa inicial
    def calc_center_of_mass(tree_list):
        if not tree_list:
            return 0.0, 0.0
        cx = sum(float(t.center_x) for t in tree_list) / len(tree_list)
        cy = sum(float(t.center_y) for t in tree_list) / len(tree_list)
        return cx, cy
    
    best_trees = copy.deepcopy(trees)
    best_score = initial_score
    
    print(f"\n-> Iniciando compactación gradual ({steps} pasos, step_size={step_size})...")
    print("   Estrategia: mover hacia centro de masa, rotar si hay colisión\n")
    
    # Visualización
    viz = TreeVisualizer(SCALE_FACTOR) if visualize else None
    if viz:
        # Calcular límites iniciales fijos para que la cámara no se mueva
        all_polys = [t.polygon for t in trees]
        from shapely.ops import unary_union
        minx, miny, maxx, maxy = unary_union(all_polys).bounds
        
        # Agregar margen del 20%
        width = maxx - minx
        height = maxy - miny
        margin = max(width, height) * 0.2
        
        fixed_xlim = (minx - margin, maxx + margin)
        fixed_ylim = (miny - margin, maxy + margin)
        
        solver._init_visualization(viz, save_frames=True)
        
        # Fijar límites en el eje
        # solver._viz_ax.set_xlim(fixed_xlim)
        # solver._viz_ax.set_ylim(fixed_ylim)
    
    improvements = 0
    
    for step in range(steps):
        # Calcular centro de masa actual
        com_x, com_y = calc_center_of_mass(solver.trees)
        
        # Ordenar árboles por distancia al centro de masa (de más lejos a más cerca)
        # Esto prioriza mover los árboles externos hacia adentro para reducir el bounding box
        tree_distances = []
        for i, tree in enumerate(solver.trees):
            dx = float(tree.center_x) - com_x
            dy = float(tree.center_y) - com_y
            dist = math.sqrt(dx*dx + dy*dy)
            tree_distances.append((dist, i))
        
        # Ordenar descendente (los más lejanos primero)
        tree_distances.sort(key=lambda x: x[0], reverse=True)
        
        # Intentar mover cada árbol hacia el centro de masa (en orden de distancia)
        moved = False
        
        for dist, i in tree_distances:
            tree = solver.trees[i]
            
            # Vector hacia el centro de masa
            dx = com_x - float(tree.center_x)
            dy = com_y - float(tree.center_y)
            # dist ya lo tenemos, pero recalculamos por claridad/precisión si cambió algo
            dist_to_com = math.sqrt(dx*dx + dy*dy)
            
            if dist_to_com < 0.001:  # Ya está en el centro
                continue
            
            # Normalizar y escalar por step_size
            # AUMENTAR RUIDO: +/- 0.8 radianes (~45 grados) para explorar más
            angle_to_com = math.atan2(dy, dx)
            noise = random.uniform(-0.8, 0.8) 
            noisy_angle = angle_to_com + noise
            
            move_x = math.cos(noisy_angle) * step_size
            move_y = math.sin(noisy_angle) * step_size
            
            # Guardar posición original
            old_x = tree.center_x
            old_y = tree.center_y
            old_angle = tree.angle
            
            # Mover árbol (INTENTO 1: Hacia el centro con ruido)
            tree.center_x = Decimal(float(tree.center_x) + move_x)
            tree.center_y = Decimal(float(tree.center_y) + move_y)
            tree.update_polygon()
            
            # Verificar colisión
            others = solver.trees[:i] + solver.trees[i+1:]
            has_collision = solver._check_strict_collision(tree, others)
            
            if has_collision:
                # Revertir movimiento
                tree.center_x = old_x
                tree.center_y = old_y
                tree.update_polygon()
                
                # Intentar rotar agresivamente para desbloquear (±45°)
                angles_to_try = [a for a in range(-45, 46, 5) if a != 0]
                # Barajar para no probar siempre lo mismo
                random.shuffle(angles_to_try)
                
                for angle_delta in angles_to_try:
                    tree.angle = Decimal(float(old_angle) + angle_delta)
                    tree.update_polygon()
                    
                    if not solver._check_strict_collision(tree, others):
                        moved = True
                        break
                else:
                    # No se pudo rotar, revertir rotación
                    tree.angle = old_angle
                    tree.update_polygon()
                    
                    # ESTRATEGIA JITTER (VIBRACIÓN):
                    # Si no podemos ir al centro ni rotar, intentar un movimiento 
                    # puramente aleatorio (Brownian motion) para "desatascar"
                    # Probabilidad del 20% para no hacerlo siempre
                    if random.random() < 0.2:
                        jitter_angle = random.uniform(0, 2*math.pi)
                        jitter_x = math.cos(jitter_angle) * step_size
                        jitter_y = math.sin(jitter_angle) * step_size
                        
                        tree.center_x = Decimal(float(old_x) + jitter_x)
                        tree.center_y = Decimal(float(old_y) + jitter_y)
                        tree.update_polygon()
                        
                        if not solver._check_strict_collision(tree, others):
                            moved = True
                        else:
                            # Revertir todo si falla también el jitter
                            tree.center_x = old_x
                            tree.center_y = old_y
                            tree.update_polygon()
                    else:
                        # Revertir todo
                        tree.center_x = old_x
                        tree.center_y = old_y
                        tree.update_polygon()
            else:
                moved = True
        
        # Calcular score actual
        current_score = solver.evaluator.calculate_kaggle_score(solver.trees)
        
        # Si mejoró, guardar
        if current_score < best_score:
            best_score = current_score
            best_trees = copy.deepcopy(solver.trees)
            improvements += 1
        
        # Visualizar cada 20 pasos
        if viz and step % 20 == 0:
            # Forzar re-cálculo de límites basado en los datos actuales
            solver._viz_ax.relim()
            solver._viz_ax.autoscale_view()
            
            # Asegurar aspecto igual para no deformar
            solver._viz_ax.set_aspect('equal', adjustable='box')
            solver._update_visualization(viz, step, steps, 1.0, 
                                        context_info=f"Step {step}/{steps} (Best: {best_score:.4f})")

        
        # Mostrar progreso cada 100 pasos
        if step % 100 == 0 and step > 0:
            improvement_pct = ((initial_score - current_score) / initial_score) * 100
            print(f"   Paso {step}/{steps}: Score={current_score:.6f} "
                  f"(mejora: {improvement_pct:.2f}%, mejoras: {improvements})")

        # ESTRATEGIA DE TELEPORTACIÓN (Salto al Hueco)
        # Cada 50 pasos, intentar mover el árbol más lejano a un hueco interior
        if step % 50 == 0:
            teleport_farthest_tree(solver, com_x, com_y)


    
    # Restaurar mejor solución
    solver.trees = best_trees
    final_score = best_score
    
    improvement = ((initial_score - final_score) / initial_score) * 100
    
    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")
    print(f"Mejoras encontradas: {improvements}")
    
    if final_score < initial_score:
        print(f"✓ MEJORA: {improvement:.2f}% mejor")
        
        backup_file = f"solutions/T{solution_id}_backup.csv"
        print(f"\n-> Guardando backup en {backup_file}")
        save_solution(backup_file, trees, initial_score)
        
        print(f"-> Sobrescribiendo {solution_file}")
        save_solution(solution_file, best_trees, final_score)
    else:
        print("= SIN MEJORA")
    
    print(f"{'='*60}\n")
    
    # Visualizar resultado
    if visualize:
        print("-> Mostrando visualización final...")
        viz = TreeVisualizer(SCALE_FACTOR)
        viz.plot(best_trees, check_collisions=solver.detect_collisions)

def teleport_farthest_tree(solver, com_x, com_y):
    """
    Intenta teletransportar el árbol más lejano a un hueco geométrico (diferencia de polígonos).
    """
    # 0. Verificar si hay suficientes árboles
    if len(solver.trees) < 2:
        return

    # 1. Encontrar el árbol más lejano
    max_dist = -1
    farthest_idx = -1
    
    for i, tree in enumerate(solver.trees):
        dx = float(tree.center_x) - com_x
        dy = float(tree.center_y) - com_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > max_dist:
            max_dist = dist
            farthest_idx = i
            
    if farthest_idx == -1: return
    
    target_tree = solver.trees[farthest_idx]
    old_x, old_y = target_tree.center_x, target_tree.center_y
    old_angle = target_tree.angle
    
    # 2. Calcular espacio vacío (Bounding Box - Unión de otros árboles)
    others = solver.trees[:farthest_idx] + solver.trees[farthest_idx+1:]
    others_polys = [t.polygon for t in others]
    others_union = unary_union(others_polys)
    
    # Usar el bounding box actual como límite
    minx, miny, maxx, maxy = others_union.bounds
    # Un poco más pequeño para forzar compactación? No, usemos el actual.
    bbox = box(minx, miny, maxx, maxy)
    
    try:
        empty_space = bbox.difference(others_union)
    except Exception as e:
        print(f"   [TELEPORT] Error calculando diferencia: {e}")
        return

    candidates = []
    if empty_space.geom_type == 'Polygon':
        candidates = [empty_space]
    elif empty_space.geom_type == 'MultiPolygon':
        candidates = list(empty_space.geoms)
        
    # Filtrar candidatos muy pequeños (área menor al árbol)
    tree_area = target_tree.polygon.area
    valid_candidates = []
    for poly in candidates:
        if poly.area >= tree_area * 0.8: # Un poco de tolerancia
            # Calcular distancia del centro del hueco al centro de masa global
            p_x, p_y = poly.centroid.x, poly.centroid.y
            dx = p_x - com_x
            dy = p_y - com_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Solo considerar si está más cerca que el árbol actual (con margen significativo)
            if dist < max_dist * 0.8:
                valid_candidates.append((dist, poly))
    
    # Ordenar por cercanía al centro
    valid_candidates.sort(key=lambda x: x[0])
    
    found_spot = False
    
    # 3. Probar colocar el árbol en los huecos candidatos
    for dist_hole, poly in valid_candidates:
        # Puntos a probar: Centroide y punto representativo
        test_points = [poly.centroid, poly.representative_point()]
        random.shuffle(test_points) # Probar puntos en orden aleatorio
        
        for point in test_points:
            target_tree.center_x = Decimal(point.x)
            target_tree.center_y = Decimal(point.y)
            
            # Probar rotaciones
            angles_to_try = [0, 45, 90, 135, 180, 225, 270, 315]
            random.shuffle(angles_to_try) # Barajar rotaciones
            
            for angle in angles_to_try:
                target_tree.angle = Decimal(angle)
                target_tree.update_polygon()
                
                # Verificar si está contenido en el hueco (rápido)
                # if poly.contains(target_tree.polygon): ... (demasiado estricto a veces)
                
                # Verificar colisión real (lento pero seguro)
                if not solver._check_strict_collision(target_tree, others):
                    found_spot = True
                    print(f"   [TELEPORT] Árbol {target_tree.id} saltó de dist {max_dist:.2f} a {dist_hole:.2f}!")
                    break
            if found_spot: break
        if found_spot: break
        
    if not found_spot:
        # Restaurar
        target_tree.center_x = old_x
        target_tree.center_y = old_y
        target_tree.angle = old_angle
        target_tree.update_polygon()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tune_gradual.py <solution_id> [steps] [step_size]")
        print("Ejemplo: python tune_gradual.py 10")
        print("         python tune_gradual.py 10 2000 0.02")
        sys.exit(1)
    
    solution_id = int(sys.argv[1])
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    step_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    
    gradual_compaction(solution_id, steps=steps, step_size=step_size)
