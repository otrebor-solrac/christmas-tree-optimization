import math
import random
import copy
from decimal import Decimal
from shapely.geometry import box
from shapely.ops import unary_union

# Asumo que importas esto de tu entorno, NO toco el Config
from .packing import PackingAlgorithm
from .config import GravityConfig

class GravityCompactor:
    """
    Algorithm that compacts the layout by pulling trees towards the center of mass.
    Simulates a gravitational pull with stochastic noise and teleportation for local optima escape.
    """
    def __init__(self, scale_factor, config: GravityConfig = None):
        self.scale_factor = scale_factor
        self.config = config if config else GravityConfig()
        # Use PackingAlgorithm for collision detection and scoring helpers
        self.solver = PackingAlgorithm(scale_factor)

    def compact(self, trees, visualizer=None, strategy='closest_first'):
        """
        Executes the gravity compaction process.
        
        Args:
            trees: List of tree objects.
            visualizer: Visualization object (optional).
            strategy (str): 'farthest_first' (original) or 'closest_first' (layers from center).
        """
        self.solver.trees = trees
        initial_score = self.solver.evaluator.calculate_kaggle_score(trees)
        
        best_trees = copy.deepcopy(trees)
        best_score = initial_score
        improvements = 0
        
        if visualizer and self.config.visualize:
            self.solver._init_visualization(visualizer, save_frames=True)

        for step in range(self.config.steps):
            # 1. Calculate Center of Mass (CoM)
            com_x, com_y = self._calc_center_of_mass(self.solver.trees)
            
            # 2. Sort trees by distance to CoM
            tree_distances = []
            for i, tree in enumerate(self.solver.trees):
                dx = float(tree.center_x) - com_x
                dy = float(tree.center_y) - com_y
                dist = math.sqrt(dx*dx + dy*dy)
                tree_distances.append((dist, i))
            
            # --- MODIFICACIÓN AQUÍ ---
            if strategy == 'closest_first':
                # Ordenar ascendente: Mueve primero los cercanos al centro (capas internas)
                tree_distances.sort(key=lambda x: x[0], reverse=False)
            else:
                # Ordenar descendente: Mueve primero los lejanos (original)
                tree_distances.sort(key=lambda x: x[0], reverse=True)
            # -------------------------
            
            # 3. Move trees
            for dist, i in tree_distances:
                tree = self.solver.trees[i]
                
                dx = com_x - float(tree.center_x)
                dy = com_y - float(tree.center_y)
                dist_to_com = math.sqrt(dx*dx + dy*dy)
                
                if dist_to_com < 0.001: continue
                
                # Calculate movement vector with noise
                angle_to_com = math.atan2(dy, dx)
                noise = random.uniform(-self.config.noise_range, self.config.noise_range)
                noisy_angle = angle_to_com + noise
                
                move_x = math.cos(noisy_angle) * self.config.step_size
                move_y = math.sin(noisy_angle) * self.config.step_size
                
                # Store old state
                old_state = (tree.center_x, tree.center_y, tree.angle)
                
                # Apply movement
                tree.center_x += Decimal(move_x)
                tree.center_y += Decimal(move_y)
                tree.update_polygon()
                
                # Check collision
                others = self.solver.trees[:i] + self.solver.trees[i+1:]
                if self.solver._check_strict_collision(tree, others):
                    # Revert
                    tree.center_x, tree.center_y, tree.angle = old_state
                    tree.update_polygon()
                    
                    # Try rotation to fit
                    self._try_rotation_escape(tree, others, old_state[2])

            # 4. Teleportation Strategy
            if self.config.enable_teleport and step % self.config.teleport_interval == 0:
                self._teleport_farthest_tree(com_x, com_y)

            # 5. Evaluate
            current_score = self.solver.evaluator.calculate_kaggle_score(self.solver.trees)
            if current_score < best_score:
                best_score = current_score
                best_trees = copy.deepcopy(self.solver.trees)
                improvements += 1
            
            # 6. Visualize
            if visualizer and self.config.visualize and step % self.config.viz_step == 0:
                self.solver._update_visualization(visualizer, step, self.config.steps, 1.0, 
                                                context_info=f"Gravity Step {step} ({strategy})")

        print(f"   ✓ Gravity Compaction ({strategy}): {improvements} improvements. {initial_score:.6f} -> {best_score:.6f}")
        return best_trees

    def _calc_center_of_mass(self, tree_list):
        if not tree_list: return 0.0, 0.0
        cx = sum(float(t.center_x) for t in tree_list) / len(tree_list)
        cy = sum(float(t.center_y) for t in tree_list) / len(tree_list)
        return cx, cy

    def _try_rotation_escape(self, tree, others, old_angle):
        angles_to_try = [a for a in range(-45, 46, 5) if a != 0]
        random.shuffle(angles_to_try)
        
        for angle_delta in angles_to_try:
            tree.angle = Decimal(float(old_angle) + angle_delta)
            tree.update_polygon()
            if not self.solver._check_strict_collision(tree, others):
                return True
        
        tree.angle = old_angle
        tree.update_polygon()
        return False

    def _teleport_farthest_tree(self, com_x, com_y):
        if len(self.solver.trees) < 2: return

        # Find farthest
        farthest_idx = -1
        max_dist = -1
        for i, tree in enumerate(self.solver.trees):
            dist = math.hypot(float(tree.center_x) - com_x, float(tree.center_y) - com_y)
            if dist > max_dist:
                max_dist = dist
                farthest_idx = i
        
        if farthest_idx == -1: return
        
        target_tree = self.solver.trees[farthest_idx]
        old_state = (target_tree.center_x, target_tree.center_y, target_tree.angle)
        
        # Calculate empty space
        others = self.solver.trees[:farthest_idx] + self.solver.trees[farthest_idx+1:]
        others_polys = [t.polygon for t in others]
        others_union = unary_union(others_polys)
        
        if others_union.is_empty:
            return
            
        minx, miny, maxx, maxy = others_union.bounds
        
        if any(math.isnan(x) for x in (minx, miny, maxx, maxy)):
            return
            
        bbox = box(minx, miny, maxx, maxy)
        
        try:
            empty_space = bbox.difference(others_union)
        except Exception:
            return

        # Try to place in empty space
        if empty_space.is_empty: return
        
        # Simplified: Try centroid of empty space
        sf = float(self.scale_factor)
        target_tree.center_x = Decimal(empty_space.centroid.x / sf)
        target_tree.center_y = Decimal(empty_space.centroid.y / sf)
        target_tree.update_polygon()
        
        if self.solver._check_strict_collision(target_tree, others):
            target_tree.center_x, target_tree.center_y, target_tree.angle = old_state
            target_tree.update_polygon()