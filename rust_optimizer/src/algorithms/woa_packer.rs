use crate::entities::Tree;
use crate::utils::geometry::{calculate_kaggle_score, calculate_overlap_area};
use crate::utils::io::save_if_better;
use crate::algorithms::soft_sa::{refine_soft_sa, SoftSARefineConfig};
use rand::Rng;
use std::path::Path;
use rayon::prelude::*;

#[derive(Clone)]
struct Whale {
    trees: Vec<Tree>,
    cost: f64,
}

impl Whale {
    fn new(trees: Vec<Tree>) -> Self {
        let mut w = Whale {
            trees,
            cost: 0.0,
        };
        w.evaluate();
        w
    }

    fn evaluate(&mut self) {
        let overlap = calculate_overlap_area(&self.trees);
        let kaggle = calculate_kaggle_score(&self.trees);
        
        // Penalización de gravedad al centro para compactación
        let gravity: f64 = self.trees.iter()
            .map(|t| (t.x.powi(2) + t.y.powi(2)).sqrt())
            .sum::<f64>() / (self.trees.len() as f64) * 0.05;

        // El costo para la población de ballenas usa el área de solapamiento (soft)
        self.cost = kaggle + (overlap * 5000.0) + gravity;
    }
}

pub fn optimize_with_woa(
    trees: Vec<Tree>,
    max_iters: usize,
    pop_size: usize,
    output_path: &Path,
) -> Vec<Tree> {
    let n_trees = trees.len();
    
    // Configuración Híbrida: Island Model
    let migration_interval = 10;
    let island_steps_per_iter = 500; // Pasos SoftSA por ballena por iteración

    println!("🐋 [Parallel WOA] Iniciando Island Model (N={}, Pop={}, Iters={})", n_trees, pop_size, max_iters);

    // 1. Inicialización de Población (Islas)
    let mut population: Vec<Whale> = (0..pop_size).into_par_iter().map(|i| {
        let mut rng = rand::thread_rng();
        if i == 0 && !trees.is_empty() {
            Whale::new(trees.clone())
        } else {
            let mut current_trees = if trees.is_empty() {
                let side = (n_trees as f64).sqrt() * 1.5;
                (0..n_trees).map(|id| {
                    let mut t = Tree::new(id, rng.gen_range(-side..side), rng.gen_range(-side..side), rng.gen_range(0.0..360.0));
                    t.update_poly();
                    t
                }).collect()
            } else {
                trees.clone()
            };

            for t in &mut current_trees {
                t.x += rng.gen_range(-2.0..2.0);
                t.y += rng.gen_range(-2.0..2.0);
                t.angle += rng.gen_range(-45.0..45.0);
                t.update_poly();
            }
            Whale::new(current_trees)
        }
    }).collect();

    let mut leader = population.iter().min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap()).unwrap().clone();

    for t in 0..max_iters {
        // A. Evolución Paralela Independiente (Island Mode)
        // Cada ballena corre un Soft SA local para mejorar su propia posición
        population.par_iter_mut().for_each(|whale| {
            let sa_config = SoftSARefineConfig {
                iterations: island_steps_per_iter,
                initial_temp: 0.1, // Temp moderada para micromejoras
                cooling_rate: 0.95,
                overlap_penalty: 5000.0,
                step_scale: 0.1,
            };
            
            let improved_trees = refine_soft_sa(whale.trees.clone(), sa_config);
            let improved_whale = Whale::new(improved_trees);
            
            if improved_whale.cost < whale.cost {
                *whale = improved_whale;
            }
        });

        // B. Migración (Sincronización Genética)
        // Cada cierto tiempo, reemplazamos las peores islas con clones mutados de las mejores
        if t % migration_interval == 0 {
            // Ordenar por costo (menor es mejor)
            population.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
            
            let best_half_count = pop_size / 2;
            let worst_half_count = pop_size - best_half_count;

            // Clonamos los mejores para usarlos como base de mutación
            let best_whales: Vec<Whale> = population.iter().take(best_half_count).cloned().collect();

            // Reemplazamos la mitad inferior
            // (Nota: Esto es una operación secuencial rápida, no cuello de botella)
            for i in 0..worst_half_count {
                let parent = &best_whales[i % best_half_count]; // Round-robin de los mejores
                let mut child_trees = parent.trees.clone();
                
                // Mutación agresiva para que la nueva isla explore otra zona cercana
                let mut rng = rand::thread_rng();
                for tree in &mut child_trees {
                    if rng.gen_bool(0.3) {
                        tree.x += rng.gen_range(-0.5..0.5);
                        tree.y += rng.gen_range(-0.5..0.5);
                        tree.angle += rng.gen_range(-10.0..10.0);
                        tree.update_poly();
                    }
                }
                
                // Índice en population es: best_half_count + i
                population[best_half_count + i] = Whale::new(child_trees);
            }
        }

        // C. Actualizar Líder Global y Guardar Récords
        let current_best = population.iter().min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap()).unwrap();
        if current_best.cost < leader.cost {
            leader = current_best.clone();
            
            let improved_trees = leader.trees.clone();
            let overlap = calculate_overlap_area(&improved_trees);
            
            // Si el solapamiento es virtualmente cero, intentamos Pulido Final
            if overlap < 1e-6 {
                let mut best_valid = improved_trees.clone();
                let mut found_valid = false;
                
                use crate::utils::collisions::check_collisions;

                // Intento 1: Pulido Fino
                let polish_1 = refine_soft_sa(best_valid.clone(), SoftSARefineConfig {
                    iterations: 2000,
                    initial_temp: 0.05,
                    cooling_rate: 0.99,
                    overlap_penalty: 100000.0,
                    step_scale: 0.05,
                });
                if !check_collisions(&polish_1) {
                    best_valid = polish_1;
                    found_valid = true;
                }
                
                // Intento 2: Pulido Agresivo (si el fino falló)
                if !found_valid {
                    let polish_2 = refine_soft_sa(best_valid.clone(), SoftSARefineConfig {
                        iterations: 5000,
                        initial_temp: 0.01,
                        cooling_rate: 0.99,
                        overlap_penalty: 1000000.0,
                        step_scale: 0.01,
                    });
                    if !check_collisions(&polish_2) {
                        best_valid = polish_2;
                        found_valid = true;
                    }
                }

                if found_valid {
                    let real_kaggle = calculate_kaggle_score(&best_valid);
                    let saved = save_if_better(output_path, &best_valid, real_kaggle).unwrap_or(false);
                    if saved {
                        println!("   ✨ Iter {}: Nuevo Récord Guardado (Kaggle: {:.6})", t, real_kaggle);
                        leader.trees = best_valid;
                        leader.evaluate();
                    }
                }
            }
        }

        if t % 10 == 0 {
            println!("   Iter {}/{} | Leader Cost: {:.4}", t, max_iters, leader.cost);
        }
    }

    leader.trees
}
