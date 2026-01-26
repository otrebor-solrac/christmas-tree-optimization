use crate::entities::Tree;
use crate::utils::geometry::{calculate_kaggle_score, calculate_overlap_area};
use crate::utils::io::save_if_better;
use rand::Rng;
use std::path::Path;

#[derive(Clone)]
struct FoodSource {
    trees: Vec<Tree>,
    cost: f64,
    fitness: f64,
    trial: usize,
}

impl FoodSource {
    fn new(trees: Vec<Tree>) -> Self {
        let mut fs = FoodSource {
            trees,
            cost: 0.0,
            fitness: 0.0,
            trial: 0,
        };
        fs.evaluate();
        fs
    }

    fn evaluate(&mut self) {
        let overlap = calculate_overlap_area(&self.trees);
        let kaggle = calculate_kaggle_score(&self.trees);
        
        // Penalización de gravedad al centro
        let gravity: f64 = self.trees.iter()
            .map(|t| (t.x.powi(2) + t.y.powi(2)).sqrt())
            .sum::<f64>() / (self.trees.len() as f64) * 0.05;

        self.cost = kaggle + (overlap * 5000.0) + gravity;
        self.fitness = 1.0 / (1.0 + self.cost);
    }
}

pub fn optimize_with_abc(
    initial_trees: Vec<Tree>,
    max_iters: usize,
    colony_size: usize,
    limit: usize,
    output_path: &Path,
) -> Vec<Tree> {
    let n_trees = initial_trees.len();
    let n_employed = colony_size / 2;
    let mut rng = rand::thread_rng();

    println!("🐝 [ABC Packer] Iniciando optimización (N={}, Pob={}, Limit={})", n_trees, colony_size, limit);

    // 1. Inicialización
    // Poblamos con variaciones de la solución inicial + algunas aleatorias si es necesario
    let mut population: Vec<FoodSource> = (0..n_employed).map(|i| {
        if i == 0 {
            FoodSource::new(initial_trees.clone())
        } else {
            // Mutación inicial fuerte para dar variedad
            let mut trees = initial_trees.clone();
            for t in &mut trees {
                t.x += rng.gen_range(-1.0..1.0);
                t.y += rng.gen_range(-1.0..1.0);
                t.angle += rng.gen_range(-20.0..20.0);
                t.update_poly();
            }
            FoodSource::new(trees)
        }
    }).collect();

    let mut best_fs = population[0].clone();

    for iteration in 0..max_iters {
        // --- FASE ABEJAS EMPLEADAS ---
        for i in 0..n_employed {
            let next_fs = mutate(&population[i], &population, i, &best_fs, &mut rng);
            if next_fs.fitness > population[i].fitness {
                population[i] = next_fs;
                population[i].trial = 0;
            } else {
                population[i].trial += 1;
            }
        }

        // --- FASE ABEJAS OBSERVADORAS ---
        let total_fitness: f64 = population.iter().map(|f| f.fitness).sum();
        for _ in 0..n_employed {
            let mut pick = rng.gen_range(0.0..total_fitness);
            let mut idx = 0;
            for (i, fs) in population.iter().enumerate() {
                pick -= fs.fitness;
                if pick <= 0.0 {
                    idx = i;
                    break;
                }
            }

            let next_fs = mutate(&population[idx], &population, idx, &best_fs, &mut rng);
            if next_fs.fitness > population[idx].fitness {
                population[idx] = next_fs;
                population[idx].trial = 0;
            } else {
                population[idx].trial += 1;
            }
        }

        // --- FASE ABEJAS EXPLORADORAS ---
        for i in 0..n_employed {
            if population[i].trial > limit {
                // Scouting: Mutación fuerte del mejor para no perder estructura
                let mut trees = best_fs.trees.clone();
                for t in &mut trees {
                    if rng.gen_bool(0.3) {
                        t.x += rng.gen_range(-2.0..2.0);
                        t.y += rng.gen_range(-2.0..2.0);
                        t.angle += rng.gen_range(-45.0..45.0);
                        t.update_poly();
                    }
                }
                population[i] = FoodSource::new(trees);
            }
        }

        // --- TRACK MEJOR ---
        for fs in &population {
            if fs.cost < best_fs.cost {
                best_fs = fs.clone();
                let real_kaggle = calculate_kaggle_score(&best_fs.trees);
                let overlap = calculate_overlap_area(&best_fs.trees);
                println!("   ✨ Iter {}: Nuevo Récord (Cost: {:.6} | Kaggle: {:.6} | Overlap: {:.6})", 
                         iteration, best_fs.cost, real_kaggle, overlap);
                
                // Guardado preventivo
                if overlap < 1e-9 {
                    let _ = save_if_better(output_path, &best_fs.trees, real_kaggle);
                }
            }
        }

        if iteration % 100 == 0 {
            println!("   Iter {}/{} | Best Cost: {:.4}", iteration, max_iters, best_fs.cost);
        }
    }

    best_fs.trees
}

fn mutate(
    current: &FoodSource,
    pop: &Vec<FoodSource>,
    idx: usize,
    best: &FoodSource,
    rng: &mut impl Rng,
) -> FoodSource {
    let mut neighbor_idx = idx;
    while neighbor_idx == idx {
        neighbor_idx = rng.gen_range(0..pop.len());
    }
    let neighbor = &pop[neighbor_idx];

    let mut new_trees = current.trees.clone();
    
    // Mutar entre 1 y el 20% de los árboles
    let n_mut = (1 + (new_trees.len() as f64 * 0.2) as usize).min(new_trees.len());
    
    for _ in 0..n_mut {
        let t_idx = rng.gen_range(0..new_trees.len());
        
        let phi = rng.gen_range(-1.0..1.0); // Exploración (ABC Tradicional)
        let psi = rng.gen_range(0.0..1.5);  // Atracción al mejor (G-Best)
        
        // Mutar X
        new_trees[t_idx].x += phi * (current.trees[t_idx].x - neighbor.trees[t_idx].x) 
                             + psi * (best.trees[t_idx].x - current.trees[t_idx].x);
        
        // Mutar Y
        new_trees[t_idx].y += phi * (current.trees[t_idx].y - neighbor.trees[t_idx].y) 
                             + psi * (best.trees[t_idx].y - current.trees[t_idx].y);
                             
        // Mutar Angulo
        new_trees[t_idx].angle += phi * (current.trees[t_idx].angle - neighbor.trees[t_idx].angle) 
                                 + psi * (best.trees[t_idx].angle - current.trees[t_idx].angle);
        
        new_trees[t_idx].update_poly();
    }

    FoodSource::new(new_trees)
}
