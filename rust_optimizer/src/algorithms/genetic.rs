use crate::entities::Tree;
use crate::algorithms::fine_tuning::fine_tuning;
use crate::algorithms::repair::repair_solution;
use crate::algorithms::gravity::apply_gravity;
use crate::utils::collisions::check_collisions;
use crate::utils::geometry::{calculate_score, calculate_kaggle_score, calculate_optimization_score};
use crate::utils::io::save_solution;
use rand::prelude::*;
use rayon::prelude::*;
use std::path::Path;

/// Genera un layout completamente aleatorio con N árboles
/// Los árboles se distribuyen en un área y luego se compactan con gravedad
fn create_random_layout(n: usize, rng: &mut ThreadRng) -> Vec<Tree> {
    let mut trees = Vec::with_capacity(n);
    
    // Calcular un área aproximada basada en N
    let side = (n as f64).sqrt() * 2.0;
    
    for id in 1..=n {
        let x = rng.gen_range(0.0..side);
        let y = rng.gen_range(0.0..side);
        let angle = rng.gen_range(0.0..360.0);
        trees.push(Tree::new(id, x, y, angle));
    }
    
    // Compactar con gravedad para hacerlo válido
    apply_gravity(&mut trees, 200, 0.15);
    
    trees
}

pub fn run_genetic_algorithm(
    initial_trees: Vec<Tree>, 
    pop_size: usize, 
    generations: usize, 
    gravity_steps: usize,
    groups: Vec<Vec<usize>>,
    output_path: &Path,
    pure_random: bool,  // NEW: 100% random mode
) -> Vec<Tree> {
    
    let n = initial_trees.len();
    
    if pure_random {
        println!("🧬 [Rust Optimizer] Iniciando GA PURO RANDOM: Pop={}, Gen={}, N={}", pop_size, generations, n);
        println!("   -> Modo: 100% poblaciones ALEATORIAS (ignorando solución existente)");
    } else {
        println!("🧬 [Rust Optimizer] Iniciando GA EXPLORATORIO: Pop={}, Gen={}, GravSteps={}", pop_size, generations, gravity_steps);
        println!("   -> Modo: 80% poblaciones RANDOM + 20% variaciones del original");
    }

    let mut initial_trees = initial_trees;
    
    // 1. Verificación Inicial (solo si no es pure random)
    if !pure_random {
        let _ = repair_solution(&mut initial_trees, true, true);
    }

    let initial_score = if check_collisions(&initial_trees) {
        f64::INFINITY
    } else {
        calculate_kaggle_score(&initial_trees)
    };
    
    if !pure_random {
        println!("   -> Score Base (Input): {:.6}", initial_score);
    }

    let mut population: Vec<Vec<Tree>> = Vec::with_capacity(pop_size);
    let mut rng = thread_rng();
    
    if pure_random {
        // --- MODO PURO RANDOM: 100% poblaciones aleatorias ---
        for _ in 0..pop_size {
            let ind = create_random_layout(n, &mut rng);
            population.push(ind);
        }
        println!("   -> Población inicial: {} individuos 100% aleatorios", pop_size);
    } else {
        // --- MODO MIXTO: Original + variaciones + random ---
        population.push(initial_trees.clone());
        
        for i in 1..pop_size {
            if i < (pop_size as f64 * 0.2) as usize {
                let mut ind = initial_trees.clone();
                for t in ind.iter_mut() {
                    t.x += rng.gen_range(-1.0..1.0);
                    t.y += rng.gen_range(-1.0..1.0);
                    t.angle += rng.gen_range(-30.0..30.0);
                    t.update_poly();
                }
                apply_gravity(&mut ind, 50, 0.1);
                population.push(ind);
            } else {
                let ind = create_random_layout(n, &mut rng);
                population.push(ind);
            }
        }
        
        println!("   -> Población inicial: {} random, {} variaciones", 
            (pop_size as f64 * 0.8) as usize, 
            (pop_size as f64 * 0.2) as usize);
    }

    // En modo pure_random, empezar con INFINITY para trackear progreso interno
    // Pero guardar solo si supera el archivo existente
    let file_score = initial_score;  // El score del archivo actual
    let mut best_global_score = if pure_random { f64::INFINITY } else { initial_score };
    let mut best_global_sol = initial_trees.clone();

    let mut generations_without_improvement = 0;
    let stagnation_threshold = 25; // Si no mejora en 25 gens -> Terremoto

    for gen in 0..generations {
        // --- PROBABILIDAD DINÁMICA DE ESTRATEGIA (PHASED EVOLUTION) ---
        // Al principio (gen 0): 80% Gravedad (Compactar)
        // Al final (gen N): 20% Gravedad (Pulir)
        let progress = gen as f64 / generations as f64;
        let gravity_prob = 0.8 - (0.6 * progress); // De 0.8 a 0.2

        let mut scored_pop: Vec<(f64, Vec<Tree>)> = population.par_iter()
            .map(|ind| {
                let mut optimized = ind.clone();
                let mut rng = thread_rng();
                
                if rng.gen::<f64>() < gravity_prob {
                    apply_gravity(&mut optimized, gravity_steps, 0.01);
                } else {
                    optimized = fine_tuning(optimized, gravity_steps, false);
                }
                
                let opt_score = if check_collisions(&optimized) {
                    f64::INFINITY
                } else {
                    calculate_optimization_score(&optimized)
                };
                (opt_score, optimized)
            })
            .collect();

        scored_pop.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // El mejor de la generación (en términos de optimización)
        // Pero calculamos su score de Kaggle real para reportar
        let best_gen_opt_score = scored_pop[0].0;
        let best_gen_kaggle_score = if best_gen_opt_score.is_finite() {
            calculate_kaggle_score(&scored_pop[0].1)
        } else {
            f64::INFINITY
        };
        
        if best_gen_kaggle_score < best_global_score {
            best_global_score = best_gen_kaggle_score;
            best_global_sol = scored_pop[0].1.clone();
            generations_without_improvement = 0; // Reset
            
            // Mostrar progreso y comparar con archivo
            if best_global_score < file_score {
                println!("🏆 Gen {}: SUPERA ARCHIVO -> {:.6} (era {:.6})", gen, best_global_score, file_score);
                if let Err(e) = save_solution(output_path, &best_global_sol, best_global_score) {
                    eprintln!("⚠️ Error guardando mejora: {}", e);
                }
            } else {
                println!("✨ Gen {}: Nuevo Récord GA -> {:.6} (archivo: {:.6})", gen, best_global_score, file_score);
            }
        } else {
            generations_without_improvement += 1;
            println!("   Gen {}: Mejor {:.6} (GA: {:.6}, Archivo: {:.6}) [Stag: {}/{}]", 
                gen, best_gen_kaggle_score, best_global_score, file_score, generations_without_improvement, stagnation_threshold);
        }

        let mut next_gen = Vec::with_capacity(pop_size);
        
        // Elitismo: Guardar los mejores
        for i in 0..5.min(pop_size) {
            next_gen.push(scored_pop[i].1.clone());
        }

        // --- ESTRATEGIA ANTI-ESTANCAMIENTO ---
        // Si estamos estancados, alternamos entre EXPLODE y SHAKE
        let stagnated = generations_without_improvement >= stagnation_threshold;
        let use_explode = (gen / stagnation_threshold) % 2 == 0; // Alternar cada vez
        
        if stagnated {
            if use_explode {
                println!("💥 ¡EXPLODE! Empujando árboles hacia afuera desde el centro...");
            } else {
                println!("🌋 ¡SHAKE! Sacudiendo la población aleatoriamente...");
            }
            generations_without_improvement = 0;
        }

        while next_gen.len() < pop_size {
            if stagnated {
                let mut child = best_global_sol.clone();
                let mut rng = thread_rng();
                
                if use_explode {
                    // EXPLODE: Empujar radialmente desde el centro
                    let mut cx = 0.0;
                    let mut cy = 0.0;
                    for t in &child { cx += t.x; cy += t.y; }
                    cx /= child.len() as f64;
                    cy /= child.len() as f64;
                    
                    let push_factor = rng.gen_range(0.3..0.8); // Variación en la intensidad
                    for t in child.iter_mut() {
                        let dx = t.x - cx;
                        let dy = t.y - cy;
                        let dist = (dx*dx + dy*dy).sqrt().max(0.1);
                        t.x += (dx / dist) * push_factor;
                        t.y += (dy / dist) * push_factor;
                        t.angle += rng.gen_range(-5.0..5.0);
                        t.update_poly();
                    }
                } else {
                    // SHAKE: Sacudida aleatoria (como antes)
                    for t in child.iter_mut() {
                        t.x += rng.gen_range(-0.5..0.5);
                        t.y += rng.gen_range(-0.5..0.5);
                        t.angle += rng.gen_range(-10.0..10.0);
                        t.update_poly();
                    }
                }
                next_gen.push(child);
            } else {
                // --- INYECCIÓN DE SANGRE FRESCA ---
                // Si ya llenamos el 90% de la población, el último 10% son individuos TOTALMENTE nuevos
                // Esto impide que la población converja completamente a una sola forma
                if next_gen.len() > (pop_size as f64 * 0.9) as usize {
                    let mut fresh = best_global_sol.clone(); 
                    let mut rng = thread_rng();
                    // Caos Total
                    for t in fresh.iter_mut() {
                        let r = rng.gen_range(0.0..12.0);
                        let theta = rng.gen_range(0.0..360.0_f64).to_radians();
                        t.x = r * theta.cos();
                        t.y = r * theta.sin();
                        t.angle = rng.gen_range(0.0..360.0);
                        t.update_poly();
                    }
                    // Compactación inicial rápida para que no sea basura total
                    apply_gravity(&mut fresh, 100, 0.2); 
                    next_gen.push(fresh);
                } else {
                    // Evolución normal (Crossover)
                    let p1 = tournament(&scored_pop);
                    let p2 = tournament(&scored_pop);
                    
                    let mut child = crossover(p1, p2);
                    let mut rng = thread_rng();
                    if rng.gen_bool(0.2) {
                        mutate(&mut child, &groups);
                    }
                    next_gen.push(child);
                }
            }
        }
        
        population = next_gen;
    }

    println!("\n🏆 Compactación Final en Rust...");
    let mut final_candidate = best_global_sol.clone();
    apply_gravity(&mut final_candidate, 2000, 0.01);
    
    let final_score = if check_collisions(&final_candidate) { f64::INFINITY } else { calculate_score(&final_candidate) };
    
    if final_score < best_global_score {
        final_candidate
    } else {
        best_global_sol
    }
}

fn tournament(scored: &[(f64, Vec<Tree>)]) -> &Vec<Tree> {
    let mut rng = thread_rng();
    let k = 3;
    let mut best = &scored[rng.gen_range(0..scored.len())];
    
    for _ in 1..k {
        let candidate = &scored[rng.gen_range(0..scored.len())];
        if candidate.0 < best.0 {
            best = candidate;
        }
    }
    &best.1
}

fn crossover(p1: &Vec<Tree>, p2: &Vec<Tree>) -> Vec<Tree> {
    let mut rng = thread_rng();
    let mut child = Vec::with_capacity(p1.len());
    for (t1, t2) in p1.iter().zip(p2.iter()) {
        if rng.gen_bool(0.5) {
            child.push(t1.clone());
        } else {
            child.push(t2.clone());
        }
    }
    child
}

fn rotate_point(x: f64, y: f64, cx: f64, cy: f64, angle_deg: f64) -> (f64, f64) {
    let rad = angle_deg.to_radians();
    let cos = rad.cos();
    let sin = rad.sin();
    let dx = x - cx;
    let dy = y - cy;
    (
        cx + dx * cos - dy * sin,
        cy + dx * sin + dy * cos
    )
}

fn mutate(trees: &mut Vec<Tree>, groups: &[Vec<usize>]) {
    let mut rng = thread_rng();
    
    if !groups.is_empty() && rng.gen_bool(0.5) {
        let group_idx = rng.gen_range(0..groups.len());
        let group = &groups[group_idx];
        
        let dx = rng.gen_range(-1.0..1.0);
        let dy = rng.gen_range(-1.0..1.0);
        let d_angle = rng.gen_range(-15.0..15.0);
        
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &idx in group {
            cx += trees[idx].x;
            cy += trees[idx].y;
        }
        cx /= group.len() as f64;
        cy /= group.len() as f64;
        
        for &idx in group {
            let t = &mut trees[idx];
            
            t.x += dx;
            t.y += dy;
            
            let (rx, ry) = rotate_point(t.x, t.y, cx + dx, cy + dy, d_angle);
            t.x = rx;
            t.y = ry;
            t.angle += d_angle;
            
            t.update_poly();
        }
        
    } else {
        let idx = rng.gen_range(0..trees.len());
        let t = &mut trees[idx];
        
        t.x += rng.gen_range(-1.0..1.0);
        t.y += rng.gen_range(-1.0..1.0);
        t.angle += rng.gen_range(-45.0..45.0);
        t.update_poly();
    }
}
