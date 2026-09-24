use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;
use clap::Parser;

use christmas_tree_optimizer::entities::Tree;
use christmas_tree_optimizer::utils::geometry::calculate_kaggle_score;
use christmas_tree_optimizer::utils::io::{load_trees, save_if_better, get_score_from_file};
use christmas_tree_optimizer::algorithms::annealing::optimize_with_annealing;
use christmas_tree_optimizer::algorithms::genetic::run_genetic_algorithm;
use christmas_tree_optimizer::algorithms::cmaes_optimizer::optimize_with_cmaes;
use christmas_tree_optimizer::algorithms::fine_tuning::fine_tuning;
use christmas_tree_optimizer::algorithms::gravity::apply_gravity;
use christmas_tree_optimizer::algorithms::strategies::{generate_mosaic, generate_grid_zipper};
use christmas_tree_optimizer::algorithms::zip_skew::optimize_zip_skew;
use christmas_tree_optimizer::algorithms::repair::repair_solution;
use christmas_tree_optimizer::algorithms::rim_pressure::apply_rim_pressure;
use christmas_tree_optimizer::algorithms::molecular::apply_molecular_dynamics;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: Option<PathBuf>,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long)]
    target_n: Option<usize>,

    #[arg(long, default_value = "annealing")]
    strategy: String,

    #[arg(long, default_value_t = 100)]
    generations: usize,

    #[arg(long, default_value_t = 50)]
    pop_size: usize,

    #[arg(long, default_value_t = 0)]
    gravity_steps: usize,

    #[arg(long, default_value_t = 0)]
    fine_tune_iters: usize,

    #[arg(long, default_value_t = false)]
    pure_random: bool,

    #[arg(long, default_value_t = false)]
    no_smart_lookup: bool,

    #[arg(long, default_value_t = false)]
    force: bool,

    #[arg(long, default_value_t = false)]
    use_groups: bool,

    #[arg(long, default_value_t = 0.0)]
    freeze_inner: f64,

    #[arg(long, default_value_t = 0.0)]
    freeze_outer: f64,

    #[arg(long)]
    base_n: Option<usize>,

    #[arg(long)]
    parts: Option<String>,

    #[arg(long, default_value_t = false)]
    freeze_base: bool,

    #[arg(long, default_value_t = 1.0)]
    sa_init_temp: f64,

    #[arg(long, default_value_t = 0.001)]
    sa_final_temp: f64,

    #[arg(long, default_value_t = 0.3)]
    sa_temp_power: f64,

    #[arg(long, default_value_t = 1.0)]
    sa_step_scale: f64,

    #[arg(long, default_value_t = 15)]
    cmaes_stagnation: usize,

    #[arg(long, default_value_t = false)]
    live_vis: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let start = Instant::now();

    // 1. Determinar N objetivo
    let target_n = if let Some(n) = args.target_n {
        n
    } else {
        // Intentar inferir de input o output
        let mut n_inferred = 0;
        let paths = [&args.input, &args.output];
        for p in paths.iter().filter_map(|p| p.as_ref()) {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem.starts_with('T') {
                if let Ok(val) = stem[1..].chars().take_while(|c| c.is_numeric()).collect::<String>().parse::<usize>() {
                    n_inferred = val;
                    break;
                }
            } else if let Ok(val) = stem.chars().take_while(|c| c.is_numeric()).collect::<String>().parse::<usize>() {
                 n_inferred = val;
                 break;
            }
        }
        if n_inferred == 0 {
            return Err("No se pudo determinar el N objetivo. Use --target-n".into());
        }
        n_inferred
    };

    // 2. Determinar archivo de salida por defecto
    let output_path = args.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!("../solutions/T{}.csv", target_n))
    });

    println!(">>> Pipeline T{} | Estrategia: {} | Salida: {} <<<", target_n, args.strategy, output_path.display());

    // 3. Preparar Semilla (Input)
    let mut trees = match &args.input {
        Some(in_path) if in_path.exists() => {
            println!("📂 Cargando input de: {}", in_path.display());
            load_trees(in_path)?
        }
        _ => {
            // Si no hay input, intentar cargar del output para continuar el progreso
            if output_path.exists() {
                println!("📂 Resumiendo desde archivo de salida: {}", output_path.display());
                load_trees(&output_path).unwrap_or_default()
            } else {
                vec![]
            }
        }
    };

    let strategy = args.strategy.to_lowercase();

    // Ajustar número de árboles si es necesario
    // PERO NO si la estrategia es 'pruning' o 'incremental' (ellas manejan esto)
    let skip_auto_adjust = strategy == "pruning" || strategy == "incremental";
    
    if !trees.is_empty() && !skip_auto_adjust {
        if trees.len() < target_n {
            println!("➕ Incrementando de {} a {} árboles...", trees.len(), target_n);
            let original_trees = trees.clone();
            trees = christmas_tree_optimizer::algorithms::strategies::smart_increment(trees, target_n)
                .unwrap_or_else(|| {
                    println!("⚠️ Falló incremento inteligente. Usando secuencial.");
                    let mut t = original_trees;
                    for i in t.len() + 1..=target_n {
                        t.push(Tree::new(i, i as f64 * 0.5, 0.0, 0.0));
                    }
                    t
                });
        } else if trees.len() > target_n {
            println!("✂️ Podando inteligentemente de {} a {} árboles...", trees.len(), target_n);
            trees = christmas_tree_optimizer::algorithms::strategies::smart_prune(trees, target_n);
        }
    } else if trees.is_empty() || args.pure_random {
        if args.pure_random {
            println!("🎲 Generando base RANDOM (pure-random)...");
            use rand::Rng;
            let mut rng = rand::thread_rng();
            trees = (1..=target_n).map(|i| {
                let x = rng.gen_range(-2.0..2.0);
                let y = rng.gen_range(-2.0..2.0);
                let angle = rng.gen_range(0.0..360.0);
                Tree::new(i, x, y, angle)
            }).collect();
        } else {
            println!("⚠️ Sin solución previa. Generando base secuencial...");
            trees = (1..=target_n).map(|i| Tree::new(i, i as f64 * 0.5, 0.0, 0.0)).collect();
        }
    }

    // 4. Ejecución de Estrategia
    let result_trees = match strategy.as_str() {
        "zipskew" => {
            optimize_zip_skew(target_n, args.generations * 10, true)
                .unwrap_or_else(|| trees.clone())
        }
        "zipper" => {
            generate_grid_zipper(target_n, 1.1, 0.8)
                .unwrap_or_else(|| trees.clone())
        }
        "mosaic" => {
            if target_n % 4 != 0 { return Err("Mosaic requiere que N sea múltiplo de 4".into()); }
            let base = args.base_n.unwrap_or(target_n / 4);
            println!("🧩 Generando Mosaico {}x{} usando base N={}", 2, 2, base);
            generate_mosaic(target_n, base).unwrap_or_else(|| trees.clone())
        }
        "deca" => {
            println!("📅 Generando Deca-Mosaico (Bloques de 10) para N={}", target_n);
            use christmas_tree_optimizer::algorithms::strategies::generate_deca_mosaic;
            generate_deca_mosaic(target_n).unwrap_or_else(|| trees.clone())
        }
        "pruning" => {
            let mut base_trees = trees.clone();
            // Smart lookup: Si la solución actual tiene <= target_n, intentar cargar T{n+1}
            // DESHABILITADO si --no-smart-lookup está activo
            if !args.no_smart_lookup && base_trees.len() <= target_n {
                let source_path = format!("../solutions/T{}.csv", target_n + 1);
                if let Ok(loaded) = load_trees(Path::new(&source_path)) {
                    println!("✂️ Intentando podar desde {} (mejor base encontrada)", source_path);
                    base_trees = loaded;
                }
            }
            println!("--> Pruning (smart_lookup={})", !args.no_smart_lookup);
            christmas_tree_optimizer::algorithms::strategies::smart_prune(base_trees, target_n)
        }
        "incremental" => {
            let mut base_trees = trees.clone();
            
            // Caso 1: Usuario especifica base explícita (ej. --base-n 30)
            if let Some(base_n) = args.base_n {
                 let source_path = format!("../solutions/T{}.csv", base_n);
                 if let Ok(loaded) = load_trees(Path::new(&source_path)) {
                     println!("➕ Cargando base explícita desde {}", source_path);
                     base_trees = loaded;
                 } else {
                     println!("⚠️ No se pudo cargar base explícita {}", source_path);
                 }
            } 
            // Caso 2: Smart lookup automático (si no se deshabilitó)
            else if !args.no_smart_lookup && base_trees.len() >= target_n && target_n > 0 {
                let source_path = format!("../solutions/T{}.csv", target_n - 1);
                if let Ok(loaded) = load_trees(Path::new(&source_path)) {
                    println!("➕ Intentando crecer desde {} (mejor base encontrada)", source_path);
                    base_trees = loaded;
                }
            }

            println!("--> Incremental (smart_lookup={})", !args.no_smart_lookup);
            let t_clone = base_trees.clone();
            christmas_tree_optimizer::algorithms::strategies::smart_increment(base_trees, target_n)
                .unwrap_or(t_clone)
        }
        "ga" => {
            let groups = if args.use_groups {
                println!("👥 Group mode enabled (coupling pairs of trees for GA)");
                (0..trees.len()).step_by(2)
                    .filter(|&i| i + 1 < trees.len())
                    .map(|i| vec![i, i + 1])
                    .collect()
            } else {
                vec![]
            };
            run_genetic_algorithm(trees.clone(), args.pop_size, args.generations, args.gravity_steps, groups, &output_path, args.pure_random)
        }
        "sa" => {
            // Calcular freeze_inner si se usa --freeze-base con --base-n
            let mut freeze_inner = args.freeze_inner;
            if args.freeze_base {
                if let Some(base) = args.base_n {
                     freeze_inner = base as f64 / target_n as f64;
                     println!("❄️ Auto-Freezing Base: Congelando el {:.1}% interior ({} árboles)", freeze_inner * 100.0, base);
                }
            }
            optimize_with_annealing(
                trees.clone(), 
                args.generations * 500, 
                true, 
                freeze_inner, 
                args.freeze_outer,
                args.sa_init_temp,
                args.sa_final_temp,
                args.sa_temp_power,
                args.sa_step_scale
            )
        }
        "sa2" => {
            christmas_tree_optimizer::algorithms::annealing2::optimize_with_annealing(
                trees.clone(),
                args.generations * 500,
                true,
                args.freeze_inner,
                args.freeze_outer,
                args.sa_init_temp,
                args.sa_final_temp,
                args.sa_temp_power,
                args.sa_step_scale,
                if args.live_vis { Some(PathBuf::from("../solutions/live.csv")) } else { None }
            )
        }
        "cmaes" => {
            optimize_with_cmaes(trees.clone(), args.generations * 5, args.pop_size, args.cmaes_stagnation, true)
        }
        "soft-sa" => {
            use christmas_tree_optimizer::algorithms::soft_sa::{optimize_soft_sa, SoftSAConfig};
            let config = SoftSAConfig {
                max_iterations: args.generations * 1000, 
                initial_temp: args.sa_init_temp, // Use user value directly
                cooling_rate: 0.99995,
                penalty_factor_start: 100.0,
                penalty_factor_end: 1_000_000.0,
                sa_step_scale: args.sa_step_scale,
            };
            optimize_soft_sa(trees.clone(), Some(config), true)
        }
        "finetune" => {
            let iters = if args.fine_tune_iters > 0 { args.fine_tune_iters } else { 1000 };
            fine_tuning(trees.clone(), iters, true)
        }
        "gravity" => {
            let mut t = trees.clone();
            apply_gravity(&mut t, args.gravity_steps, 0.01);
            t
        }
        "rim_pressure" => {
            let mut t = trees.clone();
            let steps = if args.gravity_steps > 0 { args.gravity_steps } else { 10 };
            println!("🔘 Aplicando Rim Pressure ({} pasos)...", steps);
            for _ in 0..steps {
                apply_rim_pressure(&mut t, 0.1);
            }
            t
        }
        "molecular" => {
            let mut t = trees.clone();
            let steps = if args.gravity_steps > 0 { args.gravity_steps } else { 200 };
            let step_size = 0.05;
            println!("⚛️ Aplicando Dinámica Molecular ({} pasos, step={})...", steps, step_size);
            apply_molecular_dynamics(&mut t, steps, step_size);
            t
        }
        "refine-geometric" => {
            use christmas_tree_optimizer::algorithms::refine_geometric::refine_geometric;
            refine_geometric(trees.clone(), args.generations * 10, &output_path)
        }
        "abc" => {
            use christmas_tree_optimizer::algorithms::abc_packer::optimize_with_abc;
            optimize_with_abc(trees.clone(), args.generations, args.pop_size, 100, &output_path)
        }
        "woa" => {
            use christmas_tree_optimizer::algorithms::woa_packer::optimize_with_woa;
            optimize_with_woa(trees.clone(), args.generations, args.pop_size, &output_path)
        }
        "square" => {
            println!("🔲 Aplicando Square Pressure (Cuadrando el círculo)...");
            let t = trees.clone();
            // DESHABILITADO TEMPORALMENTE
            // let iters = args.generations * 100;
            // for i in 0..iters {
            //     // Alternar: 1 vez Rim (compactar huecos), 5 veces Square (aplastar eje largo)
            //     if i % 6 == 0 { apply_rim_pressure(&mut t, 0.02); }
            //     else { apply_square_pressure(&mut t, 0.01); }
            // }
            t
        }
        "all" => {
             run_auto_pipeline(trees.clone(), &args, target_n)?
        }
        _ => return Err(format!("Estrategia desconocida: {}", strategy).into()),
    };

    // 5. Verificación y Guardado (Kaggle Score)
    let final_kaggle = calculate_kaggle_score(&result_trees);
    
    // Si --force está activo, guardar siempre sin comparar
    let improved = if args.force {
        use christmas_tree_optimizer::utils::io::save_solution;
        save_solution(&output_path, &result_trees, final_kaggle)?;
        println!("\n💾 Guardado forzado (--force): Score: {:.6}", final_kaggle);
        true
    } else {
        save_if_better(&output_path, &result_trees, final_kaggle)?
    };

    if improved && !args.force {
        println!("\n✨ ¡NUEVO RÉCORD de Kaggle! Score: {:.6}", final_kaggle);
    } else if !improved {
        println!("\n❌ No se mejoró el récord existente en {}.", output_path.display());
        println!("   Mejor actual: {:.6} | Intento: {:.6}", get_score_from_file(&output_path), final_kaggle);
    }

    println!("⏱️ Tiempo total: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn run_auto_pipeline(mut trees: Vec<Tree>, args: &Args, target_n: usize) -> Result<Vec<Tree>, Box<dyn Error>> {
    println!("🚀 Pipeline Automático Iniciado...");
    
    // Si no es válido (colisiones), intentar una base geométrica
    if christmas_tree_optimizer::utils::collisions::check_collisions(&trees) {
         println!("-> Solution base inválida, intentando ZipSkew...");
         if let Some(zs) = optimize_zip_skew(target_n, 100, true) {
             trees = zs;
         }
    }

    // Refinamiento estándar
    println!("-> Aplicando refinamiento...");
    repair_solution(&mut trees, true, true);
    
    if args.gravity_steps > 0 {
        apply_gravity(&mut trees, args.gravity_steps, 0.01);
    }

    let ft_iters = if args.fine_tune_iters > 0 { args.fine_tune_iters } else { 200 };
    trees = fine_tuning(trees, ft_iters, false);

    Ok(trees)
}
