use macroquad::prelude::*;
use clap::Parser;
use christmas_tree_optimizer::utils::io::{load_trees, save_solution};
use christmas_tree_optimizer::utils::geometry::calculate_score;
use christmas_tree_optimizer::algorithms::fine_tuning::fine_tuning;
use christmas_tree_optimizer::algorithms::gravity::apply_gravity;
use christmas_tree_optimizer::algorithms::cmaes_optimizer::optimize_with_cmaes;
use christmas_tree_optimizer::algorithms::annealing2::optimize_with_annealing;
use christmas_tree_optimizer::algorithms::rim_pressure::apply_rim_pressure;
use christmas_tree_optimizer::entities::Tree; 

use geo::Intersects;
use ::rand::Rng;
use std::collections::HashSet;
use std::path::PathBuf;
use macroquad::ui::{root_ui, widgets, hash};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct VizArgs {
    /// Input file path (e.g. ../solutions/T6.csv)
    #[arg(value_name = "FILE")]
    input: Option<String>,

    #[arg(long, default_value_t = 5000000)]
    generations: usize,

    #[arg(long, default_value_t = 10.0)]
    sa_init_temp: f64,

    #[arg(long, default_value_t = 0.00001)]
    sa_final_temp: f64,

    #[arg(long, default_value_t = 1.5)]
    sa_step_scale: f64,

    #[arg(long, default_value_t = 0.0)]
    freeze_inner: f64,

    #[arg(long, default_value_t = 0.0)]
    freeze_outer: f64,

    #[arg(long)]
    base_n: Option<usize>,

    #[arg(long, default_value_t = false)]
    freeze_base: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Tree Optimizer - Precision Mode".to_owned(),
        window_width: 1200,
        window_height: 900,
        high_dpi: true, 
        ..Default::default()
    }
}

/// KAGGLE-COMPATIBLE collision detection: intersects AND NOT touches
/// Since geo crate doesn't have a direct `touches` method, we check if intersection
/// area is non-zero to distinguish overlap from touching.
fn get_colliding_indices(trees: &[Tree]) -> (HashSet<usize>, Vec<(usize, usize)>) {
    use geo::{BooleanOps, Area};
    
    let mut colliding = HashSet::new();
    let mut pairs = Vec::new();
    
    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            use geo::BoundingRect;
            // Quick bounding box check first
            if trees[i].poly.bounding_rect().unwrap().intersects(&trees[j].poly.bounding_rect().unwrap()) {
                // KAGGLE LOGIC: Check if intersection has non-zero area
                // Touching polygons have zero intersection area
                let intersection = trees[i].poly.intersection(&trees[j].poly);
                let area = intersection.unsigned_area();
                if area > 1e-12 {  // Non-zero area means real overlap
                    colliding.insert(i);
                    colliding.insert(j);
                    pairs.push((i, j));
                }
            }
        }
    }
    (colliding, pairs)
}


fn get_solutions_dir() -> PathBuf {
    if std::path::Path::new("solutions").is_dir() {
        PathBuf::from("solutions")
    } else {
        PathBuf::from("../solutions")
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = VizArgs::parse();
    
    let mut file_path = if let Some(ref s) = args.input {
        let p = PathBuf::from(s);
        if p.exists() { 
            p 
        } else if get_solutions_dir().join(s).exists() { 
            get_solutions_dir().join(s) 
        } else if PathBuf::from("..").join(s).exists() { 
            PathBuf::from("..").join(s) 
        } else {
            p
        }
    } else {
        get_solutions_dir().join("T25.csv") 
    };

    println!("Cargando: {}", file_path.display());
    let mut trees = load_trees(&file_path).unwrap_or_else(|e| {
        println!("Error: {}. Generando árboles iniciales...", e);
        vec![]
    });

    if trees.is_empty() {
        let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let n: usize = stem.trim_start_matches(|c: char| !c.is_numeric())
            .chars().take_while(|c| c.is_numeric()).collect::<String>()
            .parse().unwrap_or(20);
        let target_count = if n == 0 { 20 } else { n };
        println!("🌱 Generando semilla inicial de {} árboles para {}...", target_count, stem);
        trees = (1..=target_count).map(|i| {
            let cols = ((target_count as f64).sqrt().ceil() as usize).max(1);
            let row = (i - 1) / cols;
            let col = (i - 1) % cols;
            Tree::new(i, (col as f64 - cols as f64 / 2.0) * 0.9, (row as f64 - cols as f64 / 2.0) * 0.9, 0.0)
        }).collect();
    }
    
    // Modo de entrada de texto para cargar soluciones
    let mut input_mode = false;
    let mut input_buffer = String::new();
    let mut load_target: Option<PathBuf> = None;

    // --- CÁLCULO DE ENCUADRE INICIAL ---
    let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
    for t in &trees {
        use geo::BoundingRect;
        if let Some(rect) = t.poly.bounding_rect() {
            if rect.min().x < min_x { min_x = rect.min().x; }
            if rect.min().y < min_y { min_y = rect.min().y; }
            if rect.max().x > max_x { max_x = rect.max().x; }
            if rect.max().y > max_y { max_y = rect.max().y; }
        }
    }
    
    let center_x = if trees.is_empty() { 0.0 } else { (min_x + max_x) / 2.0 };
    let center_y = if trees.is_empty() { 0.0 } else { (min_y + max_y) / 2.0 };
    let side = if trees.is_empty() { 10.0 } else { (max_x - min_x).max(max_y - min_y) };

    let mut cam_target = vec2(center_x as f32, center_y as f32);
    let mut zoom_scale = 0.8 / (side as f32); // Ajustar zoom para que quepa el 80% de la pantalla
    
    let mut dragging_idx: Option<usize> = None;
    let mut last_mouse_world = vec2(0., 0.);
    
    let mut score = calculate_score(&trees);
    let (mut colliding_indices, mut collision_pairs) = get_colliding_indices(&trees);
    let mut history: Option<Vec<Tree>> = None;
    
    // Auto-Reload para visualización en tiempo real
    let mut auto_reload = false;
    let mut last_reload_time = 0.0;
    
    // --- PARÁMETROS INTERACTIVOS ---
    let mut sa_gens = args.generations as f32;
    let mut sa_init_t = args.sa_init_temp as f32;
    let mut sa_final_t = args.sa_final_temp as f32;
    let mut sa_step_scale = args.sa_step_scale as f32;
    let mut f_inner = args.freeze_inner as f32;
    let mut f_outer = args.freeze_outer as f32;
    let mut auto_freeze_base = args.freeze_base;
    let mut base_n_val = args.base_n.unwrap_or(0) as f32;

    let color_collision = Color::new(0.9, 0.2, 0.2, 1.0);
    let color_selected = Color::new(1.0, 0.6, 0.0, 1.0);
    let color_border = YELLOW;
    let color_outline = BLACK;

    loop {
        // --- CENTRALIZED LOADING LOGIC ---
        if let Some(path) = load_target.take() {
            println!("🔄 Intentando cargar: {}", path.display());
            match load_trees(&path) {
                Ok(new_trees) => {
                    trees = new_trees;
                    file_path = path;
                    (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                    score = calculate_score(&trees);
                    
                    // Recalcular encuadre (Auto-Zoom)
                    let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
                    let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
                    for t in &trees {
                        use geo::BoundingRect;
                        if let Some(rect) = t.poly.bounding_rect() {
                            if rect.min().x < min_x { min_x = rect.min().x; }
                            if rect.min().y < min_y { min_y = rect.min().y; }
                            if rect.max().x > max_x { max_x = rect.max().x; }
                            if rect.max().y > max_y { max_y = rect.max().y; }
                        }
                    }
                    if !trees.is_empty() {
                        let center_x = (min_x + max_x) / 2.0;
                        let center_y = (min_y + max_y) / 2.0;
                        let side = (max_x - min_x).max(max_y - min_y);
                        cam_target = vec2(center_x as f32, center_y as f32);
                        zoom_scale = 0.8 / (side as f32);
                    }
                    println!("✅ Cargado: {} árboles, score: {:.6}", trees.len(), score);
                }
                Err(_) => {
                    println!("🌱 Archivo {} no existe. Inicializando árboles...", path.display());
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    let n: usize = stem.trim_start_matches(|c: char| !c.is_numeric())
                        .chars().take_while(|c| c.is_numeric()).collect::<String>()
                        .parse().unwrap_or(20);
                    let target_count = if n == 0 { 20 } else { n };
                    let cols = ((target_count as f64).sqrt().ceil() as usize).max(1);
                    trees = (1..=target_count).map(|i| {
                        let row = (i - 1) / cols;
                        let col = (i - 1) % cols;
                        Tree::new(i, (col as f64 - cols as f64 / 2.0) * 0.9, (row as f64 - cols as f64 / 2.0) * 0.9, 0.0)
                    }).collect();
                    file_path = path;
                    (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                    score = calculate_score(&trees);
                    cam_target = vec2(0.0, 0.0);
                    zoom_scale = 0.8 / (cols as f32 * 1.2);
                }
            }
        }

        clear_background(WHITE);
        let screen_ratio = screen_width() / screen_height();
        
        let cam = Camera2D {
            zoom: vec2(zoom_scale, zoom_scale * screen_ratio),
            target: cam_target, 
            ..Default::default()
        };

        // --- INPUTS ---
        
        // Modo de entrada de texto (L para cargar solución)
        if input_mode {
            // Capturar texto
            if let Some(c) = get_char_pressed() {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    input_buffer.push(c);
                }
            }
            if is_key_pressed(KeyCode::Backspace) && !input_buffer.is_empty() {
                input_buffer.pop();
            }
            if is_key_pressed(KeyCode::Enter) {
                // Intentar cargar la solución
                load_target = Some(get_solutions_dir().join(format!("{}.csv", input_buffer)));
                input_mode = false;
                input_buffer.clear();
            }
            if is_key_pressed(KeyCode::Escape) {
                input_mode = false;
                input_buffer.clear();
            }
            
            // Dibujar overlay de entrada
            draw_rectangle(0.0, 0.0, screen_width(), screen_height(), Color::new(0.0, 0.0, 0.0, 0.5));
            draw_rectangle(screen_width() / 2.0 - 200.0, screen_height() / 2.0 - 40.0, 400.0, 80.0, WHITE);
            draw_text("Cargar solución:", screen_width() / 2.0 - 180.0, screen_height() / 2.0 - 10.0, 24.0, BLACK);
            draw_text(&format!("{}_", input_buffer), screen_width() / 2.0 - 180.0, screen_height() / 2.0 + 25.0, 28.0, DARKBLUE);
            draw_text("(Enter: cargar, Esc: cancelar)", screen_width() / 2.0 - 180.0, screen_height() / 2.0 + 55.0, 16.0, GRAY);
            
            next_frame().await;
            continue;  // Saltar el resto del loop mientras estamos en modo entrada
        }
        
        // Activar modo de entrada con L
        if is_key_pressed(KeyCode::L) {
            input_mode = true;
            input_buffer.clear();
            // Drenar el buffer de teclado para evitar caracteres acumulados
            while get_char_pressed().is_some() {}
        }

        // Auto-Reload Toggle (V)
        if is_key_pressed(KeyCode::V) {
            auto_reload = !auto_reload;
        }
        
        let now = get_time();
        if auto_reload && now - last_reload_time > 0.1 { // 10Hz limit
            last_reload_time = now;
            let live_path = get_solutions_dir().join("live.csv");
            if live_path.exists() {
                 if let Ok(new_trees) = load_trees(&live_path) {
                     trees = new_trees;
                     (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                     score = calculate_score(&trees);
                     
                     // Auto-Center Camera
                     let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
                     let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
                     for t in &trees {
                         use geo::BoundingRect;
                         if let Some(rect) = t.poly.bounding_rect() {
                             if rect.min().x < min_x { min_x = rect.min().x; }
                             if rect.min().y < min_y { min_y = rect.min().y; }
                             if rect.max().x > max_x { max_x = rect.max().x; }
                             if rect.max().y > max_y { max_y = rect.max().y; }
                         }
                     }
                     if !trees.is_empty() {
                         let cx = (min_x + max_x) / 2.0;
                         let cy = (min_y + max_y) / 2.0;
                         // Suavizado simple (lerp) para que no tiembie tanto
                         cam_target.x = cam_target.x + (cx as f32 - cam_target.x) * 0.5;
                         cam_target.y = cam_target.y + (cy as f32 - cam_target.y) * 0.5;
                     }
                 }
            }
        }
        
        // Pan (Botón derecho)
        if is_mouse_button_down(MouseButton::Right) {
            let delta = mouse_delta_position();
            cam_target.x -= delta.x / zoom_scale;
            cam_target.y += delta.y / (zoom_scale * screen_ratio);
        }

        // --- NAVIGATION LOGIC ---
        let mut trigger_nav = 0;
        if is_key_pressed(KeyCode::Right) { trigger_nav = 1; }
        if is_key_pressed(KeyCode::Left) { trigger_nav = -1; }

        if trigger_nav != 0 {
            if let Some(stem) = file_path.file_stem().and_then(|s| s.to_str()) {
                if stem.starts_with('T') {
                    // Split "T113-1" into ["113", "1"]
                    let parts: Vec<&str> = stem[1..].split('-').collect();
                    if let Ok(n) = parts[0].parse::<usize>() {
                        let next_n = if trigger_nav == 1 { n + 1 } else { n.saturating_sub(1) };
                        if next_n > 0 {
                            // Try multiple candidates
                            let mut candidates = vec![format!("T{}", next_n)];
                            if parts.len() > 1 {
                                candidates.insert(0, format!("T{}-{}", next_n, parts[1]));
                            }
                            
                            for c_stem in candidates {
                                let next_path = file_path.with_file_name(format!("{}.csv", c_stem));
                                if next_path.exists() {
                                    load_target = Some(next_path);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Zoom
        let mouse_wheel = mouse_wheel();
        if mouse_wheel.1 != 0.0 {
            if mouse_wheel.1 > 0.0 { zoom_scale *= 1.1; } else { zoom_scale /= 1.1; }
        }

        let (mouse_x, mouse_y) = mouse_position();
        let mouse_world = cam.screen_to_world(vec2(mouse_x, mouse_y));

        if is_mouse_button_pressed(MouseButton::Left) {
            history = Some(trees.clone());
            for (i, tree) in trees.iter().rev().enumerate() {
                use geo::Contains;
                let p = geo::Point::new(mouse_world.x as f64, mouse_world.y as f64);
                if tree.poly.contains(&p) {
                    dragging_idx = Some(trees.len() - 1 - i);
                    last_mouse_world = mouse_world;
                    break;
                }
            }
        }

        if is_mouse_button_down(MouseButton::Left) {
            if let Some(idx) = dragging_idx {
                let dx = mouse_world.x - last_mouse_world.x;
                let dy = mouse_world.y - last_mouse_world.y;
                trees[idx].x += dx as f64;
                trees[idx].y += dy as f64;
                trees[idx].update_poly();
                last_mouse_world = mouse_world;
                (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                score = calculate_score(&trees);
            }
        }
        if is_mouse_button_released(MouseButton::Left) { dragging_idx = None; }

        if let Some(idx) = dragging_idx {
            let rotation_step = if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) { 0.01 } else { 0.5 };
            if is_key_pressed(KeyCode::Q) || is_key_pressed(KeyCode::E) {
                history = Some(trees.clone());
            }
            if is_key_down(KeyCode::Q) {
                trees[idx].angle += rotation_step;
                trees[idx].update_poly();
                (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                score = calculate_score(&trees);
            }
            if is_key_down(KeyCode::E) {
                trees[idx].angle -= rotation_step;
                trees[idx].update_poly();
                (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                score = calculate_score(&trees);
            }
        }

        if is_key_pressed(KeyCode::P) { 
            if let Err(e) = save_solution(&file_path, &trees, score) {
                println!("Error guardando: {}", e);
            } else {
                println!("Guardado en {}", file_path.display());
            }
        }
        if is_key_pressed(KeyCode::O) {
            history = Some(trees.clone());
            trees = fine_tuning(trees, 100, false);
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
        }
        if is_key_pressed(KeyCode::G) {
            history = Some(trees.clone());
            apply_gravity(&mut trees, 50, 0.05);
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
        }

        // Terremoto (X) - Sacudida aleatoria
        if is_key_pressed(KeyCode::X) {
            history = Some(trees.clone());
            let mut rng = ::rand::thread_rng();
            for t in trees.iter_mut() {
                t.x += rng.gen_range(-0.5..0.5);
                t.y += rng.gen_range(-0.5..0.5);
                t.angle += rng.gen_range(-5.0..5.0);
                t.update_poly();
            }
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
        }

        // Explode (F) - Empujar hacia afuera desde el centro
        if is_key_pressed(KeyCode::F) {
            history = Some(trees.clone());
            let mut cx = 0.0;
            let mut cy = 0.0;
            for t in &trees { cx += t.x; cy += t.y; }
            cx /= trees.len() as f64;
            cy /= trees.len() as f64;

            // Factor de expansión homogénea (Escalado)
            // Aplica un "zoom out" muy fino desde el centro de masas para abrir épsilon entre todos
            let scale_factor = 1.001;

            for t in trees.iter_mut() {
                t.x = cx + (t.x - cx) * scale_factor;
                t.y = cy + (t.y - cy) * scale_factor;
                t.update_poly();
            }
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
            println!("   💥 Expansión Homogénea (x{:.4}) aplicada. Nuevo score: {:.6}", scale_factor, score);
        }

        // Rim Pressure (R) - Empujar bordo hacia adentro
        if is_key_pressed(KeyCode::R) {
            history = Some(trees.clone());
            // Aplicar presión (factor 0.1 para que sea visible pero controlado)
            apply_rim_pressure(&mut trees, 0.1); 
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
            println!("   🔘 Rim Pressure aplicado. Nuevo score: {:.6}", score);
        }

        // Undo (Ctrl+Z)
        if (is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl)) && is_key_pressed(KeyCode::Z) {
            if let Some(prev) = history.take() {
                trees = prev;
                (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
                score = calculate_score(&trees);
            }
        }

        // CMA-ES (C)
        if is_key_pressed(KeyCode::C) {
            history = Some(trees.clone());
            println!("\n🧬 Ejecutando CMA-ES...");
            trees = optimize_with_cmaes(trees, 50, 0, 50, true);
            (colliding_indices, collision_pairs) = get_colliding_indices(&trees);
            score = calculate_score(&trees);
            println!("   Nuevo score: {:.6}", score);
        }

        // UI de parámetros
        let mut start_sa = false;
        widgets::Window::new(hash!(), vec2(screen_width() - 260.0, 70.0), vec2(250.0, 480.0))
            .label("SA Configuration")
            .titlebar(true)
            .ui(&mut *root_ui(), |ui| {
                ui.label(None, "SA Performance:");
                ui.slider(hash!(), "Gens (k)", 10.0..5000.0, &mut sa_gens);
                ui.label(None, &format!("  Total: {} iterations", (sa_gens * 1000.0) as usize));
                
                ui.separator();
                ui.label(None, "Temperature:");
                ui.label(None, &format!("  Init T: {:.10}", sa_init_t));
                ui.slider(hash!(), "Init Temp", 0.000000001..0.1, &mut sa_init_t);
                
                ui.label(None, &format!("  Final T: {:.10}", sa_final_t));
                ui.slider(hash!(), "Final Temp", 0.0000000001..0.001, &mut sa_final_t);
                
                ui.separator();
                ui.label(None, &format!("Step Scale: {:.3}", sa_step_scale));
                ui.slider(hash!(), "Step", 0.001..1.0, &mut sa_step_scale);

                ui.separator();
                ui.label(None, "Onion Peeling (Freeze):");
                ui.checkbox(hash!(), "Auto-Freeze Base", &mut auto_freeze_base);
                if auto_freeze_base {
                    ui.slider(hash!(), "Base N", 0.0..trees.len() as f32, &mut base_n_val);
                    f_inner = (base_n_val / trees.len() as f32).min(1.0);
                } else {
                    ui.slider(hash!(), "Inner %", 0.0..1.0, &mut f_inner);
                }
                ui.label(None, &format!("  Inner Frozen: {:.1}%", f_inner * 100.0));
                
                ui.slider(hash!(), "Outer %", 0.0..1.0, &mut f_outer);
                ui.label(None, &format!("  Outer Frozen: {:.1}%", f_outer * 100.0));

                ui.separator();
                if ui.button(None, "🔥 START ANNEALING (A)") {
                    start_sa = true;
                }
            });

        widgets::Window::new(hash!(), vec2(screen_width() - 260.0, 10.0), vec2(250.0, 120.0))
            .label("Solution Navigator")
            .titlebar(false)
            .ui(&mut *root_ui(), |ui| {
                if ui.button(None, "<< Back (Left Arrow)") {
                    trigger_nav = -1;
                }
                if ui.button(None, "Next >> (Right Arrow)") {
                    trigger_nav = 1;
                }
                ui.label(None, &format!("File: T{}", file_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or("?".to_string())));
            });
        
        // El trigger_nav de los botones de la UI debe procesarse aquí o en el siguiente frame
        // Como ya pasó la lógica de navigation arriba, lo procesamos explícitamente si cambió.
        if trigger_nav != 0 && load_target.is_none() {
            // Re-ejecutar lógica de búsqueda de archivo (podríamos refactorizar esto a una función)
            if let Some(stem) = file_path.file_stem().and_then(|s| s.to_str()) {
                if stem.starts_with('T') {
                    let parts: Vec<&str> = stem[1..].split('-').collect();
                    if let Ok(n) = parts[0].parse::<usize>() {
                        let next_n = if trigger_nav == 1 { n + 1 } else { n.saturating_sub(1) };
                        if next_n > 0 {
                            let mut candidates = vec![format!("T{}", next_n)];
                            if parts.len() > 1 {
                                candidates.insert(0, format!("T{}-{}", next_n, parts[1]));
                            }
                            for c_stem in candidates {
                                let next_path = file_path.with_file_name(format!("{}.csv", c_stem));
                                if next_path.exists() {
                                    load_target = Some(next_path);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Simulated Annealing 2 (Adaptive - Background)
        if is_key_pressed(KeyCode::A) || start_sa {
            println!("\n🔥 Ejecutando SA Pro en segundo plano...");
            let trees_clone = trees.clone();
            
            // Usar parámetros dinámicos del UI
            let iters = (sa_gens * 1000.0) as usize;
            let init_t = sa_init_t as f64;
            let final_t = sa_final_t as f64;
            let step_scale = sa_step_scale as f64;
            let f_in = f_inner as f64;
            let f_out = f_outer as f64;

            std::thread::spawn(move || {
                optimize_with_annealing(
                    trees_clone, 
                    iters, 
                    true, 
                    f_in, 
                    f_out, 
                    init_t, 
                    final_t, 
                    0.3, 
                    step_scale,
                    Some(get_solutions_dir().join("live.csv"))
                );
            });
            auto_reload = true; // Activar visualización en vivo
            println!("   🚀 Proceso iniciado. Visualización en vivo ACTIVADA. (Frozen: Inner {:.1}%, Outer {:.1}%)", f_in*100.0, f_out*100.0);
        }

        // Kaggle Collision Check (K) - Run Python checker and update visual highlighting
        if is_key_pressed(KeyCode::K) {
            println!("\n🐍 Ejecutando Python Kaggle Checker...");
            let output = std::process::Command::new("python3")
                .arg("../check_kaggle_collisions.py")
                .arg(file_path.to_str().unwrap_or(""))
                .output();
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    println!("{}", stdout);
                    if !stderr.is_empty() { println!("stderr: {}", stderr); }
                    
                    // Parse output to find colliding tree IDs and update visual
                    // Format: "   Árbol 52 <-> Árbol 53"
                    colliding_indices.clear();
                    collision_pairs.clear();
                    
                    for line in stdout.lines() {
                        if line.contains("<->") {
                            // Parse "   Árbol X <-> Árbol Y"
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            // Expected: ["Árbol", "52", "<->", "Árbol", "53"]
                            if parts.len() >= 4 {
                                if let (Ok(id1), Ok(id2)) = (
                                    parts.get(1).unwrap_or(&"0").parse::<usize>(),
                                    parts.get(4).unwrap_or(&"0").parse::<usize>()
                                ) {
                                    // Find indices by tree ID
                                    for (idx, t) in trees.iter().enumerate() {
                                        if t.id == id1 || t.id == id2 {
                                            colliding_indices.insert(idx);
                                        }
                                    }
                                    // Find pair indices
                                    let idx1 = trees.iter().position(|t| t.id == id1);
                                    let idx2 = trees.iter().position(|t| t.id == id2);
                                    if let (Some(i1), Some(i2)) = (idx1, idx2) {
                                        collision_pairs.push((i1, i2));
                                    }
                                }
                            }
                        }
                    }
                    println!("   🎨 Marcados {} árboles como colisionados", colliding_indices.len());
                }
                Err(e) => println!("Error ejecutando Python: {}", e),
            }
        }

        set_camera(&cam);

        draw_line(-100.0, 0.0, 100.0, 0.0, 0.02, GRAY);
        draw_line(0.0, -100.0, 0.0, 100.0, 0.02, GRAY);

        // Bounding Box
        let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
        for t in &trees {
            use geo::BoundingRect;
            if let Some(rect) = t.poly.bounding_rect() {
                if rect.min().x < min_x { min_x = rect.min().x; }
                if rect.min().y < min_y { min_y = rect.min().y; }
                if rect.max().x > max_x { max_x = rect.max().x; }
                if rect.max().y > max_y { max_y = rect.max().y; }
            }
        }
        if !trees.is_empty() {
             let side = (max_x - min_x).max(max_y - min_y);
             let cx = (min_x + max_x) / 2.0;
             let cy = (min_y + max_y) / 2.0;
             draw_rectangle_lines((cx - side/2.0) as f32, (cy - side/2.0) as f32, side as f32, side as f32, 0.05, BLUE);
        }

        // Identificar árboles que tocan los extremos (Border trees)
        let mut border_indices = HashSet::new();
        let eps = 1e-4;
        for (i, t) in trees.iter().enumerate() {
            use geo::BoundingRect;
            if let Some(rect) = t.poly.bounding_rect() {
                if (rect.min().x - min_x).abs() < eps || (rect.max().x - max_x).abs() < eps ||
                   (rect.min().y - min_y).abs() < eps || (rect.max().y - max_y).abs() < eps {
                    border_indices.insert(i);
                }
            }
        }

        for (i, tree) in trees.iter().enumerate() {
            let is_colliding = colliding_indices.contains(&i);
            let is_selected = Some(i) == dragging_idx;

            use geo::CoordsIter;
            let coords: Vec<Vec2> = tree.poly.exterior().coords_iter()
                .map(|c| vec2(c.x as f32, c.y as f32))
                .collect();
            
            let center = vec2(tree.x as f32, tree.y as f32);

            // Body
            let body_color = if is_colliding { color_collision } 
                           else if is_selected { color_selected }
                           else if border_indices.contains(&i) { color_border }
                           else { Color::new(0.13, 0.55, 0.13, 1.0) };

            for j in 0..coords.len() - 1 {
                draw_triangle(center, coords[j], coords[j+1], body_color);
            }
            if coords.len() > 1 {
                draw_triangle(center, coords[coords.len()-1], coords[0], body_color);
            }

            // Outline
            let thickness = 0.003 / zoom_scale; 
            for j in 0..coords.len() - 1 {
                draw_line(coords[j].x, coords[j].y, coords[j+1].x, coords[j+1].y, thickness, color_outline);
            }
            if coords.len() > 1 {
                draw_line(coords[coords.len()-1].x, coords[coords.len()-1].y, coords[0].x, coords[0].y, thickness, color_outline);
            }

            draw_circle(center.x, center.y, thickness * 2.0, Color::new(0.5, 0.25, 0.0, 0.8));
            
            // Draw tree ID label
            let font_size = 0.08 / zoom_scale;
            let id_text = format!("{}", tree.id);
            let text_color = if is_colliding { WHITE } else { BLACK };
            // Draw text at tree center (offset slightly up)
            draw_text(&id_text, center.x - font_size * 0.3, center.y + 0.3, font_size, text_color);
        }

        set_default_camera();
        let status_color = if colliding_indices.is_empty() { DARKGREEN } else { RED };
        let status_text = if colliding_indices.is_empty() { 
            "VALID (Kaggle-compatible)".to_string() 
        } else { 
            format!("COLLISION: {} pairs", collision_pairs.len()) 
        };
        
        // Calcular dimensiones del bounding box
        let width = max_x - min_x;
        let height = max_y - min_y;
        let side = width.max(height);
        
        draw_text(&format!("Kaggle Score: {:.6}", score), 20.0, 30.0, 30.0, BLACK);
        draw_text(&format!("Status: {}", status_text), 20.0, 60.0, 30.0, status_color);
        
        // Show collision pairs
        if !collision_pairs.is_empty() {
            let pairs_str: String = collision_pairs.iter().take(5)
                .map(|(i, j)| format!("{}↔{}", trees[*i].id, trees[*j].id))
                .collect::<Vec<_>>().join(", ");
            let suffix = if collision_pairs.len() > 5 { format!("... +{}", collision_pairs.len() - 5) } else { String::new() };
            draw_text(&format!("Pairs: {}{}", pairs_str, suffix), 20.0, 90.0, 22.0, RED);
        }
        
        // Show selected tree info
        let selected_info = if let Some(idx) = dragging_idx {
            format!("Selected: Tree #{} (x:{:.3}, y:{:.3}, deg:{:.1})", trees[idx].id, trees[idx].x, trees[idx].y, trees[idx].angle)
        } else {
            "Selected: None".to_string()
        };
        draw_text(&selected_info, 20.0, 120.0, 22.0, DARKBLUE);
        
        draw_text(&format!("Side: {:.4} | W: {:.4} | H: {:.4}", side, width, height), 20.0, 150.0, 22.0, DARKBLUE);
        draw_text(&format!("Trees: {} | Area: {:.4}", trees.len(), side * side), 20.0, 180.0, 22.0, DARKBLUE);
        // Mostrar archivo actual
        let current_file = file_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or("?".to_string());
        let live_tag = if auto_reload { " [LIVE]" } else { "" };
        draw_text(&format!("📁 {}{}", current_file, live_tag), screen_width() - 250.0, 30.0, 24.0, if auto_reload { RED } else { DARKGREEN });
        
        draw_text("L: Load | V: Auto-Live | Q/E: Rotate | O: Fine-Tune | G: Gravity | R: Rim | C: CMA | A: SA | X: Shake | K: Kaggle | P: Save", 20.0, screen_height() - 40.0, 17.0, DARKGRAY);

        next_frame().await
    }
}