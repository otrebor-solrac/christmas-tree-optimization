use christmas_tree_optimizer::algorithms::strategies::generate_grid_zipper;
use christmas_tree_optimizer::utils::geometry::calculate_score;

fn main() {
    let target_n = 190;
    println!(">>> Optimizing Grid Zipper for N={} <<<", target_n);

    let mut best_stride_x = 0.0;
    let mut best_row_height = 0.0;
    let mut best_score = f64::INFINITY;

    // Grid search range - expanded to find valid params
    let stride_range: Vec<f64> = (41000..42000).step_by(10).map(|x| x as f64 / 100000.0).collect(); // 0.40 to 0.50
    let height_range: Vec<f64> = (82000..83000).step_by(10).map(|y| y as f64 / 100000.0).collect(); // 0.80 to 1.00

    println!("Searching... (Exploration can take a moment)");

    use rayon::prelude::*;
    use std::sync::Mutex;

    let best_found = Mutex::new((f64::INFINITY, 0.0, 0.0));

    stride_range.into_par_iter().for_each(|stride_x| {
        for &row_height in &height_range {
            if let Some(trees) = generate_grid_zipper(target_n, stride_x, row_height) {
                let score = calculate_score(&trees);
                let mut best = best_found.lock().unwrap();
                if score < best.0 {
                    *best = (score, stride_x, row_height);
                    println!("New Best! stride_x: {:.4}, row_height: {:.4}, score: {:.6}", 
                             stride_x, row_height, score);
                }
            }
        }
    });

    let (final_best_score, final_best_stride_x, final_best_row_height) = *best_found.lock().unwrap();
    best_score = final_best_score;
    best_stride_x = final_best_stride_x;
    best_row_height = final_best_row_height;

    println!("\n✅ Optimization Finished!");
    println!("Final results for N={}:", target_n);
    println!("Best stride_x:   {:.4}", best_stride_x);
    println!("Best row_height: {:.4}", best_row_height);
    println!("Best score:      {:.6}", best_score);
    
    println!("\nSuggested update in strategies.rs:");
    println!("let stride_x = {:.4};", best_stride_x);
    println!("let row_height = {:.4};", best_row_height);
}
