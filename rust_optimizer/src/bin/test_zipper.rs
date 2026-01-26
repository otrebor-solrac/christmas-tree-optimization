use christmas_tree_optimizer::algorithms::strategies::generate_grid_zipper;
use christmas_tree_optimizer::utils::geometry::calculate_score;
use christmas_tree_optimizer::utils::collisions::check_collisions;

fn main() {
    let target_n = 200;
    
    println!("Testing Grid Zipper parameters for N={}", target_n);
    
    // Test exact values that optimize_zipper reported
    let test_cases = [
        (0.4125, 0.82),
        (0.4125, 0.8200),
        (0.4138, 0.8200),
        (0.4150, 0.8200),
        // Try some larger row heights
        (0.4125, 0.85),
        (0.4125, 0.90),
        (0.42, 0.85),
    ];
    
    for (stride_x, row_height) in test_cases {
        if let Some(trees) = generate_grid_zipper(target_n, stride_x, row_height) {
            let has_collisions = check_collisions(&trees);
            let score = calculate_score(&trees);
            println!("stride_x={:.4}, row_height={:.4} -> {} trees, collisions={}, score={:.6}", 
                stride_x, row_height, trees.len(), has_collisions, score);
        } else {
            println!("stride_x={:.4}, row_height={:.4} -> None (no valid grid)", stride_x, row_height);
        }
    }
}
