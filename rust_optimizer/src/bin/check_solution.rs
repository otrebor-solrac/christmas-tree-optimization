use christmas_tree_optimizer::utils::io::load_trees;
use christmas_tree_optimizer::utils::collisions::check_collisions;
use christmas_tree_optimizer::utils::geometry::calculate_score;
use std::path::Path;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: check_solution <path_to_csv>");
        return;
    }
    
    let path = Path::new(&args[1]);
    println!("Verificando: {}", path.display());
    
    match load_trees(path) {
        Ok(trees) => {
            println!("Árboles cargados: {}", trees.len());
            
            let has_collisions = check_collisions(&trees);
            println!("¿Tiene colisiones?: {}", has_collisions);
            
            let score = calculate_score(&trees);
            println!("Score calculado: {}", score);
            
            if has_collisions {
                // Encontrar cuáles colisionan
                use geo::prelude::*;
                use geo::coordinate_position::CoordPos;
                use geo::dimensions::Dimensions;
                
                println!("\nColisiones encontradas:");
                for i in 0..trees.len() {
                    for j in (i+1)..trees.len() {
                        let bounds_i = trees[i].poly.bounding_rect().unwrap();
                        let bounds_j = trees[j].poly.bounding_rect().unwrap();
                        
                        if bounds_i.intersects(&bounds_j) {
                            let matrix = trees[i].poly.relate(&trees[j].poly);
                            if matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty {
                                println!("  Tree {} ({:.6}, {:.6}, {:.1}°) vs Tree {} ({:.6}, {:.6}, {:.1}°)", 
                                    trees[i].id, trees[i].x, trees[i].y, trees[i].angle,
                                    trees[j].id, trees[j].x, trees[j].y, trees[j].angle);
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("Error cargando: {}", e);
        }
    }
}
