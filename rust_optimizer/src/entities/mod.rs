// Entity definitions
use geo::{LineString, Polygon, Rotate, Translate};

// --- CONSTANTES DE GEOMETRÍA ---
pub const SCALE_FACTOR: f64 = 1.0;
pub const TRUNK_W: f64 = 0.15;
pub const TRUNK_H: f64 = 0.2;
pub const BASE_W: f64 = 0.7;
pub const BASE_Y: f64 = 0.0;
pub const MID_W: f64 = 0.4;
pub const TIER_2_Y: f64 = 0.25;
pub const TOP_W: f64 = 0.25;
pub const TIER_1_Y: f64 = 0.5;
pub const TIP_Y: f64 = 0.8;
pub const TRUNK_BOTTOM: f64 = -TRUNK_H;

#[derive(Clone, Debug)]
pub struct Tree {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub poly: Polygon<f64>,
}

impl Tree {
    pub fn new(id: usize, x: f64, y: f64, angle: f64) -> Self {
        let mut t = Tree {
            id,
            x,
            y,
            angle,
            poly: Polygon::new(LineString::from(Vec::<(f64, f64)>::new()), vec![]),
        };
        t.update_poly();
        t
    }

    pub fn update_poly(&mut self) {
        // Definición del polígono base con factor de escala
        let s = SCALE_FACTOR;
        let coords = vec![
            (0.0 * s, TIP_Y * s),
            (TOP_W / 2.0 * s, TIER_1_Y * s),
            (TOP_W / 4.0 * s, TIER_1_Y * s),
            (MID_W / 2.0 * s, TIER_2_Y * s),
            (MID_W / 4.0 * s, TIER_2_Y * s),
            (BASE_W / 2.0 * s, BASE_Y * s),
            (TRUNK_W / 2.0 * s, BASE_Y * s),
            (TRUNK_W / 2.0 * s, TRUNK_BOTTOM * s),
            (-TRUNK_W / 2.0 * s, TRUNK_BOTTOM * s),
            (-TRUNK_W / 2.0 * s, BASE_Y * s),
            (-BASE_W / 2.0 * s, BASE_Y * s),
            (-MID_W / 4.0 * s, TIER_2_Y * s),
            (-MID_W / 2.0 * s, TIER_2_Y * s),
            (-TOP_W / 4.0 * s, TIER_1_Y * s),
            (-TOP_W / 2.0 * s, TIER_1_Y * s),
            (0.0 * s, TIP_Y * s),
        ];

        let line_string = LineString::from(coords);
        let poly = Polygon::new(line_string, vec![]);
        let rotated = poly.rotate_around_point(self.angle, geo::Point::new(0.0, 0.0));
        self.poly = rotated.translate(self.x * s, self.y * s);
    }
}
