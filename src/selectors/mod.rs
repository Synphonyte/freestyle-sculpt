mod distance_with_falloff;
mod geodesic_with_falloff;
mod traits;

pub use distance_with_falloff::*;
pub use geodesic_with_falloff::*;

pub use traits::*;

pub type FalloffFn = fn(f32) -> f32;

pub const LINEAR_FALLOFF: FalloffFn = |x| x;

pub const SMOOTH_FALLOFF: FalloffFn = |x| {
    let x2 = x * x;
    3.0 * x2 - 2.0 * x2 * x
};

fn sphere_with_falloff_weight(
    distance: f32,
    radius: f32,
    falloff: f32,
    falloff_func: FalloffFn,
) -> f32 {
    let rf = radius + falloff;

    if distance <= radius {
        1.0
    } else if distance <= rf {
        falloff_func((rf - distance) / falloff)
    } else {
        0.0
    }
}
