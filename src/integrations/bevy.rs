use bevy::prelude::*;

use crate::ray::Ray;

impl From<Ray3d> for Ray {
    fn from(ray: Ray3d) -> Self {
        Self {
            origin: glam::Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z),
            direction: glam::Vec3::new(ray.direction.x, ray.direction.y, ray.direction.z),
        }
    }
}
