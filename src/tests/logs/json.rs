use crate::ray::Ray;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A recorded input log. The inner `Vec` is public so replaying code can consume it.
#[derive(Serialize, Deserialize)]
pub struct InputLog(pub Vec<InputLogEntry>);

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum InputLogEntry {
    #[serde(rename_all = "camelCase")]
    PointerDownPerspective {
        local_ray: RayMap,
        state: LoggedSculptState,
    },
    PointerDownSlice {
        point: Vec3Map,
        state: LoggedSculptState,
    },
    #[serde(rename_all = "camelCase")]
    PointerMovePerspective {
        local_ray: RayMap,
    },
    PointerMoveSlice {
        point: Vec3Map,
    },
    PointerUp,
    #[serde(rename_all = "camelCase")]
    RemoveWithLasso {
        points: Vec<[f64; 3]>,
        object_to_camera_matrix: Matrix4,
        projection_matrix: Matrix4,
        state: LoggedSculptState,
    },
}

/// THREE.Matrix4 serialized via JSON.stringify → { elements: [f64; 16] }
#[derive(Serialize, Deserialize)]
pub struct Matrix4 {
    pub elements: [f32; 16], // column-major, same as three.js
}

/// THREE.Vector3 serialized via JSON.stringify
#[derive(Serialize, Deserialize)]
pub struct Vec3Map {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3Map> for Vec3 {
    fn from(value: Vec3Map) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

/// THREE.Ray serialized via JSON.stringify
#[derive(Serialize, Deserialize)]
pub struct RayMap {
    pub origin: Vec3Map,
    pub direction: Vec3Map,
}

impl From<RayMap> for Ray {
    fn from(value: RayMap) -> Self {
        Self {
            origin: value.origin.into(),
            direction: value.direction.into(),
        }
    }
}

/// The sculpt state captured when an input event was recorded.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggedSculptState {
    pub strength: f32,
    pub deformation_type: u16,
    pub radius: f32,
    pub falloff: f32,
    pub axis_normalized: Vec3Map,
    pub sculpt_params: f32,
}
