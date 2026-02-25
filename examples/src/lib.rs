#[cfg(feature = "bevy")]
use bevy::prelude::*;
use freestyle_sculpt::ray::Ray;
use mesh_graph::MeshGraph;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(Resource))]
pub struct Log {
    pub mesh_graph: MeshGraph,
    pub actions: Vec<EditAction>,
}

impl Log {
    pub fn new(mesh_graph: &MeshGraph) -> Self {
        Self {
            mesh_graph: mesh_graph.clone(),
            actions: vec![],
        }
    }

    pub fn log_action(&mut self, action: EditAction) {
        self.actions.push(action);
    }
}

#[derive(Serialize, Deserialize)]
pub struct EditAction {
    pub ty: EditActionType,
    pub ray: Ray,
    pub selector: usize,
    pub deformation: usize,
}

#[derive(Serialize, Deserialize)]
pub enum EditActionType {
    MouseDown,
    MouseUp,
    MouseMove,
}
