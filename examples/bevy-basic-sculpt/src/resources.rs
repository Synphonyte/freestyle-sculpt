use bevy::prelude::*;
pub use examples::*;
use freestyle_sculpt::{deformation::DeformationField, selectors::MeshSelector};

#[derive(Resource, Copy, Clone, Default, Deref, DerefMut)]
pub struct CurrentDeformation(usize);

#[derive(Deref, DerefMut)]
pub struct AvailableDeformations(Vec<Box<dyn DeformationField>>);

impl AvailableDeformations {
    pub fn new(deformations: Vec<Box<dyn DeformationField>>) -> Self {
        Self(deformations)
    }
}

#[derive(Resource, Copy, Clone, Default, Deref, DerefMut)]
pub struct CurrentSelection(usize);

#[derive(Deref, DerefMut)]
pub struct AvailableSelections(Vec<Box<dyn MeshSelector>>);

impl AvailableSelections {
    pub fn new(selections: Vec<Box<dyn MeshSelector>>) -> Self {
        Self(selections)
    }
}
