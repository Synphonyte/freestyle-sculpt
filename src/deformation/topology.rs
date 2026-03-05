use hashbrown::{HashMap, HashSet};
use mesh_graph::{HalfedgeId, MergeVerticesOneRing, MeshGraph, VertexId};
use rapier3d::prelude::*;
use tracing::{error, instrument};

use crate::SculptParams;

pub struct TopologyManager {
    pub protected_halfedges: HashSet<HalfedgeId>,
    pub protected_vertices: HashSet<VertexId>,

    pub collision_pipeline: CollisionPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhaseBvh,
    pub narrow_phase: NarrowPhase,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,

    pub vertex_id_to_collider_handle: HashMap<VertexId, ColliderHandle>,
}

impl TopologyManager {
    #[cfg(feature = "rerun")]
    pub fn log_rerun(&self) {
        Self::log_colliders_rerun("all", self.collider_set.iter().map(|(_, col)| col));
    }

    #[cfg(feature = "rerun")]
    pub fn log_colliders_rerun<'a>(name: &str, colliders: impl IntoIterator<Item = &'a Collider>) {
        let (positions, radii): (Vec<_>, Vec<_>) = colliders
            .into_iter()
            .map(|col: &Collider| {
                (
                    mesh_graph::utils::vec3_array(col.translation()),
                    col.shape().as_ball().unwrap().radius,
                )
            })
            .unzip();

        mesh_graph::RR
            .log(
                format!("freestyle/colliders/{name}"),
                &rerun::Ellipsoids3D::from_centers_and_radii(positions, radii),
            )
            .unwrap();
    }

    pub fn create_collider(sculpt_params: &SculptParams, vertex_id: VertexId) -> ColliderBuilder {
        ColliderBuilder::ball(sculpt_params.max_thickness_half)
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .active_hooks(ActiveHooks::FILTER_INTERSECTION_PAIR)
            .active_collision_types(ActiveCollisionTypes::FIXED_FIXED)
            .user_data(vertex_id.into())
    }

    pub fn new(mesh_graph: &MeshGraph, sculpt_params: SculptParams) -> Self {
        let mut collider_set = ColliderSet::new();
        let mut vertex_id_to_collider_handle = HashMap::new();

        for (vertex_id, &pos) in &mesh_graph.positions {
            let collider = Self::create_collider(&sculpt_params, vertex_id).translation(pos);
            let collider_handle = collider_set.insert(collider);
            vertex_id_to_collider_handle.insert(vertex_id, collider_handle);
        }

        Self {
            protected_halfedges: HashSet::new(),
            protected_vertices: HashSet::new(),

            collision_pipeline: CollisionPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set,

            vertex_id_to_collider_handle,
        }
    }

    #[instrument(skip_all)]
    pub fn update_collisions_and_merge(
        &mut self,
        mesh_graph: &mut MeshGraph,
        sculpt_params: &SculptParams,
    ) -> Option<MergeVerticesOneRing> {
        #[cfg(feature = "rerun")]
        self.log_rerun();

        self.collision_pipeline.step(
            0.002,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &TopologyPhysicsHooks {
                mesh_graph,
                protected_vertices: &self.protected_vertices,
            },
            &(),
        );

        for (col_h1, col_h2, intersecting) in self.narrow_phase.intersection_pairs() {
            if !intersecting {
                continue;
            }

            let collider1 = &self.collider_set[col_h1];
            let collider2 = &self.collider_set[col_h2];

            #[cfg(feature = "rerun")]
            Self::log_colliders_rerun("collision", [collider1, collider2]);

            let v_id1 = VertexId::from(collider1.user_data);
            let v_id2 = VertexId::from(collider2.user_data);

            // with previous merges two vertices might have become connected or protected that weren't before
            if mesh_graph.vertices.contains_key(v_id1)
                && mesh_graph.vertices.contains_key(v_id2)
                && mesh_graph.halfedge_from_to(v_id1, v_id2).is_none()
                && !self.protected_vertices.contains(&v_id1)
                && !self.protected_vertices.contains(&v_id2)
            {
                #[cfg(feature = "rerun")]
                {
                    Self::log_colliders_rerun("merge", [collider1, collider2]);
                    mesh_graph.log_verts_rerun("merge", &[v_id1, v_id2]);
                }

                let merged = mesh_graph.merge_vertices_one_rings(
                    v_id1,
                    v_id2,
                    sculpt_params.min_edge_length_squared,
                    &mut self.protected_halfedges,
                    &mut self.protected_vertices,
                );

                #[cfg(feature = "rerun")]
                mesh_graph.log_rerun();

                return Some(merged);
            }
        }

        self.protected_halfedges.clear();
        self.protected_vertices.clear();

        None
    }

    pub fn sync_mesh_graph(&mut self, mesh_graph: &MeshGraph, sculpt_params: &SculptParams) {
        for (vertex_id, &position) in mesh_graph.positions.iter() {
            let collider_handle = *self
                .vertex_id_to_collider_handle
                .entry(vertex_id)
                .or_insert_with(|| {
                    self.collider_set
                        .insert(Self::create_collider(sculpt_params, vertex_id))
                });

            self.collider_set[collider_handle].set_translation(position);
        }

        let mut removed_vertices = Vec::new();
        for (&vertex_id, &collider_handle) in self.vertex_id_to_collider_handle.iter() {
            if !mesh_graph.vertices.contains_key(vertex_id) {
                self.collider_set.remove(
                    collider_handle,
                    &mut self.island_manager,
                    &mut self.rigid_body_set,
                    false,
                );

                removed_vertices.push(vertex_id);
            }
        }

        for removed_vertex in removed_vertices {
            self.vertex_id_to_collider_handle.remove(&removed_vertex);
        }
    }
}

pub struct TopologyPhysicsHooks<'a> {
    pub mesh_graph: &'a MeshGraph,
    pub protected_vertices: &'a HashSet<VertexId>,
}

impl<'a> PhysicsHooks for TopologyPhysicsHooks<'a> {
    #[instrument(skip_all)]
    fn filter_intersection_pair(&self, context: &PairFilterContext) -> bool {
        let collider1 = &context.colliders[context.collider1];
        let collider2 = &context.colliders[context.collider2];

        let v_id1 = VertexId::from(collider1.user_data);
        let v_id2 = VertexId::from(collider2.user_data);

        if self.protected_vertices.contains(&v_id1) || self.protected_vertices.contains(&v_id2) {
            #[cfg(feature = "rerun")]
            TopologyManager::log_colliders_rerun(
                "collision_filter/rejected_protected",
                [collider1, collider2],
            );

            return false;
        }

        let halfedge_from_to = self.mesh_graph.halfedge_from_to(v_id1, v_id2);

        if halfedge_from_to.is_some() {
            false
        } else {
            let Some(&norm_1) = self.mesh_graph.vertex_normals.as_ref().unwrap().get(v_id1) else {
                error!("Normal 1 not found");
                return false;
            };
            let Some(&norm_2) = self.mesh_graph.vertex_normals.as_ref().unwrap().get(v_id2) else {
                error!("Normal 2 not found");
                return false;
            };

            if norm_1.dot(norm_2) > 0.3 {
                #[cfg(feature = "rerun")]
                TopologyManager::log_colliders_rerun(
                    "collision_filter/rejected_normals",
                    [collider1, collider2],
                );

                false
            } else {
                #[cfg(feature = "rerun")]
                TopologyManager::log_colliders_rerun(
                    "collision_filter/accepted",
                    [collider1, collider2],
                );

                true
            }
        }
    }
}
