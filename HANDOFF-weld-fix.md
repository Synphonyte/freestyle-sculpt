# Handoff — intermittent weld-hole fix (mesh-graph + freestyle-sculpt)

Status: the fixture is **recovered** and the end-to-end loop runs again. Five fixes landed (see
below); the weld now survives **2000/2000** iterations in the merge order the topology manager
uses. It is **not fully fixed**: reversing that order still finds fresh cases within a handful of
iterations, so more merge configurations still open the mesh. The capture tooling for finding
them is in the tree (`WELD_CAPTURE=1`). Write date: 2026-09-15.

## The bug

`test_apply_weld_amount_applies_close` (freestyle-sculpt, mirroring voxel-wasm) intermittently
left boundary halfedges after fusing two gapped cubes (~1 in 12 runs). The equivalent
`merge_vertices_one_rings` call in the weld could leave the fused region open.

There were **two** bugs, and they compounded: a real hole-producing defect in the band planner
(the *lens* case below), and hash-order dependence that decided whether the subsequent repair
happened to paper over it. Fixing only the ordering would have made the hole deterministic
rather than absent.

## Root cause (fixed)

Nondeterministic hash iteration order. `hashbrown`'s default hasher (foldhash) seeds every
freshly-created table randomly, so iterating the same logical set across runs yields different
orders. `merge_vertices_one_rings` and the vertex-neighborhood cleanup pick elements from such
sets via `.iter().next()` (fan start halfedges, flap candidates). On **lens-shaped merges**
(two one-rings sharing 2+ vertices) the choice steers the repair branch: the band leaves the
other lens segment's rim open, and the cleanup's repair either seals it or leaves boundary
halfedges depending purely on hash seeds.

Deterministic reproduction used before fixing: `merge(1218, 1243)` with
`flip_threshold_sqr = 0.0096` (weld's min-edge). Rings `ring1 [1744, 1219, 720, 358, 1217, 1738]`,
`ring2 [1217, 1806, 1779, 1244, 720, 347]`, shared `{720, 1217}`; band spans only `1217..=720`;
strip face `f701 (347,358,720)` torn out. In-process replay of the identical merge gave clean or
boundary `[(347,358),(358,1217),(1217,347)]` / `(358,720)` / `(358,1243)` variants — different
every run.

## Root cause #2 — the lens hole (the real defect)

`merge_vertices_one_rings` plans the band between the two one-rings as a sequence of
`ConnectPair` ranges. The walk in `compute_range_pairs_to_connect` emits a pair whenever it hits
a cap (a shared or already-connected vertex), and after the loop it pushes **whatever is left
over** between the last emitted range and where the walk started.

That leftover push required a span of two or more vertices on one of the rings
(`diff1 > 1 || diff2 > 1`). A *lens* merge — two one-rings sharing two vertices that are **not
adjacent** in the rings — splits the band into two segments, and the short one is exactly one
step on each ring. So it was silently dropped. Its faces had already been torn out with the
merged vertices' stars by `remove_neighbor_faces`, and nothing re-filled them: the merge ended
with a triangular hole.

Fix: push the leftover whenever it spans anything at all, measured as the **forward** arc
(`(end + len - start) % len`) rather than a symmetric distance, and skip it when the walk has
come full circle (new `wrapped` flag) so the first range is not duplicated.

The existing shared-vertex fixtures all have their two shared vertices *adjacent*, which leaves
one segment empty — which is why this went untested for so long.

## Fix in mesh-graph (`/Users/maccesch/Programming/mesh-graph`, HEAD `59d15a7`, uncommitted)

**Approach: make the ops order-independent. Do NOT pin the hasher.** An earlier
attempt introduced a `crate::collections` module with a fixed-seed foldhash
`BuildHasher` behind `HashMap`/`HashSet` aliases. That was reverted: it only made
the wrong branch *reproducible*, it changed the public API (the marked-set types
cross the crate boundary), and hashbrown's default hasher is what the crate wants
for speed. The order-dependence lives in the algorithms and is fixed there.

Every changed site had the same shape: a set was built from a source that already
has a deterministic order (a stored star list, a slotmap scan), and then the code
picked or drained elements *through the set*, so hashbrown's per-table random seed
decided the outcome.

- `src/ops/cleanup/vertex_neighborhood.rs`
  - `split_disconnected_neighborhoods`: the fan walks are driven by the stored
    `outgoing_halfedges` list instead of `outgoing_halfedges.iter().next()`, and
    the surviving star is written back in that same list order (it used to be
    written back in hash order, scrambling the rotation order for every later
    traversal). This is the single load-bearing fix for the lens case.
  - `remove_neighboring_flaps`: `halfedges_of_faces` is a `Vec` in slotmap scan
    order, not a `HashSet`, so the `array_combinations` pair loop repairs
    candidates in a fixed order.
- `src/ops/remove.rs` (`remove_face`): `removed_halfedges` and
  `touched_start_vertices` are ordered `Vec`s (helper `push_unique`). Their order
  decides the slotmap free order — and hence which ids the *next* allocations
  reuse — plus the returned removal lists.
- `src/ops/edit.rs` (`merge_vertices`): `faces_to_remove` / `halfedges_to_remove`
  keep a `Vec` for order next to the `HashSet` for membership; the duplicate-edge
  pass iterates its end vertices in `out_hes` order rather than `HashMap` order.
- `src/ops/merge_one_ring/mod.rs` (`compute_range_pairs_to_connect`): the leftover-range push,
  see "Root cause #2" above.
- Tests:
  - `src/ops/merge_one_ring/tests.rs::test_merge_one_ring_lens_stays_watertight` — the captured
    lens region must leave its outer cut exactly unchanged. Fails without the planner fix
    (boundary 17 -> 20, a triangular hole at 347/358/1217).
  - `src/ops/merge_one_ring/tests.rs::test_merge_one_ring_lens_is_order_independent`
    (the 30-vertex/41-face lens region captured from the weld) runs the same merge
    200x in-process and requires one identical topology fingerprint. Before the
    fix: 14 distinct outcomes in 200 runs.
  - `src/ops/edit.rs::test_merge_vertices_is_order_independent` does the same for
    `merge_vertices`, allocating after the merge so the free order is observable.
    Before the fix: 200 distinct outcomes in 200 runs.
  - `src/utils.rs::topology_fingerprint` (cfg(test)): faces as canonicalised
    vertex-id triples + boundary edges + counts, raw slotmap ids included.

No public API change, so no `[patch.crates-io]`-forcing release is required.


## Validation already done

- mesh-graph: `cargo test --features gltf,serde,instrumentation --lib` → **82/82
  pass**; `cargo clippy --features gltf,serde,instrumentation --all-targets` →
  clean; `cargo fmt --check` clean.
- Each determinism test was verified to actually fail against the pre-fix code
  (see the counts above).
- **End-to-end**: with the fixture recovered, `test_apply_weld_amount_applies_close`
  passes and the mesh fuses to ~1400-1500 faces, boundary 0. The repeated loop still
  fails (see "Remaining work") — the single-run test is simply too coarse to catch it.


## Fixture recovery (done) — and what the wipe actually cost

The pre-wipe `morph_open_close.rs` was still in the object store. `git fsck` lists no useful
dangling blob by name, but scanning every object does find it:

```
git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype)' \
  | awk '$2=="blob"{print $1}' \
  | while read o; do git cat-file blob "$o" | grep -lq tessellated_cube_at && echo "$o"; done
```

11 blobs match; `2105567a3ddced2907abec784b88d13d13fb7c34` is the pre-wipe file (its only
differences from the working tree were inside the test module). It is restored.

**The tessellation was never the problem.** Nine of the eleven blobs share one
`tessellated_cube_at`, and its grid loop is *algebraically identical* to the reconstruction —
`[a, b, c, a, c, d]` with `c = a+row+1, d = a+row` emits the same two triangles as
`[a, b, d, a, d, c]` with `c = a+row, d = a+row+1`. What the wipe actually lost was one
character in the test helper:

```rust
    // apply_weld_amount
    morphological_open_close(..., -amount.abs(), ())   // original
    morphological_open_close(..., amount, ())          // reconstruction
```

The sign. A positive amount runs the *open*, which eats the mesh — hence `1536 -> 0 faces`,
and hence the vacuous pass (an empty mesh has no boundary halfedges).

Fixture signature now matches the handoff target exactly: 772 verts / 1536 faces / 0 boundary,
welding to ~1400-1500 faces.

## Defects found and fixed in the band planner / face insertion

Each was found by `WELD_CAPTURE=1` (below), reproduced as a cut-out region test in mesh-graph,
and each test was verified to fail with its fix reverted.

1. **Lens: the short segment was dropped.** `compute_range_pairs_to_connect` pushes whatever is
   left over between the last emitted range and where the walk started, but required a span of
   two or more on one ring. A lens whose two shared vertices are not adjacent leaves exactly one
   step on each ring, so it was dropped and nothing re-filled the faces that
   `remove_neighbor_faces` had already torn out. Fixed by measuring the **forward** arc
   (`(end + len - start) % len`) and pushing any non-empty leftover, with a `wrapped` flag so a
   walk that comes full circle does not re-emit its first range.
   → `test_merge_one_ring_lens_stays_watertight`
2. **Asymmetric lens: the leftover range did not reach its cap on both rings.** The walk advances
   both rings in lockstep and only lets one catch up when the loop body sees a shared vertex — and
   it stops as soon as ring 1 is back at its start, so the final cap never got that treatment.
   With the shared pair adjacent on one ring and four steps apart on the other, the leftover
   covered one step instead of four and the merge left an open quad. Fixed by applying the same
   catch-up to the leftover range before pushing it.
   → `test_merge_one_ring_asymmetric_lens_stays_watertight`
3. **Parallel edges left behind by planned faces.** `add_or_get_boundary_edge` only reuses a free
   half in the direction it was asked for; when a planned face needs `a -> b` and an earlier face
   of the same merge took that slot, it adds a *second* edge between the same two vertices. That
   is normally transient — refusing it outright regresses four other captured regions — but when
   nothing claims the free halves they stay face-less forever: the mesh reports them as boundary,
   a hole with no face missing from it. Fixed with a sweep (`weld_parallel_edges`) that runs after
   the new faces and again after the cleanup, twinning the two faced halves to each other and
   dropping the orphans.
   → `test_merge_one_ring_parallel_edge_leftover_stays_watertight`

Plus the two ordering fixes described above (mesh-graph ops, freestyle collision pairs).

## Extracting a case (automated) — and the thing that made it work

`WELD_CAPTURE=1` turns the merge loop in `topology.rs` into a case extractor. When a merge opens
a watertight mesh it cuts the neighborhood at radius 1..4, replays the merge on each cut
standalone, takes the smallest one that still opens, minimises it face by face, and prints it as
the constants `OPEN_WELD_CASES` in mesh-graph uses.

```
cd freestyle-sculpt
WELD_CAPTURE=1 WELD_LOOP_ITERATIONS=500 cargo test --release --features instrumentation \
  weld_flake_loop_until_failure -- --ignored --nocapture
```

Wrap the sort key in `topology.rs` in `std::cmp::Reverse` to make the loop fail in tens of
iterations instead of thousands.

**Two things have to be right or the extraction lies, and both cost a long detour to find:**

1. **Capture the outgoing-halfedge order.** `one_ring()` starts at the vertex's
   `outgoing_halfedge` seed and the band planner scans the ring from index 0, so the per-vertex
   order of `outgoing_halfedges` decides where the band is cut. A rebuild derives that order from
   the order faces happen to be added — not from the history the live mesh had. Until the capture
   recorded and restored it (`REGION_OUTGOING`, `restore_outgoing_order`), *no* faithful cut
   reproduced anything, at any radius. This is almost certainly why earlier attempts at replaying
   captured state kept failing.
2. **Keep both merged vertices interior to the cut.** A merged vertex sitting on the rim has a
   truncated one-ring, so the op sees a neighborhood the mesh never had. The minimiser will strip
   a region down to exactly that if you let it. `merge_pair_is_interior` rejects such cuts, and
   `test_open_weld_cases_are_faithful` guards it on the mesh-graph side.

Also: compare boundary *edge sets restricted to interior vertices*, never raw counts. A cut's rim
is boundary here but not in the mesh it came from, the ops treat boundary specially
(`merge_remaining_unconnected` fuses vertices joined by boundary halfedges), and a rim edge
closing will otherwise mask a hole opening.

When narrowing a case, note `tracing` is compiled out in release (`release_max_level_off`), so
run the replay in a **debug** build with `RUST_LOG=mesh_graph=debug` — the planner already logs
its rings, shared sets, start indices, range pairs and planned faces.

## Defects #4, #5 and #6 — found by tracing the extracted cases

4. **`range_pairs_to_connect.is_empty()` must not end the op.** It is a legitimate outcome (every
   ring vertex is shared with the other ring or already joined to it), but the op *returned*
   there — after `remove_neighbor_faces` had already taken both stars out, leaving a hole with
   nothing to close it. It now falls through with nothing planned and lets the cleanup and the
   remaining-unconnected merge repair the region, and keeps the already-connected strip (which is
   only torn out to make room for faces that in this case do not exist).
5. **A dead ring halfedge must not disqualify a ring position from starting the scan.**
   `find_shared_start_indices_from_ring` skipped any position whose halfedge was not alive — but a
   dead ring halfedge means `remove_neighbor_faces` consumed that edge, which is exactly a
   position the band still has to span. With the liveness requirement dropped, a case that planned
   nothing plans a full 13-face band across both rings.
6. **A full wrap can leave one seam triangle.** When the band goes all the way around both rings,
   the last planned face and the first meet at a quad and only one of its two triangles is
   planned: the start cap reads `AlreadyConnected`, which is true of the *edge* but not of the
   *face*. Rather than a sixth special case in the walk, `close_triangular_holes` runs at the end
   of the merge: a boundary loop of exactly three halfedges whose vertices all took part in this
   merge has exactly one triangle that closes it. Deliberately narrow — three-halfedge loops only,
   every vertex in the merge region, each with a single outgoing boundary halfedge — so a mesh
   that is meant to be open is never sealed by accident.
   → `test_weld_cases_stay_watertight`

Also considered and **not** kept: retrying the start-index scan from ring 2 when ring 1 yields
nothing. Defensible (the search is symmetric) but no case needed it once (5) was in.

## Is the input to these merges bad? Measured: no.

Three questions worth settling before patching a seventh time. `WELD_CAPTURE=1` answers all
three (the checks are in `topology.rs`, gated behind it).

1. **Is the mesh manifold going into the failing merge?** Yes. `manifold_problems` checks it
   combinatorially — an undirected vertex pair with more than two halfedges is a non-manifold
   edge, and a vertex whose incident faces number fewer than `edges - 1` has a star of more than
   one fan. Neither fires, and the boundary count is zero, right up to the merge that opens the
   mesh. The op is not being handed garbage.
2. **Are these merges ill-posed configurations that should be refused?** No — they are the normal
   case. The shared-one-ring-vertex histogram over a whole weld run is roughly
   `[23, 50, 63, 11, 1, 3]` (index = number of shared vertices): only ~15% of merges have
   disjoint one-rings, and two shared vertices is the *most common* case. A precondition like
   "refuse when the rings overlap" would refuse most of the weld. And failures do not concentrate
   there: one captured failure had **zero** shared ring vertices.
3. **Can it be repaired afterwards instead?** Only partly. `close_triangular_holes` handles the
   three-halfedge case and is worth keeping. Generalising it to close any short boundary loop the
   merge opened did **not** close the class, and the residual failures say why: classified by
   `weld_flake_loop_until_failure`, most are clean hole rims (11, 16, 19 halfedges, each paired
   with a face-carrying twin) but one run left **348 boundary halfedges of which 326 were "wire"**
   — edges with no face on either side. That is structurally broken output, not an unfilled
   polygon, and no hole-filler can repair it.

## ⚠️ Remaining work

1. **The weld still fails under a reversed merge order** — first failure between iteration 1 and
   27 across runs. The shipping order (vertex id ascending) survives 2000/2000. The class is not
   closed.
2. **The conclusion from the three measurements above**: the problem is not the input and not a
   missing precondition, and it cannot be fully repaired after the fact. It is that
   `merge_vertices_one_rings` destroys and then rebuilds — `remove_neighbor_faces` takes both
   stars out, and the band planner refills the annulus **from index ranges over the two rings**
   rather than from the hole that was actually created. Nothing ties the rebuild to the removal,
   so it can miss regions (six defects so far) and, in the wire case, emit faceless edges.

   The fix with a real chance of closing the class: after removing the stars, **walk the boundary
   loop(s) the removal actually left and stitch/triangulate those**, so "the op closes exactly
   what it opened" is structural instead of emergent. The seven captured-region tests plus the
   extractor are the safety net for that rewrite.
3. The weld is still not deterministic — the surviving face count varies by tens between runs.
   Rapier's collider handles are assigned deterministically (`positions` is a slotmap), so the
   source is elsewhere; `deformation/traits.rs` is the next place to look.

## Build environment (both repos — Xcode license blocks the linker)

Always set:
```
export RUSTFLAGS="-C linker=/Library/Developer/CommandLineTools/usr/bin/clang -C link-arg=-isysroot -C link-arg=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
```
Doctest linker failures are this artifact, not a code issue.

## Pitfalls (tried and failed — do not repeat)

- The pre-existing experimental working-tree code (seam-slot heuristics,
  `flip_planned_winding_if_slot_occupied`, `faces_incident_to_vertex`,
  `prev_prev_other_v_id` corner triangle) was discarded: it regressed 8 `merge_one_ring`
  tests on top of HEAD.
- Pinning the hash seed (a fixed-seed `BuildHasher` behind crate-wide `HashMap`/`HashSet`
  aliases) is the wrong fix and was reverted: it only freezes whichever branch the seed
  happened to select, it changes the public API (the marked-set types cross the boundary),
  and hashbrown's default hasher was chosen for speed. Fix the algorithms instead.
- Likewise, swapping in `std::collections` types is not an option — slower, same reason.
- Picking "an element" deterministically is not enough on its own: `iter().min()` on the
  fan start produced a *deterministic* hole where `iter().max()` did not. The fix that
  holds is to drive the walk from the stored star list (and write it back in that order),
  which keeps the vertex's rotation order intact for everything downstream.
- Replaying the *journal* across processes fails: slotmap IDs differ between the fixture-build
  process and a fresh build. Use the numbered `weld_seq_*_pre.json` state dumps (aligned 1:1
  with the journal's merge steps) for any future replay work.
- The `logs` test module generates one test per recorded `.glb`/`.json` pair; `punch_hole`/
  `deformation` module tests pass but the full `--lib` suite is not quick.
- In freestyle-sculpt, only ever touch `morph_open_close.rs` via careful edits — never a
  blanket `git checkout` of that file (that is what lost the fixture).
- The investigation worktree `/tmp/mg-head` was force-removed; verified gone
  (`git worktree list` shows only the main repo). Do not recreate it — the
  `[patch.crates-io]` must point at the real repo `/Users/maccesch/Programming/mesh-graph`.
- ~28 GB of `weld_pre_merge_*.json`, `weld_seq_*.json`, `weld_fail_post_*.json`,
  `weld_fail_journal.json` was deleted from the freestyle repo root. The numbered JSON dumps
  referenced above are gone too — regenerate them from a failing `weld_flake_loop_until_failure`
  run if replay work is ever needed again.

## Current working-tree state (verified 2026-09-15)

mesh-graph (`git status` — all uncommitted): modified `src/ops/edit.rs`,
`src/ops/remove.rs`, `src/ops/cleanup/vertex_neighborhood.rs`,
`src/ops/merge_one_ring/tests.rs`, `src/utils.rs`. Branch `main` at `59d15a7`,
ahead of `origin/main` by 1 commit (do not push until decided).

freestyle-sculpt (`git status` — all uncommitted): modified `Cargo.toml`
(TEMPORARY `[patch.crates-io] mesh-graph = { path = "../mesh-graph" }`),
`src/deformation/fields/hole_punch.rs`, `src/deformation/fields/morph_open_close.rs`
(reconstructed — NOT faithful, see above), `src/deformation/journal.rs`,
`src/deformation/topology.rs`, `src/tests/logs/mod.rs`. All collections are back
on plain `hashbrown`; builds clean against the patched mesh-graph.


## How to verify the fix still holds

```
# Always --release: these ops are orders of magnitude slower in a debug build.
cd /Users/maccesch/Programming/mesh-graph
export RUSTFLAGS="-C linker=/Library/Developer/CommandLineTools/usr/bin/clang -C link-arg=-isysroot -C link-arg=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
cargo test --release --features gltf,serde,instrumentation --lib
cargo clippy --features gltf,serde,instrumentation --all-targets

cd /Users/maccesch/Programming/freestyle-sculpt
cargo test --release --features rerun,instrumentation test_apply_weld_amount_applies_close  # needs the real fixture
cargo test --release --features rerun,instrumentation weld_flake_loop_until_failure         # after restoring it
```
