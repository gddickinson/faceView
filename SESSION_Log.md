# faceView — Session Log

## 2026-05-06 — Session 1: Scaffold + research

- Spec: multimodal GUI (chat / STT / TTS / face presence / identity / emotion /
  mouth-activity + visemes) drivable by Claude Code via local HTTP and MCP.
- Research (parallel agents): confirmed PySide6 over PyQt6; MediaPipe + ArcFace
  for vision; faster-whisper + silero-vad + pyttsx3/Kokoro for speech; honest
  scope on lip-reading (visemes only, not VSR).
- Decisions: private `gddickinson/faceView` repo, fresh `faceview` conda env,
  ANTHROPIC_API_KEY with demo-mode fallback, MCP + FastAPI both shipped.
- Created project tree, `pyproject.toml` with optional ML extras, `CLAUDE.md`,
  `INTERFACE.md` (full module map), this log, and `.gitignore`.
- Conda env `faceview` created (Python 3.11).

## 2026-05-07 — Session 12: Atlas rotation + MakeHuman + ARKit + research

User asked us to proceed with all 7 ranked next steps from the
realism assessment + research game-industry techniques starting
with MediaPipe and MakeHuman.

**Research findings**

  * MediaPipe FaceLandmarker outputs 478 3D landmarks + **52 ARKit-
    compatible blendshapes** per frame. Industry standard.
  * ARKit's 52-shape canonical set is FACS-derived, used by
    MetaHumans, Ready Player Me, MediaPipe, iOS Face ID, etc.
  * MetaHuman skin uses **subsurface scattering + dual specular
    lobes + scanned topology** — well beyond CPU rasterisation.
  * MakeHuman base mesh: 19K verts, **CC0 licensed**, proper
    feature topology designed for character animation.
  * USC ICT / ProductionCrate pack: **150+ MIT-licensed
    blendshapes** (mesh deltas) — future integration path.

**Multi-angle texture atlas** (`vision/face_warp_atlas.py`)
  * Renders BP3D head at 5 yaws (-45° to +45°) via the GPU
    pipeline → bundles in `assets/data/atlas/`.
  * Per frame: pick the two nearest atlas textures, warp each
    via FACS landmark deformation, crossfade by yaw distance.
  * New render mode `face_warp_3d` — photo-real face that *both*
    rotates and deforms with FACS.

**ARKit 52-blendshape compatibility** (`vision/arkit_blendshapes.py`)
  * Canonical 52-name list, two-way mapping to/from our 12 AUs.
  * `ARKitFrame` dataclass for round-tripping captures.
  * Lets external face-tracking (MediaPipe / iOS Face ID / etc)
    drive our avatar via the standard vocabulary, and lets our
    AU pipeline emit ARKit-compatible coefficients for rigging
    Unity / Unreal / RPM avatars.

**MakeHuman base mesh** (`vision/makehuman_mesh.py`)
  * Bundled `assets/data/makehuman/base.obj` (CC0, 19K verts).
  * `load_makehuman_head` parses, crops to head + neck, decimates
    via vertex clustering. Renders via QPainter Z-sort.
  * Cleaner topology than BP3D-skin-decimated for character work.
  * New render mode `makehuman_3d`, persona of same name.

**Comparison** (`docs/images/all_modes_compare.png`)
  Seven modes side-by-side: anatomical 2D / anatomical refined /
  decimated BP3D / MakeHuman / face_warp 2D / face_warp 3D atlas
  / GPU lifelike (reference).

**Tests**: 99 → 105 (new test_arkit_blendshapes covering canonical
set, AU↔ARKit round-trip, edge cases).

**Future integrations documented in roadmap**:
  - USC ICT 150+ blendshape pack (MIT) — replace synthetic FACS
    deltas with real mesh deltas.
  - MetaHuman Head FBX (52 ARKit-compatible blendshapes from
    Gumroad) — drop-in better mesh.
  - Open-mouth texture variant — current crude composite is too
    obvious; need proper jaw-rotation render via faceforge pipeline.

## 2026-05-07 — Session 11: TMJ jaw + decimated BP3D head + faceforge investigation

User said the lite 3D didn't look like a face at all and asked us
to learn from faceforge's actual pipeline (jaw on skull, muscles
on jaw, skin on muscles).

**Faceforge investigation** — read the relevant modules:

  * `anatomy/skull.py`: skull is a SceneNode hierarchy with a
    `jawPivot` node positioned at the TMJ (temporomandibular joint).
    Mandible + lower teeth are children of the pivot, so rotating
    the pivot rotates everything attached.
  * `anatomy/jaw_muscles.py`: jaw muscles deform via vertex rotation
    around the same pivot when the jaw angle changes.
  * `coordination/simulation.py`: the actual formula —
    ``jaw_angle = AU26 * 0.28 + AU25 * 0.06`` (radians).
  * Skin is layered on top via skinning weights driven by the
    deformed bones + muscles.

**TMJ jaw rotation in our pipeline** (`vision/anatomy.py`)
  * Lifted the same formula. Added `_apply_jaw_rotation` that
    rotates lower-face landmarks around `TMJ_Y = 0.50` (face-box
    normalised). Chin drops by `(y - TMJ_Y) * sin(angle)`, lower
    lip + lip corners follow. Upper face stays fixed.
  * `deform_landmarks` now does jaw rotation first, then muscle
    contraction — same order faceforge uses.
  * Verified: `AU26=1` drops chin from y=0.96 to y=1.087, lip_corner
    drops by 0.077 — proportional rigid rotation, not stretch.

**Decimated BP3D skin head** (`vision/head_decimated.py`)
  * Diagnosed why old lite 3D was bizarre: Delaunay over 86 hand-
    placed points crosses feature boundaries (eye→forehead,
    lip→cheek), creating spider-web topology.
  * Fix: start from real anatomy. Load FMA7163 (BP3D skin mesh,
    full body), crop to top 22% (head + neck), apply BP3D→screen
    reorient, then **vertex-cluster decimation** in pure NumPy:
    grid-based bucketing reduces ~30K verts to ~3500 at grid=24.
  * Render via QPainter Z-sort with backface culling. Result is a
    recognisable human head + neck + shoulders — no spider web.
  * ~120 ms/frame at grid=20 (~8 fps). Acceptable for static views;
    for real-time animation pair with the GPU mode (roadmap A24).
  * New render mode `head_decimated_3d`, persona of same name.

**Re-rendered face_warp texture** with extended neck region.

**5-way comparison** in `docs/images/five_modes_compare.png`:
stylised 2D / old lite 3D / decimated head / face warp 2D / GPU
lifelike. The new decimated head is unmistakably a human shape;
the old lite 3D was unmistakably not.

Tests: 96 → 99. Three new in `test_head_decimated.py` covering
decimation output, render frame validity, dispatcher routing —
all gated on BP3D meshes being present.

Honest limits documented in roadmap:
  * face_warp can't show open mouth (single closed-mouth texture).
  * decimated head has no visible eyes/lips at low grid (texture
    needed for that, GPU path next).
  * Lite 3D Delaunay-on-landmarks approach is now deprecated; use
    `head_decimated_3d` instead.

## 2026-05-06 — Session 10: Image-warp realistic face

User said the lite 3D still looked bizarre and asked to investigate
other methods for a realistic face. Investigated multiple approaches:

  * Better mesh topology (hand-crafted rings) — same flat-shading
    problem; high effort, modest gain.
  * MediaPipe canonical face mesh (468 verts) — proper topology with
    feature rings, but heavy work to wire FACS blendshapes.
  * Decimated BP3D skin mesh — real anatomy at lower density; still
    polygonal without textures.
  * 3DMM (FLAME / BFM) — research-grade, heavy data dependency.
  * **Image-space warp of GPU-rendered texture** — photo-real
    appearance + cheap CPU warp. Locked to front view but easily
    paired with `faceforge_3d_gpu` for rotation.

Shipped the image-warp approach as the most lifelike *animated*
option. New module `vision/face_warp.py`:
  * Loads a one-time GPU-rendered BP3D neutral face texture from
    `assets/data/neutral_face.png`.
  * Per frame: deform the 86 anatomical landmarks via the existing
    AU pipeline, build Delaunay over the deformed positions, reverse-
    map every output pixel to its source via barycentric coords,
    bilinear-sample the texture.
  * Pure NumPy + scipy — no OpenCV dependency.
  * ~25 fps on CPU at 320x320 — interactive. Real face appearance
    from the BP3D source, FACS-driven motion.

New `tools/render_neutral_face_texture.py` regenerates the texture
via the GPU lifelike pipeline + crops to the face-box convention.

New persona `face_warp_2d` and render mode of the same name routed
through the existing `render_face` dispatcher.

Tests: 92 → 96. `test_face_warp.py` covers texture-present rendering,
dispatcher routing, emotion deltas, and graceful error when the
texture is missing.

Also fixed the `MissingDependency` constructor signature usage
across `face_warp.py`, `gpu_renderer.py`, `anatomy_meshes.py`,
`faceforge_bridge.py` — was using wrong kwarg (`install_hint=`)
instead of the actual signature `(package, extra, hint=)`. Tests
flushed out the bug now that it actually fires.

## 2026-05-06 — Session 9: Smoother lite 3D + BP3D-aligned 2D proportions

User pointed out the lite 3D head looked too cuboid and asked for
the 2D faces to use BP3D proportions. Both addressed.

**Lite 3D smoothing** (`vision/head_3d_lite.py`)
- Replaced hand-tuned per-landmark Z values with a smooth ellipsoidal
  Z function (`_smooth_z`) — half-ellipsoid dome centred at face,
  radii (rx=0.45, ry=0.55, rz=0.22). Per-group / per-landmark Z
  offsets layered on top, but kept small so they don't cause seams.
  Continuous quadric surface = no more cuboid feel.
- Added 30+ midpoint vertex inserts via `_MIDPOINT_PAIRS` — every
  adjacent pair on the face oval, plus interior bridges (cheek to
  jaw, lip to chin, glabella to hairline). Densifies the front mesh
  before triangulation.
- Added `_subdivide()` — one pass of edge-midpoint subdivision after
  Delaunay. Each triangle becomes 4. Midpoint vertex groups inherit
  parent groups when both agree, fall back to skin otherwise.
- Computed per-vertex normals (averaged from incident triangles)
  and per-vertex Phong shading; each triangle's colour = mean of
  its three vertex shades. Cheap Gouraud-style smoothing.
- Removed pen outline (was creating visible mesh edges).
- ~140 vertices, ~600 triangles after subdivision. Still 20+ fps.

**BP3D-aligned 2D proportions** (`vision/anatomy.py`)
- Adjusted all 86 landmark Y positions to match BP3D-measured
  anatomy: eye line at vertical midpoint of head (was 40%), nose
  tip at ~64% (was 59%), mouth at ~78% (was 71%), narrower head
  (1:1.45 ratio). Brows shifted down to follow eye line.
- Repositioned Zygomaticus Major muscle pair so L and R sit on
  their respective anatomical sides (not on the centerline) — the
  prior layout cancelled out at the lip corners.
- All 2D render modes (stylised, anatomical, layered, anatomy_xray,
  etc.) inherit the new proportions automatically.

Tests: 92 stay green. Demo images and GUI screenshots re-rendered.

## 2026-05-06 — Session 8: Three new 3D rendering tracks

User asked for three things: (1) use the 3D model to improve other
animations, (2) GPU-accelerate the lifelike head with Apple Metal,
(3) build simplified animatable 3D models. All three shipped.

**Lite 3D animatable head** (`vision/head_3d_lite.py`)
- Existing 86 anatomical landmarks + ~19 back-of-head / scalp / neck
  closure points = ~105 vertices total.
- Hand-tuned Z depth per landmark group (nose tip protrudes, temples
  recede, ears further back, scalp dome behind hairline, neck-front
  flush, etc.).
- SciPy Delaunay over the projected front face + hand-coded back
  triangulation (~50 hand-tris) stitches the silhouette closed.
- AU-driven landmark deformation reused from the 2D pipeline, so the
  same FACS expressions / visemes drive both 2D and lite-3D.
- Triangle colour rule: feature colour only when *all three*
  vertices share the feature group; otherwise default to skin —
  prevents Delaunay-spanning lip-coloured wedges.
- Z-sorted painter's algorithm via QPainter. Per-triangle Phong
  (Lambert diffuse + specular).
- ~55 fps on a single CPU core. New render mode `head_3d_lite`,
  persona of the same name.

**GPU-accelerated lifelike head** (`vision/gpu_renderer.py`)
- New optional dep `moderngl` (one-time `pip install moderngl`).
- Standalone offscreen OpenGL 4.1 context. On Apple Silicon this is
  served by Apple's Metal compatibility layer (`Apple M1 Max
  Metal - 90.5`).
- VBO upload per BP3D mesh, cached after first frame. Phong vertex
  + fragment shaders. Per-mesh material uniforms (colour, opacity,
  shininess) preserve the catalog appearance.
- BP3D→screen reorientation moved on-CPU once at upload time so the
  shader stays simple.
- Renders the full 145-mesh head at **~36 fps** on M1 Max — the only
  path that animates the lifelike anatomy in real time. New render
  mode `faceforge_3d_gpu`.

**BP3D-derived landmark refinement** (`vision/bp3d_landmarks.py`)
- Measures anatomical reference points (chin, mandible angles, top
  of skull, temples) directly off the BP3D skull bone meshes.
- Returns a name → (x_norm, y_norm) override dict the 2D template
  could opt into. Currently exposed as infrastructure; full
  integration into the 2D landmark template is roadmap A15.

Demos in `tools/animate_3d_modes.py`:
- `head_3d_lite_emotions.png` — 6-emotion grid in lite 3D.
- `head_3d_lite_talking.gif` — lite 3D head speaking + rotating.
- `gpu_lifelike_rotate.gif` — full BP3D head rotating in real time.
- `three_d_modes_compare.png` — stylised 2D / lite 3D / GPU
  lifelike side-by-side at the same neutral pose.

Three new personas in `personas.json`: `head_3d_lite`,
`faceforge_3d_gpu` (and the existing `faceforge_3d`).

Tests: 83 → 92. Coverage: lite-3D template + dispatch + rotation
+ emotion delta + persona-driven mode; GPU import gated on moderngl
+ render gated on BP3D meshes available.

## 2026-05-06 — Session 7: Lifelike photo-anatomical face

- User asked for a *lifelike* 3D anatomical face leveraging the
  faceforge head pipeline. Investigation found ~145 STLs needed (full
  head+neck catalog), not the 28 we'd been using. Lifted faceforge's
  bundled JSON catalog directly:
  `assets/config/anatomy/{skull_bones, face_features, expression_muscles,
  jaw_muscles, neck_muscles, cervical_vertebrae, skin, eye_colors}.json`.
- New `vision/anatomy_catalog.py` unifies the 8 configs into one
  `MeshSpec` list (color, opacity, shininess, draw_order, category)
  exposing `load_catalog()`, `specs_by_category()`, and named layer-set
  resolution. `lifelike` makes skin opaque on top; `xray` keeps it
  translucent; `muscles`, `features`, `skull_only`, `vertebrae` are
  subsets.
- Renderer upgraded: per-mesh material (color/opacity/shininess), Phong
  lighting (ambient + diffuse + specular), draw-order layering so bones
  draw first and skin last, NumPy-vectorised per-triangle shading,
  alpha-aware QPainter blending. BP3D coordinate transform extended
  with a 180° Y-flip so the face points toward the camera.
- Renderer auto-scales to the *bone* bbox (skull) when bones are
  present in the layer set. Stops the full-body skin mesh from zooming
  the head out to a tiny ant in the frame.
- copy_anatomy_meshes now reads the unified catalog → copies all
  ~145 STLs from a BodyParts3D dump. Exercised against the user's
  `/Volumes/GeorgeDrive/.../bodyparts3D/stl` mirror — 143 of 145 STLs
  present in that snapshot.
- Demos: 4-panel `anatomy_meshes_grid.png` (skull / muscles / features /
  lifelike) plus front + 3/4 view stills of the lifelike face. Skull
  rotation GIF kept; full-mesh GIFs deferred to the OpenGL upgrade
  (CPU rasteriser is too slow at 145 meshes × 5000 tris/frame).
- Tests: 76 → 83. `test_anatomy_catalog.py` covers load, layer sets,
  opacity rules, color shape, unknown-set error path. Existing
  `test_anatomy_layers.py` continues to verify the dispatcher routes
  `faceforge_3d` correctly when meshes are present.

## 2026-05-06 — Session 6: Layered anatomy + photo-anatomical bridge

- Added six new render modes spanning two tracks:
  - **Stylised illustrative anatomy** — `anatomy_skull`, `anatomy_brain`,
    `anatomy_eyeballs`, `anatomy_muscles`, `anatomy_xray`,
    `anatomy_layers`. Five new modules: `anatomy_skull.py` (cranium +
    orbits + pyriform aperture + mandible + teeth), `anatomy_brain.py`
    (4 cerebral lobes + cerebellum + brainstem with gyri/sulci),
    `anatomy_eyeballs.py` (full sphere globes + iris + optic nerve),
    `anatomy_muscle_masses.py` (solid 43-muscle layer oriented along
    fiber direction), `sim_face_layered.py` (compositor with per-layer
    alpha and preset-name lookup).
  - **Photo-anatomical** — `faceforge_3d`. New `anatomy_meshes.py`
    parses BP3D binary STLs with NumPy + struct, computes per-tri
    normals, applies BP3D→screen reorientation. New
    `faceforge_bridge.py` exposes `render_face_faceforge()` and
    `faceforge_status()`. Z-sorted Lambert with double-sided shading.
- New `tools/copy_anatomy_meshes.py` copies the head + neck FMA subset
  from a local BodyParts3D dump into `assets/anatomy_meshes/` (gitignored).
  Tested with `/Volumes/GeorgeDrive/claude_test/face_app/bodyparts3D/stl`
  — 22 of 28 expected STLs present (some FMA codes missing from this
  particular BP3D mirror; the renderer adapts).
- New `tools/animate_anatomy_layers.py` renders
  `docs/images/anatomy_layers_grid.png` (6-panel grid),
  `anatomy_peel.gif` (peel-away skin → muscles → skull → brain),
  `anatomy_meshes_rotate.gif` (BP3D head rotating).
- Persona JSON gains 7 new presets: `anatomy_layers`, `anatomy_skull`,
  `anatomy_brain`, `anatomy_muscles`, `anatomy_xray`, `anatomy_eyeballs`,
  `faceforge_3d`. All routed through the existing `render_face`
  dispatcher — the talking-avatar pipeline picks them up via
  `set_persona`.
- `sim_face.render_face` dispatch now covers four families: stylised
  (default), 2D anatomical, layered illustration, photo-anatomical.
- Tests: 63 → 76. New `test_anatomy_layers.py` covers preset rendering,
  dispatcher routing, layer-name validation, faceforge bridge fallback
  + on-disk path.

## 2026-05-06 — Session 5: Anatomical renderer

- Investigated faceforge (3D OpenGL anatomy app at
  `/Volumes/GeorgeDrive/claude_test/face_app/faceforge`). It's far
  heavier than would fit cleanly into faceView (BodyParts3D STL
  meshes, OpenGL skinning, ~6,300 lines of anatomy code), but its
  43-muscle expression catalogue with AU maps was directly liftable.
- Bundled `assets/config/expression_muscles.json` — trimmed catalogue
  (name + AU map only), no STL refs.
- New `vision/anatomy.py` (382 lines): 86-point landmark template
  generated programmatically at canonical face proportions
  (rule-of-thirds, eye spacing, lip rest), plus a `MUSCLE_LAYOUT`
  table giving each muscle a 2D centroid, fiber direction, and
  influence radius. `deform_landmarks(base, au_values)` applies
  every active muscle's pull to every landmark within its radius.
- New `vision/sim_face_anatomical.py` (198 lines) + helper module
  `sim_face_anatomical_parts.py` (~520 lines) — anatomically grounded
  2D renderer. Layered: hair-back, skin oval, side/brow/nasolabial
  shading, cheek apples, hair-front, brows with hair strokes, eyes
  (sclera + iris with limbal ring + pupil + specular + lashes), nose
  (bridge shadow + alar wings + nostrils + AU9 wrinkle), mouth
  (cupid's bow + cavity + teeth strip), mentolabial sulcus.
- Three render modes via `Persona.render_mode`:
  `anatomical` (default for anatomical persona), `anatomy_overlay`
  (translucent muscle layer with fiber-direction ticks), `wireframe`
  (landmark dots + group polylines). The stylised renderer remains
  the default for backward compatibility.
- `sim_face.render_face` now dispatches based on `params.render_mode`,
  so the avatar pipeline picks up the new modes via persona only —
  no other code changes.
- Three new personas in `personas.json`: `anatomical`,
  `anatomy_overlay`, `wireframe`.
- Demo: `tools/animate_anatomical.py` produces
  `docs/images/anatomical_talking.gif`,
  `anatomical_overlay.gif`, `anatomical_compare.gif` (side-by-side
  stylised vs anatomical) and an emotion grid PNG.
- Tests: 48 → 63 (+8 anatomy unit tests, +7 render-mode dispatch
  smoke tests). All green.

## 2026-05-06 — Session 4: Roadmap + personas + coarticulation + CI

- Added `ROADMAP.md` — five tracks (R/L/A/S/X) covering reliability, the
  real-time loop, avatar depth, server surface, and stretch goals. Marks
  what's now in flight vs queued vs later.
- New `vision/personas.py` + `assets/config/personas.json` — 7 bundled
  appearance presets (default / claude / warm_tan / fair_blonde /
  cool_pale / warm_dark / auburn). `Persona` is a static overlay
  applied to every `FaceParams` after `face_state_to_params`, so
  appearance is fully decoupled from the FACS animation.
- Coarticulation in `vision/speech.py`: new `viseme_blend_at(timeline,
  t, attack=0.04, release=0.06)` returns a per-AU weighted-max blend
  across active viseme envelopes. `TalkingAvatar` now uses the blend
  as its mouth-AU target instead of stepped `viseme_at`, giving
  continuous trajectories across phoneme boundaries.
- New Service ops `set_emotion`, `set_persona`, `avatar_say`,
  `list_personas` reach the avatar through `Service.bind_camera_worker`.
  Wired through HTTP (`POST /avatar/{emotion,persona,say}`,
  `GET /avatar/personas`) and MCP (`set_emotion`, `set_persona`,
  `avatar_say`, `list_personas` — total MCP tool count: 9).
- New `tools/render_personas.py` produces `docs/images/personas.png`
  (4-col contact sheet with persona name labels).
- New `.github/workflows/test.yml` runs pytest + headless smoke on
  every push and PR; uploads the headless smoke PNG as an artefact.
- Tests: 31 → 48. New `test_personas.py` (6), `test_coarticulation.py`
  (5), `test_service_avatar.py` (6). All green.

## 2026-05-06 — Session 3: Richer renderer

- User asked for a better renderer. Split `vision/sim_face.py` into
  `sim_face.py` (303 lines, top-level layered draw) +
  `sim_face_parts.py` (492 lines, brow/eye/cheek/nose/mouth helpers)
  to stay under the 500-line budget.
- Extended `FaceParams` with 9 AU-grade fields (mouth_pucker,
  mouth_stretch, cheek_raise, nose_wrinkle, upper_lid_raise,
  inner/outer_brow_raise, brow_lower, lip_corner_drop) so visemes
  and expression presets reach the renderer with full per-AU
  intensity instead of being collapsed into smile/jaw_open.
- New layered drawing: background vignette → ears with inner shadow →
  head skin (radial gradient + side shading + rim light) → AU6 cheek
  apples → hair cap + fringe path with strand highlights → tangent-
  aligned brow strokes (12 hairs + solid body) → almond eyes (radial
  iris, eyelashes, AU6 lid crease) → nose bridge with AU9 wrinkle →
  mouth with cupid's bow, asymmetric smile/frown, teeth strip with
  vertical dividers, chin shadow.
- Mouth geometry settled after several iterations: separate
  `corner_dy` (capped) and `mid_dy` (asymmetric pos/neg) plus
  `upper_h_scale` floor of 0.30 so frowns no longer wedge into
  pointed triangles and smiles get a proper ∪ curve.
- All 31 tests still pass. Re-rendered `docs/images/` (main, happy,
  speaking, surprised, face_neutral/happy/sad/surprised, avatar
  GIF + strip + monitor).
- Committed as `84bd56b`. Push to remote pending user authorisation.

## 2026-05-06 — Session 2: FACS-based talking avatar

- User pointed out the related `face_app/faceforge` project — pulled the
  FACS model, expression presets, viseme table, and a compact CMU
  pronouncing dictionary as bundled assets. Replaces my hand-rolled
  char→viseme heuristic with proper ARPAbet phoneme lookup + 15-class
  viseme alphabet keyed to AU activations.
- Added `vision/face_state.py` (FACS FaceState with 12 AUs + pose + gaze),
  `vision/expressions.py` (preset loader), `vision/speech.py` (SpeechEngine),
  `vision/avatar.py` (TalkingAvatar with AutoBlink + AutoBreathing +
  AutoSaccade + lip-sync). Renderer unchanged — `face_state_to_params()`
  bridges the AU model to the existing `FaceParams` renderer.
- `SimCameraWorker` gained an "avatar" scenario and an LLM_REPLY hook so
  the camera panel can become Claude's animated face when
  `FACEVIEW_AVATAR=1` is set.
- New `tools/animate_talking.py` records `avatar_talking.gif`,
  `avatar_strip.png`, `avatar_monitor.png` for the README — visible
  lip-sync + blinks + breathing.
- Test suite: 17 → 31. New tests cover FACS preset loading, AU→FaceParams
  bridge, CMU dict + letter-rule fallback, viseme mapping, idle blink
  occurrence, jaw-motion during speech, frame variation across an utterance.

## 2026-05-06 — Session 1, continued: working build

- Core, GUI, LLM, speech, vision, server all implemented per `INTERFACE.md`.
- Procedural face renderer added (`vision.sim_face` + `SimCameraWorker`)
  after user request: full pipeline now runs without a real webcam.
- 17 pytest tests passing (event bus, conversation, screenshot helper,
  service layer, sim-face determinism, offscreen smoke).
- Captured 5 GUI scenes + 4 face crops into `docs/images/` via
  `tools.capture_gui_screenshots`.
- README written with embedded screenshots, honest lip-reading scope, and
  full HTTP+MCP usage docs.

## Next

- `git init`, push to private `gddickinson/faceView`.
- Optional follow-ups: real Auto-AVSR ONNX upgrade for true VSR; Kokoro TTS;
  enrol-owner CLI; auto-start camera/audio toggles in the GUI menu.
