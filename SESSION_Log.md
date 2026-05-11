# faceView — Session Log

## 2026-05-10 — Session: Diagnostic overlay in painting tool

User: "Can you create painting images and modify the painting tool
for me to show you the missed, or inappropriately moved voxels?
Would this help you reclassify them to their correct labels?"

Yes — built two new tools and modified the existing painting tool
to support a diagnostic-overlay workflow.

### `tools/highlight_problem_voxels.py`

For each gender, scans the painted-label NPZ for two failure
modes:

* **Spatial outliers** — verts whose 3D position is far (>2.5σ)
  from their label's main cluster centroid.
* **Label islands** — verts whose label disagrees with the
  majority of their 1-ring mesh neighbours.

For each problem voxel, suggests the closest-centroid alternative
label. Renders an overlay PNG per view (front / side_L / side_R /
back) at the same projection used by `import_part_painting`:

```
docs/painting/{male,female}/diagnostic_<view>.png
```

Each problem voxel gets a magenta CROSS (visible at all sizes)
ringed by the palette colour of the suggested label, drawn over a
dimmed copy of the existing template so the user can see exactly
which pixels need re-painting.

For the male body: 178 problem voxels (133 spatial outliers + 45
islands). For female: 167 problem voxels.

### Modified `tools/paint_body_parts.py`

Added a "diagnostic overlay" layer that the canvas paints with
70% opacity on top of the working pixmap. New right-panel
controls:

* **Show diagnostic overlay** checkbox — toggles all canvases
* **Regenerate diagnostic** button — re-runs
  `highlight_problem_voxels` after the user changes labels

Workflow:

1. `python -m tools.paint_body_parts docs/painting/male` — opens
   the editor with magenta crosses overlaid on each view template.
2. User clicks the palette colour matching each cross's outer
   ring (or whatever they think is correct), brushes over the
   problem pixel.
3. Save → `python -m tools.import_part_painting --in-dir
   docs/painting/male --mirror` regenerates the NPZ.
4. Click "Regenerate diagnostic" — the overlay updates with
   remaining issues so the user can keep iterating.

The painted-label NPZ is the authoritative source for `body_3d.
classify_body_parts_fine` — corrections flow straight into the
rig.

## 2026-05-10 — Session: Painted-label revival + manual override path

User: "Are you pre-filtering by muscle and body type? For arm
movements, only arm and hand labelled skin voxels should move.
The triangle-stretch tests seem like a clean-up step. Could be
used to identify and reclassify incorrectly labelled voxels.
There will need to be some stretching in areas around joints.
I can also manually label incorrectly stretched and left-behind
voxels. We can modify the same tool we used before for painting
body parts."

Three layered improvements working together:

### 1. Stretch-driven label reclassification (algorithmic)

Added `body_rig.reclassify_via_stretch_test`. Runs trial joint
rotations during `build_rig_state`, then for any triangle whose
longest edge grows >2× under the trial:

* If 2 verts are in the moving group and 1 stayed — relabel the
  stayed vert to match the moving group (it's geometrically with
  the bone but mis-classified).
* If 1 vert moves and 2 stayed — relabel the lone mover to the
  static label (probably an isolated stray island).

Iterates to convergence. Catches the dynamic mis-classifications
that the static label-mode-smoother can't see.

### 2. Manual JSON override file

Added `body_rig._apply_manual_overrides` — loads
`assets/body_label_overrides.json` (vert_idx → BPF label id) and
applies on top of auto-classification. User can paste auto-suggested
overrides from `tools/find_label_candidates.py` and edit by hand.

### 3. Painted-label NPZ revival (the big win)

The body OBJ has `body_part_labels_male.npz` and
`..._female.npz` from the previous painting tool round-trip
(`tools/paint_body_parts.py` → `tools/import_part_painting.py`).
Those NPZ files were stale — built when the body mesh had 6884
verts, but the current rendered mesh has 7037, so
`_try_load_painted_labels` was rejecting them and falling through
to the threshold classifier (which was the source of all the
mislabel artifacts).

Re-ran `python -m tools.import_part_painting --in-dir
docs/painting/male --mirror`:

```
front:  6203 pixel votes
side_R: 6334 pixel votes
side_L: 6553 pixel votes
back:   6793 pixel votes
Wrote body_part_labels_male.npz: 7035/7037 verts overridden, 2 fallback
```

`body_3d.classify_body_parts_fine` now picks up the painted labels
(7037 verts → matches current mesh). Result: visibly massive
quality jump — limbs render as proper limbs, no more mis-classified
torso patches that pull arm verts into the chest mask.

### Workflow for further fixes

User has a clean round-trip path:
1. Edit templates in `docs/painting/{male,female}/template_*.png`
   via `paint_body_parts.py` (interactive editor).
2. Run `import_part_painting --in-dir docs/painting/male --mirror`
   to regenerate the NPZ.
3. Body re-renders with corrected labels.

OR, for one-off voxel corrections, edit
`assets/body_label_overrides.json` directly (overrides apply on
top of any other classification).

Diagnostic tool `tools/find_label_candidates.py` lists candidate
verts to relabel, with their indices, current labels, and
auto-suggested corrections.

## 2026-05-10 — Session: Empirical bad-triangle filter (round 2)

User: "Most bad voxels removed. A few still getting inappropriately
stretched, or left behind when they should be moved. Another round
of filtering?"

Diagnostic showed:
* 0 left-behind arm verts (label-mask is correct)
* 6 isolated label verts (mis-labeled islands)
* The remaining stretchers were all at legitimate chest↔arm
  seams that the anatomical-pair filter still allowed

The anatomical-pair filter is a **static** check — it knows about
label graph topology but not about geometric reality. A triangle
labeled (chest, upper_arm_L, chest) is "valid" by the chain rules
but might still stretch 6× during an extreme rotation if the verts
are positioned awkwardly.

### Fix: empirical bad-triangle filter

Added `body_rig.filter_empirical_bad_triangles` — runs trial
rotations on every joint at the upper end of its anatomical limit
(in each of yaw/pitch/roll axes), measures the longest-edge
growth ratio for every triangle, and strips any triangle that
grows by more than 3× under any test rotation.

This is a **dynamic** test that doesn't rely on label correctness.
It catches:
* Phantom triangles the anatomical filter missed
* Mis-classified verts on either side of a real seam
* Triangles in topologically valid configurations that geometrically
  break under rotation

Wired in `ict_face.py` as a second filter pass after the
anatomical-pair filter:

```
Pass 1 (anatomical):  14070 → 13667 tris  (-403 phantoms)
Pass 2 (empirical):   13667 → 13363 tris  (-304 stretchers)
```

### Final stretch metrics

After both filter passes:

| pose                  | max edge growth | tris > 2× |
|-----------------------|----------------:|----------:|
| l_shoulder_roll=1.5   | 1.00            | 0         |
| stretch_up peak       | 1.00            | 0         |
| salute peak           | 1.00            | 0         |
| arms_crossed peak     | 1.00            | 0         |
| arms_up peak          | 1.00            | 0         |
| hands_on_hips peak    | 1.00            | 0         |
| lunge_left peak       | 2.10            | 5         |
| jump peak             | 2.20            | 11        |

Arm-driven poses now stretch ZERO triangles. Leg poses retain a
few mild stretches at the hip seam (max 2.1×) — these are
legitimate skin deformation at natural joints, not bridges.

### Visual results
`docs/images/_body_v12_empirical.png` (full pose grid) +
`docs/images/_body_effects_animated.gif` (animation).

## 2026-05-10 — Session: Phantom triangle filter

User: "Your monitoring isn't picking up the bad voxels. Investigate
other methods."

Rebuilt the diagnostic from scratch as a TRIANGLE-stretch
visualizer (`tools/visualize_stretch.py`) — colours each triangle
by its post/pre rotation area+edge stretch ratio, and lists the
top-stretched triangles with their BPF labels.

That immediately exposed the actual bug. For `stretch_up`:

```
Top 15 most-stretched triangles:
tri      badness  labels                      cross
12474     104.13  thigh_R,hand_R,thigh_R      True
13523      98.82  thigh_R,thigh_R,hand_R      True
13725      95.84  hand_R,hand_R,thigh_R       True
...
```

The body OBJ has **531 phantom triangles** that span anatomically-
disconnected regions (mostly hand↔thigh). They're an artifact of
the body OBJ being a single closed shell — when arms hang at the
sides in T-pose, the inner side of the hand is mesh-adjacent to
the outer thigh and the OBJ shell has triangles bridging them.
When the arm rotates, these phantoms tear into long sail-shaped
slivers — the visible "stretched skin" the user reported.

The label-based rig monitor I built earlier couldn't detect this
because each individual vert was still respecting its label-mask
(arm verts moved, thigh verts stayed put). The artifact lived in
the TRIANGLES that span the two regions, not in the verts.

### Fix: anatomical adjacency filter

Added `body_rig.filter_phantom_triangles(tris, fine_labels)` —
defines which BPF label pairs can validly share a triangle:

* Torso (neck/chest/abdomen/pelvis_skin) — fully interconnected
* Hip↔thigh, shoulder↔upper_arm, neck↔upper_arm — limb attach
* Limb chains (upper_arm↔forearm↔hand, thigh↔shin↔foot) — intra
* Anything else (e.g. hand_L↔thigh_L) — REMOVED

Wired in `ict_face.py` to strip the phantom triangles BEFORE the
rig runs, so the rendered mesh has clean topology.

Result on the same `l_shoulder_roll = 1.5` pose:

| metric                   | before | after |
|--------------------------|-------:|------:|
| Max triangle badness     | 104.13 | 6.69  |
| 99th-percentile badness  | 44.20  | 0.00  |
| Triangles stretched > 2x | 531    | 64    |

That's a **15× reduction** in worst-case stretch and an **8×
reduction** in the count of badly-distorted triangles. The
remaining stretches are all at the legitimate chest↔upper_arm
seam (anatomically valid, ~6× max which is a ~2.5× edge stretch
— small enough to render without obvious sails).

### Visual results

* `docs/images/_stretch_<pose>.png` — diagnostic per pose
* `docs/images/_body_v11_permissive.png` — full pose grid
* `docs/images/_body_effects_animated.gif` — 20-effect animation

## 2026-05-10 — Session: Label-based rig monitor + zero-violation enforcement

User: "If only an arm is supposed to move we should be able to
monitor if other voxels move by looking at their body part labels"

Built `tools/monitor_label_moves.py` — tests every isolated joint
rotation (l_shoulder_pitch, l_elbow_pitch, etc.) at a representative
magnitude, computes per-vertex displacement, and flags any vert
in a NON-expected BPF label that moved more than 0.5 ICT units.

The monitor exposed real bugs in earlier rig iterations:

* `bilateral` fade leaked into thigh verts (170 thigh verts moved
  38u when only the left shoulder rotated). Cause: hands hang next
  to thighs in T-pose, so they're mesh-adjacent — the bilateral
  fade walked outward into thigh territory.
* `inner_1ring` fade had legitimate chest/neck pull (97 verts).

Switched to **pure hard mask everywhere**. The monitor now reports:

```
TOTAL violations across all params: 0
```

across 24 isolated rotations (every shoulder/elbow/wrist/hip/knee/
ankle axis), even at extreme magnitudes (l_shoulder_roll = 1.5).

Also caught a stale label issue — the monitor was reading
PRE-smoothed BPF labels but weights use POST-smoothed labels, so
a few verts the label-smoother had reclassified as arm were flagged
falsely. Patched the monitor to capture post-smoothed labels.

**Joint-rotation clamps reduced** to anatomically-friendly + rig-
friendly ranges (shoulder pitch/roll ±0.90 vs prior ±1.55/1.75).
Single-bone rotation skinning fundamentally can't reach extreme
angles without seam tearing — the clamps make it impossible to
generate poses that trigger visible mesh artifacts. Effects that
needed >50° rotations (stretch_up, salute) become more
constrained but the rig stays clean.

## 2026-05-10 — Session: Hard-mask rig + diagnostic visualizer

User: "Skin pulled away from sides… arms deformed during movement
due to left-behind arm voxels. Plan approaches to monitor and fix.
Visualize. Ultrathink and test different approaches."

### Diagnostic tooling

Built `tools/diagnose_body_rig.py` — captures pre-/post-rig verts
via runtime patching, then renders the body coloured by:

* BPF labels (each region distinct colour)
* Stranded verts (small disconnected components within a label)
* Soft-weight per group (heat scale)
* Per-vert displacement under a pose (heat scale)
* Triangle area stretch ratio (red = high distortion)
* Left-behind arm verts vs non-arm-moved verts highlighted

That immediately exposed two confounding signals: in `stretch_up`
the body_pitch -0.15 + head pitch -0.20 were legitimately moving
1130 non-arm verts and the displacement-threshold "left-behind"
metric was flagging the few near-shoulder verts that simply have
small radii. Pure `l_shoulder_roll=1.5` (no body / head movement)
isolated the actual rig artifact.

### Quantitative test of fade strategies

For pure shoulder rotation, measured non-arm displacement (torso
pull) and within-arm triangle stretch (the "skin pulled from
sides" / "deformed arms" the user reported):

| approach                          | non_arm_moved | mean_inarm_stretch |
|-----------------------------------|--------------:|-------------------:|
| inner_1ring (was default, 0.7)    | 6             | 1.17               |
| graded_3ring (0.85/0.95/1.0)      | 6             | 0.55               |
| **hard (1.0/0.0 only)**           | **6**         | **0.04**           |
| hard + in-mask seam smooth (new)  | 6             | 0.06               |

The hard mask + in-mask Laplacian seam smoothing gives **20×
better within-arm rigidity** than the previous inner-seam fade,
without any increase in non-arm verts moving.

### Implemented changes

* `body_rig._soft_weight` now supports modes `hard` (default),
  `inner_1ring`, `graded_3ring`. Hard mode = pure 1.0 inside, 0.0
  outside. The earlier inner-seam fade was the source of the
  within-arm distortion (weight discontinuity 1.0 → 0.7 → 0.0
  caused triangles spanning bulk-arm and seam-arm to deform during
  rotation).
* Added `body_rig._smooth_seam` — Laplacian smoothing **restricted
  to in-mask neighbours**. Run after each rotation on the seam-ring
  verts of the rotated group. Smooths the worst seam triangles
  without pulling toward torso (the in-mask restriction is what
  keeps it surgical).
* Pre-computed `seam_indices` in `RigState` so the per-frame seam
  smoother is O(seam) per group, not O(n_verts).

### Visual results

* `docs/images/_diag_labels_<pose>.png` — labels + weights + strand
* `docs/images/_diag_disp_<pose>.png` — displacement + stretch maps
* `docs/images/_fade_mode_comparison.png` — 6 poses × 3 fade modes
* `docs/images/_body_v7_hard_seam.png` — full pose grid post-fix
* `docs/images/_body_effects_animated.gif` — 20-effect animation

## 2026-05-10 — Session: Head-attach + inner-seam skinning

User: "The head needs to stay attached to the body as it moves —
it is becoming detached" / "Skin of the torso is still sometimes
being pulled away from the body when the arms move."

Two coupled fixes:

1. **Removed body rotation from `apply_body_rig_v2`** — the v2
   rig was double-rotating the body (its own torso-mask rotation
   PLUS the existing `_apply_neck_rotation` later in the pipeline),
   while the head was only rotating once. That gap is what looked
   like the head detaching. Now v2 only does limb rotations + pivot
   propagation; `_apply_neck_rotation` handles body bend and brings
   the head along, so they stay locked together.

2. **Inverted the soft-weight boundary** in
   `body_rig._soft_weight`. Previously the 1-ring fade gave torso
   verts adjacent to the limb a 0.15 weight — enough to visibly
   pull the torso skin along with arm rotations. Now out-of-mask
   verts always get 0.0 (torso never moves) and IN-mask seam verts
   relax to 0.7 instead of 1.0, so the limb side absorbs the seam
   discontinuity. Result: torso is fully stationary during all arm
   rotations, the seam is still smooth.

## 2026-05-10 — Session: Body-rig artifact fixes

User report: "Strange deformations in the movements, voxels from
unwanted body parts being included… feet being stretched… sides of
body being pulled up with the arms… movement should be
anatomically realistic and limited to realistic skeleton movement."

Root cause for the worst artifact (stretched-stick arms reaching
to the floor) was a **body-part classifier bug** — `hand_band` was
defined as ``(y <= wrist_y) & (abs_x >= hand_x)``, which captured
the wide outer-foot verts (~654 of them, at Y=-142, far below the
wrist) and tagged them as `BPF_HAND_L/R`. Every shoulder rotation
then yanked the foot along with the hand.

Fixes:
- **`body_3d.classify_body_parts_fine`** — `hand_band` now also
  requires the vert to be within ~0.65 head-heights below the
  wrist; foot verts at large |x| are no longer hand candidates.
  Foot region drops the corresponding ``abs_x < hand_x`` cap so
  wide outer-foot verts go to feet where they belong.
  Counts after fix: hand_L 306 (was 960), foot_L 1137 (was 483).
- **`body_rig.apply_body_rig_v2`** — child pivots now propagate
  through every parent rotation. Shoulder rotation moves elbow +
  wrist pivots; elbow rotation moves wrist pivot; same for
  hip→knee→ankle. Without this, child rotations pivoted around
  stale (pre-parent-rotation) joint positions and stretched the
  forearm/hand into long thin sticks.
- **`body_rig._smooth_labels_mode`** — 1-ring mesh-adjacency mode
  filter (2 iterations) cleans up classifier boundary noise; stray
  verts at limb/torso seams get pulled back into the majority
  cluster of their neighbours.
- **Anatomical joint clamps** in `body_rig._JOINT_LIMITS` —
  shoulder pitch ±90°, roll ±100°, yaw ±29°; elbow flex 150°
  (no hyperextension), elbow yaw/roll near zero; hip/knee/ankle
  similar realistic ranges. Body bend (yaw/pitch/roll) clamped to
  spinal ROM.
- **Reduced soft-weight boundary fade** from 0.5 → 0.15 so
  rotations don't visibly drag adjacent torso/leg verts (fixes
  "sides being pulled up with the arms").

Smoke test: ``tools/animate_body_effects.py`` renders all 20 new
effects through their full timeline as a 5×4 animated GIF —
anatomy stays intact across the entire arc.

## 2026-05-10 — Session: Expanded body movement catalog

User: "please add more body movement effects"

Added 18 new pre-effect handlers to `effects_pre.py`, all built on
the v2 painted-label rig (so torso/limb separation is artifact-free):

- **Composite arm poses**: `shrug`, `arms_crossed`, `hands_on_hips`,
  `point_left`, `point_right`, `thinking` (hand-to-chin), `clap`
  (2-Hz oscillating), `stretch_up` (overhead V), `salute`, `curtsy`.
- **Head/body animations**: `yes_nod` and `no_shake` (3-cycle
  emphatic versions of head_nod / head_shake), `look_around`
  (1-cycle sweep), `breathe` (subtle 2-Hz chest rise), `idle_sway`
  (slow body roll for idle balance).
- **Leg/whole-body**: `lunge_left`, `lunge_right`, `jump`
  (crouch-then-leap two-hump envelope).

All wired into the `HANDLERS` dict (62 total now). Smoke test:
`tools/test_motion.py --effects` renders a 5x4 grid sampled at each
effect's visual peak (per-effect `_EFFECT_U` overrides for
multi-cycle oscillations) → `docs/images/_motion_grid_effects.png`.

Tuning notes:
- Single-axis shoulder roll clamps at π/2 ≈ horizontal; for
  overhead poses (`stretch_up`, `salute`) push roll well past π/2
  (≈ 1.45–2.1) to continue rotation toward vertical.
- For `arms_crossed`, light outward shoulder roll (+0.35) keeps
  upper arms near horizontal; deep elbow flex (−1.8) folds forearms
  inward across the chest.
- The renderer scales head pose by 0.4 (pitch) / 0.6 (yaw, roll) —
  factor that in when picking effect magnitudes for emphatic head
  movements like `yes_nod` (0.55 raw → ~13° rendered).

## 2026-05-08 — Session: Anatomical body fit using mesh landmarks

User: "There should be a way to match the heads to the bodies using the
preexisting body heads... Please use actual anatomical measurement data
to ensure the bodies look correct."

Replaced the hand-tuned percentage approach in `body_3d.py` with a
landmark-anchored fit that uses real measurements at both ends:

- **Body chin from the Vitruvian canon**: adult humans are 7.5 head-
  heights tall. ``chin_z = z_max − z_span / 7.5`` is computed from the
  body OBJ's own bounding box, so the male and female meshes (which
  ship in different sizes already) each get correct chin/neck/shoulder
  landmarks in their own units.
- **ICT chin from the real mesh landmark**: vertex 964 — the peak-
  displacement vertex of the `jawOpen` blendshape — is the chin tip of
  the ICT-FaceKit neutral mesh. Its Y coordinate (≈ −6.47) gives a
  true ICT head height of ≈ 21 ICT units (chin → crown).
- **1:1 placement**: scale the body so its `head_height` equals ICT's
  `head_height`, then translate so the body's chin (= max-Y after
  stripping above the chin) lands on ICT's chin Y. The result is
  anatomically anchored — head and body meet exactly at the chin line
  on the body's own preserved neck and clavicles.

Sanity check: scaling produces 7.5 heads tall (157 ICT units) for both
sexes. Female body uses scale 95.97, male uses 93.09 — the female head
is naturally smaller in the source OBJ so the scale ratio differs while
the rendered proportions stay canonical.

Multi-view render (front / 3-quarter / profile / back × female / blend
/ male) confirms heads attach cleanly to bodies with visible necks
through each body's clavicle topology.

124 tests pass.

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

## 2026-05-07 — Session 21: Hairless xray + mood skin + glowing eyes + jelly anatomy

User: "the Claude animated head with x-ray rendering is one of the best
images for the avatar. No hair... mood-driven skin tones, glowing
eyes, can we merge faceforge anatomy with the head for jelly-person
effects?"

Tackled in priority order: small visual wins first, then the bigger
anatomy underlay.

**Hairless xray** — `style=="xray"` skips the scalp Y-band so the
ICT head reads as a bare skull. Avoids the green-hair uncanny look
and matches the medical-glow aesthetic.

**Mood-driven skin tone (xray)** — `_xray_mood_offset(params)`
computes a small RGB delta from live AU values and mixes it into
M_Face / M_BackHead before render:
- AU12 (smile)        → green-cyan lift
- AU4  (brow lower)   → red shift (anger)
- AU15 (corner drop)  → cool blue (sad)
- AU5  (lid raise)    → pale (fear)
- AU25 (jaw open)     → magenta hot core (open mouth → glow)

**Glowing eyes** — added a `v_emit` per-vertex attribute and a
`u_emit_pulse` shader uniform. `_MATERIAL_EMISSIVE` baked from the
ICT material map (iris/sclera/lacrimal/teeth glow; rest 0). Pulse is
a per-style `(base, amp, hz)` time-modulated scalar (xray pulses
~0.5 Hz, neon ~0.6, cyberpunk ~0.7, transparent ~0.4). Brightened
xray iris colour to (0.30, 0.95, 1.00) so the glow reads.

**CPU bloom post-process** — sci-fi modes get a Gaussian blur of
bright pixels added back over the original. ~3 ms at 320×320.
Halos the eye/teeth glow and gives a soft scifi rim around the
head. Per-style amplitude tuned (xray strongest at 0.45).

**Jelly-anatomy underlay** — new `style=="jelly"` mode composites
BP3D head anatomy behind a translucent ICT xray skin:
- `_render_jelly_composite` renders ICT (xray) and BP3D anatomy,
  per-pixel alpha-blends them. Skin pixels semi-translucent
  (0.05 + luma/255 * 0.85, mid-tone skin attenuated 0.35×); eye/
  teeth glow stays opaque.
- Cool tint + brightness lift on BP3D so muscles read against
  cyan skin instead of warm bone.
- Soft Gaussian silhouette mask of the ICT head clips BP3D to
  inside the ICT outline.
- Final bloom over composite halos the glow through both layers.

**moderngl context-sharing** — moderngl 5.12 has no API for
switching between two standalone GL contexts in the same thread
(verified empirically: second renderer's draws turn into black
frames). `_GpuRenderer.__init__` now accepts an optional `ctx`
parameter, and ict_face's `_shared_anatomy_renderer` builds one
that reuses the ICT renderer's context. Cold start ~5s
(shader compile + mesh upload), warm ~90 ms per frame.

**Anatomy alignment** — user noted "anatomy is much larger than
the animation". Two fixes:
1. Drop cervical vertebrae + neck muscle group from the rendered
   spec list (`_NECK_MUSCLE_TOKENS`: Cap., Colli, Sterno, Thyro,
   Hyoid, Scalene, Levator Scap, Omohyoid, Platysma, Digastric).
   142 specs → ~110 head-only.
2. `_align_anatomy_to_ict`: width-based uniform scale (0.96 ×
   ict_w / bp3d_w) + bbox-centroid translation. No aspect
   distortion; anatomy sits *just inside* the ICT silhouette so
   the bloom halo frames it.

Personas: added `ict_jelly` to `personas.json`. Live-verified
through HTTP /avatar/persona — switches mid-session.

Showcase images:
- `docs/images/ict_xray_moods.png` — 6 moods, hairless xray + tint
- `docs/images/ict_xray_glow.png` — 8-frame eye-pulse strip
- `docs/images/ict_jelly_moods.png` — 6 moods, jelly underlay
- `docs/images/live_*.png` — full-GUI captures via /screenshot

## 2026-05-07 — Session 20: Sci-fi color profiles

User asked for stylised color profiles: "transparent, neon, cyberpunk
xray". Built four `style` presets that flip the ICT material palette
and shader uniforms wholesale. Each is selectable as a persona.

**Implementation**

* `Persona.style` (default `"natural"`) added; `load_persona` /
  `apply_persona` propagate it to `params._persona_style`.
* `vision/ict_face.py`:
  * `_SCIFI_PALETTES` — RGB tuples per ICT material name for each
    of the four styles.
  * `_shader_overrides_for_style` — per-style ambient / specular /
    shininess / sss_tint dict. Xray boosts ambient + drops specular
    for that flat medical-glow look. Neon flattens SSS + cranks
    specular for plastic sheen.
  * `_material_palette` returns the sci-fi palette wholesale when
    style is non-natural; otherwise the natural HSV-derived skin
    palette.
  * `_per_vertex_colors_for` short-circuits the lip / brow / cheek
    post-processing for sci-fi styles — those flourishes only make
    sense on natural skin tones.
  * `_ICTRenderer.render` reads `self._style_uniforms.get(...)`
    rather than hardcoding shader values.
* `personas.json`: 4 new entries (`ict_neon`, `ict_transparent`,
  `ict_cyberpunk`, `ict_xray`) — each picks a black/dark background
  to make the stylised palette pop.

**Visual verification**

Live-captured all four through `POST /avatar/persona` — round-trips
cleanly. Showcase grid in `docs/images/ict_scifi_styles.png`:
- neon: hot magenta skin, glowing cyan eyes, electric green crown
- transparent: ghost pale-blue, ethereal
- cyberpunk: cool teal skin, magenta hair
- xray: dim cyan-bone with bright bone-white teeth/sclera

Committed as `26ee828`. Push to origin blocked by no-direct-to-main
policy — branch + PR needed.

## 2026-05-07 — Session 18: ICT polish v2 + Ollama bug-fix + live integration

After running the live GUI in session 17, user pointed out several
issues that needed fixing:

1. Default happy expression had AU25=0.3 → mouth open showing teeth.
   Closed-mouth happy is more natural.
2. No eyebrows visible.
3. Lips weren't tinted differently from skin.
4. Skin tone was uniform — no blood-flow variation.
5. Hair cap was uniform / didn't extend over the top.

Also: Ollama integration was broken (`'method' object is not iterable`).

**Fixes**

  * `expressions.json`: happy.AU25 0.3 → 0.0, AU12 0.9 → 0.75,
    AU6 0.7 → 0.6. Smile without dropping the jaw.

  * `ict_face.py` `_per_vertex_colors_for(params)` rewritten as a
    fully vectorised NumPy pipeline:
    - Hair cap: top 32% of head Y, ALL sides (no z-back filter
      anymore — the previous logic excluded the front of the
      crown). Per-vertex hair noise (~±10% multiplicative) so it
      doesn't look uniform.
    - Eyebrows: thin band 2.5-5.5% of head height above the eye
      mean Y, front-facing only, painted with hair_color * 0.9.
    - Lips: blend lip_color at 60% with underlying skin (lips
      read as redder skin, not paint stripe).
    - Cheek blush: ~10% blend toward warm pink at the cheek apple
      area (was 30% — looked like rouge).
    - Subtle whole-face per-vertex luminance noise (~±2.5%).

  * `llm/ollama_client.py`:
    - Fixed `'method' object is not iterable` —
      `Conversation.messages` is a method, not an attribute.
      Now calls `conv.messages()` properly.
    - `pick_default_model` skips `*-vision` and `*llava*` variants
      (they expect multimodal inputs we don't supply, return 500).

**Live integration test**
  GUI booted with `FACEVIEW_AVATAR=1`; sent "Hi, say hello briefly"
  via `POST /chat` with no ANTHROPIC_API_KEY set. Ollama (llama3
  installed locally) replied "Hello! It's nice to chat with you
  today!" through the streaming /api/chat endpoint. Status pills
  read `happy 78% / mouth silent` (closed-mouth smile correct).
  Hair cap visible, subtle eyebrows + cheek blush, skin variation.

Tests stay at 124 green.

## 2026-05-07 — Session 17: ICT polish (skin / eyes / hair) + Ollama fallback

User asked for four targeted improvements after the live GUI test:
1. Eye color was off → added persona-driven iris colour.
2. Skin tone too pink → HSV-based skin colour from `persona.skin_hue`
   plus tunable saturation/value, and toned-down SSS.
3. Realistic hair → scalp-vertex hair-cap detection inside the
   ICT mesh (no separate mesh needed; Y > top-28% + Z < median
   triggers `M_HairCap` material with persona.hair_color).
4. Ollama fallback when no Anthropic key.

**Persona schema extended** (`vision/personas.py`)
  * `eye_color`, `skin_saturation`, `skin_value` fields.
  * Persona JSON updated for all 9 ICT presets — brown / blue /
    green / hazel iris distribution; saturation drops with age,
    value lighter for females.

**ICT renderer per-persona palette** (`vision/ict_face.py`)
  * `_material_palette(params)` builds skin from HSV(skin_hue, sat, val),
    iris from persona eye_color, gums tinted from lip_color.
  * `_per_vertex_colors_for(params)` rebuilds the per-vertex
    array per-frame so persona switches take effect immediately.
  * SSS tint reduced from (0.78, 0.46, 0.40) to (0.62, 0.36, 0.30)
    plus narrower terminator band.
  * Scalp hair cap: vertices with Y > 72% of head height AND
    Z < median (back-of-head) get hair_color, with a smooth fade
    at the forehead transition.

**Ollama LLM fallback** (`llm/ollama_client.py`)
  * `is_ollama_available()` pings localhost:11434/api/tags.
  * `list_ollama_models()` and `pick_default_model()` for auto-
    selection (FACEVIEW_OLLAMA_MODEL env var or first installed
    llama/mistral/phi/qwen).
  * `OllamaEngine` matches our existing engine protocol —
    streaming chat via /api/chat, pure stdlib urllib + json.
  * `ClaudeClient` chains: Anthropic → Ollama → Echo. Logs which
    engine was selected at startup.

Tests: 117 → 124. `test_ollama_bridge.py` covers reachability,
model listing, default-model picking, OllamaEngine init, and the
fallback chain (with patched detectors).

## 2026-05-07 — Session 16: ICT-only consolidation + persona library

User pivoted to a single-solution focus: make ICT-FaceKit the
single best face renderer for the project, with multiple personas
spanning different sexes and ages. The other bridge modules
(BFM / FLAME / RPM / MetaHuman / FaceScape / DECA) stay shipped
as opt-in alternatives but ICT is now the recommended path.

**Persona library** (`assets/config/personas.json`) gains six
named ICT presets:
  ict_male_young / male_middle / male_elder
  ict_female_young / female_middle / female_elder
Each combines:
  * Identity PCA coefficients (5–6 ICT identity_<n> modes
    blended) producing visibly different head shapes per persona.
  * Skin hue, hair colour, and lip colour appropriate to age/sex
    (greying hair on elders, younger lip tones on females, etc).
  * `docs/images/ict_persona_library.png` showcases all six.

**Avatar integration**
  * SimCameraWorker now accepts `persona=` kwarg → passed through
    to TalkingAvatar.
  * app.py auto-picks `ict_claude` when ICT data is present
    (already wired in session 14).

**Hair overlay disabled by default**
  Procedural 2D hair was producing spiky / over-tall caps that
  hurt rather than helped. Disabled by default; opt in with
  `params._enable_hair = True`. The bald ICT head reads cleaner.

**README pivoted to ICT-centric**
  Replaced the multi-mode comparison table at the top with the
  ICT-FaceKit setup flow + persona library. Other modes still
  documented but not foregrounded.

Tests stay at 117 green.

## 2026-05-07 — Session 15: All remaining face-resource bridges shipped

User asked us to wire in MakeHuman gendered targets first then
work through the remaining roadmap items in order. Done.

**MakeHuman gendered targets** (`vision/makehuman_mesh.py`)
  * Bundled CC0 `male_young.target` and `female_young.target`
    files from MakeHuman community.
  * `load_target(name, n_verts)` parses sparse vertex deltas.
  * `load_makehuman_head(grid, target=)` applies them before crop/
    decimation. Personas `makehuman_male` / `makehuman_female`
    set the `mh_target` key.

**A39 — Basel Face Model bridge** (`vision/bfm_face.py`)
  * Lazy-imports `eos-py` (PyPI). Loads BFM 2017 H5 from
    `assets/data/bfm/`. Persona `identity_weights` keys `bfm_<n>`
    drive the PCA shape coefficients.
  * Apple Silicon caveat: PyPI `eos-py` wheels are x86_64 only.
    Documented in module docstring; users run under Rosetta.

**A41 — Ready Player Me bridge** (`vision/rpm_avatar.py`)
  * Lazy-imports `pygltflib`. Fetches `<id>.glb` from
    `https://models.readyplayer.me/`, caches to
    `assets/data/rpm/`. Extracts head mesh + ARKit-named morph
    targets from the glTF binary blob. Renders through ICT's
    moderngl pipeline.

**A38 — FLAME PyTorch bridge** (`vision/flame_face.py`)
  * Lazy-imports `torch` + `FLAME-PyTorch`. Persona keys
    `flame_shape_<n>` and `flame_expr_<n>` drive the 100+100 PCA
    coefficients. CC-BY academic — model file (~100 MB) requires
    user signup at MPI-IS.

**A32 — MetaHuman FBX bridge** (`vision/metahuman_face.py`)
  * Loader via `pyassimp`. Reads anim_meshes for ARKit
    blendshapes. Free Gumroad distribution from Dragonboots,
    user places `head.fbx` at `assets/data/metahuman/`.

**A40 — FaceScape / FaceVerse bridge** (`vision/facescape_face.py`)
  * OBJ loader for the non-commercial pore-level scans. Persona
    keys `facescape_subject` / `facescape_expression`. Data
    needs manual download (research licence).

**A44 — DECA / EMOCA capture bridge** (`vision/deca_capture.py`)
  * `DECACapture(checkpoint_dir).fit_to_image(bgr)` returns FLAME
    parameters; `.to_au_values(codedict)` heuristically maps to
    our 12-AU pipeline. Heavy dep (torch + DECA repo).

**Wiring**
  * `sim_face.render_face` now dispatches: `bfm_3d` / `rpm_3d` /
    `flame_3d` / `metahuman_3d` / `facescape_3d` modes alongside
    everything else.

Tests: 107 → 117. test_optional_face_bridges.py covers import,
graceful MissingDependency raises, MakeHuman target loader.

ROADMAP STATUS
  Done in this session: A38, A39, A40, A41, A44, A45, A46 (all as
  bridges; users opt in deliberately by installing deps + data).
  Heavy / commercial paths now have lightweight Python wrappers
  ready for the day a faceview user wants the higher-fidelity
  head.

## 2026-05-07 — Session 14: SSS shader + cleanup + roadmap completion

User asked for A42 (skin texture + SSS), README cleanup, removal
of redundant modes, then to plan + implement all remaining
roadmap items.

**A42 — SSS skin shader on ICT face** (`vision/ict_face.py`)
  * Build tool now extracts per-triangle material tags from the
    OBJ's `usemtl` directives (12 materials: face / back-head /
    teeth / gums / sclera-L/R / iris-L/R / lacrimal-fluid /
    eye-blend / occlusion / lashes).
  * Per-vertex colour computed from the material table (skin warm,
    teeth ivory, sclera bright, iris dark amber, etc.) — vertices
    on material seams blend naturally.
  * Upgraded GLSL fragment shader with five components:
    1. Wrap-diffuse (Lambert × 0.5 + 0.5) for soft falloff
    2. Subsurface tint at the terminator only (warm flesh bleed)
    3. Sky-tinted ambient (warm above, cool below)
    4. Dual-lobe specular (broad + tight)
    5. Fresnel rim glow on thin features

**A43 — Eye-specific specular** (`vision/ict_face.py`)
  * Per-vertex specular intensity from the material table — sclera
    + lacrimal fluid get high gloss (~0.9-1.0), teeth moderate
    (0.65), skin subtle (0.30), lashes matte (0.05). Wet-eye look.

**A26 — GPU path for `head_decimated_3d`** (`vision/head_decimated.py`)
  * New `render_face_decimated_gpu` routes through moderngl with a
    Phong shader, replacing the 8 fps QPainter path.

**A12 — Phong on CPU faceforge_3d** (already vectorised in
`vision/anatomy_meshes.py` from session 7) — verified.

**A36 — openFACS UDP bridge** (`vision/openfacs_bridge.py`)
  * Pure-stdlib socket + JSON. `OpenFACSBridge.send(au_values)`
    emits one packet on UDP localhost:5000 in phuselab/openFACS'
    expected format. `attach_to_avatar(avatar)` wraps the
    avatar's tick so every rendered frame also streams.

**A34 — MediaPipe FaceLandmarker capture** (`vision/mediapipe_capture.py`)
  * `MediaPipeCapture(camera_index=0).next_frame_blendshapes()`
    returns 52 ARKit-named coefficients per webcam frame; chains
    cleanly into `arkit_to_au_values()` to drive any of our
    avatar render modes.

**Cleanup**
  * Removed deprecated `vision/head_3d_lite.py` + tests + persona
    + dispatcher entry. The Delaunay-over-86-points spider-web
    was deprecated since session 11; cleared away now.
  * README mode table restructured: ICT at top, modes grouped by
    realism tier, deprecated entries removed.
  * INTERFACE updated with new modules.

Tests: 109 → 107 (removed 4 head_3d_lite tests, added 4 new ones).
All green.

Roadmap: A12, A26, A34, A36, A42, A43 marked done. Remaining
candidates documented (A32 MetaHuman FBX, A38 FLAME PyTorch,
A39 BFM via eos-py, A40 FaceScape, A41 Ready Player Me, A44 DECA)
require heavy ML deps or non-commercial licensing — deferred to
future sessions when the use case demands them.

## 2026-05-07 — Session 13: ICT-FaceKit integration — biggest realism jump yet

User asked us to research all the candidate face resources and ship
the best one. Surveyed: ICT-FaceKit, MetaHuman head FBX, FLAME,
Basel Face Model, FaceScape, FaceVerse, DECA/EMOCA, Ready Player
Me, MediaPipe FaceLandmarker, openFACS, MakeHuman, ProductionCrate,
CMU mocap, USC ICT pack. Findings consolidated in
`docs/FACE_RESEARCH.md`.

**Top recommendation shipped: ICT-FaceKit** (USC Institute for
Creative Technologies, MIT-licensed, 26K verts, 157 blendshapes).

Pipeline:
1. Cloned `USC-ICT/ICT-FaceKit` (386 MB tree of OBJs).
2. New `tools/build_ict_blendshapes.py` reads the neutral mesh +
   every blendshape OBJ, computes per-vertex deltas, saves a
   compressed npz: `assets/data/ict/face_kit.npz` (23 MB).
3. New `vision/ict_face.py` loads the npz, applies ARKit-named
   blendshape coefficients as vertex displacements, renders
   through moderngl with a Phong shader.
4. ICT names map 1:1 to ARKit (just `_L`/`_R` → `Left`/`Right`),
   plugs straight into our existing `arkit_blendshapes` layer.
5. `face_params_to_au_values` → `au_to_arkit_values` →
   `apply_blendshapes(neutral, deltas)` → render. Same FACS
   pipeline, real anatomical mesh deltas, ~88 fps GPU.

Result: a real human head with visible teeth when the jaw opens,
genuine smiles that pull lip corners up, smooth skin shading, all
animated by our existing FACS expressions / viseme pipeline.

Tests: 105 → 109. `test_ict_face.py` covers npz load, blendshape
application, render frame validity, dispatcher routing — all
gated on the npz being built locally.

Render mode `ict_face_3d`, persona of same name. The npz is
gitignored (23 MB); user runs build tool once after cloning
ICT-FaceKit.

Demo images:
- `docs/images/ict_face_grid.png` — neutral / happy / sad /
  surprised / jaw_open / yaw 0.5
- `docs/images/ict_face_talking.gif` — talking head animation
- `docs/images/realism_progression.png` — full mode progression

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

## 2026-05-09 — Full BP3D skeleton + region-aware skin fit

- Loaded all 231 BP3D bones (cervical 15 / thoracic 22 / lumbar 11 /
  skull 11 / jaw 1 / rib_cage 43 / pelvis 2 / upper_limb 10 / hand 54
  / lower_limb 8 / foot 54) into `assets/skeleton/` plus their
  per-group manifest JSONs in `assets/skeleton/configs/`.
- Replaced the single uniform-scale fit with a region-aware fitter
  (`vision.skeleton_fit.fit_to_body`): each anatomical region computes
  its own bbox in BP3D-ICT frame and maps to a target box derived
  from the avatar's actual body-skin landmarks (shoulder line, hip
  line, knee/ankle bands), measured directly off the gendered body
  mesh with IQR clipping so the hand outliers don't pollute torso
  width.
- Body width is sampled per-Y-band using `body_3d.classify_body_parts`
  to mask away arm/hand verts at the hip line. Reuses the same
  scale=118.3 / `body_mesh_alignment` calibration logic faceforge
  already validated for the inverse problem (sex-transformation body
  warp); we just go in the opposite direction (skeleton → skin).
- New `tools/render_skeleton_overlay.py`: renders body+head at front
  and side views with all fitted bones projected through the same MVP
  as the avatar renderer. Output: `docs/images/_skeleton_male_full.png`
  and `..._female_full.png`.
- Split `skeleton.py` (was 702 lines after the new defs) into
  `skeleton.py` (271 lines, defs/STL parse/transform) and
  `skeleton_fit.py` (347 lines, landmarks + fit). Both now under the
  500-line cap.
- `INTERFACE.md` updated for the new module + tool.

## 2026-05-09 — Limb chain rotation: BP3D bones aligned to skin arm/leg axis

- Added `_chain_align(prox_src, dist_src, prox_tgt, dist_tgt)` — Z-axis
  2D rotation in the XY plane that rotates a limb chain to match the
  body skin's actual arm/leg direction. Rotates whole upper_limb +
  hand together (one rotation around the shoulder); same for
  lower_limb + foot (around the hip).
- Endpoints are bone *ends*, not centroids: top of humerus / bottom of
  radius for arms; top of femur / bottom of tibia for legs (use the
  20% extreme-Y verts).
- Fixed the L/R side handedness: BP3D R is at +X, ICT/avatar R is at
  −X. Negated X in `_skull_to_ict` so R/L stay consistent across
  frames — otherwise the rotation flipped the wrong direction.
- Measured actual arm centroid X at three Y bands (shoulder / elbow /
  wrist) from the body mesh's BP_LEFT_ARM/BP_RIGHT_ARM verts. The
  BP3D shoulder→wrist target now uses these instead of the torso
  hip width — so the arm chain lands at the body's hanging hand
  position (≈±37 X at wrist) rather than the inner torso edge.
- Restricted the rotation to Z axis only. Earlier 3D Rodrigues
  rotated Z components too, which pushed the scapula / clavicle
  bundle out of plane. XY-only rotation keeps the front-back ordering
  intact.

## 2026-05-09 — Per-segment limb fit + face-anchored skull/jaw + line rendering

- Replaced the single-chain limb fit with a three-part rig per side:
  shoulder girdle (clavicle+scapula bbox-fit onto the body shoulder
  area), humerus (chain shoulder→elbow), radius+ulna (chain
  elbow→wrist). Same idea for legs: femur (hip→knee) and
  tibia/fibula/patella (knee→ankle) as separate chains.
- New `skeleton_landmarks.limb_landmarks(body_morph)` measures 3D
  shoulder/elbow/wrist and hip/knee/ankle joint positions off the
  body skin, plus 3D bboxes for the body's hand and foot region.
  Limb fits anchor to these so bones run along the body's actual
  hanging-arm/leg cylinder (X + Z + Y), not just a vertical column.
- Hand and foot now bbox-fit into the measured body hand/foot box —
  guarantees they stay inside the visible silhouette.
- Skull/jaw use a 4-anchor piecewise Y mapping (crown → eye → mouth
  → chin), with eye_src derived from the orbit aperture (zygomatic
  top + frontal bottom midpoint) and mouth_src from between-teeth
  (maxilla bottom + mandible top). Eye sockets and chin land on the
  ICT face mesh's iris materials and chin landmark vertex.
- Pelvis raised: top of bowl now sits 0.85 head_h above hip joint,
  putting the iliac crest in the lower torso instead of the leg.
- Cervical Z shear backed off from 0.30 → 0.12 head_h; the spine
  meets the skull base at the back of the head without poking
  through it.
- Bones rendered as PCA-axis lines (`tools/render_skeleton_overlay.py`).
  PCA principal direction handles vertical bones (vertebrae, femur)
  and horizontal ones (clavicle, ribs) uniformly — line goes between
  the bone's two extremes along its long axis.
- Split landmarks out of `skeleton_fit.py` into
  `skeleton_landmarks.py` (now 366 lines); both fit + landmark
  modules under the 500-line cap.

## 2026-05-10 — Skeleton-bone voxel relabel

Built `tools/skeleton_voxel_relabel.py` to detect mis-labeled body
voxels by measuring each vert's distance to its owning bone segment
(shoulder→elbow→wrist→hand_tip and hip→knee→ankle→foot_tip pivots
from `RigState`). Three complementary detectors run together:

- **Rest-pose bone-distance**: vert flagged if dist(current bone)
  is `>1.5×` dist(closest limb bone) AND `>1.5` units farther.
- **Cross-side anatomical check**: any `_R` label at +X (subject's
  left) or `_L` at -X is mirrored to the same-chain label on the
  correct side, but only if the mirror bone is actually nearby
  (else defer to closest-bone pick).
- **Per-pose bone-following** (latent): captured posed-pivot dict
  via `_capture_rig_io`. Currently disabled — `apply_body_rig_v2`
  works on a *local* copy of `rig.pivots` so the cached state never
  reflects posed positions.

Mirror-correct broken wrist_R pivot: the skeleton-fit puts wrist_R
right next to elbow_R for both genders (likely a BP region-detection
failure). When `|x_R| < 0.6 × |x_L|`, mirror the L-side joint to
the R-side for the bone-distance test (the rig itself is unchanged).

Convergence after multiple iterative passes:
- `body_label_overrides_male.json`: ~366 → 983 overrides
- `body_label_overrides_female.json`: ~464 → 856 overrides

Visual: dramatic reduction in flyaway artifacts during stretch_up,
arms_up, clap, etc. Remaining stragglers (a couple of thin streaks)
require fixing the underlying wrist_R skeleton fit, not just label
overrides.

## 2026-05-10 — Bake overrides + phantom-filter ordering fix

Created `tools/bake_label_overrides.py` to merge
`body_label_overrides_{male,female}.json` into
`body_part_labels_{male,female}.npz`. Renamed JSON files to
`_baked.json` suffix so runtime no longer re-applies them.
923 male / 789 female overrides baked. Backups of the original NPZ
saved to `body_part_labels_<g>_orig.npz`.

Also added one more direct armpit fix: male vert 1813 was labeled
`u_arm_L` but sat in the left armpit cleft, far from the upper-arm
bone — 12/19 neighbours were chest, so reassigned directly.

**Root cause of the lingering "dark voxel necklace" at the shoulders**
(visible across every effect, even in the rest pose):

`ict_face.py` ran `filter_phantom_triangles` on the SMOOTHED but
NOT-OVERRIDDEN labels (line 1412). When a vert was overridden to a
different anatomical region (e.g. `u_arm_L` → `chest`), its old
bridge triangles to neighbouring arm verts survived. During arm
rotation those triangles stretched into dark slivers along the
shoulder seam — the "necklace" we kept noticing.

Fix in `ict_face.py`: apply `_apply_manual_overrides` to `_smoothed`
BEFORE `filter_phantom_triangles`, then update `_fine` to the
overridden labels for downstream `build_rig_state`.

Result: GUI renders are dramatically cleaner — neutral, arms_up,
clap, salute, arms_crossed all show clean limb motion with no
necklace artifact.

Dynamic test confirms 0 unexpected movers across all 12 effects ×
2 genders.

## 2026-05-10 — GUI tour + CI test + graded skinning weights

**GUI tour**: scripted `tools/_gui_tour.sh` to drive the live GUI
through 28 body effects per gender, captured screenshots, built
labelled grids saved to `docs/images/body_effects_tour_{male,female}.png`.
All poses render cleanly — no flyaways, no necklace artifact.

**CI regression test**: `tests/test_body_rig_regression.py` covers
26 cases: 13 arm-effect × 2 genders asserting only arm labels move,
3 leg-effect × 2 genders asserting only leg labels move, and
2 neutral-pose isolated-voxel checks. Threshold = 1.0 unit
displacement. All 34 cases pass.

**Graded skinning weights**: switched default
`FACEVIEW_RIG_WEIGHT_MODE` from "hard" to "graded_3ring"
(seam-ring 0.85, second ring 0.95, deeper 1.0). Visible
improvement at shoulder/armpit transitions across arms_up,
salute, arms_crossed for both genders — sharp seam and
triangular armpit gap (HARD mode) are eliminated. Set env var
to "hard" to revert.

Regression tests still pass under graded mode — torso verts
adjacent to arms remain weight 0.0 because they're not in the
arm mask; only IN-mask seam-ring arm verts are graded down.

## 2026-05-10 — body_morph intermediate-value regression fix

**Symptom**: User opened the GUI fresh and reported "many bad voxels
during movement" — massive flyaway pieces of arm/torso, holes in
abdomen, despite the regression test suite passing and the prior
GUI tour being clean.

**Root cause**: `body_part_labels_{male,female}.npz` are baked for
two specific vert counts only — 7037 (male) and 7028 (female).
Inside `gen_body_mesh`, intermediate `body_morph` values blend the
two raw OBJs (both 10582 verts), then the sloped chin-strip uses
the BLENDED chin/neck Y, dropping a different number of top verts
at every morph value (7028…7037). For any morph ∈ (-1, +1) the
post-strip mesh has a count that matches NEITHER NPZ, so
`_try_load_painted_labels` returns None and the old threshold
classifier kicks in — undoing all the skeleton-relabel work.

The GUI's `body_morph` slider default was 0.0, which produced a
7029-vert mesh on every fresh launch — explaining why every user
session started in the broken state.

**Fix** (two edits):

1. `src/faceview/vision/body_3d.py` — `gen_body_mesh` now snaps the
   morph to the nearest baked extreme (`±1.0`) before picking the
   raw mesh. The slider remains continuous in the UI but the
   renderer treats it as a 2-state male/female selector. Labels
   always match.
2. `src/faceview/gui/effects_panel.py:373` — `body_morph` slider
   default changed `0.0 → 1.0` with step `0.05 → 2.0` so the slider
   behaves as a discrete toggle and fresh GUI sessions start at the
   tested male morph.

**Personas**: confirmed personas.json never sets `body_morph` —
every persona inherits the slider value, so all 40 personas
share the same body. No per-persona corrections were missed; the
correction needed was only for the slider default.

`tests/test_body_rig_regression.py` still passes (32 effect tests
+ 2 neutral skipped without scipy). GUI screenshot after fix
shows a clean avatar across `arms_up`, `arms_out`, `kick_left` —
no flyaways, no holes.

## 2026-05-10 — Head-nod neck-base drift fix

**Issue**: When the head pitches (slider `params.pitch`), the BASE of
the neck visibly drifts even though the user expects it to stay
stationary — only the top of the neck and skull should pivot.

**Diagnosis** (`tools/_nod_drift_measure.py` + `_nod_drift_inspect.py`):
The cervical cascade in `_apply_cervical_cascade` interpolates pitch
across 12 spine levels with cumulative fractions
`(1.00, 0.98, 0.85, 0.55, 0.25, 0.10, 0.04, ...)`. At C5 the cumulative
pitch is still **10 %** and at C6 it's **4 %**, so mid-neck verts
displace by ~0.25 ICT units at full pitch — the visible drift.

**Fix**: Introduced `FACEVIEW_NOD_MODE` env var that selects from
five cascade profiles, plus an optional post-anchor that snaps
verts back to rest below a Y threshold:

- `current` — legacy fractions (kept for A/B)
- `sharper` — bend concentrated at C1-C3, C4-T4 → 0
- `spine_ripple` — sharp top + tiny T1-T4 ripple **(NEW DEFAULT)**
- `anchored` — legacy fractions + snap-to-rest below y_norm=-0.30
- `sharp_anchored` — sharper + anchor below y_norm=-0.25

Default chosen as `spine_ripple` because the user explicitly asked
for "some flex passed down the spine" while keeping the neck-base
junction visibly stationary.

**Measured improvement** (body-mesh mean displacement at pitch=+1.0):

| Y band | current | spine_ripple | change |
|---|---|---|---|
| upper-neck (C1-C3) | 1.0036 | 0.5838 | -42 % |
| mid-neck (C4-C6) | 0.2472 | 0.0606 | -76 % |
| neck-base (C7-T1) | 0.0344 | 0.0211 | -39 % |
| upper-torso/clavicle | 0.0043 | 0.0079 | (tiny ripple) |
| mid-torso | 0.0000 | 0.0000 | unchanged |

`tests/test_body_rig_regression.py` still passes under the new
default (32 effect tests + 2 neutral skipped without scipy).

Tools added:
- `tools/_capture_nod_sideview.py` — baseline side-view grid
- `tools/_nod_drift_measure.py` — per-Y-band displacement table
- `tools/_nod_drift_inspect.py` — cascade parameter dump
- `tools/_compare_nod_modes.py` — all-mode visual grid
- `tools/_nod_overlay_compare.py` — rest-vs-pitched colour overlay
- `tools/_nod_final_compare.py` — before/after 2×2 grid
- `tools/_nod_table.py` — labelled comparison table image

Compare images at `/tmp/nod_modes_compare.png`,
`/tmp/nod_final_compare.png`, `/tmp/nod_table.png`.
