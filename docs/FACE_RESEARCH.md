# Face research — comprehensive resource survey (2026-05-07)

What follows is a thorough survey of every realistic-face resource
the project has investigated, with licensing, file format, Python
integration path, and recommendation status. We've already
integrated the green ones; the others are documented as candidates
for future work.

## Quick decision matrix

| Resource | License | Quality | Python? | Texture? | Status |
|---|---|---|---|---|---|
| **ICT-FaceKit** | MIT | High (USC) | numpy, openmesh | Has UV (subset) | ✅ **integrated** — `ict_face_3d` |
| **MakeHuman base** | CC0 | Medium | OBJ parser | No | ✅ integrated — `makehuman_3d` |
| **BodyParts3D** | non-comm research | High | STL loader | No (medical scan) | ✅ integrated — `faceforge_3d_gpu` etc |
| **ARKit 52 blendshapes** | Apple spec (open) | n/a | Names list | n/a | ✅ integrated — `arkit_blendshapes` |
| **MetaHuman head FBX** (Gumroad) | Free (license unclear) | Highest | FBX parser needed | Yes | candidate, A32 |
| **FLAME** | CC-BY academic | High | PyTorch (heavy dep) | UV map | candidate, A37 |
| **Basel Face Model 2017** | academic | High | `pip install eos-py` | OBJ + texture | candidate, A38 |
| **FaceScape** | non-commercial | Highest (pore-level) | Python pipeline | Yes | future, A39 |
| **FaceVerse** | non-commercial | Highest (DLSR rig) | Python pipeline | Yes | future, A40 |
| **DECA / EMOCA** | research | Medium (image→3D) | PyTorch3D | UV | future capture pipeline |
| **Ready Player Me** | free for use | Medium-high | GLB → Blender | Yes | possible, A41 |
| **MediaPipe FaceLandmarker** | Apache 2.0 | n/a (capture only) | mediapipe pkg | n/a | future input bridge, A34 |
| **openFACS** | MIT (Unreal) | High render | Python UDP API | n/a | future output bridge, A36 |
| **CMU mocap** | research-license | Body, not face | BVH/ASF/AMC | n/a | out of scope (body) |
| **ProductionCrate facial pack** | (USC ICT) | High | OBJ files | UV | superseded by direct ICT-FaceKit |

## ICT-FaceKit — our pick (`ict_face_3d`)

**Why it won the comparison:**
- MIT license — matches faceview's, can ship freely.
- Released by USC's Institute for Creative Technologies Vision &
  Graphics Lab. Used in research-grade work and the source of the
  ProductionCrate pack.
- 26,719 vertices, 52,220 triangles, 157 blendshapes total
  (52 ARKit-aligned + 100 PCA identity + 5 misc).
- Each blendshape ships as a full deformed OBJ; we pre-compute
  per-vertex deltas and store as a 23 MB compressed `.npz`.
- Names are 1:1 ARKit-compatible (`jawOpen`, `mouthSmile_L`,
  `eyeBlinkLeft`, …) so our existing AU↔ARKit mapping plugs in
  unchanged.
- Renders at **~88 fps** through our existing moderngl path (Apple
  Metal). The mouth opens with visible teeth and tongue, smiles
  pull lip corners up — actual facial muscle behaviour from real
  scans.

**What this enables:**
- A real, animated, lifelike avatar — the highest-realism mode the
  project ships.
- Drop-in compatibility with any external face tracker that emits
  ARKit blendshapes (MediaPipe FaceLandmarker, iOS Face ID, etc).
- An identity space (100 PCA shape modes) for generating
  many distinct faces — future work to expose this as personas.

## MetaHuman head FBX (candidate)

Free download from a Gumroad creator (Dragonboots) extracts the
52-blendshape head + teeth + eye models from Epic Games' MetaHumans.
Highest visible quality but:
- License terms aren't clear (Gumroad freebie, derived from Epic's
  EULA).
- FBX format requires a heavy parser (`PyMeshLab` / `aspose-3d` /
  Blender headless). Not lightweight.
- Already covered by ICT-FaceKit's coverage of ARKit blendshapes
  for now.

## FLAME (candidate)

Statistical 3D head model from Max Planck Institute, learned from
33,000+ aligned scans. Differentiable PyTorch implementation.
Articulated jaw + neck + eyeballs as separate joints + pose-
dependent corrective blendshapes + global expression blendshapes.

Huge pro: differentiable → can fit to images / fit to FACS curves.
Huge con: PyTorch dep is heavy (~2 GB), CC-BY academic licence
restricts commercial use. Most useful as the basis for a
**capture pipeline** (image → fitted FLAME → drive our avatar)
rather than as a primary renderer.

## Basel Face Model 2017 (candidate)

University of Basel's classic 3DMM. Easy Python access via
`eos-py` on PyPI. Outputs OBJ + texture from PCA shape +
expression coefficients. Lower polygon count than ICT, less
detailed expression rig, but the simplest *pip-installable*
realistic-head option. Could supplement ICT for cases where users
don't want to clone the GitHub repo.

## FaceScape / FaceVerse (future)

Highest-quality scans available — pore-level detail from DLSR
rig captures, 18,760 textured 3D faces (FaceScape) or 2,688
high-quality scans (FaceVerse). Restricted to non-commercial
research use. Right answer if we ever build a *training* pipeline
for face reconstruction; overkill for our renderer.

## Game-industry techniques (per MetaHuman docs)

| Technique | Effect | Feasible in faceview? |
|---|---|---|
| Subsurface scattering (SSS) | Light penetrates skin → soft realistic shading | GPU only, custom shader; possible upgrade to ICT renderer |
| Dual specular lobes | Two-tone skin sheen | GPU only, shader work |
| Detail texture (pores) | Close-up realism | Need texture map |
| Dual-quaternion skinning | Smooth bone deformation | Already implicit via blendshapes |
| Eye specular + tear film | Wet-eye highlights | Need eye-specific material |
| Asymmetric blinks | Personality / liveness | Cheap — already supported via ARKit L/R |

These are the next realism tier above what we ship today; all
require shader work on top of ICT's geometry.

## Recommendation

**Today's commit makes ICT-FaceKit the project's highest-realism
animated avatar mode** — `ict_face_3d`. It pairs naturally with
our ARKit blendshape compatibility layer, runs at GPU real-time
speed, and doesn't depend on any heavy ML or commercial libraries.

**Next realism step** (when the project wants to push further):
1. Add a skin texture map to ICT's UV layout (it ships with one)
2. Add a custom GLSL fragment shader doing SSS + dual specular
3. Eye-specific specular material for the wet look
4. Integrate FLAME as a *capture* path: image → FLAME fit → ARKit
   coefficients → ICT renderer

Sources:
- [USC-ICT/ICT-FaceKit (MIT)](https://github.com/USC-ICT/ICT-FaceKit)
- [ICT Face Model overview](https://vgl.ict.usc.edu/Data/ICTFaceModel/)
- [MakeHuman Community](https://static.makehumancommunity.org)
- [ARKit Face Blendshapes](https://arkit-face-blendshapes.com/)
- [MediaPipe — Realistic Virtual Humans](https://developers.googleblog.com/mediapipe-enhancing-virtual-humans-to-be-more-realistic/)
- [FLAME model](https://flame.is.tue.mpg.de/)
- [Basel Face Model 2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html)
- [FaceScape](https://nju-3dv.github.io/projects/FaceScape/)
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse-Dataset)
- [DECA](https://github.com/yfeng95/DECA) / [EMOCA](https://github.com/radekd91/emoca)
- [MetaHuman Head FBX (Gumroad)](https://dragonboots.gumroad.com/l/metahumanhead)
- [Ready Player Me docs](https://docs.readyplayer.me/)
- [phuselab/openFACS](https://github.com/phuselab/openFACS)
- [CMU mocap](https://mocap.cs.cmu.edu/)
- [ProductionCrate free blendshapes](https://news.productioncrate.com/free-3d-facial-expression-blendshapes-pack/)
- [Yelzkizi — Hyper-Realistic MetaHuman in UE5](https://yelzkizi.org/create-a-hyper-realistic-metahuman-in-unreal-engine-5/)
