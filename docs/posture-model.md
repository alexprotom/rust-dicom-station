# Posture model

*Tools ▸ Posture model* shows **the way from one position of a patient to
another**, and measures along it. One patient is often imaged in more than
one position: the planning CT lying down and the treatment CT seated or
upright, prone and supine, arms up and arms down, a full and an empty
bladder. A registration ([registration.md](registration.md)) says where every
point of one scan went in the other; the posture model builds what lies in
between. Each study is a **state**; between two neighbouring states the
model is a path `Phi_t`, `t` in `[0, 1]`, with `Phi_0` the identity and
`Phi_1` the fitted transform, and the window

- **plays it**: the CT, the contours and the 3D scene move from one state
  to the other, forwards, backwards or in a loop, and can be filmed;
- **measures it**: a table of what each structure, each bone and each joint
  did - how far it went, how it turned, how much it changed shape and
  volume.

It is an interpolation of anatomy that was observed at the end points, not
a biomechanical simulation. A state halfway is "the smoothest invertible
path between what was scanned", which is the right thing to look at and to
measure along, and the wrong thing to plan a treatment on. The window says
so under its bar, and every export carries it.

The same model is a CLI (`examples/posture_cli.rs`), an MCP tool
(`build_posture_model`, [mcp.md](mcp.md)) and a workflow step (*Posture
model*, [workflows.md](workflows.md)); all four call
`crate::workflow::posture`.

## The model

### States and the anchor frame

A **state** is one image series of a study with the structure set drawn on
it (else the study's first). Two states in two positions have two frames of
reference, and "how far did the bladder move" has no answer until something
is held still. So the model has an **anchor**:

| Anchor | How the states are laid onto the first |
|---|---|
| a structure on every state (the sacrum, a vertebra, the target) | the rigid surface fit of that structure (`registration::shape`, distance maps, no grey values), with the principal-axes start below |
| *Whole image* | a rigid intensity registration (elastix rigid) |
| *Scanner frame* | nothing: for scans that share a frame of reference |

Every state's image and structures are carried into the first state's
patient coordinates (LPS millimetres) - the **anchor frame** - on the first
state's grid coarsened to the analysis spacing (2 mm). All displacements of
the tables are in that frame, and the animation shows the anchor standing
still while the rest moves around it. Changing the anchor changes the
numbers, by design; the tables name it.

### Three layers

A posture change is not one smooth deformation. Bones move as rigid bodies,
often through large angles (a hip flexed 60 degrees between lying and
sitting); soft tissue follows them and deforms on top. A single B-spline
across both bends bones and folds at the joints. The model has three
layers:

1. **Rigid bodies.** Each structure marked rigid - by default the bones
   every state has (TotalSegmentator's skeleton names, and anything named
   *bone*, *femur*, *pelvis*, *vertebra*, a spine level ...: `fit::is_bone`)
   - gets its own rigid motion from the surface fit of that one structure.
   That fit searches locally from the centroids; a bone turned through
   sixty degrees is out of its reach, so when the surfaces end further apart
   than 1.5 mm it runs again from each of the four proper rotations that lay
   the body's principal axes onto its partner's, and takes such a start
   only when it leaves the surfaces clearly closer (an RMS under 0.8 of the
   local fit's: a nearly symmetric bone laid on itself backwards can lie a
   hair closer by noise). The anchor, if it is a structure, is a body
   standing still. A structure is **not** a rigid body of the pair - and is
   left out, with a warning, the tissue there following its neighbours -
   when it is under three analysis voxels thick along an axis on a state (a
   mark drawn on one slice turns nowhere in a fit), when its contoured
   volume differs between the states by more than a quarter (a bone keeps
   its volume: the scans or the segmentation saw different parts of it -
   the lower ribs at the top of a pelvic scan), or when its best fit leaves
   the surfaces more than 3 mm apart or overlapping with a Dice under 0.6.
   Costal cartilage is not taken for bone.
2. **Poly-rigid blend** (`posture::polyrigid`). Each body's motion is
   written as its matrix logarithm, a *twist* `xi_i` (`posture::se3`); the
   velocity field is `xi(x) = sum_i w_i(x) xi_i`, the log-Euclidean
   poly-affine framework of Arsigny et al. restricted to rigid components.
   A body's weight is one in the region it **sweeps** on its screw from
   `t = 0` to `t = 1`, grown by a lattice node; elsewhere the weights are
   Shepard's inverse-distance weights with an exponential reach (40 mm),
   normalised. Where swept regions meet, a node goes to the body that is
   there at the start - judged on the bodies as they are before their
   one-node margins, so at a joint narrower than a lattice step the femoral
   head's nodes stay the femur's and the socket's the pelvis's. The
   deformation is the flow of that field
   (`posture::svf`): scaling and squaring, the first small step taken
   exactly per node as the rigid motion of that node's twist. A node whose
   path stays in one body's region is given that body's motion exactly, so
   a bone moves rigidly at every `t` however far it turns.
3. **Residual** (optional, *Residual*). What the bodies leave - organ
   filling, soft tissue - is recovered by the plastimatch B-spline engine
   between the first image and the second pulled back through the blend,
   faded to nothing inside the bodies (`1 - rho`, `rho` one inside a body
   at the start and falling over two lattice steps), and turned into a
   stationary velocity field `u` by the logarithm of a displacement field.

### The path

For a stationary velocity field the deformation at `t` is its flow for time
`t`, so

```
Phi_t = exp(t v) o exp(t u),        Phi_t^-1 = exp(-t u) o exp(-t v)
```

- `Phi_0` is the identity and `Phi_1` the fitted transform;
- every `Phi_t` is invertible, so contours and meshes are pushed forward and
  images pulled back without inverting a field numerically;
- a rigid body travels on a screw: constant axis, constant angular rate;
- the path is symmetric: back is forward walked backwards.

The residual acts first, in the first state's configuration where the
fading is exact; the plan this page grew from wrote the two the other way
round, which would let the residual bend a bone that had already moved.

**Linear mode** walks `x + t d(x)` on an end-to-end displacement: the
registration between the two workspaces when there is one, else the
diffeomorphic model's own `Phi_1` - so the two modes share their end points
and differ only in the path. It needs no fitting of its own and is
indistinguishable for small changes; on a turning bone it cuts the chord
instead of following the arc and the bone shrinks halfway (the *Stretch*
column shows it), and the Model fold says when its Jacobian goes to zero or
below on the way.

### More than two states

States `S_0 .. S_n` (treatment-day CTs, chair angles) make a chain: a model
per neighbouring pair, `t` running over `[0, n]`, piecewise, every later
state laid onto the first by the anchor. Each state can carry a **posture
parameter** (an angle, a day, a filling): the time reads in it beside `t`,
and the charts are drawn against it when every state has one.

**Smooth chains** (*Smooth chain*, on by default; `posture::chain`). Each
segment on its own turns every body at a constant rate, so where two
segments meet the body stops one motion and starts the other: a kink in
every curve and a jolt in the animation. With three or more states each
body instead rides a cubic curve in SE(3) through every state - the
cumulative cubic Hermite form of Kim, Kim and Shin, written for motions
acting on the left,

```
h(s) = exp(b3(s) w_c) exp(b2(s) w_b) exp(b1(s) w_a)
b1 = 1 - (1 - s)^3,  b2 = 3 s^2 - 2 s^3,  b3 = s^3
w_a = m_i / 3,  w_c = m_{i+1} / 3,  exp(w_b) = exp(-w_c) g exp(-w_a)
```

with `g` the segment's fitted motion and `m_k` the body's velocity at state
`k`, the mean of the two neighbouring segments' twists (Catmull-Rom; at the
chain's two ends the one segment's own). The curve lands exactly on `g`,
leaves and arrives with exactly those velocities, and is the stationary
screw itself when the neighbours agree. Between the bodies the field at `s`
is `s v_i + sum_b w_b (log h_b(s) - s xi_b)`: the segment's blend corrected
body by body with the blend's own weights (the four largest per node), zero
at `s = 0`, the segment's field at `s = 1`, the body's own curve wherever
its weight is one (its exact region keeps it rigid). So the path stays
invertible at every time, each body's velocity is continuous at a state
exactly, and the tissue between to first order. The residual stays
stationary per segment: it is what one pair of scans leaves unexplained,
with no velocity of its own to match.

A **statistical model** over many states (principal modes of the per-state
logarithms, a mean posture) is not built: one patient's imaging gives two
to a handful of states, too few for any mode, and modes across patients
need them all in one anatomy first, an atlas registration the station does
not make.

### What it cannot represent

- **Sliding.** Bowel against the abdominal wall, lung against the chest
  wall: a smooth field shears where anatomy slips.
- **Content that changes.** Gas, stool and bladder filling are not a
  deformation of the same material: the model moves the boundary, and the
  grey values inside are what *Cross-fade* mixes.
- **Anything outside both scans.** The model is the overlap of the fields
  of view; the Model fold gives its share and its SI extent.
- **The couch and the chair.** They are not anatomy; an image anchor sees
  them, a structure anchor does not.
- **Dose on the moved anatomy.** The window can draw the first state's dose
  carried along the path (below); that is the planned dose moved with the
  tissue, for looking at - not a dose recalculated on the new geometry, and
  no dose number is derived from it.

## The tables

Every number is in the anchor frame, in patient directions RL, AP, SI. The
time-dependent columns are sampled at ten times per segment and read at the
slider's time by linear interpolation: "from the first state to here".

**States** - label (the workspace), date and series, the patient position
the header gives, the posture parameter, the grid, the rigid transform that
brought the state into the anchor frame (its six numbers about the anchor's
centroid), and the body volume (the EXTERNAL / BODY structure when every
state has one, else the tissue above -300 HU inside the overlap) with its
change.

**Structures** - one row per structure every state has: the volume change
as contoured (each state's own grid), and - two scans of two positions
rarely cover the same length of the patient - whether a scan **cuts** the
structure (it reaches the edge of what that scan covers) and its volume
change inside the part every scan covers, the one to read for a cut
structure; the centroid displacement as
contoured and as the model carries it; the length of the centroid's path;
the rotation of the best rigid fit of the structure's own motion
(Procrustes of its surface points against where the model puts them); what
is left after that fit (Dice / HD95 of the rigidly moved first contour
against the last); the Jacobian inside (mean, 5th and 95th percentile); the
mean HU on the first and last state; the **model check** - the first
contour carried by the model to the last state against that state's own
(Dice / HD95) - and whether the structure took part in the fit. With
*WEPL* ticked (on by default, gantry 90, couch 0), also the
**water-equivalent path length** to the structure's centroid along a beam
at those IEC 61217 angles (read in the first state's patient position; head
first supine when the header gives none) and its change since the first
state: the image at the slider's time - the segment's two scans carried to
it and cross-faded - read from the centroid back towards the source until
the ray leaves the image, through an HU to relative stopping power table
(the generic curve of the synthetic CT's WET rows, `sct::eval::Hlut`), by
the midpoint rule in half-voxel steps. The beam is fixed in the anchor
frame: the column says how far the target's depth changes as the patient
moves while the beam stays where it was set up. At the states it reads the
scans; between them it reads the model's interpolation.

**Bones and joints** - per rigid body its screw over the whole path (angle,
axis, travel along it, the pivot nearest its centroid), its fit, and the
**largest stretch** inside it along the path, and the 95th percentile
over its points a lattice step or more from every other bone. A bone keeps
its shape away from its joints; within a lattice cell of a close neighbour
(the femoral head in its socket, the sacrum at the sacroiliac joint) the
field blends the two motions, a thin sliding layer the largest value finds
and a finer lattice thins. Per joint - a pair of bodies - the second's rotation
relative to the first, split about RL (flexion), AP (abduction, side
bending) and SI (axial rotation), about the first state's axes (the station
has no joint landmarks for anatomical frames). Joints are found by name
(each hip, neighbouring vertebrae, lumbosacral; two bodies and nothing else,
the two) and can be added below the table.

**Model** - mode, anchor, bodies, lattice and analysis grid, and the
checks: inverse consistency (`Phi_t^-1 o Phi_t`, max and 95th percentile at
half way and at the end, over the lattice nodes the path keeps on the
lattice), the smallest Jacobian over the path and the folded fraction (none
in the diffeomorphic mode), the mean model-check Dice of structures used in
the fit and of those left out, the overlap of the scans, and the provenance
of every piece.

A click on a name focuses the structure: its contour drawn thicker in the
views, the other surfaces of the 3D scene faint. 📈 on a row puts its curves
(displacement and path, rotation, Jacobian; a body's angle and stretch; a
joint's three angles) into the chart pane, against time or the posture
parameter; a click on the chart moves the slider there.

## The window

- **Top bar**: the states (workspaces holding an image) each with its
  posture parameter, ➕ for a chain; the anchor; the mode; *Rigid* (the
  bodies, ticked from the structures every state has); *Residual*; the
  lattice and the reach; *Smooth chain* (three states or more); *WEPL* and
  its gantry and couch angles; *▶ Build* (a background job with progress
  and Cancel) and *Open model*. Under it, the caveat, and for each state
  that lacks structures another has, which ones - with *Contour B with
  ...* when an automatic model of the registry makes some of them (the one
  that makes the most, the smaller on a tie,
  `workflow::posture::model_for_missing`). The button runs it on that
  workspace through the *Structure auto tools* module, whose *Results*
  fold then lists what it found; add them there and press *Build* again.
- **Centre**: the three planes, the 3D scene and the chart of the **model
  workspace**, drawn by the same `pane` function as the main rows, so the
  crosshair, zoom, window, structure visibility and the 3D camera are the
  ordinary ones.
- **Bottom bar**: ⏮ ▶/⏸ ⏭, the time slider with a tick at each state, the
  time (and the parameter), frames a second, the end mode (loop, bounce,
  once: the *Image playback* module's), what is shown, the overlay and its
  arrow spacing, *Jacobian wash*, *Dose (display only)* when the first
  state has a dose, *Live slices* or *Key frames* and the number of steps,
  the recorder (the selected view, ⏺ / ⏹).
- **Right**: the exports, *Save GIF* and the table.

**What is shown**: *A moving forward* (the segment's first state carried to
`t`), *B moving back*, *Cross-fade* (`(1 - s) A + s B`, exactly each scan at
its end), *Difference* (A minus B, carried), *Jacobian* (of the path from
the first state, times 1000, in a window of its own). *Overlay* draws the
displacement from the first state as arrows or a deformed grid
(`registration::dvf`), in the colours and at the arrow scale of the main
viewer's vector field; the millimetres beside it (*Arrow spacing*, 12 by
default, 2 to 60) are how far apart the arrows (the grid's lines) are,
this window's own. Arrows closer together are drawn shorter in
proportion (`dvf::arrow_length_factor`: the arrow scale holds at 12 mm),
so a dense field does not pile up and the lengths still compare. *Jacobian wash* colours the local volume change
since the first state over whatever is shown, read where the tissue now is
(`apply::jacobian_now`, `apply::wash_plane`): clear within about one per
cent of no change, blue compressed, red expanded (full colour at half or
twice the volume), magenta folded.

**Dose, for display only.** When the first state's workspace shows a dose,
*Dose (display only)* draws it carried along the path,
`D_t(y) = D(to_first(y))` (`apply::carry_dose`), with the dose display
module's colours and the plan's maximum. It is the planned dose moved with
the tissue - not a dose computed for the new anatomy, which would cross
different tissue on the way in - and nothing is derived from it: the model
workspace is the window's, so no DVH, dose readout or export reaches it.

**Live slices** (the default). While the slider moves or a run plays, each
step computes only the three planes on screen, on the full grid at the
step's exact time: on the inference GPU when there is one
(`posture::gpu`: the states' images uploaded once as 3-D textures when the
model is built, the frame's pull-back fields per step as storage buffers,
one compute shader that is `apply::Sampler::value` transcribed), else on
the CPU (`apply::render_planes`, the same pixels the full render gives
those planes, to the bit). The path at the step's time is computed on the
lattice in the same background job; the clock waits for each step, so a
run shows (and films) every one. The control's tooltip says where they run.

**Key frames** instead: whole images (a 512 x 512 x 274 CT is 144 MB a
frame) at 21 evenly spaced times by default, rendered on a working grid
that fits the *Image playback* budget and held in memory as a 4D group's
phases are. Either way, when the slider rests the exact frame at its time
is rendered on the full grid in the background. Structures, meshes, dose
and wash are carried when a frame is shown. Playback is one more
`PlayTarget` (`Model`), so the clock and the end modes are the module's.

**A window, not a module.** Every other tool that works on the images on
screen is a module of the right panel; this one stays a tool window, by
decision. It runs a workspace of its own whose planes, scene and chart it
draws in itself, with a time slider and a table wider than the panel; as a
module its views would have to take one of the main row's workspaces,
which is exactly what holding the model workspace off the main screen
avoids.

**The model workspace** is one of the four letters - the first free one
past A - kept off the main screen while the window holds it (it is not in
the data tree, *View ▸ Workspaces* names it, no load or copy is offered
into it). Closing the window lets it go; the model stays and comes back
with the window. With four workspaces in use, *Build* asks for one to be
freed.

### What it writes

| Button | File |
|---|---|
| 💾 CSV | the four tables, one file, the motion report's writer |
| 💾 Model | `.rdsposture` (JSON): the lattice, grids, anchor transforms, states, per segment its bodies (motion, fit, the lattice nodes they cover) and the residual and linear fields; the blend is rebuilt on reading. *Open model* reads it back onto the same states and recomputes the tables, nothing refitted |
| 📤 REG | the whole path, first state to last, as a DICOM Deformable Spatial Registration from the first state's frame into the last state's own coordinates |
| 📤 Image at t | the image at the slider's time as a derived CT series of the first state's study (on its grid, Image Type DERIVED, the time and the caveat in the series description and derivation) and the first state's structures carried to that time as an RT Structure Set on it, each ROI's description saying it was interpolated |
| ⏺ | the film of a pane as GIF or PNG frames, one cycle of the run, taken from the screen |
| Save GIF | the path as an animated GIF rendered from the model: all views (the three planes and the 3D scene) or any one of them, with the vector field if asked (below) |

**Save GIF** (`posture::film`). A menu. **View**: *All views* (the
default) - axial, sagittal and coronal side by side in cells of the panes'
shape and the 3D scene under them across the whole width, as the window
lays them out - or *Axial*, *Sagittal*, *Coronal* or *3D* alone. A click
in a pane (any button), or the selected-view list beside ⏺, picks that
pane here too; any view can be picked in the menu without one. *3D* is
greyed out when no structure of the first state has a surface. **Vector
field**: the displacement over the planes as the views draw it - arrows,
or the deformed grid while the overlay list says so (arrows when the
overlay is off) - at the bar's arrow spacing, in the views' colours and
proportions. Then *Width* (800 px by default; the height follows the
panes' shape) and *There and back* (the way back added without repeating
an end, for a smooth loop). *Save...* asks for the file and writes it on a
background job, with a progress bar and *Cancel* in place of the button (a
cancelled GIF leaves no file).

Nothing is taken off the screen: every frame is one exact time of the
model, rendered for the film. The run's times - the bar's number of
steps, evenly over the whole path, at the bar's frames a second - are each
rendered exactly as a live step is (`render_planes_with`: on the inference
GPU when there is one, else the CPU), on the full grid, at the slices the
views show (a key frame's coarser slice is put back on the full grid), with
what is shown and the window of the model workspace, so the same choices
write the same file. The view's zoom and pan are not taken: each cell holds
the whole slice, its millimetres square, resampled bilinearly. Over the
grey: the first state's structures ticked visible as one-pixel outlines in
their colours, where their tissue is at the time (a pixel is inside when
the point it came from on the first state, `to_first(y)`, is inside the
structure's mask there: the same pull the image goes through); with
*Jacobian wash* on, the wash; with *Vector field*, the field of that time
(`dvf::glyphs_on_plane` / `deformed_grid_on_plane` on the frame's forward
displacement, the colour by magnitude over that frame's largest, the
arrow's length, head and the disc for an arrow leaving the plane as the
views have them), one pixel wide and opaque, clipped to its cell. The dose
is not drawn.

**The 3D scene in the film** is drawn by the film, not filmed off the
screen. The 3D pane is a software renderer already: it turns, shades and
depth-sorts the meshes on the CPU and hands egui's painter the triangles.
The film does the same with the same functions (`mesh3d::turn`,
`headlight`, `scene_scale`, `depth_key`, `EXTERNAL_ALPHA`): the first
state's meshes carried to the frame's time (`apply::carry_meshes`, as the
window carries them), the 3D pane's camera (turn, zoom, pan, the sphere it
fits), its opacity and each structure's tick, colour and opacity, far
triangles first, each blended over what is behind it with its colours
interpolated, filled one pixel centre at a time. So a frame is exactly
its time - no waiting for a repaint, no screenshot - the CLI can film the
scene, and a film made twice is the same file. What the pane draws over
the surfaces (the orientation cube, the text, the beams) and egui's
anti-aliased edges are not in it. The other way - stepping the window
through the times and cutting each frame out of a screenshot of it - was
not taken: it would film exactly the screen, but only while the window is
on screen and drawn, a frame per repaint, and not in the CLI or the
headless tests; ⏺ remains for filming the screen as it is.

One palette for the whole film (the GIF's global table), no dithering, so
no frame is quantized on its own and nothing flickers: greys (192; 177
with the field, 131 with the scene, 113 with both), 32 structure colours
(the structures' own; past the 32nd distinct colour the nearest), the
wash as two hues at two strengths over 7 greys each, magenta for folded
tissue; with the field, the arrows' 16-step ramp of the views' colour map,
their pale yellow and the grid's colour; with the scene, 64 colours for
the shaded surfaces, a median cut of the colours the scene shows at up to
five times of the run (both ends among them), each pixel of the scene
then the nearest of those and the greys. A palette per frame would fit
each frame better and change the greys from one frame to the next, which
reads as flicker on a CT; the surfaces are what needs the colours, and
they change little along a path. `posture_cli --phantom --gif out.gif`
writes one without a window, the scene included (`--gif-view
all|axial|sagittal|coronal|3d`, `--gif-width`, `--gif-frames`, `--fps`,
`--bounce`, `--wash`, `--gif-field arrows|grid` with `--arrow-spacing`
and `--arrow-scale`; the scene's camera is a fresh pane's).

## Code

| Path | Contents |
|---|---|
| `crates/rds-domain/src/posture/mod.rs` | the method's page; `Mode`, `Anchor`, `StateInfo`, `Segment`, `PostureModel`, `Frame`, `coarsen` |
| `posture/se3.rs` | matrix log / exp of rigid transforms, screws, patient-axis angles, symmetric 3x3 eigen, stretch, polar factor |
| `posture/svf.rs` | `Lattice`, displacement `Field` (sample, compose, inverse, gradient, Jacobian), `Svf` (exp by scaling and squaring with exact regions), `log` |
| `posture/polyrigid.rs` | occupancy, swept regions, distances, weights, the blend, rigidity |
| `posture/fit.rs` | the three-layer fit of a segment, principal-axes seeds, `is_bone` |
| `posture/chain.rs` | bodies on cubic Hermite curves through a chain's states; the sparse blend weights; the corrected field at a time |
| `posture/apply.rs` | the path on points, meshes, structure sets, masks, images and doses; `Show`, `Sampler`; the three planes on screen; key frames; the working grid; the Jacobian wash |
| `posture/gpu.rs` | the three planes on the inference GPU (WGSL), and the CPU fallback |
| `posture/measure.rs` | the four tables, the checks, the WEPL along a beam, CSV |
| `posture/file.rs` | the `.rdsposture` file |
| `posture/film.rs` | *Save GIF*: the run's times, the layout, the palette, the outlines, the wash and the field over the planes, the 3D scene drawn as the pane draws it, written as an animated GIF |
| `crates/rds-domain/src/workflow/posture.rs` | `build`, `reopen`, `export_frame`, `export_registration`, preset joints, `lacking` and `model_for_missing` |
| `crates/rds-domain/src/gen_test_data.rs` | `articulated_pair` / `generate_articulated`: the articulated phantom |
| `src/app/posture_win.rs`, `posture_table.rs` | the window, its workspace, playback and recording; the table and chart |
| `crates/rds-domain/src/mcp/tools/posture.rs` | `build_posture_model` |
| `crates/rds-domain/src/workflow/graph/nodes/posture.rs` | the *Posture model* step |
| `examples/posture_cli.rs` | headless: folders (or `--phantom`, or `--simulate` on one CT) in, tables, model file, REG, key frames and the image at a time out; `--wepl GANTRY[,COUCH]`, `--no-smooth`, `--bench-slices N` (live steps timed on the CPU and the GPU, and how far apart they are), `--gif FILE` (the film; `--gif-view` all, axial, sagittal, coronal or 3d, `--gif-width`, `--gif-frames`, `--fps`, `--bounce`, `--wash`, `--gif-field` arrows or grid, `--arrow-spacing`, `--arrow-scale`) |

## Verification

- **Mathematics** (unit tests in `posture/`): `exp(log(T)) = T` to rounding
  for every angle up to and including half a turn; halves and thirds of a
  twist compose to the whole; one twist flows to its motion exactly at every
  `t`; the flow backwards undoes the flow forwards to lattice precision; the
  logarithm of an exponential comes back; a bar turning 50 degrees about a
  hinge moves rigidly at every `t` while its neighbour stays still, and
  nothing folds; a Hermite curve with equal end velocities is the screw
  itself, and lands on its end leaving and arriving as asked; a beam's
  direction follows IEC 61217 for supine, prone and a turned couch; a dose
  is drawn where its tissue went; the wash reads a uniform growth.
- **The articulated phantom** (`tests/posture.rs`): two bones on a hinge in
  soft tissue, an organ that grows, the second state turned 40 degrees and
  placed elsewhere in its scanner. The anchor's placement, the lower bone's
  screw (angle, axis, no travel, the pivot on the hinge), the joint angle,
  half the angle half way, bones that keep their shape and volume, no
  folding, inverse consistency, the organ's volume change and the model
  check are held to the analytic truth; the linear mode shares the end point
  and shrinks the bone; the model file reads back the same path and reopens
  with the same tables; the cross-fade lands on each scan; a chain of three
  states; a chain turning 10 then 30 degrees whose lower bone's speed is
  the same either side of the middle state on curves and jumps without
  them, the states where they were, the bone rigid on its curve, the
  curves kept by the model file; the live planes are the rendered volume's
  planes to the bit, for every *Show*; the WEPL to the organ is its depth
  in water on each state; the phantom through a DICOM round trip.
- **The GPU** (`#[ignore]`d, `RDS_GPU_TESTS=1 cargo test --release --test
  posture -- --ignored gpu`): the live planes on the GPU against the CPU's,
  for every *Show* at three times.
- **The interface** (`tests/ui.rs`): the window opens on the phantom, the
  model builds, the tables name the bodies and carry the WEPL columns, live
  slices are the default, the wash ticks on, the end reads the hinge's
  angle, playing moves the time, every widget has a name; a state lacking
  a structure (*liver*) is named and offered the model that makes it;
  *Save GIF* lists the five views (all views the default, a click in a
  pane choosing it, any picked from the list) and *Vector field*, and the
  arrow spacing is on the window's bar and in *Settings ▸ Viewer*.
- **The film** (`tests/posture.rs`, unit tests in `posture/film.rs`): the
  GIF decodes back to one frame per time at the size its layout gives -
  the three planes, the planes with the scene under them, a plane alone,
  the scene alone; outlines in colour; the arrows only where tissue has
  moved (none at the first state); the scene's surfaces moving with the
  hinge; the same file when written again. The scene's renderer draws a
  surface facing the light in its own colour, an edge-on one at the
  headlight's floor, the near surface over the far whatever their order,
  a body translucent over what is inside; the palette makes room for the
  arrows and the scene from the greys, the arrow ramp is the views' colour
  map, the median cut keeps the colours it can.
- **The step and the tool**: `tests/workflow_steps.rs`, `tests/mcp_tools.rs`,
  `tests/mcp_phi.rs` (the tool returns tables and handles only).
- **Known transforms on real anatomy** (`#[ignore]`d, needs `data-test/` or
  `RDS_POSTURE_DATA`): the simulator's rigid motion of the bundled 4DCT
  phase read back by the three largest structures (run: 3 degrees read back
  within 0.7).
- **A real pair** (`#[ignore]`d, `RDS_POSTURE_PAIR` and `RDS_POSTURE_ANCHOR`;
  no data or path in the repository): no ground truth, so what any model
  of a real pair must be - no fold, inverse consistency, every body fitting
  its contours and rigid away from its joints, the anchor still. Run on a
  supine planning CT and an upright CT of one patient from the station's
  archive, both with TotalSegmentator v3 (1.5 mm) structures, the sacrum as
  anchor: eight bodies (both hip bones, both femurs, the sacrum, L3 to L5;
  the lower ribs and L2, cut differently by the two scans, left out by the
  rules above), built in 8 s; hips about 7 to 8 degrees about RL, the
  lumbar spine 6 to 10 degrees, the lumbosacral junction 6; no fold
  (Jacobian minimum 0.40), inverse consistency at the end 0.07 mm at the
  95th percentile (3.2 mm at worst); stretch off the joints at the 95th
  percentile 0.03 for the hip bones, 0.10 for the femurs and L5, at most
  0.06 elsewhere - 0.02 and below for the femurs on a 3 mm lattice (14 s).
  That pair is what found the rules above: before them, ribs cut by the
  scan turned 50 to 145 degrees, the path folded (Jacobian minimum -8.6) and
  the inverse missed by 54 mm at worst.

## Defaults chosen

The plan left these open; the window starts from:

1. **Name**: *Posture model*.
2. **Anchor**: picked by the user; the default when none is picked is the
   whole image (rigid). A structure is the better anchor when there is one.
3. **Rigid bodies**: every bone the states share, editable under *Rigid*.
4. **Structures on both states**: matched by name. A state lacking some is
   offered the automatic model of the registry that makes the most of them
   (the window does not guess which model made the others: it matches the
   names against every model's classes).
5. **Time axis**: `t` always, the posture parameter beside it when every
   state has one (the charts then use it).
6. **Intermediate CTs** are exported, marked as interpolated in the series
   description, the derivation and every ROI's description.
7. **Dose** is carried along the path for display only; no dose quantity
   is recomputed or derived (see *Dose, for display only*).
8. **Where it lives**: a tool window with its own workspace, not a module
   (see *A window, not a module*).

## Not built

- a statistical model over many states (see *Smooth chains* for why);
- a residual layer with velocities continuous across a chain's states.

## References

- Arsigny V, Commowick O, Ayache N, Pennec X. A fast and log-Euclidean
  polyaffine framework for locally linear registration. *J Math Imaging
  Vis* 2009;33(2):222-238. doi:10.1007/s10851-008-0135-9. The blend.
- Arsigny V, Commowick O, Pennec X, Ayache N. A log-Euclidean framework for
  statistics on diffeomorphisms. *MICCAI 2006*, LNCS 4190, 924-931.
  Stationary velocity fields, scaling and squaring, the logarithm.
- Vercauteren T, Pennec X, Perchant A, Ayache N. Symmetric log-domain
  diffeomorphic registration: a demons-based approach. *MICCAI 2008*, LNCS
  5241, 754-761. The time-scaled, inverse-consistent path.
- Alexa M. Linear combination of transformations. *ACM Trans Graph*
  2002;21(3):380-387 (SIGGRAPH 2002). Mixing transforms through their
  logarithms.
- Murray RM, Li Z, Sastry SS. *A Mathematical Introduction to Robotic
  Manipulation*. CRC Press, 1994, ch. 2. Twists, screws, the exponential.
- Kim M-J, Kim M-S, Shin SY. A general construction scheme for unit
  quaternion curves with simple high order derivatives. *SIGGRAPH 1995*,
  369-376. doi:10.1145/218380.218486. The cumulative curves of a chain.
