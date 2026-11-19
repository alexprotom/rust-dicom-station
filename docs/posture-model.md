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
(`build_posture_model`, and for templates `build_posture_template`,
`list_posture_templates` and `predict_posture`, [mcp.md](mcp.md)) and a
workflow step (*Posture
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
| *Auto* (the default) | a structure chosen when the model is built (`workflow::posture::auto_anchor`), then as below: among the bones of at least 10 cm3 that every scan sees **whole** - on no state reaching the edge of the scan, their contoured volumes within 10 % - the sacrum; else the middle one, by height, of the vertebrae; else the largest; none, the whole image. The model records what was chosen ("anchor chosen: ..." in its provenance, and the anchor name in the tables and the file) |
| a structure on every state (the sacrum, a vertebra, the target) | the rigid surface fit of that structure (`registration::shape`, distance maps, no grey values), with the principal-axes start below |
| *Whole image* | a rigid intensity registration (elastix rigid) |
| *Scanner frame* | nothing: for scans that share a frame of reference |

Why *Auto* takes those: the frame every number is read in should be a
bone the anatomy pivots about and the scans agree on. A bone one scan cut
is fitted to its whole self and reads a turn that never happened - on a
supine and upright pair framed on the chest, an upright scan ending at S1
made a sacrum anchor read 47 degrees and the whole spine bend 49 (2026-11,
three supine and upright pairs); the middle vertebra both scans see keeps
every other level's motion small. A structure anchor picked by hand that a
scan does not see whole is warned about.

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
   head's nodes stay the femur's and the socket's the pelvis's. A body's
   swept region beyond its own nodes gives back to the blend every node
   nearer another body's region than the two bodies' velocities there
   differ, by more than a lattice step (`polyrigid::separate`): a shoulder
   blade or an arm sweeping up
   to ribs that stay still would otherwise make the velocity jump by tens
   of millimetres from one node to the next, and the discrete flow folded
   the tissue between them (the armpit, the lateral chest wall) on ten of
   twelve thoracic supine and upright pairs; bodies that move alike keep
   their regions, and a body's own nodes are never given back. The
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
   Two scans of two positions rarely cover the same part of the patient,
   and what lies outside the end scan is its lowest value: left to itself
   the registration drags the tissue at one scan's edge onto the empty
   space beyond the other's and folds it. So it is fitted only where both
   scans have tissue (`fit::residual_region`: the start voxels whose
   blended position the end scan covers, inside the EXTERNAL structure
   grown by 15 mm, the edge shrunk by a lattice step), faded to nothing
   outside that region over a lattice step, and halved until its flow
   keeps the Jacobian at 0.2 or more (the provenance says by how much). Its
   defaults are mutual information, a 30 mm lattice and a bending penalty
   of 1 (`workflow::posture::default_residual`): on the supine and upright
   pairs mean squares folded wherever the content changed (a bladder with
   contrast on one scan only, gas) and mutual information with the stiffer
   lattice moved the organs the furthest without folding.

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

### The patient only

*Patient only* (on by default, `PostureRequest::patient_only`): each
state's image with what lies outside its EXTERNAL structure (the type, or
the names BODY, EXTERNAL, SKIN, OUTLINE), grown by 3 mm, set to air - -1000
HU, or the image's lowest value when that is higher (an MR)
(`workflow::posture::patient_only`). The couch of a lying scan and the
chair of a seated one are not anatomy: carried along the path they bend
with the tissue around them, and the image of the other position should
hold neither. The model is fitted on these images (an image anchor no
longer sees the couch) and draws them; a state without an EXTERNAL
structure is kept as scanned, and the provenance says which.

### Carrying another image

Another image taken in the first state's position - an MR, a PET, a
contrast series - is put on the first state's grid
(`workflow::posture::carry_image`: in the scanner's frame, by a rigid
registration by mutual information, or through a transform the station
holds) and given to the renderer in place of the first state's own image,
so *A moving forward* carries it along the path: a supine MR with a supine
and an upright CT gives an upright MR. Only the geometry is carried; what
the other position does to the grey values is not modelled. In this
version the CLI does it (`--carry`, frames and GIF); the window and the MCP
tool are to follow, with a derived-series writer of the image's own
modality.

### How far to trust it

`measure::Tables::verdict` reads the model's checks into a word - *good*,
*check*, *poor* - and its findings, the worst first: a path that folds
(poor); the way back missing the way there by more than 1 mm at the 95th
percentile; an anchor that fits a state only to more than 2.5 mm RMS (cut
by a scan?); bones fitting looser than 2.5 mm; organs whose model check is
under a Dice of 0.6 (they moved on their own - gravity, filling - which the
bones do not carry); scans overlapping on under half of the first. The
window heads its side with it, the MCP tool and the CLI print it.

### What it cannot represent

- **Sliding.** Bowel against the abdominal wall, lung against the chest
  wall, a shoulder blade over the ribs: a smooth field shears where anatomy
  slips. The blend gives the space between two bones that move apart room
  to change smoothly (above), but a shoulder blade within a node or two of
  a rib can still fold a node or two; `posture_cli --folds` says where.
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

## Templates: a cohort, and one scan predicted

A model of one patient needs both scans. A **posture template**
(`posture::template`, `.rdsposture-template`) is what a cohort of such
models says about the change of posture itself, so that a patient with
only the first scan can be given the second:

- **A sample per model** (`PostureRequest::sample`, `posture_cli --out` on
  a pair writes `sample.json`, as does `build_posture_model` with
  `sample`): per bone its centroid, seven points on its
  principal axes and its motion; per pair of bones that touch at the first
  state (contours within 4 mm) the **joint centre**, the middle of the two
  contact patches; per organ (not a bone, not the outline, not cut, 5 cm3
  or more) its centroid at the start, as contoured at the end, and as the
  bones alone carry it, with the bones' local gradient there. No name or
  identifier: the label is what the caller gives.
- **Relations.** For bones `a` and `b`, `M = T_a^-1 o T_b` is `b`'s motion
  relative to `a` in the first state's coordinates - the same whatever bone
  anchored that patient's model, so a cohort anchored on the sacrum, T4 or
  L1 speaks of one joint. Kept as a rotation vector and the centre's own
  displacement, for joints about the contact centre and for every pair
  directly about the middle of the two centroids; the template's estimate
  is the componentwise median over the patients that have it, its spread
  the median distance from it (one misfitted rib moves neither). A
  relation is used when three patients or more have it.
- **Organs.** After the bones, an organ that sags when the patient sits up
  is elsewhere than the bones carry it; the offset is kept pulled back
  through the bones' gradient (`G^-1 delta`, the displacement to apply at
  the first state, before the bones), its median used. Bladder, bowel,
  stomach, rectum, oesophagus and lungs change content or breath between
  scans and give no offset.
- **Prediction** (`workflow::posture::predict`): the scan's bones, the
  joints found the same way, a root (`auto_anchor` on the one scan), and
  every bone walked out from it along the relation whose accumulated
  spread at its centroid is least (Dijkstra: a rib straight from a vertebra
  near it rather than down the whole spine); the organs with offsets moved
  as wholes, faded at their edges, as a residual velocity before the bones
  (halved until it keeps the Jacobian at 0.2); the rest blended as for a
  fit. The outcome is a model like any - it plays, measures, exports (the
  image at `t = 1` is the predicted second scan) - whose second state is
  labelled *predicted*, whose tables check nothing against a scan, and
  whose verdict says so.
- **Leave one out** (`template::leave_one_out`, `posture_cli --loo`,
  `build_posture_template`; `template::loo_groups` and `loo_organs` sum it
  up by group of bones and by organ): each
  patient predicted from the others, its bones against its own fitted
  motions relative to the root (mean distance at the seven points), its
  organs against their contoured centroids, beside the error of predicting
  no motion at all.

On the `U2S_ModelTest` cohort (15 supine and upright CT pairs, TotalSegmentator
v3 structures: 12 thorax and neck, 1 pelvis, 1 thorax-abdomen-pelvis, 1
thorax-abdomen; 2026-11-19), leave one out, mean / median / 90th percentile
over every bone of every patient:

| Bones | n | template, mm | no motion, mm |
|---|---|---|---|
| thoracic vertebrae | 122 | 3.0 / 1.3 / 7.4 | 4.1 / 2.1 / 10.4 |
| ribs | 195 | 6.4 / 4.8 / 12.9 | 6.9 / 5.0 / 12.8 |
| cervical vertebrae | 41 | 6.1 / 4.4 / 12.3 | 6.1 / 4.7 / 12.8 |
| shoulder girdle, arms, sternum | 65 | 17.2 / 16.4 / 30.2 | 25.9 / 23.3 / 48.3 |

The thoracic spine and the ribs follow a common pattern; the shoulders move
the most and the template takes a third off their error; the neck does not
improve (the head's position on its rest differs from patient to patient);
the lumbar spine and the pelvis have too few patients (one sacrum) to say
anything yet. The organs' offsets take the heart from 17.6 to 8.3 mm, the
superior vena cava from 21.9 to 12.4 mm and the subclavian arteries from
8 - 10 to 5.2 - 5.7 mm (centroids, mean, from where the patient's own
fitted bones carry them), and leave the thyroid where no motion at all
would (9.6 mm). On the same cohort the per-patient models no longer fold
on ten of the twelve thoracic pairs but on three, each at a few lattice
nodes beside a clavicle or a shoulder blade on the first ribs.

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

One line makes a model, one card says how far to trust it, and the rest
is a click away in a menu.

- **Top bar**, one line: the states (workspaces holding an image), → between
  them, ➕ for a chain; *Anchor* (*Auto* until one is picked, reading
  *Auto: Sacrum* once built); *▶ Build* (a background job with progress
  and Cancel); *⚙ Options*: the mode, *Rigid* (the bodies, ticked from the
  structures every state has), *Residual*, the lattice and the reach,
  *Smooth chain* (three states or more), *Patient only*, each state's
  posture parameter, *WEPL* and its gantry and couch angles, and *Open
  model...*. Under the bar, only when there is one: for each state that
  lacks structures another has, which ones - with *Contour B with ...*
  when an automatic model of the registry makes some of them (the one that
  makes the most, the smaller on a tie,
  `workflow::posture::model_for_missing`). The button runs it on that
  workspace through the *Structure auto tools* module, whose *Results*
  fold then lists what it found; add them there and press *Build* again.
  The second state's list ends with the templates in the data folder's
  `posture-templates/` (*Template: name*) and *Open a template...*, which
  copies the file there; with one picked, the second state is predicted
  from the first, *Root: auto* stands in for the anchor, and *▶ Predict*
  replaces *▶ Build*.
- **Centre**: the three planes, the 3D scene and the chart of the **model
  workspace**, drawn by the same `pane` function as the main rows, so the
  crosshair, zoom, window, structure visibility and the 3D camera are the
  ordinary ones.
- **Bottom bar**: ⏮ ▶/⏸ ⏭, the time slider with a tick at each state, the
  time (and the parameter), what is shown, *Display* - the overlay and its
  arrow spacing, *Jacobian wash*, *Dose (display only)* when the first
  state has a dose, *Live slices* or *Key frames* and the number of steps,
  frames a second, the end mode (loop, bounce, once: the *Image playback*
  module's) - and *Save GIF* (below). Inside the
  *Display* and *Options* menus every choice is one click (a row of
  buttons, not a drop-down: a drop-down in a menu is a popup in a popup,
  and the outer one takes the click), and every number - arrow spacing,
  steps, frames a second, lattice, reach - has ◀ ▶ arrows a step either
  side: 1 below 15, 5 from 15 to 50, 10 above, landing on a multiple of the
  step.
- **Right**: the verdict (a coloured dot, *Good model*, *Model to check* or
  *Poor model*, and its findings), the build's warnings folded as
  *Notes*, the caveat, *📤 Export* (the tables as CSV, the model file, the
  DICOM REG, the image at t), and the tables.

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
| Save GIF (on the bottom bar) | the path as an animated GIF rendered from the model: all views (the three planes and the 3D scene) or any one of them, with the vector field if asked (below) |

**Save GIF** (`posture::film`). A menu. **View**: *All views* (the
default) - axial, sagittal and coronal side by side in cells of the panes'
shape and the 3D scene under them across the whole width, as the window
lays them out - or *Axial*, *Sagittal*, *Coronal* or *3D* alone. The
GIF takes what is picked here and nothing else: a click in a pane does
not change it (it once did, and a GIF of one pane came out of a click
meant for something else). *3D* is greyed out when no structure of the
first state has a surface. **Vector
field**: the displacement over the planes as the views draw it - arrows,
or the deformed grid while *Display*'s field says so (arrows when the
field is off) - at *Display*'s arrow spacing, in the views' colours and
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
headless tests. The window had such a recorder too (⏺, one pane cut out
of the screen), and it went: *Save GIF* does the same better.

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
| `posture/template.rs` | posture templates: samples, joints and their centres, the relations' medians and spreads, organ offsets, the bones' motions walked out from a root, leave one out, the `.rdsposture-template` file |
| `crates/rds-domain/src/workflow/posture.rs` | `build` (and its sample), `reopen`, `predict` (one scan and a template), `export_frame`, `export_registration`, preset joints, `lacking` and `model_for_missing` |
| `crates/rds-domain/src/gen_test_data.rs` | `articulated_pair` / `generate_articulated`: the articulated phantom |
| `src/app/posture_win.rs`, `posture_table.rs` | the window, its workspace, playback and *Save GIF*; the table and chart |
| `crates/rds-domain/src/mcp/tools/posture.rs` | `build_posture_model` (with `sample`), `build_posture_template`, `list_posture_templates`, `predict_posture` |
| `crates/rds-domain/src/workflow/graph/nodes/posture.rs` | the *Posture model* step |
| `examples/posture_cli.rs` | headless: folders (or `--phantom`, or `--simulate` on one CT) in, tables, the verdict, model file, REG, key frames and the image at a time out; `--anchor auto` by default, `--whole-scan`, `--residual-grid` / `--residual-reg` / `--residual-metric`, `--carry DIR` with `--carry-align rigid\|scanner` (another image of the first position carried), `--wepl GANTRY[,COUCH]`, `--no-smooth`, `--bench-slices N` (live steps timed on the CPU and the GPU, and how far apart they are), `--gif FILE` (the film; `--gif-view` all, axial, sagittal, coronal or 3d, `--gif-width`, `--gif-frames`, `--fps`, `--bounce`, `--wash`, `--gif-field` arrows or grid, `--arrow-spacing`, `--arrow-scale`); `--folds` (where the end folds, by the two bodies nearest); a cohort: `--out` on a pair writes `sample.json` (`--label`), `--make-template OUT SAMPLE...`, `--loo TEMPLATE`, `--template FILE` with one state (`--no-organs`) |

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
  *Save GIF* is on the bottom bar with no recorder beside it, lists the
  five views (all views the default, any picked from the list, a click in
  a pane changing nothing) and *Vector field*, and the arrow spacing is
  under the window's *Display* and in *Settings ▸ Viewer*.
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
- **Iteration of 2026-11** (`tests/posture.rs`): *Auto* holds one of the
  phantom's bones and the joint reads the hinge; the residual neither folds
  nor stretches the bones and leaves the organ no worse; *patient only*
  clears a couch under the phantom and keeps the patient; an image of the
  first position carried in the scanner's frame renders as the first
  state's own. The rule of *Auto* (sacrum, middle vertebra, largest bone,
  none) is a unit test of `workflow::posture`.
- **Templates** (unit tests in `posture/template.rs`, `tests/posture.rs`):
  a joint reads the same whatever bone anchored the model; the median
  patient is predicted exactly and one misfit 60 degrees off does not move
  the median; an organ's offset is kept before the bones and a bladder
  gives none; four phantoms hinged 30, 40, 40 and 50 degrees (their bones
  24 mm apart, so the direct relation, turning about the middle of the
  centroids on the hinge, carries it) leave a template that reads back with
  the median 40 degrees, a leave-one-out within 1.5 mm, and one scan alone
  given a second state whose bones turn 40 degrees, its image the first
  moved, its verdict *predicted*, nothing folded.
- **The step and the tool**: `tests/workflow_steps.rs`, `tests/mcp_tools.rs`
  (the summary and the full answer, the verdict, the image at `export_t`),
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
2. **Anchor**: *Auto* (above) until the user picks one; the CLI and the MCP
   tool default to it too.
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

- a statistical model over many states (see *Smooth chains* for why): a
  template keeps medians per joint and per organ, not a model of the
  deformation's modes, and reads rotations in the first state's patient
  axes rather than in anatomical bone frames;
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
