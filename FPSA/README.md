# Function-Preserving Shape Augmentation (FPSA)

This directory implements the mesh-augmentation component described in [Function-Preserving Data Generation for Zero-Shot Real-to-Sim-to-Real Manipulation](https://arxiv.org/abs/2609.18293). FPSA changes nonfunctional object geometry while preserving annotated task-critical regions, mesh topology, and vertex correspondence. Those correspondences are then used to transfer task poses and convex collision proxies to every generated mesh.

For environment setup, simulation, policy training, and deployment, start with the [repository README](../README.md).

## Method overview

An FPSA asset is generated in four stages:

1. **Annotate constraints and handles.** Constraints identify task-critical regions that must remain valid; handles specify the vertices and directions used to drive a deformation.
2. **Deform the mesh.** Slippage-preserving reshaping is used for stretching edits that retain local surface invariants. ARAP is used for structural bending edits. Multiple edits may be composed as a chain.
3. **Transfer task geometry.** Because topology and vertex indices are preserved, the implementation transfers a grasp/tool frame from a corresponding local surface patch and transfers the source CoACD proxy with stored face/barycentric coordinates.
4. **Export an asset bundle.** Each sample contains the visual mesh, collision mesh, transferred task metadata, deformation metadata, and optional debug information.

The input mesh must be triangulated, vertex-manifold, edge-manifold with boundary edges allowed, and orientable. `ShapeAugmentor` rejects invalid inputs instead of silently modifying their topology.

## Entry points

| Entry point | Intended objects | Transferred task data |
|---|---|---|
| [`grasp_randomizer.py`](grasp_randomizer.py) | bracket, assembly tool, gear | initial grasp/TCP pose and CoACD collision proxy |
| [`tool_randomizer.py`](tool_randomizer.py) | wrench and similar held tools | tool frame, wrench-to-TCP transform, and CoACD collision proxy |
| [`FPSA_annotation_helper_gui_axis.py`](FPSA_annotation_helper_gui_axis.py) | custom input meshes | constrained vertices, handle vertices, and handle directions |

Shared configuration parsing, deterministic linspace sampling, chained deformation expansion, multiprocessing, solver overrides, and manifest writing live in [`src.py`](src.py). The mesh deformation and transfer implementation lives in [`FPSA.py`](FPSA.py).

## Included configurations

| Config | Runner | Default sample count | Output root |
|---|---|---:|---|
| [`configs/bracket/bracket_meta.yaml`](configs/bracket/bracket_meta.yaml) | `grasp_randomizer.py` | 210 | `data/objects/bracket/fpsa_aug_outputs` |
| [`configs/assembly/assembly_meta.yaml`](configs/assembly/assembly_meta.yaml) | `grasp_randomizer.py` | 255 | `data/objects/assembly/tool/fpsa_aug_outputs` |
| [`configs/gear/gear_meta.yaml`](configs/gear/gear_meta.yaml) | `grasp_randomizer.py` | 60 | `data/objects/gear_extraction/gear/fpsa_aug_outputs` |
| [`configs/wrench/wrench_meta.yaml`](configs/wrench/wrench_meta.yaml) | `tool_randomizer.py` | 105 | `data/objects/wrench/fpsa_aug_outputs` |

Counts reflect the current checked-in `sampler.labels` and `linspace_points_per_stage`. Confirm them with `--dry-run` after changing a configuration.

## Quick start

Run commands from the repository root:

```bash
conda activate PhyDomain
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Inspect a batch without running the deformation solver:

```bash
python FPSA/grasp_randomizer.py \
  --meta FPSA/configs/assembly/assembly_meta.yaml \
  --dry-run
```

Generate grasp-based assets:

```bash
# Pick-up bracket
python FPSA/grasp_randomizer.py \
  --meta FPSA/configs/bracket/bracket_meta.yaml

# Assembly tool
python FPSA/grasp_randomizer.py \
  --meta FPSA/configs/assembly/assembly_meta.yaml

# Gear
python FPSA/grasp_randomizer.py \
  --meta FPSA/configs/gear/gear_meta.yaml
```

Generate wrench tool-use assets:

```bash
python FPSA/tool_randomizer.py \
  --meta FPSA/configs/wrench/wrench_meta.yaml
```

Common command-line overrides are available without editing YAML:

```bash
python FPSA/grasp_randomizer.py \
  --meta FPSA/configs/assembly/assembly_meta.yaml \
  --labels chain_x_and_slippage \
  --workers 8 \
  --obj-path /path/to/tool.obj \
  --initial-grasp-path /path/to/tool_grasp.yaml \
  --output-root /path/to/output
```

`tool_randomizer.py` supports `--labels`, `--workers`, `--obj-path`, `--output-root`, and `--dry-run`; it does not take an initial-grasp file because the tool frame comes from the `tcp` section of its metadata.

## Sampling semantics

Every primitive is sampled deterministically with `numpy.linspace` over its configured `range`. Every chain expands to the full Cartesian product of its stages. A two-stage ARAP + slippage chain with `N` points per stage therefore produces `N²` samples: every value from the first stage is paired with every value from the second.

`linspace_points_per_stage` may be a single integer:

```yaml
sampler:
  labels: [chain_y_and_slippage]
  linspace_points_per_stage: 5
```

or a per-primitive mapping:

```yaml
sampler:
  labels: [x_stretch, y_stretch]
  linspace_points_per_stage:
    x_stretch: 120
    y_stretch: 80
```

If a chain contains `x_stretch` followed by `y_stretch`, the second example yields `120 × 80 = 9,600` samples for that chain. Selecting both primitives independently yields `120 + 80` samples instead.

## Configuration reference

A metadata file has four main sections:

```yaml
object:
  name: example
  obj_path: /path/to/example.obj
  initial_grasp_path: /path/to/example_grasp.yaml  # grasp runner only

sampler:
  max_workers: 8
  labels: [bend_then_stretch]
  linspace_points_per_stage: 5

output:
  root: /path/to/fpsa_aug_outputs
  overwrite: false
  write_coacd: true
  save_debug: true

deformations:
  - label: bend_then_stretch
    type: chain
    chain: [bend, stretch]

  - label: bend
    method: ARAP
    constrained_ids: [10, 11, 12, 20, 21]
    reshaped_ids: [20, 21]
    reshaped_vector: [0.0, 1.0, 0.0]
    range: [-0.02, 0.02]
    max_iters: 200

  - label: stretch
    method: slippage
    constrained_ids: [10, 11, 12, 20, 21]
    reshaped_ids: [20, 21]
    reshaped_vector: [1.0, 0.0, 0.0]
    range: [-0.01, 0.03]
    max_iters: 120
```

Important fields:

- `constrained_ids` contains fixed vertices and handle vertices.
- `reshaped_ids` identifies the handles whose targets are displaced.
- `reshaped_vector` may be one 3-vector shared by all handles or one vector per handle. Vectors are normalized by the sampler.
- `range` is expressed in the mesh's length unit and may be shared or specified per handle.
- `method` is case-insensitive and supports `slippage` and `ARAP`.
- `max_iters` overrides the method default for a primitive.
- `solver.vertex_solve_params` forwards named overrides to the slippage solver. The gear config, for example, raises `bc_weight` to `1.0e7`; other included configs use `ShapeAugmentor` defaults.
- `output.overwrite: false` is recommended while developing a new config so an existing sample is not replaced accidentally.

## Annotating a custom mesh

Launch the Open3D annotation helper:

```bash
python FPSA/FPSA_annotation_helper_gui_axis.py \
  --mesh /path/to/object.obj \
  --label object_x_stretch \
  --out /tmp/object_x_stretch.yaml \
  --single-record
```

The main keyboard controls are:

| Key | Action |
|---|---|
| `C` | select constrained vertices |
| `H` | select handle/reshaped vertices |
| `D` | select a handle for direction editing |
| `X`, `Y`, `Z` | assign the selected handle to the positive axis |
| `1`, `2`, `3` | assign the selected handle to the negative X, Y, or Z axis |
| `A` | apply the selected direction to every handle |
| `Backspace` | delete the selected annotation |
| `S` | save |
| `Q` or `Esc` | close |

The helper outputs the deformation record, but you should still choose the displacement `range`, method, solver settings, sampling density, and task-pose metadata based on the object's geometry and task tolerances. Avoid mesh processing after annotation: changing vertex order or topology invalidates the saved vertex IDs and transfer correspondence.

For grasp-based objects, prepare an initial grasp YAML containing a 4 × 4 `T_mesh_hand` or `T_mesh_hand_tcp` transform and the gripper opening. An existing file such as [`../data/objects/bracket/bracket_grasp.yaml`](../data/objects/bracket/bracket_grasp.yaml) can be used as a schema example.

## Outputs

Each generated sample is stored in its own directory. A grasp-based bundle contains:

```text
<sample_name>/
├── <sample_name>.obj
├── <sample_name>_coacd.obj
├── <sample_name>_grasp.yaml
├── <sample_name>_sample.yaml
└── <sample_name>_debug.yaml
```

A tool-use bundle replaces the grasp file with `<sample_name>_wrench_to_tcp.yaml`. Material and texture files are copied when present. The batch output root also contains:

- `manifest.csv` for quick inspection and filtering;
- `manifest.jsonl` for structured downstream processing; and
- one result row per requested sample, including failures and tracebacks.

The batch process completes all jobs it can and records failed samples rather than discarding the entire run. Treat the manifest's `ok` field as the authoritative completion check.

## Troubleshooting

### The input mesh is rejected as non-manifold

Repair and triangulate the mesh before annotation, then regenerate all vertex annotations. Topology-changing repairs performed after annotation invalidate the IDs in the config.

### A batch consumes too much memory

Reduce `sampler.max_workers` in YAML or pass `--workers 1`. Every worker loads the mesh, initializes the deformation solver, and creates/updates collision geometry.

### A chain produces more samples than expected

Chains use a Cartesian product, not a zip. Run with `--dry-run` and inspect `grid_counts`. Reduce `linspace_points_per_stage`, select fewer labels, or configure smaller per-stage counts.

### The transferred grasp or tool frame is incorrect

Confirm that the source transform is expressed in the source mesh frame, the mesh was not reindexed, and the local anchor region remains within the intended task interface. Inspect the generated debug YAML and visualize the transferred frame before using the asset in simulation.

### The slippage module cannot be imported

Install the bundled CPython 3.9/Linux wheel or rebuild the binding as described in the [root installation guide](../README.md#install-the-slippage-preserving-reshaping-binding). The bundled wheel is not portable across Python ABIs or operating systems.

## References

The deformation implementation builds on:

- C. Araújo, N. Vining, S. Burla, M. Ruivo de Oliveira, E. Rosales, and A. Sheffer, “Slippage-Preserving Reshaping of Human-Made 3D Content,” *ACM Transactions on Graphics*, 2023.
- O. Sorkine and M. Alexa, “As-Rigid-As-Possible Surface Modeling,” *Symposium on Geometry Processing*, 2007.
- X. Wei, M. Liu, Z. Ling, and H. Su, “Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search,” *ACM Transactions on Graphics*, 2022.
