"""Shared configuration, sampling, and batch helpers for FPSA randomizers."""

from __future__ import annotations

import copy
import csv
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - reported when a YAML file is loaded
    yaml = None


DEFAULT_METHOD_MAX_ITERS = {"arap": 200, "slippage": 120}


@dataclass(frozen=True)
class PrimitiveSpec:
    label: str
    method: str
    constrained_ids: list[int]
    reshaped_ids: list[int]
    reshaped_vector: Any
    range: Any
    max_iters: Optional[int] = None


@dataclass(frozen=True)
class ChainSpec:
    label: str
    steps: list[str]


Spec = Union[PrimitiveSpec, ChainSpec]
Worker = Callable[[dict[str, Any]], dict[str, Any]]


def load_meta(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs")
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config suffix: {path.suffix}")
    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping")
    return data


def to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    return repr(value)


def dump_data(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_builtin(data)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to write YAML outputs")
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned or "sample"


def matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    return matrix


def unique_ints(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _chain_steps(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        steps = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        steps = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(f"Chain '{label}' must be a list or comma-separated string")
    if not steps:
        raise ValueError(f"Chain '{label}' has no stages")
    return steps


def parse_specs(meta: Mapping[str, Any]) -> dict[str, Spec]:
    entries = meta.get("deformations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("deformations must be a non-empty list")

    specs: dict[str, Spec] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "label" not in entry:
            raise ValueError("Every deformation requires a label")
        label = str(entry["label"])
        if label in specs:
            raise ValueError(f"Duplicate deformation label: {label}")
        if str(entry.get("type", "")).lower() == "chain" or "chain" in entry:
            specs[label] = ChainSpec(label, _chain_steps(entry.get("chain"), label))
            continue

        required = ("constrained_ids", "reshaped_ids", "reshaped_vector", "range")
        missing = [key for key in required if key not in entry]
        if missing:
            raise KeyError(f"Primitive '{label}' is missing {missing}")
        specs[label] = PrimitiveSpec(
            label=label,
            method=str(entry.get("method", "slippage")),
            constrained_ids=[int(value) for value in entry["constrained_ids"]],
            reshaped_ids=[int(value) for value in entry["reshaped_ids"]],
            reshaped_vector=entry["reshaped_vector"],
            range=entry["range"],
            max_iters=(
                int(entry["max_iters"])
                if entry.get("max_iters") is not None
                else None
            ),
        )

    for label in specs:
        resolve_primitives(label, specs)
    return specs


def resolve_primitives(
    label: str,
    specs: Mapping[str, Spec],
    stack: tuple[str, ...] = (),
) -> list[str]:
    if label not in specs:
        raise KeyError(f"Unknown deformation '{label}'; available: {list(specs)}")
    if label in stack:
        raise ValueError(f"Cyclic chain: {' -> '.join((*stack, label))}")
    spec = specs[label]
    if isinstance(spec, PrimitiveSpec):
        return [label]
    result: list[str] = []
    for child in spec.steps:
        result.extend(resolve_primitives(child, specs, (*stack, label)))
    return result


def _direction(value: Any, index: int, count: int) -> np.ndarray:
    axes = {
        "x": (1.0, 0.0, 0.0),
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    if isinstance(value, str):
        try:
            vector = np.asarray(axes[value.lower().strip()], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"Unknown direction: {value}") from exc
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (3,):
            vector = array
        elif array.shape == (1, 3):
            vector = array[0]
        elif array.shape == (count, 3):
            vector = array[index]
        else:
            raise ValueError(
                f"reshaped_vector must be (3,), (1, 3), or ({count}, 3); got {array.shape}"
            )
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("reshaped_vector cannot be zero")
    return vector / norm


def _handle_range(value: Any, index: int, count: int) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (2,):
        selected = array
    elif array.shape == (1, 2):
        selected = array[0]
    elif array.shape == (count, 2):
        selected = array[index]
    else:
        raise ValueError(
            f"range must be [low, high], [[low, high]], or ({count}, 2); got {array.shape}"
        )
    low, high = float(selected[0]), float(selected[1])
    if low > high:
        raise ValueError(f"Invalid range [{low}, {high}]")
    return low, high


def _linspace_value(bounds: tuple[float, float], index: int, count: int) -> float:
    if count <= 0:
        raise ValueError("linspace point count must be positive")
    low, high = bounds
    if count == 1:
        return 0.5 * (low + high)
    return float(np.linspace(low, high, count, dtype=np.float64)[index])


def sample_primitive(spec: PrimitiveSpec, index: int, count: int) -> dict[str, Any]:
    handle_count = len(spec.reshaped_ids)
    if handle_count == 0:
        raise ValueError(f"Primitive '{spec.label}' has no reshaped_ids")
    magnitudes = [
        _linspace_value(_handle_range(spec.range, i, handle_count), index, count)
        for i in range(handle_count)
    ]
    displacements = [
        (magnitudes[i] * _direction(spec.reshaped_vector, i, handle_count)).tolist()
        for i in range(handle_count)
    ]
    return {
        "label": spec.label,
        "method": spec.method,
        "constraint_ids": unique_ints((*spec.constrained_ids, *spec.reshaped_ids)),
        "reshaped_ids": list(spec.reshaped_ids),
        "magnitudes": magnitudes,
        "displacements": displacements,
        "max_iters": spec.max_iters,
        "linspace_index": index,
        "linspace_count": count,
    }


def _points_for_stage(sampler: Mapping[str, Any], label: str) -> int:
    configured = sampler.get("linspace_points_per_stage")
    if configured is None:
        raise KeyError("sampler.linspace_points_per_stage is required")
    if isinstance(configured, Mapping):
        if label not in configured and "default" not in configured:
            raise KeyError(f"No linspace point count configured for '{label}'")
        configured = configured.get(label, configured.get("default"))
    points = int(configured)
    if points <= 0:
        raise ValueError(f"linspace point count for '{label}' must be positive")
    return points


def make_jobs(
    meta: dict[str, Any],
    labels: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Create the complete per-chain Cartesian product of linspace samples."""
    specs = parse_specs(meta)
    sampler = meta.get("sampler", {})
    selected = list(labels) if labels else list(sampler.get("labels", []))
    if not selected:
        selected = [label for label, spec in specs.items() if isinstance(spec, ChainSpec)]
        if not selected:
            selected = list(specs)
    unknown = [label for label in selected if label not in specs]
    if unknown:
        raise KeyError(f"Unknown selected labels: {unknown}")

    jobs: list[dict[str, Any]] = []
    for top_label in selected:
        primitive_labels = resolve_primitives(top_label, specs)
        counts = [_points_for_stage(sampler, label) for label in primitive_labels]
        for indices in product(*(range(count) for count in counts)):
            stages: list[dict[str, Any]] = []
            for position, (label, index, count) in enumerate(
                zip(primitive_labels, indices, counts), start=1
            ):
                spec = specs[label]
                if not isinstance(spec, PrimitiveSpec):
                    raise AssertionError("Resolved chain stage is not primitive")
                stage = sample_primitive(spec, int(index), int(count))
                stage["stage_position"] = position
                stages.append(stage)
            sample_id = len(jobs)
            jobs.append(
                {
                    "sample_id": sample_id,
                    "top_label": top_label,
                    "sampled_stages": stages,
                    "grid_indices": [int(value) for value in indices],
                    "grid_counts": counts,
                    "meta": meta,
                }
            )
    return jobs


def method_max_iters(stage: Mapping[str, Any], solver: Mapping[str, Any]) -> int:
    if stage.get("max_iters") is not None:
        return int(stage["max_iters"])
    method = str(stage["method"]).lower()
    configured = solver.get("method_max_iters", {})
    for key, value in configured.items():
        if str(key).lower() == method:
            return int(value)
    if "max_iters" in solver:
        return int(solver["max_iters"])
    return DEFAULT_METHOD_MAX_ITERS.get(method, 120)


def apply_vertex_solve_params(augmentor: Any, solver: Mapping[str, Any]) -> None:
    """Apply explicitly configured slippage solver weights to one augmentor."""
    overrides = solver.get("vertex_solve_params", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("solver.vertex_solve_params must be a mapping")
    params = augmentor.vertex_solve_params
    for name, value in overrides.items():
        if not hasattr(params, name):
            raise KeyError(f"Unknown VertexSolveParams field: {name}")
        setattr(params, name, value)


def apply_stages(augmentor: Any, job: Mapping[str, Any], sample_name: str) -> list[dict[str, Any]]:
    solver = job["meta"].get("solver", {})
    apply_vertex_solve_params(augmentor, solver)
    records: list[dict[str, Any]] = []
    for stage in job["sampled_stages"]:
        stage_index = int(stage["stage_position"])
        input_name = (
            f"{sample_name}_step{stage_index:02d}_"
            f"{safe_name(stage['method'])}_{safe_name(stage['label'])}"
        )
        max_iters = method_max_iters(stage, solver)
        vertices = augmentor.displacement_reshape(
            constraint_ids=stage["constraint_ids"],
            displace_idxs=stage["reshaped_ids"],
            displacements=np.asarray(stage["displacements"], dtype=np.float64),
            max_iters=max_iters,
            handle_error_distrib_enabled=False,
            input_name=input_name,
            reshape_method=str(stage["method"]),
        )
        records.append(
            {
                **stage,
                "input_name": input_name,
                "used_max_iters": max_iters,
                "result_shape": list(np.asarray(vertices).shape),
            }
        )
    return records


def object_path(meta: Mapping[str, Any]) -> str:
    try:
        return str(meta["object"]["obj_path"])
    except KeyError as exc:
        raise KeyError("object.obj_path is required") from exc


def output_paths(meta: Mapping[str, Any], sample_name: str) -> dict[str, Path]:
    output = meta.get("output", {})
    root = Path(output.get("root", "fpsa_outputs"))
    sample_dir = root / sample_name if output.get("layout", "per_shape_dir") == "per_shape_dir" else root
    return {
        "sample_dir": sample_dir,
        "obj": sample_dir / f"{sample_name}.obj",
        "grasp": sample_dir / f"{sample_name}_grasp.yaml",
        "wrench_to_tcp": sample_dir / f"{sample_name}_wrench_to_tcp.yaml",
        "meta": sample_dir / f"{sample_name}_sample.yaml",
        "debug": sample_dir / f"{sample_name}_debug.yaml",
    }


def prepare_output(paths: Mapping[str, Path], protected: Sequence[str], overwrite: bool) -> None:
    paths["sample_dir"].mkdir(parents=True, exist_ok=True)
    existing = [paths[key] for key in protected if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs exist and overwrite=false: {existing}")


def apply_overrides(
    meta: dict[str, Any],
    obj_path: Optional[str],
    output_root: Optional[str],
    initial_grasp_path: Optional[str] = None,
) -> dict[str, Any]:
    result = copy.deepcopy(meta)
    if obj_path is not None:
        result.setdefault("object", {})["obj_path"] = obj_path
    if initial_grasp_path is not None:
        result.setdefault("object", {})["initial_grasp_path"] = initial_grasp_path
    if output_root is not None:
        result.setdefault("output", {})["root"] = output_root
    return result


def split_labels(value: Optional[str]) -> Optional[list[str]]:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def dry_run_summary(jobs: Sequence[Mapping[str, Any]], limit: int = 8) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in jobs:
        label = str(job["top_label"])
        counts[label] = counts.get(label, 0) + 1
    preview = []
    for job in jobs[:limit]:
        preview.append(
            {
                "sample_id": job["sample_id"],
                "top_label": job["top_label"],
                "grid_indices": job["grid_indices"],
                "grid_counts": job["grid_counts"],
                "stages": [
                    {
                        "label": stage["label"],
                        "method": stage["method"],
                        "magnitude": stage["magnitudes"][0],
                    }
                    for stage in job["sampled_stages"]
                ],
            }
        )
    return {
        "num_jobs": len(jobs),
        "jobs_per_label": counts,
        "sampling": "independent Cartesian product of per-stage linspace values",
        "preview": preview,
    }


def write_manifest(root: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def execute_jobs(
    jobs: Sequence[dict[str, Any]],
    worker: Worker,
    workers: Optional[int],
    fields: Sequence[str],
    runner_name: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    meta = jobs[0]["meta"]
    sampler = meta.get("sampler", {})
    requested = int(workers if workers is not None else sampler.get("max_workers", os.cpu_count() or 1))
    max_workers = max(1, min(requested, len(jobs)))
    if max_workers == 1:
        rows = [worker(job) for job in jobs]
    else:
        context = mp.get_context(str(sampler.get("mp_start_method", "spawn")))
        rows = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            futures = [executor.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: int(row.get("sample_id", 10**12)))
    root = Path(meta.get("output", {}).get("root", "fpsa_outputs"))
    write_manifest(root, rows, fields)
    succeeded = sum(bool(row.get("ok")) for row in rows)
    print(f"[{runner_name}] {succeeded}/{len(rows)} succeeded; manifest: {root / 'manifest.csv'}")
    return rows
