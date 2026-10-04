"""FPSA batch randomizer for objects used as tools.

Unlike grasp_randomizer, this runner transfers the wrench frame, resets the
deformed mesh to that frame, and writes the resulting wrench-to-TCP transform.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    from .src import (
        apply_overrides,
        apply_stages,
        dry_run_summary,
        dump_data,
        execute_jobs,
        load_meta,
        make_jobs,
        matrix4,
        object_path,
        output_paths,
        prepare_output,
        safe_name,
        split_labels,
        to_builtin,
    )
except ImportError:  # Direct execution: python FPSA/tool_randomizer.py
    from src import (  # type: ignore
        apply_overrides,
        apply_stages,
        dry_run_summary,
        dump_data,
        execute_jobs,
        load_meta,
        make_jobs,
        matrix4,
        object_path,
        output_paths,
        prepare_output,
        safe_name,
        split_labels,
        to_builtin,
    )


MANIFEST_FIELDS = (
    "ok",
    "sample_id",
    "sample_name",
    "label",
    "obj_path",
    "coacd_path",
    "wrench_to_tcp_path",
    "meta_path",
    "debug_path",
    "error",
)


def _shape_augmentor_class() -> Any:
    try:
        from .FPSA import ShapeAugmentor
    except ImportError:
        from FPSA import ShapeAugmentor  # type: ignore
    return ShapeAugmentor


def _initial_tcp(meta: dict[str, Any]) -> np.ndarray:
    tcp = meta.get("tcp", {})
    if "T_init_tcp" in tcp:
        return matrix4(tcp["T_init_tcp"], "tcp.T_init_tcp")
    translation = np.asarray(tcp.get("translation", [0.06989, 0.0, 0.0]), dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"tcp.translation must contain 3 values, got {translation.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sample_id = int(job["sample_id"])
    try:
        meta = job["meta"]
        obj = object_path(meta)
        if not Path(obj).is_file():
            raise FileNotFoundError(f"Input OBJ not found: {obj}")

        object_name = safe_name(str(meta.get("object", {}).get("name", Path(obj).stem)))
        sample_name = f"{object_name}_{safe_name(job['top_label'])}_{sample_id:06d}"
        paths = output_paths(meta, sample_name)
        output = meta.get("output", {})
        prepare_output(
            paths,
            ("obj", "wrench_to_tcp", "meta"),
            bool(output.get("overwrite", False)),
        )

        augmentor = _shape_augmentor_class()(obj_path=obj, initial_grasp_path=None)
        stages = apply_stages(augmentor, job, sample_name)
        solver = meta.get("solver", {})
        transferred_frame, anchor, transfer_debug = augmentor.transfer_grasp_SE3(
            T_grasp_old=np.eye(4, dtype=np.float64),
            k_ring=int(solver.get("k_ring", 3)),
            use_distance_weights=True,
            quat_order="xyzw",
            patch_method=str(solver.get("patch_method", "k_ring")),
        )
        transferred_frame = matrix4(transferred_frame, "transferred wrench frame")
        frame_inverse = np.linalg.inv(transferred_frame)
        wrench_to_tcp = _initial_tcp(meta) @ frame_inverse
        augmentor.apply_transformation_to_mesh(frame_inverse)
        final_obj, coacd = augmentor.write_augment_obj(
            output_path=str(paths["obj"]),
            write_coacd=bool(output.get("write_coacd", True)),
            return_paths=True,
        )
        transform_path = dump_data(
            paths["wrench_to_tcp"],
            {
                "mesh_path": final_obj,
                "reference_frame": "randomized_wrench_mesh",
                "target_frame": "tcp",
                "wrench_to_tcp_T": wrench_to_tcp,
                "deformation_label": job["top_label"],
            },
        )
        record = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "label": job["top_label"],
            "grid_indices": job["grid_indices"],
            "grid_counts": job["grid_counts"],
            "stages": stages,
            "source_obj_path": obj,
            "final_obj_path": final_obj,
            "coacd_path": coacd,
            "wrench_to_tcp_path": str(transform_path),
            "wrench_to_tcp_T": wrench_to_tcp,
        }
        meta_path = dump_data(paths["meta"], record)
        debug_path = ""
        if bool(output.get("save_debug", True)):
            debug_path = str(
                dump_data(
                    paths["debug"],
                    {
                        "sample": record,
                        "mesh_status": augmentor.mesh_status(),
                        "frame_anchor": anchor,
                        "frame_transfer_debug": transfer_debug,
                        "transferred_frame": transferred_frame,
                        "frame_inverse": frame_inverse,
                    },
                )
            )
        return {
            "ok": True,
            "sample_id": sample_id,
            "sample_name": sample_name,
            "label": job["top_label"],
            "obj_path": final_obj,
            "coacd_path": coacd or "",
            "wrench_to_tcp_path": str(transform_path),
            "meta_path": str(meta_path),
            "debug_path": debug_path,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "sample_id": sample_id,
            "sample_name": "",
            "label": job.get("top_label", ""),
            "obj_path": "",
            "coacd_path": "",
            "wrench_to_tcp_path": "",
            "meta_path": "",
            "debug_path": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def run_batch(
    meta_path: str | Path,
    labels: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
    obj_path_override: Optional[str] = None,
    output_root: Optional[str] = None,
) -> list[dict[str, Any]]:
    meta = apply_overrides(load_meta(meta_path), obj_path_override, output_root)
    jobs = make_jobs(meta, labels=labels)
    return execute_jobs(jobs, _worker, workers, MANIFEST_FIELDS, "FPSA tool")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPSA tool-use randomizer")
    parser.add_argument("--meta", required=True, help="Tool reshaping YAML/JSON config")
    parser.add_argument("--labels", help="Comma-separated deformation labels")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--obj-path")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = split_labels(args.labels)
    meta = apply_overrides(load_meta(args.meta), args.obj_path, args.output_root)
    jobs = make_jobs(meta, labels=labels)
    if args.dry_run:
        print(json.dumps(to_builtin(dry_run_summary(jobs)), indent=2, ensure_ascii=False))
        return
    execute_jobs(jobs, _worker, args.workers, MANIFEST_FIELDS, "FPSA tool")


if __name__ == "__main__":
    main()
