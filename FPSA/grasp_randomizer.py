"""FPSA batch randomizer for objects whose grasp pose must follow reshaping.

Assembly, bracket, and gear use this runner.  Every chain is sampled as the
complete Cartesian product of per-stage linspace values.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

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
except ImportError:  # Direct execution: python FPSA/grasp_randomizer.py
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
    "grasp_path",
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


def _initial_grasp_path(meta: dict[str, Any]) -> str:
    try:
        return str(meta["object"]["initial_grasp_path"])
    except KeyError as exc:
        raise KeyError("object.initial_grasp_path is required by grasp_randomizer") from exc


def _write_grasp(
    path: Path,
    grasp: dict[str, Any],
    mesh_path: str,
    source_path: str,
    job: dict[str, Any],
) -> Path:
    tcp = grasp.get("T_mesh_hand_tcp", grasp.get("T_mesh_hand"))
    hand = grasp.get("T_mesh_hand", tcp)
    if tcp is None:
        raise KeyError("Transferred grasp has no T_mesh_hand_tcp or T_mesh_hand")
    opening = float(grasp.get("opening_width_m", grasp.get("opening", 0.06)))
    return dump_data(
        path,
        {
            "mesh_path": mesh_path,
            "reference_frame": grasp.get("reference_frame", "mesh_local_frame"),
            "hand_frame": grasp.get("hand_frame", "panda_hand"),
            "opening_width_m": opening,
            "pregrasp_opening_width_m": float(
                grasp.get("pregrasp_opening_width_m", grasp.get("pregrasp_opening", opening))
            ),
            "finger_joint_m": float(
                grasp.get("finger_joint_m", grasp.get("finger_joint", opening / 2.0))
            ),
            "T_mesh_hand": matrix4(hand, "T_mesh_hand"),
            "T_mesh_hand_tcp": matrix4(tcp, "T_mesh_hand_tcp"),
            "source_grasp_path": source_path,
            "deformation_label": job["top_label"],
        },
    )


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sample_id = int(job["sample_id"])
    try:
        meta = job["meta"]
        obj = object_path(meta)
        grasp_source = _initial_grasp_path(meta)
        if not Path(obj).is_file():
            raise FileNotFoundError(f"Input OBJ not found: {obj}")
        if not Path(grasp_source).is_file():
            raise FileNotFoundError(f"Initial grasp not found: {grasp_source}")

        object_name = safe_name(str(meta.get("object", {}).get("name", Path(obj).stem)))
        sample_name = f"{object_name}_{safe_name(job['top_label'])}_{sample_id:06d}"
        paths = output_paths(meta, sample_name)
        output = meta.get("output", {})
        prepare_output(paths, ("obj", "grasp", "meta"), bool(output.get("overwrite", False)))

        augmentor = _shape_augmentor_class()(obj_path=obj, initial_grasp_path=grasp_source)
        stages = apply_stages(augmentor, job, sample_name)
        solver = meta.get("solver", {})
        grasp, anchor, transfer_debug = augmentor.transfer_initial_grasp_guess(
            k_ring=int(solver.get("k_ring", 3)),
            use_distance_weights=True,
            quat_order="xyzw",
            patch_method=str(solver.get("patch_method", "k_ring")),
            return_format="dict",
        )
        final_obj, coacd = augmentor.write_augment_obj(
            output_path=str(paths["obj"]),
            write_coacd=bool(output.get("write_coacd", True)),
            return_paths=True,
        )
        grasp_path = _write_grasp(paths["grasp"], grasp, final_obj, grasp_source, job)

        record = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "label": job["top_label"],
            "grid_indices": job["grid_indices"],
            "grid_counts": job["grid_counts"],
            "stages": stages,
            "source_obj_path": obj,
            "source_grasp_path": grasp_source,
            "final_obj_path": final_obj,
            "coacd_path": coacd,
            "grasp_path": str(grasp_path),
            "transferred_grasp": grasp,
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
                        "grasp_anchor": anchor,
                        "grasp_transfer_debug": transfer_debug,
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
            "grasp_path": str(grasp_path),
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
            "grasp_path": "",
            "meta_path": "",
            "debug_path": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def run_batch(
    meta_path: str | Path,
    labels: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
    obj_path_override: Optional[str] = None,
    initial_grasp_path: Optional[str] = None,
    output_root: Optional[str] = None,
) -> list[dict[str, Any]]:
    meta = apply_overrides(
        load_meta(meta_path), obj_path_override, output_root, initial_grasp_path
    )
    jobs = make_jobs(meta, labels=labels)
    return execute_jobs(jobs, _worker, workers, MANIFEST_FIELDS, "FPSA grasp")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPSA grasp-pose randomizer")
    parser.add_argument("--meta", required=True, help="Object reshaping YAML/JSON config")
    parser.add_argument("--labels", help="Comma-separated deformation labels")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--obj-path")
    parser.add_argument("--initial-grasp-path")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = split_labels(args.labels)
    meta = apply_overrides(
        load_meta(args.meta), args.obj_path, args.output_root, args.initial_grasp_path
    )
    jobs = make_jobs(meta, labels=labels)
    if args.dry_run:
        print(json.dumps(to_builtin(dry_run_summary(jobs)), indent=2, ensure_ascii=False))
        return
    execute_jobs(jobs, _worker, args.workers, MANIFEST_FIELDS, "FPSA grasp")


if __name__ == "__main__":
    main()
