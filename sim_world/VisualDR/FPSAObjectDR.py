from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FPSAObjectDR:
    """Randomly select a consistent FPSA assembly-parent asset bundle.

    Every augmented sample is treated atomically:
      - visual mesh: ``<sample>.obj``
      - collision mesh: ``<sample>_coacd.obj``
      - grasp initial guess: ``<sample>_grasp.yaml`` / ``.yml``

    Sampling is always random with replacement.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(int(seed))
        self.sample_pool: List[Dict[str, Any]] = []
        self.last_sample: Optional[Dict[str, Any]] = None
        self._pool_signature: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def _as_path(path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    @staticmethod
    def _sample_id_from_name(path: Path) -> Optional[int]:
        match = re.search(r"_(\d+)$", path.stem)
        return int(match.group(1)) if match else None

    @staticmethod
    def _matching_grasp_path(obj_path: Path) -> Optional[Path]:
        for suffix in (".yaml", ".yml"):
            candidate = obj_path.with_name(f"{obj_path.stem}_grasp{suffix}")
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _matching_collision_path(obj_path: Path) -> Optional[Path]:
        candidate = obj_path.with_name(f"{obj_path.stem}_coacd.obj")
        return candidate if candidate.is_file() else None

    @classmethod
    def _asset_from_obj(cls, obj_path: Path) -> Optional[Dict[str, Any]]:
        # Do not treat the convex-decomposition cache as a visual sample.
        if obj_path.stem.endswith("_coacd"):
            return None

        grasp_path = cls._matching_grasp_path(obj_path)
        collision_path = cls._matching_collision_path(obj_path)
        if grasp_path is None or collision_path is None:
            return None

        return {
            "obj_path": str(obj_path),
            "collision_path": str(collision_path),
            "grasp_path": str(grasp_path),
            "source": "fpsa_aug",
            "sample_id": cls._sample_id_from_name(obj_path),
        }

    def build_sample_pool(
        self,
        base_mesh_path: Optional[str] = None,
        base_collision_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> List[Dict[str, Any]]:
        signature = (
            str(base_mesh_path),
            str(base_collision_path),
            str(base_grasp_path),
            str(fpsa_aug_root),
            bool(include_base),
        )
        if signature == self._pool_signature and self.sample_pool:
            return self.sample_pool

        pool: List[Dict[str, Any]] = []

        if include_base:
            if base_mesh_path is None or base_grasp_path is None:
                raise ValueError(
                    "include_base=True requires base_mesh_path and base_grasp_path"
                )

            base_mesh = self._as_path(base_mesh_path)
            base_grasp = self._as_path(base_grasp_path)
            if not base_mesh.is_file():
                raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
            if not base_grasp.is_file():
                raise FileNotFoundError(f"Base grasp pose not found: {base_grasp}")

            collision = None
            if base_collision_path is not None:
                collision = self._as_path(base_collision_path)
                if not collision.is_file():
                    raise FileNotFoundError(
                        f"Base collision mesh not found: {collision}"
                    )

            pool.append(
                {
                    "obj_path": str(base_mesh),
                    "collision_path": str(collision) if collision is not None else None,
                    "grasp_path": str(base_grasp),
                    "source": "base",
                    "sample_id": -1,
                }
            )

        if fpsa_aug_root is not None:
            root = self._as_path(fpsa_aug_root)
            if not root.is_dir():
                raise FileNotFoundError(
                    f"FPSA augmented object folder not found: {root}"
                )

            # Supports both flat output and per-shape subfolders.
            for obj_path in root.rglob("*.obj"):
                asset = self._asset_from_obj(obj_path)
                if asset is not None:
                    pool.append(asset)

        if not pool:
            raise FileNotFoundError(
                "No valid FPSA asset bundle found. Each augmented sample needs "
                "<sample>.obj, <sample>_coacd.obj, and <sample>_grasp.yaml."
            )

        self.sample_pool = pool
        self._pool_signature = signature
        return self.sample_pool

    def reset(self, seed: Optional[int] = None) -> None:
        """Optionally reseed the random sampler and clear the last sample."""
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.last_sample = None

    def sample_asset(
        self,
        base_mesh_path: Optional[str] = None,
        base_collision_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> Dict[str, Any]:
        self.build_sample_pool(
            base_mesh_path=base_mesh_path,
            base_collision_path=base_collision_path,
            base_grasp_path=base_grasp_path,
            fpsa_aug_root=fpsa_aug_root,
            include_base=include_base,
        )

        index = int(self.rng.integers(len(self.sample_pool)))
        sample = dict(self.sample_pool[index])
        sample["pool_index"] = index
        sample["sampling_mode"] = "random"
        self.last_sample = sample
        return sample

    # Backward-compatible two-path API.
    def sample(
        self,
        base_mesh_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> Tuple[str, str]:
        sample = self.sample_asset(
            base_mesh_path=base_mesh_path,
            base_grasp_path=base_grasp_path,
            fpsa_aug_root=fpsa_aug_root,
            include_base=include_base,
        )
        return sample["obj_path"], sample["grasp_path"]

    def sample_FPSA_augmented_object(
        self,
        base_mesh_path: Optional[str] = None,
        base_grasp_path: Optional[str] = None,
        fpsa_aug_root: Optional[str] = None,
        include_base: bool = True,
    ) -> Tuple[str, str]:
        return self.sample(
            base_mesh_path=base_mesh_path,
            base_grasp_path=base_grasp_path,
            fpsa_aug_root=fpsa_aug_root,
            include_base=include_base,
        )