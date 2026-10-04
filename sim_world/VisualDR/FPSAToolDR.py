"""FPSA tool-mesh domain randomizer.

This module samples an augmented tool mesh together with the collision mesh and
its mesh-frame -> TCP transform produced by the FPSA chained randomizer.

Expected sample layout (``per_shape_dir`` output):

    fpsa_aug_root/
      wrench_chain_y_and_slippage_000000/
        wrench_chain_y_and_slippage_000000.obj
        ... COACD collision OBJ ...
        wrench_chain_y_and_slippage_000000_wrench_to_tcp.yaml

The YAML must contain ``wrench_to_tcp_T`` as a 4x4 homogeneous transform.  The
returned quaternion follows PyBullet / scipy convention: ``[x, y, z, w]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


_MESH_SUFFIXES = {".obj", ".ply", ".stl"}
_COLLISION_TOKENS = ("coacd", "convex", "vhacd", "collision")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read FPSA wrench_to_tcp YAML files")
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError("Unsupported metadata suffix: %s" % path.suffix)

    if not isinstance(data, dict):
        raise TypeError("Metadata root must be a mapping: %s" % path)
    return data


def _as_matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("%s must be a 4x4 matrix, got %s" % (name, matrix.shape))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("%s contains non-finite values" % name)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("%s has an invalid homogeneous last row" % name)
    return matrix


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a nearly-orthogonal 3x3 matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to normalized [x, y, z, w]."""
    r = _project_rotation(rotation)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Could not convert rotation matrix to a valid quaternion")
    quat /= norm

    # q and -q represent the same rotation.  Canonicalize for reproducible logs.
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _quat_xyzw_to_rotation_matrix(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pos: Sequence[float], orn_xyzw: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quat_xyzw_to_rotation_matrix(orn_xyzw)
    result[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return result


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


class FPSAToolDR(object):
    """Randomly select FPSA-augmented tool assets and their TCP transform.

    ``sample`` returns:

        visual_mesh_path, collision_mesh_path,
        wrench_to_tcp_pos, wrench_to_tcp_orn

    Both paths are strings. Position and orientation are NumPy arrays, and the
    orientation order is ``[x, y, z, w]``.
    """

    def __init__(self, seed: int = 0, require_collision: bool = True):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.require_collision = bool(require_collision)
        self.last_sample: Optional[Dict[str, Any]] = None
        self._discovery_cache: Dict[str, List[Dict[str, Any]]] = {}

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def clear_cache(self) -> None:
        self._discovery_cache.clear()

    @staticmethod
    def _resolve_declared_path(
        value: Any,
        metadata_path: Path,
        root: Path,
    ) -> Optional[Path]:
        if value is None or str(value).strip() == "":
            return None

        declared = Path(os.path.expandvars(os.path.expanduser(str(value))))
        candidates: List[Path] = []
        if declared.is_absolute():
            candidates.append(declared)
        else:
            candidates.extend([metadata_path.parent / declared, root / declared, declared])

        # A batch directory may have been moved after generation. Prefer a file
        # with the same basename next to the metadata, then search under root.
        candidates.append(metadata_path.parent / declared.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        matches = sorted(root.rglob(declared.name))
        if matches:
            matches.sort(key=lambda p: (len(p.parts), str(p)))
            return matches[0].resolve()
        return None

    @staticmethod
    def _metadata_candidates(root: Path) -> List[Path]:
        paths: List[Path] = []

        # Use the manifest first when present, but still scan afterward because
        # a partially generated batch may have valid samples missing from it.
        manifest_jsonl = root / "manifest.jsonl"
        if manifest_jsonl.is_file():
            for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not bool(row.get("ok", True)):
                    continue
                value = row.get("wrench_to_tcp_path")
                if value:
                    declared = Path(os.path.expandvars(os.path.expanduser(str(value))))
                    if declared.is_file():
                        paths.append(declared.resolve())
                    else:
                        local = root / declared.name
                        if local.is_file():
                            paths.append(local.resolve())
                        else:
                            paths.extend(p.resolve() for p in root.rglob(declared.name))

        patterns = (
            "*_wrench_to_tcp.yaml",
            "*_wrench_to_tcp.yml",
            "*_wrench_to_tcp.json",
        )
        for pattern in patterns:
            paths.extend(path.resolve() for path in root.rglob(pattern))

        unique: Dict[str, Path] = {}
        for path in paths:
            if path.is_file():
                unique[_path_key(path)] = path
        return sorted(unique.values(), key=lambda p: str(p))

    @staticmethod
    def _collision_score(candidate: Path, visual_mesh: Path) -> Tuple[int, str]:
        name = candidate.name.lower()
        relative_parts = [part.lower() for part in candidate.parts]
        visual_stem = visual_mesh.stem.lower()
        score = 0
        if visual_stem in candidate.stem.lower():
            score += 20
        if "coacd" in name:
            score += 15
        elif "vhacd" in name:
            score += 12
        elif "convex" in name:
            score += 10
        elif "collision" in name:
            score += 8
        if any("coacd" in part for part in relative_parts):
            score += 6
        if candidate.suffix.lower() == ".obj":
            score += 3
        return (-score, str(candidate))

    @classmethod
    def _find_collision_mesh(
        cls,
        visual_mesh: Path,
        metadata: Dict[str, Any],
        metadata_path: Path,
        root: Path,
    ) -> Optional[Path]:
        for key in (
            "collision_mesh_path",
            "coacd_mesh_path",
            "coacd_path",
            "collision_path",
            "vhacd_mesh_path",
        ):
            resolved = cls._resolve_declared_path(metadata.get(key), metadata_path, root)
            if resolved is not None:
                return resolved

        stem = visual_mesh.stem
        parent = visual_mesh.parent
        exact_names = [
            "%s_coacd.obj" % stem,
            "%s.coacd.obj" % stem,
            "%s_convex.obj" % stem,
            "%s_vhacd.obj" % stem,
            "%s_collision.obj" % stem,
            "%s_coacd.ply" % stem,
            "%s_coacd.stl" % stem,
        ]
        exact_paths: List[Path] = []
        for name in exact_names:
            exact_paths.extend(
                [
                    parent / name,
                    parent / "coacd" / name,
                    parent / "collision" / name,
                ]
            )
        for path in exact_paths:
            if path.is_file() and path.resolve() != visual_mesh.resolve():
                return path.resolve()

        # COACD writers use different directory/file conventions. Search only
        # inside the selected sample directory first, then its immediate parent.
        search_roots = [parent]
        if parent != root:
            search_roots.append(parent.parent)

        # Stop at the first search root containing collision assets. This avoids
        # accidentally pairing a generic ``decomposed.obj`` from another sample
        # when all samples live under the same batch root.
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            matches: Dict[str, Path] = {}
            for candidate in search_root.rglob("*"):
                if not candidate.is_file() or candidate.suffix.lower() not in _MESH_SUFFIXES:
                    continue
                if candidate.resolve() == visual_mesh.resolve():
                    continue
                lowered = "/".join(part.lower() for part in candidate.parts)
                if not any(token in lowered for token in _COLLISION_TOKENS):
                    continue
                matches[_path_key(candidate)] = candidate.resolve()

            if matches:
                ranked = sorted(
                    matches.values(),
                    key=lambda p: cls._collision_score(p, visual_mesh),
                )
                return ranked[0]
        return None

    @classmethod
    def _sample_from_metadata(
        cls,
        metadata_path: Path,
        root: Path,
        require_collision: bool,
    ) -> Dict[str, Any]:
        metadata = _load_yaml_or_json(metadata_path)
        if "wrench_to_tcp_T" not in metadata:
            raise KeyError("Missing wrench_to_tcp_T in %s" % metadata_path)

        transform = _as_matrix4(metadata["wrench_to_tcp_T"], "wrench_to_tcp_T")
        visual_mesh = cls._resolve_declared_path(
            metadata.get("mesh_path"), metadata_path, root
        )
        if visual_mesh is None:
            base_name = metadata_path.name
            for suffix in ("_wrench_to_tcp.yaml", "_wrench_to_tcp.yml", "_wrench_to_tcp.json"):
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            fallback = metadata_path.parent / (base_name + ".obj")
            if fallback.is_file():
                visual_mesh = fallback.resolve()

        if visual_mesh is None:
            raise FileNotFoundError("Could not resolve visual mesh for %s" % metadata_path)

        collision_mesh = cls._find_collision_mesh(
            visual_mesh=visual_mesh,
            metadata=metadata,
            metadata_path=metadata_path,
            root=root,
        )
        if require_collision and collision_mesh is None:
            raise FileNotFoundError(
                "Could not find the COACD/collision mesh accompanying %s" % visual_mesh
            )

        return {
            "is_base": False,
            "visual_mesh_path": str(visual_mesh),
            "collision_mesh_path": str(collision_mesh) if collision_mesh is not None else None,
            "wrench_to_tcp_T": transform,
            "wrench_to_tcp_pos": transform[:3, 3].copy(),
            "wrench_to_tcp_orn": _rotation_matrix_to_quat_xyzw(transform[:3, :3]),
            "metadata_path": str(metadata_path),
            "chain_labels": list(metadata.get("chain_labels", [])),
            "stages": metadata.get("stages", []),
        }

    def discover(
        self,
        fpsa_aug_root: str,
        refresh: bool = False,
        skip_invalid: bool = True,
    ) -> List[Dict[str, Any]]:
        root = Path(os.path.expandvars(os.path.expanduser(str(fpsa_aug_root)))).resolve()
        if not root.is_dir():
            raise FileNotFoundError("FPSA tool augmentation root does not exist: %s" % root)

        cache_key = "%s|collision=%s" % (_path_key(root), self.require_collision)
        if not refresh and cache_key in self._discovery_cache:
            return list(self._discovery_cache[cache_key])

        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for metadata_path in self._metadata_candidates(root):
            try:
                samples.append(
                    self._sample_from_metadata(
                        metadata_path=metadata_path,
                        root=root,
                        require_collision=self.require_collision,
                    )
                )
            except Exception as exc:
                errors.append("%s: %s: %s" % (metadata_path, type(exc).__name__, exc))
                if not skip_invalid:
                    raise

        if not samples:
            detail = "\n".join(errors[:10])
            raise FileNotFoundError(
                "No valid FPSA tool samples found under %s.%s"
                % (root, ("\n" + detail) if detail else "")
            )

        self._discovery_cache[cache_key] = samples
        return list(samples)

    def _make_base_sample(
        self,
        base_mesh_path: str,
        base_collision_mesh_path: Optional[str],
        base_wrench_to_tcp_pos: Sequence[float],
        base_wrench_to_tcp_orn: Sequence[float],
    ) -> Dict[str, Any]:
        visual_mesh = Path(
            os.path.expandvars(os.path.expanduser(str(base_mesh_path)))
        ).resolve()
        if not visual_mesh.is_file():
            raise FileNotFoundError("Base wrench mesh does not exist: %s" % visual_mesh)

        collision_mesh: Optional[Path] = None
        if base_collision_mesh_path is not None:
            collision_mesh = Path(
                os.path.expandvars(os.path.expanduser(str(base_collision_mesh_path)))
            ).resolve()
            if not collision_mesh.is_file():
                raise FileNotFoundError("Base wrench collision mesh does not exist: %s" % collision_mesh)
        else:
            collision_mesh = self._find_collision_mesh(
                visual_mesh=visual_mesh,
                metadata={},
                metadata_path=visual_mesh,
                root=visual_mesh.parent,
            )

        if self.require_collision and collision_mesh is None:
            raise FileNotFoundError(
                "include_base=True requires base_collision_mesh_path or a discoverable "
                "COACD mesh beside %s" % visual_mesh
            )

        transform = _pose_to_matrix(base_wrench_to_tcp_pos, base_wrench_to_tcp_orn)
        return {
            "is_base": True,
            "visual_mesh_path": str(visual_mesh),
            "collision_mesh_path": str(collision_mesh) if collision_mesh is not None else None,
            "wrench_to_tcp_T": transform,
            "wrench_to_tcp_pos": transform[:3, 3].copy(),
            "wrench_to_tcp_orn": _rotation_matrix_to_quat_xyzw(transform[:3, :3]),
            "metadata_path": None,
            "chain_labels": ["base"],
            "stages": [],
        }

    def sample(
        self,
        base_mesh_path: Optional[str],
        fpsa_aug_root: str,
        include_base: bool = False,
        base_collision_mesh_path: Optional[str] = None,
        base_wrench_to_tcp_pos: Sequence[float] = (0.06989, 0.0, 0.0),
        base_wrench_to_tcp_orn: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        refresh: bool = False,
    ) -> Tuple[str, Optional[str], np.ndarray, np.ndarray]:
        candidates = self.discover(fpsa_aug_root, refresh=refresh)

        if include_base:
            if base_mesh_path is None:
                raise ValueError("include_base=True requires base_mesh_path")
            candidates = list(candidates)
            candidates.append(
                self._make_base_sample(
                    base_mesh_path=base_mesh_path,
                    base_collision_mesh_path=base_collision_mesh_path,
                    base_wrench_to_tcp_pos=base_wrench_to_tcp_pos,
                    base_wrench_to_tcp_orn=base_wrench_to_tcp_orn,
                )
            )

        index = int(self.rng.integers(0, len(candidates)))
        selected = dict(candidates[index])
        selected["sample_index"] = index
        selected["num_candidates"] = len(candidates)
        self.last_sample = selected

        return (
            str(selected["visual_mesh_path"]),
            selected["collision_mesh_path"],
            np.asarray(selected["wrench_to_tcp_pos"], dtype=np.float64).copy(),
            np.asarray(selected["wrench_to_tcp_orn"], dtype=np.float64).copy(),
        )
