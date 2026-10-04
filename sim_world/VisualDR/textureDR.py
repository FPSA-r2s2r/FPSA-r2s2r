import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


class TextureDR:
    """Procedural texture domain randomization for PyBullet bodies.

    This randomizer is intended for non-task bodies such as the robot, table,
    wall, or floor. It generates a texture image on disk, loads it with
    PyBullet loadTexture(), and applies it through changeVisualShape().

    Supported procedural patterns:
        - checkers
        - gradient
        - noise
        - plain
    """

    PATTERNS = ("checkers", "gradient", "noise", "plain")

    def __init__(self, bullet_client, seed=42, cache_dir=None):
        self.bullet_client = bullet_client
        self.rng = np.random.default_rng(seed)
        self.cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="texture_dr_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_cfg = None
        self._counter = 0
        self.loaded_texture_ids = []
        self._original_visual_state = {}

    # ----------------------------- sampling utils -----------------------------
    @staticmethod
    def _clip_rgb(rgb):
        rgb = np.asarray(rgb, dtype=float).reshape(3)
        return np.clip(rgb, 0.0, 1.0)

    def _sample_color(self, value_range=(0.05, 0.95)):
        lo, hi = value_range
        return self.rng.uniform(float(lo), float(hi), size=3)

    def _sample_color_pair(self, min_dist=0.30):
        c0 = self._sample_color()
        c1 = self._sample_color()
        for _ in range(32):
            if np.linalg.norm(c0 - c1) >= float(min_dist):
                break
            c1 = self._sample_color()
        return self._clip_rgb(c0), self._clip_rgb(c1)

    def _sample_specular(self, specular_range):
        if specular_range is None:
            return None
        lo, hi = specular_range
        s = float(self.rng.uniform(float(lo), float(hi)))
        return [s, s, s]

    def _normalize_patterns(self, patterns):
        if patterns is None:
            return self.PATTERNS
        patterns = tuple(str(p).lower().strip() for p in patterns)
        unknown = [p for p in patterns if p not in self.PATTERNS]
        if unknown:
            raise ValueError(f"Unknown texture patterns {unknown}. Available: {self.PATTERNS}")
        if len(patterns) == 0:
            raise ValueError("patterns must contain at least one pattern name.")
        return patterns

    # -------------------------- procedural generators -------------------------
    def _make_checkers(self, size, c0, c1):
        num_tiles = int(self.rng.integers(4, 14))
        yy, xx = np.indices((size, size))
        tile = max(1, size // num_tiles)
        mask = ((xx // tile + yy // tile) % 2).astype(bool)
        image = np.empty((size, size, 3), dtype=np.float32)
        image[~mask] = c0
        image[mask] = c1
        return image, {"num_tiles": num_tiles}

    def _make_gradient(self, size, c0, c1):
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        t = xx * np.cos(angle) + yy * np.sin(angle)
        t = (t - t.min()) / max(float(t.max() - t.min()), 1e-8)
        image = (1.0 - t[..., None]) * c0 + t[..., None] * c1
        return image.astype(np.float32), {"angle_rad": angle}

    def _make_noise(self, size, c0, c1):
        low = int(self.rng.integers(8, 33))
        noise = self.rng.random((low, low), dtype=np.float32)
        noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        noise = np.clip(noise, 0.0, 1.0)
        image = (1.0 - noise[..., None]) * c0 + noise[..., None] * c1
        return image.astype(np.float32), {"low_res_size": low}

    def _make_plain(self, size, c0, c1=None):
        image = np.broadcast_to(c0.reshape(1, 1, 3), (size, size, 3)).copy()
        return image.astype(np.float32), {}

    def _make_texture_image(self, pattern, size):
        c0, c1 = self._sample_color_pair()
        if pattern == "checkers":
            image, extra = self._make_checkers(size, c0, c1)
        elif pattern == "gradient":
            image, extra = self._make_gradient(size, c0, c1)
        elif pattern == "noise":
            image, extra = self._make_noise(size, c0, c1)
        elif pattern == "plain":
            image, extra = self._make_plain(size, c0, c1)
        else:
            raise ValueError(f"Unknown texture pattern {pattern!r}.")

        image = np.clip(image, 0.0, 1.0)
        return image, c0, c1, extra

    def sample_procedural_texture(self, patterns=None, texture_size=128):
        patterns = self._normalize_patterns(patterns)
        size = int(texture_size)
        if size <= 0:
            raise ValueError("texture_size must be positive.")

        pattern = str(self.rng.choice(patterns))
        image, c0, c1, extra = self._make_texture_image(pattern, size)

        self._counter += 1
        path = self.cache_dir / f"procedural_{self._counter:06d}_{pattern}.png"
        image_u8 = (255.0 * image).round().astype(np.uint8)
        ok = cv2.imwrite(str(path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write procedural texture: {path}")

        texture_id = self.bullet_client.loadTexture(str(path))
        self.loaded_texture_ids.append(texture_id)

        return {
            "pattern": pattern,
            "texture_size": size,
            "path": str(path),
            "texture_id": int(texture_id),
            "color0": c0.tolist(),
            "color1": c1.tolist(),
            **extra,
        }

    # ----------------------- PyBullet visual-state utils -----------------------
    @staticmethod
    def _looks_like_rgba(x):
        try:
            arr = np.asarray(x, dtype=float)
        except Exception:
            return False
        return arr.shape == (4,) and np.all(np.isfinite(arr))

    def _extract_rgba_from_visual_row(self, row):
        for item in reversed(row):
            if self._looks_like_rgba(item):
                rgba = np.asarray(item, dtype=float)
                rgba[:3] = np.clip(rgba[:3], 0.0, 1.0)
                rgba[3] = np.clip(rgba[3], 0.0, 1.0)
                return rgba.astype(float)
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    def _extract_alpha_from_visual_row(self, row):
        return float(self._extract_rgba_from_visual_row(row)[3])

    @staticmethod
    def _extract_texture_id_from_visual_row(row):
        # Common PyBullet layout:
        #   (..., rgbaColor, textureUniqueId)
        # Some versions omit textureUniqueId; -1 means no texture.
        if len(row) >= 9:
            try:
                return int(row[8])
            except Exception:
                return -1
        return -1

    def _get_visual_rows(self, body_id):
        try:
            rows = self.bullet_client.getVisualShapeData(body_id)
        except Exception:
            rows = None
        return list(rows) if rows else []

    def get_visual_link_indices(self, body_id, link_indices=None):
        rows = self._get_visual_rows(body_id)
        if link_indices is not None:
            keep = {int(x) for x in link_indices}
        else:
            keep = None

        links = []
        for row in rows:
            try:
                link_idx = int(row[1])
            except Exception:
                link_idx = -1
            if keep is None or link_idx in keep:
                links.append(link_idx)

        if not links and keep is None:
            links = [-1]
        return sorted(set(links))

    def get_alpha_by_link(self, body_id, link_indices=None):
        rows = self._get_visual_rows(body_id)
        if link_indices is not None:
            link_indices = {int(x) for x in link_indices}

        grouped = {}
        for row in rows:
            try:
                link_idx = int(row[1])
            except Exception:
                link_idx = -1
            if link_indices is not None and link_idx not in link_indices:
                continue
            grouped.setdefault(link_idx, []).append(self._extract_alpha_from_visual_row(row))

        if not grouped:
            return {-1: 1.0}
        return {int(k): float(np.mean(v)) for k, v in grouped.items()}

    def _cache_original_visual_state(self, body_id):
        body_id = int(body_id)
        if body_id in self._original_visual_state:
            return

        state = {}
        for row in self._get_visual_rows(body_id):
            try:
                link_idx = int(row[1])
            except Exception:
                link_idx = -1

            # Keep one visual state per link. That is sufficient for the Panda
            # URDF used here and matches changeVisualShape(body, link).
            if link_idx not in state:
                rgba = self._extract_rgba_from_visual_row(row)
                state[link_idx] = {
                    "rgbaColor": rgba.tolist(),
                    "textureUniqueId": self._extract_texture_id_from_visual_row(row),
                }

        if not state:
            state[-1] = {
                "rgbaColor": [1.0, 1.0, 1.0, 1.0],
                "textureUniqueId": -1,
            }

        self._original_visual_state[body_id] = state

    def restore_original_texture(self, body_id, link_indices=None):
        """Restore the cached pre-DR visual state for a body.

        This is used for the sampled ``original`` branch. It is intentionally
        an active restore, not a no-op, because the Panda body persists across
        episodes and may have received a randomized texture in a previous one.
        """
        body_id = int(body_id)
        self._cache_original_visual_state(body_id)
        original_state = self._original_visual_state[body_id]

        links = self.get_visual_link_indices(body_id, link_indices=link_indices)
        link_cfgs = []
        for link_idx in links:
            state = original_state.get(int(link_idx))
            if state is None:
                continue

            rgba = [float(x) for x in state["rgbaColor"]]
            texture_id = int(state.get("textureUniqueId", -1))
            kwargs = {"rgbaColor": rgba}
            # -1 removes the previously assigned procedural texture when the
            # original visual did not use a texture.
            kwargs["textureUniqueId"] = texture_id
            self.bullet_client.changeVisualShape(body_id, int(link_idx), **kwargs)
            link_cfgs.append({
                "linkIndex": int(link_idx),
                "rgbaColor": rgba,
                "textureUniqueId": texture_id,
                "applied": False,
            })

        cfg = {
            "body_id": body_id,
            "mode": "original",
            "applied": False,
            "link_cfgs": link_cfgs,
        }
        self.last_cfg = cfg
        return cfg

    def _apply_texture(self, body_id, link_idx, texture_id, alpha=1.0, specular=None):
        rgba = [1.0, 1.0, 1.0, float(np.clip(alpha, 0.0, 1.0))]
        kwargs = {
            "textureUniqueId": int(texture_id),
            "rgbaColor": rgba,
        }
        if specular is not None:
            kwargs["specularColor"] = [float(x) for x in specular]
        self.bullet_client.changeVisualShape(body_id, int(link_idx), **kwargs)
        return rgba

    # ------------------------------- public API -------------------------------
    def sample_and_apply_texture_randomization(
        self,
        body_id,
        patterns=None,
        texture_size=128,
        per_link=True,
        link_indices=None,
        specular_range=(0.02, 0.25),
        alpha=None,
        original_texture_prob=0.10,
    ):
        """Apply procedural textures to a PyBullet body.

        Only the explicitly passed body is modified. Callers remain responsible
        for deciding which scene bodies should receive texture randomization.

        original_texture_prob gives an episode-level chance of keeping the
        body's currently loaded visual appearance unchanged. The sampled
        ``original`` branch is a no-op: it does not call changeVisualShape(),
        because forcing cached rgba / texture ids back onto the body can erase
        the URDF material colors and make some links appear white.
        Set it to 0.0 to always apply procedural texture DR.
        """
        body_id = int(body_id)
        self._cache_original_visual_state(body_id)

        links = self.get_visual_link_indices(body_id, link_indices=link_indices)
        patterns = self._normalize_patterns(patterns)

        original_texture_prob = float(np.clip(original_texture_prob, 0.0, 1.0))
        if self.rng.random() < original_texture_prob:
            # Keep the visual state exactly as PyBullet currently has it.
            # Do not call restore_original_texture()/changeVisualShape() here:
            # re-applying cached rgba or texture ids can discard URDF material
            # colors and turn links white.
            cfg = {
                "body_id": body_id,
                "mode": "original",
                "applied": False,
                "patterns": list(patterns),
                "texture_size": int(texture_size),
                "per_link": bool(per_link),
                "original_texture_prob": original_texture_prob,
                "reason": "sampled_original_robot_texture_noop",
                "link_cfgs": [],
            }
            self.last_cfg = cfg
            return cfg

        alpha_by_link = self.get_alpha_by_link(body_id, link_indices=links)
        specular = self._sample_specular(specular_range)

        shared_texture = None
        if not per_link:
            shared_texture = self.sample_procedural_texture(
                patterns=patterns,
                texture_size=texture_size,
            )

        link_cfgs = []
        for link_idx in links:
            texture_cfg = shared_texture or self.sample_procedural_texture(
                patterns=patterns,
                texture_size=texture_size,
            )
            apply_alpha = alpha_by_link.get(int(link_idx), 1.0) if alpha is None else float(alpha)
            applied_rgba = self._apply_texture(
                body_id=body_id,
                link_idx=link_idx,
                texture_id=texture_cfg["texture_id"],
                alpha=apply_alpha,
                specular=specular,
            )
            link_cfgs.append({
                "linkIndex": int(link_idx),
                "rgbaColor": applied_rgba,
                "texture": texture_cfg,
            })

        cfg = {
            "body_id": int(body_id),
            "mode": "procedural",
            "applied": True,
            "patterns": list(patterns),
            "texture_size": int(texture_size),
            "per_link": bool(per_link),
            "original_texture_prob": original_texture_prob,
            "specularColor": specular,
            "link_cfgs": link_cfgs,
        }
        self.last_cfg = cfg
        return cfg

    def sample_and_apply_robot_texture_randomization(
        self,
        body_id,
        patterns=None,
        texture_size=128,
        per_link=True,
        link_indices=None,
        specular_range=(0.02, 0.25),
        alpha=None,
        original_texture_prob=0.10,
    ):
        """Backward-compatible robot-specific entry point."""
        return self.sample_and_apply_texture_randomization(
            body_id=body_id,
            patterns=patterns,
            texture_size=texture_size,
            per_link=per_link,
            link_indices=link_indices,
            specular_range=specular_range,
            alpha=alpha,
            original_texture_prob=original_texture_prob,
        )

    def reset(self, body_id=None, restore_original=False):
        """Reset stored config, optionally restoring the cached original texture."""
        if restore_original and body_id is not None:
            self.restore_original_texture(body_id)
        self.last_cfg = None
