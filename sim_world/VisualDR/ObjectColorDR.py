import colorsys
import numpy as np


class ObjectColorDR:
    """Object-level color/material domain randomization for PyBullet.

    Put this class in VisualDR.py, next to LightingDR, ImgNoiseDR, PoseDR,
    and DistractorDR. It is designed for objects that are already loaded into
    PyBullet. It only calls changeVisualShape(), so geometry, collision, mass,
    dynamics, pose, friction, and grasp labels are unchanged.

    Public modes:
        1. mode="bounded": read the current visual rgbaColor from PyBullet and
           apply a small HSV jitter around it. If the current rgbaColor is white
           or gray, the hue is ambiguous, so the class samples a mild colored
           tint instead. This is useful for textured OBJ assets whose material
           color is [1, 1, 1, 1]: the original PNG texture is not edited, but the
           global rgbaColor tint can still recolor the rendered object while
           keeping texture details visible.

        2. mode="recolor": read the current visual rgbaColor from PyBullet and
           replace its hue with a target/palette hue, e.g. green -> red or
           white -> blue. This is stronger than bounded jitter, but it still
           works as a global tint; it does not replace or overwrite the PNG
           texture.
    """

    DEFAULT_RECOLOR_PALETTE = {
        "red":    (0.90, 0.08, 0.06),
        "orange": (0.95, 0.38, 0.05),
        "yellow": (0.92, 0.75, 0.08),
        "green":  (0.10, 0.65, 0.16),
        "blue":   (0.08, 0.22, 0.90),
        "cyan":   (0.05, 0.65, 0.80),
        "purple": (0.55, 0.12, 0.80),
        "pink":   (0.95, 0.35, 0.55),
        "gray":   (0.45, 0.45, 0.45),
        "white":  (0.90, 0.90, 0.90),
    }

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.rng = np.random.default_rng(seed)
        self.last_cfg = None

    # ----------------------------- color utils -----------------------------
    @staticmethod
    def _clip_rgb(rgb):
        rgb = np.asarray(rgb, dtype=float).reshape(3)
        return np.clip(rgb, 0.0, 1.0)

    @staticmethod
    def _rgb_to_hsv(rgb):
        r, g, b = ObjectColorDR._clip_rgb(rgb)
        return np.array(colorsys.rgb_to_hsv(float(r), float(g), float(b)), dtype=float)

    @staticmethod
    def _hsv_to_rgb(hsv):
        h, s, v = np.asarray(hsv, dtype=float).reshape(3)
        h = float(h % 1.0)
        s = float(np.clip(s, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=float)

    def _parse_color(self, color):
        """Accept an RGB tuple/list/array or a color name."""
        if isinstance(color, str):
            key = color.lower().strip()
            if key not in self.DEFAULT_RECOLOR_PALETTE:
                raise ValueError(
                    f"Unknown color name {color!r}. Available: "
                    f"{sorted(self.DEFAULT_RECOLOR_PALETTE.keys())}"
                )
            return np.array(self.DEFAULT_RECOLOR_PALETTE[key], dtype=float)
        return self._clip_rgb(color)

    def _sample_specular(self, specular_range):
        if specular_range is None:
            return None
        lo, hi = specular_range
        s = float(self.rng.uniform(float(lo), float(hi)))
        return [s, s, s]

    # ----------------------- PyBullet visual-state utils --------------------
    @staticmethod
    def _looks_like_rgba(x):
        try:
            arr = np.asarray(x, dtype=float)
        except Exception:
            return False
        return arr.shape == (4,) and np.all(np.isfinite(arr))

    def _extract_rgba_from_visual_row(self, row):
        """Robustly find the rgbaColor tuple in one getVisualShapeData() row.

        PyBullet versions can differ slightly in row layout, but rgbaColor is a
        length-4 numeric tuple. We scan from the end so the common last-field
        rgbaColor is found first.
        """
        for item in reversed(row):
            if self._looks_like_rgba(item):
                rgba = np.asarray(item, dtype=float)
                rgb = np.clip(rgba[:3], 0.0, 1.0)
                alpha = float(np.clip(rgba[3], 0.0, 1.0))
                return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=float)
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    def _get_visual_rows(self, body_id):
        try:
            rows = self.bullet_client.getVisualShapeData(body_id)
        except Exception:
            rows = None
        return list(rows) if rows else []

    def get_current_visual_rgba_by_link(self, body_id, link_indices=None):
        """Read current visual rgbaColor from PyBullet, grouped by linkIndex.

        Returns:
            dict: {link_index: rgba np.ndarray shape (4,)}

        For a simple OBJ body this is usually {-1: rgba}. For textured objects,
        PyBullet often reports [1, 1, 1, 1], which is still useful: applying a
        non-white rgbaColor later acts as a global tint over the texture.
        """
        rows = self._get_visual_rows(body_id)
        if not rows:
            return {-1: np.array([1.0, 1.0, 1.0, 1.0], dtype=float)}

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

            grouped.setdefault(link_idx, []).append(self._extract_rgba_from_visual_row(row))

        if not grouped:
            return {-1: np.array([1.0, 1.0, 1.0, 1.0], dtype=float)}

        rgba_by_link = {}
        for link_idx, rgba_list in grouped.items():
            arr = np.stack(rgba_list, axis=0)
            rgba_by_link[int(link_idx)] = np.mean(arr, axis=0)
        return rgba_by_link

    def _apply_visual_color(self, body_id, link_idx, rgb, alpha=1.0, specular=None):
        rgb = self._clip_rgb(rgb)
        alpha = 1.0 if float(alpha) >= 0.5 else 0.0  # PyBullet alpha is effectively binary.
        rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)]

        kwargs = {"rgbaColor": rgba}
        if specular is not None:
            kwargs["specularColor"] = [float(x) for x in specular]

        self.bullet_client.changeVisualShape(body_id, int(link_idx), **kwargs)
        return rgba

    # ---------------------------- sampling modes ----------------------------
    def _sample_bounded_rgb(self, base_rgb, strength=0.35):
        """Sample a mild color tint around the current visual color."""
        strength = float(np.clip(strength, 0.0, 1.0))
        h, s, v = self._rgb_to_hsv(base_rgb)

        hue_jitter_deg = 4.0 + 26.0 * strength
        hue_delta = self.rng.uniform(-hue_jitter_deg, hue_jitter_deg) / 360.0
        val_scale = self.rng.uniform(1.0 - 0.55 * strength, 1.0 + 0.80 * strength)

        # If rgbaColor is white/gray, hue is not meaningful and s is near 0.
        # In that case, sample a mild colored tint so textured objects with
        # white material can still receive visible object-color randomization.
        achromatic = s < 0.06
        if achromatic:
            sampled_h = self.rng.uniform(0.0, 1.0)
            sampled_s = self.rng.uniform(0.04, 0.10 + 0.45 * strength)
            sat_scale = None
        else:
            sat_scale = self.rng.uniform(1.0 - 0.45 * strength, 1.0 + 0.45 * strength)
            sampled_h = (h + hue_delta) % 1.0
            sampled_s = np.clip(s * sat_scale, 0.0, 1.0)

        sampled_v = np.clip(v * val_scale, 0.0, 1.0)
        sampled_hsv = np.array([sampled_h, sampled_s, sampled_v], dtype=float)
        sampled_rgb = self._hsv_to_rgb(sampled_hsv)

        return sampled_rgb, sampled_hsv, {
            "achromatic_base": bool(achromatic),
            "hue_delta_deg": float(hue_delta * 360.0) if not achromatic else None,
            "saturation_scale": float(sat_scale) if sat_scale is not None else None,
            "value_scale": float(val_scale),
            "strength": strength,
        }

    def _sample_recolor_rgb(
        self,
        base_rgb,
        target_rgb,
        preserve_base_value=True,
        min_saturation_for_recolor=0.45,
    ):
        """Replace base hue with target hue while preserving texture-friendly value."""
        base_h, base_s, base_v = self._rgb_to_hsv(base_rgb)
        target_h, target_s, target_v = self._rgb_to_hsv(target_rgb)

        sampled_h = target_h
        sampled_s = max(float(base_s), float(min_saturation_for_recolor))
        sampled_s = min(sampled_s, max(float(target_s), float(min_saturation_for_recolor)))
        sampled_v = float(base_v) if preserve_base_value else float(target_v)

        sampled_hsv = np.array([sampled_h, sampled_s, sampled_v], dtype=float)
        sampled_rgb = self._hsv_to_rgb(sampled_hsv)
        return sampled_rgb, sampled_hsv, {
            "base_hsv": [float(base_h), float(base_s), float(base_v)],
            "target_hsv": [float(target_h), float(target_s), float(target_v)],
            "preserve_base_value": bool(preserve_base_value),
            "min_saturation_for_recolor": float(min_saturation_for_recolor),
        }

    def sample_and_apply_object_color_randomization(
        self,
        body_id,
        mode="bounded",
        strength=0.35,
        recolor_palette=None,
        recolor_target_color=None,
        specular_range=(0.02, 0.15),
        alpha=None,
        link_indices=None,
    ):
        """Read current rgbaColor, sample a new color, and apply it.

        Args:
            body_id: PyBullet body id that has already been loaded.
            mode: "bounded" for small HSV jitter, or "recolor" for hue swap.
            strength: bounded-mode intensity in [0, 1]. Higher values create
                stronger hue/saturation/value changes.
            recolor_palette: optional palette for mode="recolor". Entries can
                be color names or RGB tuples.
            recolor_target_color: optional fixed target color for mode="recolor".
                Example: "red" or (0.9, 0.05, 0.05).
            specular_range: sampled specularColor range for changeVisualShape().
                Use None to leave specular unchanged.
            alpha: None means preserve the current visual alpha per link.
            link_indices: optional subset of visual link indices.
        """
        mode = str(mode).lower().strip()
        if mode not in ("bounded", "jitter", "bounded_jitter", "recolor", "global_recolor", "color_swap", "palette"):
            raise ValueError(f"Unknown object color DR mode {mode!r}. Use 'bounded' or 'recolor'.")

        rgba_by_link = self.get_current_visual_rgba_by_link(body_id, link_indices=link_indices)
        specular = self._sample_specular(specular_range)

        # For recolor, sample a single target hue for the whole object so all
        # visual links remain consistent.
        target_name = None
        target_rgb = None
        if mode in ("recolor", "global_recolor", "color_swap", "palette"):
            if recolor_target_color is not None:
                target_rgb = self._parse_color(recolor_target_color)
                target_name = str(recolor_target_color)
            else:
                if recolor_palette is None:
                    recolor_palette = ["red", "blue", "yellow", "orange", "purple", "pink", "cyan", "gray"]
                idx = int(self.rng.integers(0, len(recolor_palette)))
                target = recolor_palette[idx]
                target_rgb = self._parse_color(target)
                target_name = str(target)

        link_cfgs = []
        for link_idx, base_rgba in rgba_by_link.items():
            base_rgb = self._clip_rgb(base_rgba[:3])
            base_alpha = float(base_rgba[3])
            apply_alpha = base_alpha if alpha is None else float(alpha)
            base_hsv = self._rgb_to_hsv(base_rgb)

            if mode in ("bounded", "jitter", "bounded_jitter"):
                sampled_rgb, sampled_hsv, extra = self._sample_bounded_rgb(base_rgb, strength=strength)
                link_cfg = {
                    "mode": "bounded",
                    "linkIndex": int(link_idx),
                    "base_rgba": base_rgba.tolist(),
                    "base_hsv": base_hsv.tolist(),
                    "sampled_color_rgb": sampled_rgb.tolist(),
                    "sampled_hsv": sampled_hsv.tolist(),
                    **extra,
                }
            else:
                sampled_rgb, sampled_hsv, extra = self._sample_recolor_rgb(
                    base_rgb,
                    target_rgb,
                    preserve_base_value=True,
                    min_saturation_for_recolor=0.45,
                )
                link_cfg = {
                    "mode": "recolor",
                    "linkIndex": int(link_idx),
                    "base_rgba": base_rgba.tolist(),
                    "base_hsv": base_hsv.tolist(),
                    "target_color": target_name,
                    "target_color_rgb": target_rgb.tolist(),
                    "sampled_color_rgb": sampled_rgb.tolist(),
                    "sampled_hsv": sampled_hsv.tolist(),
                    **extra,
                }

            applied_rgba = self._apply_visual_color(
                body_id=body_id,
                link_idx=link_idx,
                rgb=sampled_rgb,
                alpha=apply_alpha,
                specular=specular,
            )
            link_cfg["rgbaColor"] = applied_rgba
            link_cfg["specularColor"] = specular
            link_cfgs.append(link_cfg)

        cfg = {
            "mode": "bounded" if mode in ("bounded", "jitter", "bounded_jitter") else "recolor",
            "body_id": int(body_id),
            "link_cfgs": link_cfgs,
            "specularColor": specular,
        }
        self.last_cfg = cfg
        return cfg

    def reset(self):
        """Reset only the randomizer state record.

        This does not restore a body's previous color. In this pipeline, target
        objects are newly loaded inside make_scene(), so resetting the stored
        config is sufficient.
        """
        self.last_cfg = None
