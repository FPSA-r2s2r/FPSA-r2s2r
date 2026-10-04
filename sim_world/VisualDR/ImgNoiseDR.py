import cv2
import numpy as np


class ImgNoiseDR(object):
    """Camera-specific image noise / photometric domain randomization.

    This class is intentionally image-level rather than object-level:
    - Object/material color should still be randomized through ObjectColorDR.
    - This class models per-camera sensor differences such as white balance,
      exposure, gamma, saturation, hue, color rendering matrix, vignette,
      blur, and sensor noise.

    The sampled parameters are held fixed after sample_image_noise_randomization()
    is called. In your data collection pipeline this means one episode/scene gets
    a consistent camera response instead of frame-by-frame flicker.
    """

    def __init__(self, bullet_client, seed=42, camera_name="camera"):
        self.bullet_client = bullet_client
        self.camera_name = str(camera_name)
        self.ImgNoise_rng = np.random.default_rng(seed)
        self.image_noise_cfg = None

    def reset(self):
        self.image_noise_cfg = None

    @staticmethod
    def _valid_odd_kernels(blur_kernel_choices):
        blur_kernel_choices = tuple(int(k) for k in blur_kernel_choices)
        blur_kernel_choices = tuple(k for k in blur_kernel_choices if k >= 3 and k % 2 == 1)
        return blur_kernel_choices if len(blur_kernel_choices) > 0 else (3,)

    def _sample_color_matrix(self, strength):
        """Sample a mild RGB color-rendering matrix.

        rgb_gain is mostly white balance / color temperature. This matrix models
        a stronger camera ISP / sensor spectral-response mismatch by mixing RGB
        channels. It is intentionally kept close to identity so geometry and
        object identity are preserved.
        """
        strength = float(max(strength, 0.0))
        if strength <= 1e-8:
            return np.eye(3, dtype=np.float32)

        rng = self.ImgNoise_rng
        matrix = np.eye(3, dtype=np.float32)

        # Diagonal response variation plus off-diagonal channel mixing.
        diag_delta = rng.uniform(-0.75 * strength, 0.75 * strength, size=3)
        off_delta = rng.uniform(-strength, strength, size=(3, 3))
        np.fill_diagonal(off_delta, 0.0)

        matrix += np.diag(diag_delta).astype(np.float32)
        matrix += off_delta.astype(np.float32)

        # Avoid extreme sign inversions / channel explosions.
        matrix = np.clip(matrix, -0.25, 1.45).astype(np.float32)
        return matrix

    def sample_image_noise_randomization(
        self,
        brightness_range=(-14.0, 14.0),      # additive RGB offset in [0, 255]
        contrast_range=(0.85, 1.15),         # contrast around gray 0.5
        gamma_range=(0.88, 1.15),            # <1 brighter, >1 darker
        saturation_range=(0.85, 1.18),       # HSV saturation multiplier
        rgb_gain_range=(0.92, 1.08),         # per-channel white-balance gain
        hue_shift_deg_range=(0.0, 0.0),      # global hue rotation in degrees
        color_matrix_strength_range=(0.0, 0.0),  # RGB channel mixing strength
        gray_mix_range=(0.0, 0.0),           # mix toward grayscale / mono response
        vignette_strength_range=(0.0, 0.0),  # radial darkening, useful for fisheye
        gaussian_std_range=(0.0, 5.0),       # pixel noise std in [0, 255]
        salt_pepper_prob_range=(0.0, 0.003),
        blur_prob_range=(0.0, 0.20),
        blur_kernel_choices=(3,),
    ):
        """Sample one fixed photometric/noise configuration.

        Args are ranges so the same class can be reused for different cameras.
        The strongest camera-to-camera color mismatch comes from rgb_gain,
        hue_shift, color_matrix, saturation, gamma, and gray_mix.
        """
        rng = self.ImgNoise_rng

        rgb_gain_lo, rgb_gain_hi = rgb_gain_range
        blur_kernel_choices = self._valid_odd_kernels(blur_kernel_choices)

        color_matrix_strength = float(rng.uniform(*color_matrix_strength_range))

        self.image_noise_cfg = {
            "camera_name": self.camera_name,
            "brightness": float(rng.uniform(*brightness_range)),
            "contrast": float(rng.uniform(*contrast_range)),
            "gamma": float(rng.uniform(*gamma_range)),
            "saturation": float(rng.uniform(*saturation_range)),
            "rgb_gain": rng.uniform(float(rgb_gain_lo), float(rgb_gain_hi), size=3).astype(np.float32),
            "hue_shift_deg": float(rng.uniform(*hue_shift_deg_range)),
            "color_matrix_strength": color_matrix_strength,
            "color_matrix": self._sample_color_matrix(color_matrix_strength),
            "gray_mix": float(rng.uniform(*gray_mix_range)),
            "vignette_strength": float(rng.uniform(*vignette_strength_range)),
            "gaussian_std": float(rng.uniform(*gaussian_std_range)),
            "salt_pepper_prob": float(rng.uniform(*salt_pepper_prob_range)),
            "blur_prob": float(rng.uniform(*blur_prob_range)),
            "blur_ksize": int(rng.choice(blur_kernel_choices)),
        }
        return self.image_noise_cfg

    @staticmethod
    def _apply_hsv_adjustment(img_float01, saturation=1.0, hue_shift_deg=0.0):
        """Apply HSV saturation scaling and hue rotation on RGB float image."""
        img_float01 = np.clip(img_float01, 0.0, 1.0).astype(np.float32)
        hsv = cv2.cvtColor(img_float01, cv2.COLOR_RGB2HSV)

        # OpenCV float HSV uses H in degrees [0, 360), S/V in [0, 1].
        hsv[..., 0] = (hsv[..., 0] + float(hue_shift_deg)) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0.0, 1.0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _apply_gray_mix(img_float01, gray_mix):
        """Mix RGB image toward luma grayscale."""
        gray_mix = float(np.clip(gray_mix, 0.0, 1.0))
        if gray_mix <= 1e-8:
            return img_float01

        # Rec. 601 luma is close enough for camera ISP augmentation.
        luma = (
            0.299 * img_float01[..., 0]
            + 0.587 * img_float01[..., 1]
            + 0.114 * img_float01[..., 2]
        )[..., None]
        return (1.0 - gray_mix) * img_float01 + gray_mix * luma

    @staticmethod
    def _apply_vignette(img_float01, strength):
        """Apply radial darkening. Useful for fisheye / wide-angle lenses."""
        strength = float(np.clip(strength, 0.0, 0.95))
        if strength <= 1e-8:
            return img_float01

        h, w = img_float01.shape[:2]
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h, dtype=np.float32),
            np.linspace(-1.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        rr2 = xx * xx + yy * yy
        rr2 = rr2 / max(float(rr2.max()), 1e-6)
        mask = 1.0 - strength * rr2
        mask = np.clip(mask, 1.0 - strength, 1.0)[..., None]
        return img_float01 * mask

    def apply_image_noise(self, rgb):
        if self.image_noise_cfg is None:
            return rgb

        cfg = self.image_noise_cfg
        rng = self.ImgNoise_rng

        # Work in normalized RGB for camera-response transforms.
        img = rgb.astype(np.float32) / 255.0

        # 1) Per-channel gain models white-balance / sensor color response.
        rgb_gain = np.asarray(cfg.get("rgb_gain", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
        img = img * rgb_gain
        img = np.clip(img, 0.0, 1.0)

        # 2) Camera color-rendering matrix. This gives stronger camera-specific
        # color changes than white balance alone.
        color_matrix = np.asarray(cfg.get("color_matrix", np.eye(3)), dtype=np.float32).reshape(3, 3)
        img = np.tensordot(img, color_matrix.T, axes=1)
        img = np.clip(img, 0.0, 1.0)

        # 3) Hue and saturation model ISP color rendering differences.
        img = self._apply_hsv_adjustment(
            img,
            saturation=cfg.get("saturation", 1.0),
            hue_shift_deg=cfg.get("hue_shift_deg", 0.0),
        )

        # 4) Optional mix toward grayscale/monochrome response.
        img = self._apply_gray_mix(img, cfg.get("gray_mix", 0.0))
        img = np.clip(img, 0.0, 1.0)

        # 5) Contrast and brightness/exposure-like offset.
        contrast = float(cfg.get("contrast", 1.0))
        brightness = float(cfg.get("brightness", 0.0)) / 255.0
        img = (img - 0.5) * contrast + 0.5 + brightness
        img = np.clip(img, 0.0, 1.0)

        # 6) Gamma response. gamma < 1 brightens; gamma > 1 darkens.
        gamma = max(float(cfg.get("gamma", 1.0)), 1e-6)
        img = np.power(img, gamma)
        img = np.clip(img, 0.0, 1.0)

        # 7) Radial falloff for fisheye / wide-angle cameras.
        img = self._apply_vignette(img, cfg.get("vignette_strength", 0.0))
        img = np.clip(img, 0.0, 1.0)

        # 8) Sensor noise in pixel space.
        img = img * 255.0
        gaussian_std = float(cfg.get("gaussian_std", 0.0))
        if gaussian_std > 0.0:
            img += rng.normal(0.0, gaussian_std, size=img.shape)

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        # 9) Mild defocus / motion-like blur.
        if rng.random() < float(cfg.get("blur_prob", 0.0)):
            ksize = int(cfg.get("blur_ksize", 3))
            if ksize >= 3 and ksize % 2 == 1:
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 10) Rare dead/hot pixels.
        sp_prob = float(cfg.get("salt_pepper_prob", 0.0))
        if sp_prob > 0.0:
            mask = rng.random(img.shape[:2])
            img[mask < sp_prob / 2.0] = 0
            img[(mask >= sp_prob / 2.0) & (mask < sp_prob)] = 255

        return img