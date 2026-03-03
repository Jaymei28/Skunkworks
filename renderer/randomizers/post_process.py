"""
post_process.py
===============
Post-processing overrides for synthetic data generation.

Supported effects (all are individually configurable and randomizable):

    ┌──────────────────┬───────────────────────────────────────────────────────┐
    │ Effect           │ Description                                           │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Exposure         │ Global brightness multiplier (EV stops).              │
    │ Bloom            │ Glow around bright regions (Gaussian convolution).    │
    │ Noise            │ Additive luminance / chroma noise (small or large).   │
    │ Ambient_Occlusion│ Darkening of flat / shadowed regions (SSAO approx).  │
    │ White_Balance    │ Cool ↔ Warm tint via per-channel scaling.             │
    │ Blur             │ Gaussian barrel blur over the whole frame.            │
    └──────────────────┴───────────────────────────────────────────────────────┘

Usage::

    pp = PostProcessRandomizer()

    # Randomize all effects and apply
    pp.randomize()
    output = pp.apply(composite)   # composite: (H, W, 3) float tensor [0, 1]

    # Or apply only specific effects
    output = pp.apply(composite, effects=["exposure", "noise"])
"""

import random
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, radius: int, device) -> torch.Tensor:
    """Create a 1-D Gaussian kernel tensor (1, 1, radius*2+1)."""
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(xs**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def _gaussian_blur_2d(
    image: torch.Tensor, sigma: float, radius: int
) -> torch.Tensor:
    """
    Separable Gaussian blur on an (H, W, 3) float tensor.
    Returns the same shape and dtype.
    """
    device = image.device
    # image: (H, W, 3) → (1, 3, H, W) for F.conv2d
    x = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    k = _gaussian_kernel_1d(sigma, radius, device)  # (1, 1, 2r+1)

    # Horizontal pass: treat each channel independently
    k_h = k.repeat(3, 1, 1).unsqueeze(-1)   # (3, 1, k, 1)  — actually we need (3,1,1,k)
    k_h = k.view(1, 1, 1, -1).repeat(3, 1, 1, 1)  # (3, 1, 1, width_kernel)
    pad_h = (0, radius, 0, 0)
    x_padded = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x_padded, k_h, groups=3)

    # Vertical pass
    k_v = k.view(1, 1, -1, 1).repeat(3, 1, 1, 1)  # (3, 1, height_kernel, 1)
    x_padded = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x_padded, k_v, groups=3)

    # Back to (H, W, 3)
    return x.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PostProcessRandomizer:
    """
    Randomizes and applies a suite of post-processing overrides to a
    composited image tensor.

    Each effect has its own ``_range`` parameter that controls the uniform
    sampling bounds.  All effects can be toggled on/off via ``enabled``.

    Parameters
    ----------
    exposure_range : (min_ev, max_ev)
        EV (exposure value) shift.  0 = neutral; ±1 = ×2 or ×0.5 brightness.
    bloom_intensity_range : (min, max)
        Additive weight of the blurred-bright-pixels layer [0, 1].
    bloom_threshold : float
        Pixel luminance above which bloom is applied (in [0, 1]).
    noise_mode : {"small", "large", "random"}
        Noise grain size.  ``"random"`` picks uniformly each call.
    noise_intensity_range : (min, max)
        Std-dev of the Gaussian noise added per channel.
    ao_intensity_range : (min, max)
        Darken flat-region approximation strength [0, 1].
    white_balance_temp_range : (min_k, max_k)
        Colour temperature in Kelvin.  6500 K = daylight / neutral.
        Lower → warmer; higher → cooler.
    blur_sigma_range : (min, max)
        Std-dev of the full-frame Gaussian blur in pixels.  0 = off.
    enabled : dict | None
        Map of effect-name → bool. All enabled by default.
    """

    EFFECT_NAMES = [
        "exposure",
        "bloom",
        "noise",
        "ambient_occlusion",
        "white_balance",
        "blur",
    ]

    def __init__(
        self,
        exposure_range:          tuple = (-1.0, 1.0),
        bloom_intensity_range:   tuple = (0.0, 0.35),
        bloom_threshold:         float = 0.75,
        noise_mode:              str   = "random",
        noise_intensity_range:   tuple = (0.0, 0.08),
        ao_intensity_range:      tuple = (0.0, 0.40),
        white_balance_temp_range: tuple = (3200, 9000),
        blur_sigma_range:        tuple = (0.0, 2.5),
        enabled:                 dict  | None = None,
    ):
        self.exposure_range          = exposure_range
        self.bloom_intensity_range   = bloom_intensity_range
        self.bloom_threshold         = bloom_threshold
        self.noise_mode              = noise_mode
        self.noise_intensity_range   = noise_intensity_range
        self.ao_intensity_range      = ao_intensity_range
        self.white_balance_temp_range = white_balance_temp_range
        self.blur_sigma_range        = blur_sigma_range

        # Enable / disable individual effects
        if enabled is None:
            self.enabled = {name: True for name in self.EFFECT_NAMES}
        else:
            self.enabled = {name: enabled.get(name, True) for name in self.EFFECT_NAMES}

        # Current sampled values (set by randomize())
        self.params: dict = {}
        self.randomize()

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------

    def randomize(self) -> dict:
        """
        Sample a new random value for every *enabled* effect.

        Returns:
            dict mapping effect name → sampled parameter value(s).
        """
        noise_mode = (
            random.choice(["small", "large"])
            if self.noise_mode == "random"
            else self.noise_mode
        )

        self.params = {
            "exposure":          random.uniform(*self.exposure_range),
            "bloom_intensity":   random.uniform(*self.bloom_intensity_range),
            "noise_intensity":   random.uniform(*self.noise_intensity_range),
            "noise_mode":        noise_mode,
            "ao_intensity":      random.uniform(*self.ao_intensity_range),
            "white_balance_temp": random.uniform(*self.white_balance_temp_range),
            "blur_sigma":        random.uniform(*self.blur_sigma_range),
        }
        return self.params

    def set_params(self, **kwargs):
        """Override individual params manually before calling apply()."""
        self.params.update(kwargs)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        image:   torch.Tensor,
        effects: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Apply post-processing effects to an image.

        Args:
            image:   (H, W, 3) float tensor, values in [0, 1].
            effects: Optional list of effect names to apply (respects
                     ``self.enabled`` flags).  If None, all enabled
                     effects are applied in order.

        Returns:
            (H, W, 3) float tensor, clamped to [0, 1].
        """
        active = effects if effects is not None else self.EFFECT_NAMES

        for name in active:
            if not self.enabled.get(name, True):
                continue
            image = self._dispatch(image, name)

        return torch.clamp(image, 0.0, 1.0)

    def _dispatch(self, image: torch.Tensor, name: str) -> torch.Tensor:
        dispatch = {
            "exposure":          self._apply_exposure,
            "bloom":             self._apply_bloom,
            "noise":             self._apply_noise,
            "ambient_occlusion": self._apply_ao,
            "white_balance":     self._apply_white_balance,
            "blur":              self._apply_blur,
        }
        fn = dispatch.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown post-process effect '{name}'. "
                f"Valid: {list(dispatch.keys())}"
            )
        return fn(image)

    # ------------------------------------------------------------------
    # Individual effects
    # ------------------------------------------------------------------

    # --- 1. Exposure ---
    def _apply_exposure(self, image: torch.Tensor) -> torch.Tensor:
        """
        Multiply luminance by 2^EV.
        EV=0 → no change.  EV=+1 → 2× brighter.  EV=−1 → 0.5× darker.
        """
        ev = self.params.get("exposure", 0.0)
        multiplier = 2.0 ** ev
        return image * multiplier

    # --- 2. Bloom ---
    def _apply_bloom(self, image: torch.Tensor) -> torch.Tensor:
        """
        Additive glow generated by blurring bright regions.
        """
        intensity = self.params.get("bloom_intensity", 0.0)
        if intensity < 1e-4:
            return image

        threshold = self.bloom_threshold
        # Extract bright pixels
        bright = (image - threshold).clamp(0, 1)
        # Blur them (large radius for a glow look)
        sigma  = 8.0
        radius = int(3 * sigma)
        glow   = _gaussian_blur_2d(bright, sigma=sigma, radius=radius)
        return image + glow * intensity

    # --- 3. Noise ---
    def _apply_noise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Additive Gaussian noise.
        ``noise_mode = "small"`` adds fine grain (each pixel independent).
        ``noise_mode = "large"`` adds coarser grain (2×2 blocks).
        """
        std  = self.params.get("noise_intensity", 0.0)
        mode = self.params.get("noise_mode", "small")
        if std < 1e-5:
            return image

        H, W, _ = image.shape
        device   = image.device

        if mode == "large":
            # Generate at half resolution, then upscale for coarser grain
            h2, w2 = max(1, H // 2), max(1, W // 2)
            noise_small = torch.randn(h2, w2, 3, device=device) * std
            # Nearest-neighbor upsample
            noise = (
                noise_small
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            noise = F.interpolate(noise, size=(H, W), mode="nearest")
            noise = noise.squeeze(0).permute(1, 2, 0)
        else:  # "small"
            noise = torch.randn(H, W, 3, device=device) * std

        return image + noise

    # --- 4. Ambient Occlusion (SSAO approximation) ---
    def _apply_ao(self, image: torch.Tensor) -> torch.Tensor:
        """
        Screen-space AO approximation: compare a pixel's brightness to its
        local neighbourhood average.  Pixels darker than neighbours are
        deemed 'occluded' and darkened further.
        """
        intensity = self.params.get("ao_intensity", 0.0)
        if intensity < 1e-4:
            return image

        # Convert to NHWC → NCHW for blurring
        sigma  = 10.0
        radius = int(3 * sigma)
        blurred = _gaussian_blur_2d(image, sigma=sigma, radius=radius)

        # AO factor: ratio of pixel to neighbourhood mean
        lum         = image.mean(dim=-1, keepdim=True)
        lum_blurred = blurred.mean(dim=-1, keepdim=True)
        # Ratio < 1 means locally darker (occluded)
        ao_factor = (lum / (lum_blurred + 1e-6)).clamp(0, 1)

        # Apply: darken by ao_factor weighted by intensity
        ao_factor = 1.0 - (1.0 - ao_factor) * intensity
        return image * ao_factor

    # --- 5. White Balance ---
    def _apply_white_balance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Shift colour temperature via per-channel multipliers.
        Uses a simplified Planckian locus approximation.

        colour_temp < 6500 K → warmer (more red/green, less blue)
        colour_temp > 6500 K → cooler (more blue, less red)
        """
        temp = self.params.get("white_balance_temp", 6500.0)

        r_gain, g_gain, b_gain = _kelvin_to_rgb_gains(temp)
        gains = torch.tensor(
            [r_gain, g_gain, b_gain],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 1, 3)

        return image * gains

    # --- 6. Blur ---
    def _apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Full-frame Gaussian blur (lens defocus / motion blur proxy)."""
        sigma = self.params.get("blur_sigma", 0.0)
        if sigma < 0.1:
            return image

        radius = max(1, int(3.0 * sigma))
        return _gaussian_blur_2d(image, sigma=sigma, radius=radius)


# ---------------------------------------------------------------------------
# Colour temperature helper
# ---------------------------------------------------------------------------

def _kelvin_to_rgb_gains(kelvin: float) -> tuple[float, float, float]:
    """
    Approximate RGB multipliers for a given colour temperature in Kelvin.
    Normalised so that 6500 K ≈ (1, 1, 1).

    Based on Tanner Helland's algorithm.
    Returns:
        (r, g, b) gains each in roughly [0.5, 1.5].
    """
    k = kelvin / 100.0

    # --- Red ---
    if k <= 66:
        r = 1.0
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592) / 255.0

    # --- Green ---
    if k <= 66:
        g = (99.4708025861 * math.log(k) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((k - 60) ** -0.0755148492) / 255.0

    # --- Blue ---
    if k >= 66:
        b = 1.0
    elif k <= 19:
        b = 0.0
    else:
        b = (138.5177312231 * math.log(k - 10) - 305.0447927307) / 255.0

    # Clamp to valid range
    r = max(0.01, min(1.5, r))
    g = max(0.01, min(1.5, g))
    b = max(0.01, min(1.5, b))

    # Normalise so 6500 K → (1, 1, 1)
    r_ref, g_ref, b_ref = _kelvin_raw(6500)
    return r / r_ref, g / g_ref, b / b_ref


def _kelvin_raw(kelvin: float) -> tuple[float, float, float]:
    """Raw un-normalised RGB for a given Kelvin (used internally)."""
    k = kelvin / 100.0
    r = 1.0 if k <= 66 else 329.698727446 * ((k - 60) ** -0.1332047592) / 255.0
    if k <= 66:
        g = (99.4708025861 * math.log(k) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((k - 60) ** -0.0755148492) / 255.0
    b = 1.0 if k >= 66 else (0.0 if k <= 19 else (138.5177312231 * math.log(k - 10) - 305.0447927307) / 255.0)
    return max(0.01, r), max(0.01, g), max(0.01, b)
