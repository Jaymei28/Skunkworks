import torch
import random
import math


# ---------------------------------------------------------------------------
# Weather types
# ---------------------------------------------------------------------------
WEATHER_CLEAR    = "clear"
WEATHER_RAIN     = "rain"
WEATHER_FOG      = "fog"
WEATHER_DUST     = "dust"
WEATHER_OVERCAST = "overcast"

ALL_WEATHER_TYPES = [
    WEATHER_CLEAR,
    WEATHER_RAIN,
    WEATHER_FOG,
    WEATHER_DUST,
    WEATHER_OVERCAST,
]


class WeatherRandomizer:
    """
    Applies procedural weather effects to a composited image tensor.

    Supported weather types:
        - ``clear``    – no effect applied.
        - ``rain``     – vertical streak overlay + darkened sky tint.
        - ``fog``      – additive white/grey fog based on depth-aware gradient.
        - ``dust``     – warm orange-ochre fog / reduced visibility.
        - ``overcast`` – desaturates + brightens slightly (flat lighting).

    Usage::

        weather = WeatherRandomizer(
            weights={
                "clear": 0.30,
                "rain":  0.20,
                "fog":   0.20,
                "dust":  0.15,
                "overcast": 0.15,
            }
        )

        # In the generation loop:
        weather_type, intensity = weather.randomize()
        composite = weather.apply(composite, weather_type, intensity)
    """

    def __init__(
        self,
        weights: dict | None = None,
        intensity_range: tuple = (0.2, 0.85),
    ):
        """
        Args:
            weights: Dict mapping weather type name → sampling weight.
                     All weather types default to equal weight if None.
            intensity_range: (min, max) float for weather effect strength.
        """
        if weights is None:
            weights = {w: 1.0 for w in ALL_WEATHER_TYPES}

        self.weather_types = list(weights.keys())
        self.weights       = list(weights.values())
        self.intensity_range = intensity_range

        # Current state (set by randomize())
        self.current_type      = WEATHER_CLEAR
        self.current_intensity = 0.0

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------

    def randomize(self):
        """
        Picks a random weather type and intensity.

        Returns:
            tuple(str, float): (weather_type, intensity)
        """
        self.current_type = random.choices(
            self.weather_types, weights=self.weights, k=1
        )[0]
        self.current_intensity = random.uniform(*self.intensity_range)
        return self.current_type, self.current_intensity

    def set(self, weather_type: str, intensity: float):
        """Manually set weather type and intensity."""
        if weather_type not in ALL_WEATHER_TYPES:
            raise ValueError(
                f"Unknown weather type '{weather_type}'. "
                f"Choose from {ALL_WEATHER_TYPES}"
            )
        self.current_type      = weather_type
        self.current_intensity = intensity

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        image: torch.Tensor,
        weather_type: str | None = None,
        intensity:    float | None = None,
    ) -> torch.Tensor:
        """
        Apply weather effect to an image tensor.

        Args:
            image:        (H, W, 3) float tensor in [0, 1] range.
            weather_type: Override type (uses ``current_type`` if None).
            intensity:    Override intensity (uses ``current_intensity`` if None).

        Returns:
            (H, W, 3) float tensor, weather-composited, clamped to [0, 1].
        """
        wtype = weather_type if weather_type is not None else self.current_type
        alpha = intensity    if intensity    is not None else self.current_intensity
        device = image.device

        if wtype == WEATHER_CLEAR:
            return image

        if wtype == WEATHER_RAIN:
            return self._apply_rain(image, alpha, device)

        if wtype == WEATHER_FOG:
            return self._apply_fog(image, alpha, device)

        if wtype == WEATHER_DUST:
            return self._apply_dust(image, alpha, device)

        if wtype == WEATHER_OVERCAST:
            return self._apply_overcast(image, alpha, device)

        return image

    # ------------------------------------------------------------------
    # Private effect implementations
    # ------------------------------------------------------------------

    def _apply_rain(
        self, image: torch.Tensor, intensity: float, device
    ) -> torch.Tensor:
        """
        Simulate rain via:
          1. Random vertical/slightly-angled bright streaks.
          2. Overall cool-blue darkening tint.
        """
        H, W, C = image.shape

        # --- Rain streaks ---
        num_streaks = int(intensity * 800)
        streak_layer = torch.zeros(H, W, device=device)

        for _ in range(num_streaks):
            x    = random.randint(0, W - 1)
            y0   = random.randint(0, H - 1)
            # Streak length: 5–25 px
            length = random.randint(5, 25)
            # Slight wind angle (±2 px horizontal drift for every 10 px)
            dx_per_10 = random.uniform(-2, 2)

            for dy in range(length):
                yy = y0 + dy
                xx = x + int(dy * dx_per_10 / 10)
                if 0 <= yy < H and 0 <= xx < W:
                    # Bright, semi-transparent streak
                    streak_layer[yy, xx] = random.uniform(0.4, 0.9)

        streak_rgb = streak_layer.unsqueeze(-1).expand(-1, -1, 3)

        # --- Dark, cool-blue tint ---
        tint = torch.tensor(
            [0.7, 0.75, 0.85], device=device
        ).view(1, 1, 3)

        rain_weight = intensity * 0.45
        image = image * (tint * (1 - rain_weight) + tint * rain_weight)
        # Blend streaks
        image = torch.clamp(image + streak_rgb * intensity * 0.3, 0, 1)
        return image

    def _apply_fog(
        self, image: torch.Tensor, intensity: float, device
    ) -> torch.Tensor:
        """
        Simulate fog using a top-to-bottom gradient density map.
        Fog is near-white (slightly cool).
        """
        H, W, _ = image.shape

        # Gradient: denser at horizon (bottom 60%) than top
        y_lin = torch.linspace(0, 1, H, device=device)
        # Sigmoid-shaped density: heavier near middle–bottom
        fog_density = torch.sigmoid((y_lin - 0.35) * 6) * intensity
        fog_map = fog_density.view(H, 1, 1).expand(H, W, 3)

        fog_color = torch.tensor([0.92, 0.93, 0.96], device=device).view(1, 1, 3)
        image = image * (1 - fog_map) + fog_color * fog_map
        return torch.clamp(image, 0, 1)

    def _apply_dust(
        self, image: torch.Tensor, intensity: float, device
    ) -> torch.Tensor:
        """
        Simulate dust / sandstorm with a warm ochre overlay and
        a slight noise texture to break uniformity.
        """
        H, W, _ = image.shape

        dust_color = torch.tensor([0.85, 0.72, 0.48], device=device).view(1, 1, 3)
        # Add Perlin-like noise: use rand as cheap approximation
        noise = torch.rand(H, W, 1, device=device) * 0.08
        dust_map = (intensity * 0.7 + noise).expand(-1, -1, 3).clamp(0, 1)

        image = image * (1 - dust_map) + dust_color * dust_map
        return torch.clamp(image, 0, 1)

    def _apply_overcast(
        self, image: torch.Tensor, intensity: float, device
    ) -> torch.Tensor:
        """
        Simulate overcast / cloudy sky.
        Desaturates the image and adds a slight bright, grey lift.
        """
        # Desaturate
        grey = image.mean(dim=-1, keepdim=True)
        image = image * (1 - intensity * 0.6) + grey * (intensity * 0.6)

        # Uniform cool grey lift (simulates bounced diffuse sky light)
        lift = torch.tensor([0.88, 0.88, 0.90], device=device).view(1, 1, 3)
        image = image * (1 - intensity * 0.25) + lift * (intensity * 0.25)

        return torch.clamp(image, 0, 1)
