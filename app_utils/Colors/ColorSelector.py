import matplotlib as mpl
import numpy as np


class ColorSelector:
    def __init__(self, colormap, is_dark_mode=False):
        self.colormap = colormap
        self.is_dark_mode = is_dark_mode
        self.brightness_threshold = 0.6 if is_dark_mode == 'light' else 0.5
        self.suitable_colors = self._precompute_colors()

    def _precompute_colors(self, n_samples=1000):
        """Precompute suitable colors"""
        samples = np.linspace(0, 1, n_samples)
        colors = [self.colormap(x) for x in samples]
        return [c for c in colors if self._is_suitable(c)]

    def _is_suitable(self, color):
        brightness = perceived_brightness(color)
        if self.is_dark_mode:
            return brightness > self.brightness_threshold
        else:
            return brightness < self.brightness_threshold

    def get_color(self, rng):
        """Get random suitable color"""
        return rng.choice(self.suitable_colors)


def perceived_brightness(color):
    """Calculate perceived brightness of a color (0-1 scale)"""
    # Convert to RGB if color is in other format
    if isinstance(color, str):  # Hex color
        color = mpl.colors.to_rgb(color)
    elif len(color) == 4:  # RGBA
        color = color[:3]

    # Weighted sum of RGB channels
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
