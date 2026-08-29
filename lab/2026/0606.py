from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def random_periods(
    min_modulus: float = 0.5,
    max_modulus: float = 2.5,
    min_angle_separation: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[complex, complex]:
    """
    Return two linearly-independent complex periods (ω1, ω2).

    Linear independence is enforced by requiring their arguments differ by at
    least `min_angle_separation` radians (and not be nearly π apart, which
    would make them real multiples of each other).
    """
    if rng is None:
        rng = np.random.default_rng()

    while True:
        r1, r2 = rng.uniform(min_modulus, max_modulus, size=2)
        θ1, θ2 = rng.uniform(0, 2 * np.pi, size=2)
        ω1 = r1 * np.exp(1j * θ1)
        ω2 = r2 * np.exp(1j * θ2)

        angle_diff = abs((θ2 - θ1 + np.pi) % (2 * np.pi) - np.pi)
        if angle_diff > min_angle_separation and angle_diff < np.pi - min_angle_separation:
            return ω1, ω2
def make_lattice_points(
    n: int,
    omega1: complex = 1.0,
    omega2: complex = 1j,
) -> list[complex]:
    """
    Return all non-zero lattice points m*ω1 + k*ω2 for m,k in [-n, n],
    sorted by modulus.
    """
    integers = range(-n, n + 1)
    points = [
        a * omega1 + b * omega2
        for a in integers
        for b in integers
        if (a, b) != (0, 0)
    ]
    points.sort(key=abs)
    return points
def make_partial_sum(lattice_points: list[complex], n_terms: int):
    """Return a vectorised callable for the n_terms-th partial sum of ℘."""
    active = lattice_points[:n_terms]

    def wp(z: np.ndarray) -> np.ndarray:
        result = 1.0 / z**2
        for omega in active:
            result += 1.0 / (z - omega)**2 - 1.0 / omega**2
        return result

    return wp
def domain_color(f_vals: np.ndarray, modulus_clamp: float = 10.0) -> np.ndarray:
    """
    Convert complex array to RGB via domain coloring:
      - Hue    → argument (angle)
      - Value  → log-modulus brightness bands (zeros are dark, poles are bright)
      - Saturation fixed at 1
    """
    angle = np.angle(f_vals)                          # [-π, π]
    hue = (angle / (2 * np.pi)) % 1.0                # [0, 1)

    mod = np.abs(f_vals)
    mod = np.clip(mod, 1e-10, None)
    # Log-modulus banding: repeating bright/dark rings at each order of magnitude
    log_mod = np.log(mod)
    value = 0.5 + 0.5 * np.sin(log_mod * np.pi)      # oscillates in (0, 1)

    hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
    return mcolors.hsv_to_rgb(hsv)
def plot_partial_sum(
    f,
    re_range: tuple[float, float] = (-7.0, 7.0),
    im_range: tuple[float, float] = (-7.0, 7.0),
    resolution: int = 500,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    re = np.linspace(*re_range, resolution)
    im = np.linspace(*im_range, resolution)
    Re, Im = np.meshgrid(re, im)
    Z = Re + 1j * Im

    with np.errstate(divide="ignore", invalid="ignore"):
        W = f(Z)

    rgb = domain_color(W)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        rgb,
        extent=[*re_range, *im_range],
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    return ax
def _auto_range(omega1: complex, omega2: complex, zoom: float = 4.0) -> tuple[float, float]:
    """Sensible plot range based on the period moduli."""
    half = zoom * max(abs(omega1), abs(omega2))
    return (-half, half)
def make_plots(
    n: int = 6,
    omega1: complex | None = None,
    omega2: complex | None = None,
    re_range: tuple[float, float] | None = None,
    im_range: tuple[float, float] | None = None,
    resolution: int = 500,
    outdir: str = "./weierstrass_frames/",
    n_images: int = 25,
    seed: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    if omega1 is None or omega2 is None:
        omega1, omega2 = random_periods(rng=rng)

    print(f"ω1 = {omega1:.4f},  ω2 = {omega2:.4f}")

    if re_range is None:
        re_range = _auto_range(omega1, omega2)
    if im_range is None:
        im_range = re_range

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    lattice_points = make_lattice_points(n, omega1, omega2)
    print(f"Total lattice points: {len(lattice_points)}")

    for i in range(n_images):
        f = make_partial_sum(lattice_points, i)
        title = f"℘(z)  ω1={omega1:.3f}, ω2={omega2:.3f}  [{i} terms]"
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        plot_partial_sum(f, re_range=re_range, im_range=im_range,
                         resolution=resolution, title=title, ax=ax)
        path = out / f"Weierstrass{i:03d}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
def compare_zoom(
    n: int = 6,
    n_terms: int = 20,
    omega1: complex | None = None,
    omega2: complex | None = None,
    resolution: int = 500,
    seed: int | None = None,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    if omega1 is None or omega2 is None:
        omega1, omega2 = random_periods(rng=rng)

    print(f"ω1 = {omega1:.4f},  ω2 = {omega2:.4f}")

    lattice_points = make_lattice_points(n, omega1, omega2)
    f = make_partial_sum(lattice_points, n_terms)

    outer = _auto_range(omega1, omega2, zoom=4.0)
    inner = _auto_range(omega1, omega2, zoom=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    subtitle = f"ω1 = {omega1:.3f},   ω2 = {omega2:.3f}   ({n_terms} terms)"

    plot_partial_sum(f, re_range=outer, im_range=outer,
                     resolution=resolution, title="zoomed out", ax=axes[0])
    plot_partial_sum(f, re_range=inner, im_range=inner,
                     resolution=resolution, title="zoomed in", ax=axes[1])

    # Mark the fundamental parallelogram on the zoomed-out plot
    corners = np.array([0, omega1, omega1 + omega2, omega2, 0])
    axes[0].plot(corners.real, corners.imag, "w--", lw=1.2, alpha=0.7, label="fund. domain")
    axes[0].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Weierstrass ℘-function — {subtitle}", fontsize=11)
    fig.tight_layout()
    return fig

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    """
    Weierstrass P-function visualizer using matplotlib domain coloring.

    Supports the standard square lattice (ω1=1, ω2=i) and randomly generated
    lattices from two linearly independent complex periods.

    Python 3 compatible.
    """




    # ---------------------------------------------------------------------------
    # Lattice helpers
    # ---------------------------------------------------------------------------





    # ---------------------------------------------------------------------------
    # Weierstrass P-function partial sums
    # ---------------------------------------------------------------------------



    # ---------------------------------------------------------------------------
    # Domain coloring
    # ---------------------------------------------------------------------------



    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------



    # ---------------------------------------------------------------------------
    # Batch generation
    # ---------------------------------------------------------------------------





    # ---------------------------------------------------------------------------
    # Quick side-by-side comparison (zoomed in vs zoomed out)
    # ---------------------------------------------------------------------------



    # ---------------------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------------------

    if __name__ == "__main__":
        # Random lattice, reproducible with seed=
        fig = compare_zoom(n=2, n_terms=100, omega1=1, omega2=1j, resolution=1000, seed=42)
        fig.savefig("weierstrass_preview.png", dpi=120, bbox_inches="tight")
        plt.show()

        # Standard square lattice (original behaviour):
        # fig = compare_zoom(n=6, n_terms=20, omega1=1, omega2=1j)

        # Uncomment to generate full frame sequences with a random lattice:
        # make_plots(6, outdir="./zoomed_out/", n_images=25, seed=42)
        # make_plots(6, re_range=(-1, 1), im_range=(-1, 1),
        #            outdir="./zoomed_in/", n_images=25, seed=42)

if __name__ == '__main__':
    generate()
