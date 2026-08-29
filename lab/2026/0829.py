import argparse
import gc
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import jv

from common.image_processing import ImageProcessingSettings
from common.technology import images_to_hdr_video

# Edit this list to change the colourmap used by the entire plot.
# Colours are interpolated smoothly between these anchor colours.
PALETTE = [
    "#f43227",  # red anchor (yours, unchanged)
    "#fd654e",
    "#ff8d76",
    "#ffb19f",
    "#ffd4ca",
    "#f6f6f6",  # neutral midpoint
    "#cdc8df",
    "#a39cc8",
    "#7971b1",
    "#4d4a99",
    "#0b2582",  # blue anchor (yours, unchanged)
]


def generate(settings: ImageProcessingSettings = None, hdr: bool = False):
    """Hyperbolic Chladni Labyrinths (12s 60fps 9:16 Vertical Loop).

    Mathematical Concept:
        Embeds standing-wave 2D vibration nodal curves (Chladni/Bessel eigenfunctions)
        into the complex Poincaré disk model of hyperbolic geometry. The coordinates
        undergo continuous time-dependent conformal Möbius automorphisms:
            w(z, t) = exp(i * theta(t)) * (z - alpha(t)) / (1 - conj(alpha(t)) * z)
        where alpha(t) = r0 * exp(i * t) orbits in the unit disk and theta(t) rotates
        the hyperbolic frame. Rendered in vertical 9:16 format and repeated across 3
        seamless loops (720 frames at 60 fps = 12.0s).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    bg_color = "k"
    cmap = LinearSegmentedColormap.from_list(
        "hyperbolic_labyrinth",
        PALETTE,
        N=256,
    )
    palette_colors = [
        cmap(i / max(1, len(PALETTE) - 1)) for i in range(len(PALETTE))
    ]

    # Grid setup (optimized for high-fps vector contouring)
    res = 380
    x = np.linspace(-1.06, 1.06, res)
    y = np.linspace(-1.06, 1.06, res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    R = np.abs(Z)
    mask = R < 0.992

    # Fixed harmonic vibration modes
    modes = [
        (rng.integers(2, 6), rng.integers(1, 5)),
        (rng.integers(4, 9), rng.integers(2, 6)),
        (rng.integers(6, 12), rng.integers(1, 4)),
    ]
    weights = rng.uniform(0.7, 1.3, size=len(modes))
    base_phases = rng.uniform(0, 2 * np.pi, size=len(modes))

    # Precomputed boundary circle geometry
    boundary_theta = np.linspace(0, 2 * np.pi, 300)
    glow_radii = np.linspace(0.975, 0.995, 6)

    # Contour levels setup
    num_levels = 32
    level_colors = [cmap(i / max(1, num_levels - 1)) for i in range(num_levels)]
    base_lws = np.array(
        [0.5 + 1.2 * np.sin(np.pi * i / num_levels) for i in range(num_levels)]
    )

    # Video setup: 60 fps for 4 seconds = 240 frames
    fps = 60
    num_frames = 340
    min_shift_radius = 0.35
    max_shift_radius = 0.75
    radial_cycles = 1  # Must be an integer for a seamless loop.

    # Reusable Figure with 9:16 vertical aspect ratio (1080x1920 at dpi=120)
    fig = plt.figure(figsize=(9, 16), dpi=120)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=bg_color)

    for frame_idx in range(num_frames):
        # Parametric time for seamless periodic loop [0, 2*pi)
        t = 2 * np.pi * frame_idx / num_frames

        # Conformal Möbius parameters.
        # The radius oscillates periodically between the configured limits.
        radius_midpoint = (min_shift_radius + max_shift_radius) / 2
        radius_amplitude = (max_shift_radius - min_shift_radius) / 2
        shift_radius = radius_midpoint - radius_amplitude * np.cos(
            radial_cycles * t
        )
        alpha = shift_radius * np.exp(1j * t)
        theta = t

        # Conformal mapping
        W = (Z - alpha) / (1 - np.conj(alpha) * Z) * np.exp(1j * theta)
        R_w = np.abs(W)
        Phi_w = np.angle(W)

        # Superpose Bessel + Chladni harmonics
        field = np.zeros_like(X)
        for (n, m), w_weight, ph in zip(modes, weights, base_phases):
            bessel_root_approx = (m + 2 * n) * 1.8
            radial = jv(m, bessel_root_approx * np.clip(R_w, 0, 1.5))
            angular = np.cos(n * Phi_w + ph)

            u = np.real(W) * 3.5
            v = np.imag(W) * 3.5
            chladni_cartesian = np.sin(n * u) * np.sin(m * v) - np.sin(m * u) * np.sin(
                n * v
            )
            field += w_weight * (0.65 * radial * angular + 0.35 * chladni_cartesian)

        field_masked = np.where(mask, field, np.nan)

        levels = np.linspace(
            np.nanpercentile(field_masked, 5),
            np.nanpercentile(field_masked, 95),
            num_levels,
        )

        ax.cla()
        ax.set_facecolor(bg_color)
        ax.axis("off")
        ax.set_aspect("equal")

        # Set 9:16 vertical framing with centered unit circle
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.92, 1.92)

        # Fast vectorized multi-pass glowing contours (3 calls total)
        # Outer diffuse glow
        ax.contour(
            X,
            Y,
            field_masked,
            levels=levels,
            colors=level_colors,
            linewidths=base_lws * 2.8,
            alpha=0.18,
        )
        # Mid aura
        ax.contour(
            X,
            Y,
            field_masked,
            levels=levels,
            colors=level_colors,
            linewidths=base_lws * 1.6,
            alpha=0.40,
        )
        # Sharp core line
        ax.contour(
            X,
            Y,
            field_masked,
            levels=levels,
            colors=level_colors,
            linewidths=base_lws,
            alpha=0.95,
        )

        # Highlight zero-nodal curves with neon bloom
        for z_lw, z_col, z_alpha in [
            # Coloured bloom sits below the white HDR highlight layer.
            (7.0, palette_colors[2 % len(palette_colors)], 0.18),
            (3.8, palette_colors[0 % len(palette_colors)], 0.45),
            (1.8, "#ffffff", 1.0),
        ]:
            ax.contour(
                X,
                Y,
                field_masked,
                levels=[0.0],
                colors=[z_col],
                linewidths=[z_lw],
                alpha=z_alpha,
            )

        # Perimeter horizon glow rings
        for r_glow in glow_radii:
            ring_x = r_glow * np.cos(boundary_theta)
            ring_y = r_glow * np.sin(boundary_theta)
            ax.plot(
                ring_x,
                ring_y,
                color=cmap(frame_idx / max(1, num_frames - 1)),
                lw=1.0,
                alpha=0.45 * (r_glow - 0.97) / 0.025,
            )

        # Outer boundary disk with glow halo
        for halo_lw, halo_alpha in [(4.5, 0.20), (2.4, 0.40), (1.2, 0.95)]:
            ax.plot(
                np.cos(boundary_theta) * 0.995,
                np.sin(boundary_theta) * 0.995,
                color="#ffffff" if halo_lw < 2.0 else palette_colors[0],
                lw=halo_lw,
                alpha=halo_alpha,
            )

        fig.savefig(
            settings.frames_path / f"frame{frame_idx:04d}.png",
            facecolor=bg_color,
        )

    plt.close(fig)
    gc.collect()

    # Repeat loop 3 times (240 unique frames * 3 = 720 frames = 12.0s at 60 fps)
    num_repeats = 4
    for repeat in range(1, num_repeats):
        for i in range(num_frames):
            src = settings.frames_path / f"frame{i:04d}.png"
            dst = settings.frames_path / f"frame{repeat * num_frames + i:04d}.png"
            shutil.copyfile(src, dst)

    # Encode all 720 frames to 60 fps MP4 video (12.0 seconds).
    if hdr or getattr(settings, "hdr", False):
        images_to_hdr_video(
            settings.frames_path,
            f"{settings.filename}_HDR_HLG.mp4",
            fps=fps,
        )
    else:
        settings.save_video(fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the 0829 plot.")
    parser.add_argument(
        "--hdr",
        action="store_true",
        help="also encode a 1080x1920 HEVC Main 10 Rec.2020 HLG video",
    )
    args = parser.parse_args()
    generate(hdr=args.hdr)
