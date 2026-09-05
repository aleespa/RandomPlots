"""Gargantua -- a ray-traced thin accretion disc around a Schwarzschild black hole."""

import gc
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates

from common.image_processing import ImageProcessingSettings

# RP_FPS=5 in the environment gives a quick 30-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 6
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 4  # play the rendered loop this many times back-to-back in the mp4
FIGURE_SIZE = (7.2, 12.8)  # 1080x1920 @ 150 dpi -- 9:16 for Reels/Stories
DPI = 150
WIDTH, HEIGHT = 1080, 1920
BG_COLOR = "#000000"

# Blackbody-style heat ramp indexed by tone-mapped brightness: the dim outer
# disc glows deep red, the bright beamed inner edge runs through orange and
# gold into a near-white core.
PALETTE = ["#000000", "#3a0300", "#9e0f00", "#ff3c00", "#ff8a00", "#ffd24d", "#fff6d5"]

N_WORKERS = min(os.cpu_count() or 1, 12)

_STATE = {}


# ----------------------------------------------------------------------------
# Geodesics
# ----------------------------------------------------------------------------
def _camera(cam):
    """Orthonormal camera frame. The disc lies in z = 0, the hole at the origin."""
    inc = np.deg2rad(cam["inclination_deg"])
    pos = cam["distance"] * np.array([np.sin(inc), 0.0, np.cos(inc)])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def _trace_rows(rows):
    """Trace one photon per pixel of the given rows back from the camera.

    Schwarzschild geometry (M = 1) is spherically symmetric, so every photon
    stays in the plane spanned by the camera position and its direction. In
    that plane, with u = 1/r and phi the swept angle, the null geodesic obeys

        u'' + u = 3 u^2,

    integrated here with RK4. The disc plane z = 0 is crossed at the fixed
    angles phi_1 + k*pi (they depend only on the orbital plane), so at each
    crossing r is interpolated and tested against [r_in, r_out]; the first
    hit is recorded with the disc azimuth and the redshift factor

        g = sqrt(1 - 3/r) / (1 - Omega * L_z/E),   Omega = r^(-3/2),

    for a Keplerian emitter. Rays that reach r = 2 are captured; rays moving
    outward beyond the disc can never come back and are dropped.

    Returns float32 maps (r, psi, g) with r = 0 where nothing was hit.
    """
    cam = _STATE["cam"]
    pos, fwd, right, up = _camera(cam)
    r_in, r_out, scale = cam["r_in"], cam["r_out"], cam["px_per_m"]
    dphi = cam["dphi"]

    cols = np.arange(WIDTH)
    x = (cols[None, :] + 0.5 - WIDTH / 2) / scale
    y = (HEIGHT / 2 - rows[:, None] - 0.5) / scale
    x, y = np.broadcast_arrays(x, y)
    n = x.size
    d = (
        fwd[None, :] * cam["distance"]
        + x.reshape(-1, 1) * right
        + y.reshape(-1, 1) * up
    )
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    p0 = np.broadcast_to(pos, (n, 3))
    normal = np.cross(p0, d)
    b = np.linalg.norm(normal, axis=1)  # impact parameter
    normal = normal / b[:, None]
    e1 = p0 / cam["distance"]
    e2 = np.cross(normal, e1)
    d_r = np.einsum("ij,ij->i", d, e1)
    d_t = np.einsum("ij,ij->i", d, e2)

    u = np.full(n, 1.0 / cam["distance"])
    du = -u * d_r / d_t
    # Disc-plane crossings: cos(phi) e1z + sin(phi) e2z = 0.
    phi_cross = np.arctan2(-e1[:, 2], e2[:, 2]) % np.pi
    phi_cross = np.where(phi_cross < 1e-6, np.pi, phi_cross)
    lam = b * normal[:, 2]  # L_z / E

    hit_r = np.zeros(n)
    hit_psi = np.zeros(n)
    hit_g = np.zeros(n)
    active = np.ones(n, dtype=bool)
    phi = 0.0

    def f(u, du):
        return du, 3.0 * u * u - u

    while active.any() and phi < cam["phi_max"]:
        idx = np.nonzero(active)[0]
        ua, dua = u[idx], du[idx]
        k1u, k1d = f(ua, dua)
        k2u, k2d = f(ua + 0.5 * dphi * k1u, dua + 0.5 * dphi * k1d)
        k3u, k3d = f(ua + 0.5 * dphi * k2u, dua + 0.5 * dphi * k2d)
        k4u, k4d = f(ua + dphi * k3u, dua + dphi * k3d)
        un = ua + dphi / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        dun = dua + dphi / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
        phi_new = phi + dphi

        pc = phi_cross[idx]
        crossing = (pc > phi) & (pc <= phi_new)
        if crossing.any():
            c = np.nonzero(crossing)[0]
            t = (pc[c] - phi) / dphi
            uc = ua[c] + (un[c] - ua[c]) * t
            rc = 1.0 / np.maximum(uc, 1e-9)
            on_disc = (rc >= r_in) & (rc <= r_out)
            gi = idx[c[on_disc]]
            rr = rc[on_disc]
            ph = pc[c[on_disc]]
            point = rr[:, None] * (
                np.cos(ph)[:, None] * e1[gi] + np.sin(ph)[:, None] * e2[gi]
            )
            hit_r[gi] = rr
            hit_psi[gi] = np.arctan2(point[:, 1], point[:, 0])
            omega = rr**-1.5
            hit_g[gi] = np.sqrt(1.0 - 3.0 / rr) / (1.0 - omega * lam[gi])
            active[gi] = False
            phi_cross[idx[c]] += np.pi  # next crossing for the rays that missed

        u[idx], du[idx] = un, dun
        captured = un > 0.5  # r < 2: inside the horizon
        gone = (dun < 0) & (un < 1.0 / r_out)  # outward bound, past the disc
        active[idx[captured | gone]] = False
        phi = phi_new

    shape = (len(rows), WIDTH)
    return (
        hit_r.reshape(shape).astype(np.float32),
        hit_psi.reshape(shape).astype(np.float32),
        hit_g.reshape(shape).astype(np.float32),
    )


# ----------------------------------------------------------------------------
# Disc texture and frames
# ----------------------------------------------------------------------------
def _loop_noise(grids, s, t, tau):
    """Cubic value noise on a 4-D lattice, periodic in t and exactly periodic in tau.

    s, t in [0, 1) index the lattice's first two axes; the loop phase tau
    walks a circle in the last two, so noise(tau + 1) == noise(tau).
    """
    total = np.zeros_like(s)
    amp = 1.0
    for grid in grids:
        n0, n1, nt, _ = grid.shape
        ct = nt / 2 + 0.3 * nt * np.cos(2 * np.pi * tau)
        st = nt / 2 + 0.3 * nt * np.sin(2 * np.pi * tau)
        coords = [s * n0, t * n1, np.full_like(s, ct), np.full_like(s, st)]
        total += amp * map_coordinates(grid, coords, order=3, mode="grid-wrap")
        amp *= 0.5
    return total


def _disc_intensity(tau, st):
    """Observed intensity of the disc for loop phase tau, on hit pixels only."""
    r, psi, g = st["hit_r"], st["hit_psi"], st["hit_g"]
    cam = st["cam"]
    r_in, r_out = cam["r_in"], cam["r_out"]

    s = np.log(r / r_in) / np.log(r_out / r_in)
    # Trailing spiral streaks: shear the azimuth with log-radius, and spin the
    # whole pattern by one full turn per loop.
    t = (psi + st["spiral_shear"] * np.log(r / r_in) - 2 * np.pi * tau) / (2 * np.pi)
    t = t % 1.0

    noise = _loop_noise(st["grids"], s, t, tau)
    noise = (noise - st["noise_mean"]) / st["noise_std"]
    texture = np.exp(st["turbulence"] * noise)

    emissivity = (r_in / r) ** st["emissivity_power"]
    edge = np.clip((r_out - r) / st["outer_fade"], 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return emissivity * edge * texture * g ** st["beaming_power"]


def _compose(intensity_hit, st):
    """Tone-map, colour and bloom a full frame from hit-pixel intensities."""
    lum = np.zeros((HEIGHT, WIDTH))
    lum[st["hit_mask"]] = intensity_hit
    mapped = 1.0 - np.exp(-st["gain"] * lum)
    lut = st["lut"]
    rgb = lut[(np.clip(mapped, 0, 1) * (lut.shape[0] - 1)).astype(np.int64)]
    rgb = rgb.transpose(2, 0, 1)  # (3, H, W)
    if st["bloom_weight"] > 0:
        passes = (
            (st["bloom_sigma"], st["bloom_weight"]),
            (st["bloom_sigma"] * 4, st["bloom_weight"] * 0.5),
        )
        for sigma, weight in passes:
            rgb = rgb + weight * np.stack([gaussian_filter(ch, sigma) for ch in rgb])
    return np.clip(rgb, 0, 1).transpose(1, 2, 0).astype(np.float32)


def _init_worker(state):
    _STATE.update(state)
    if "frames_path" in state:
        _STATE["frames_path"] = Path(state["frames_path"])


def _render_chunk(frame_indices):
    """Render a run of frames, reusing one Figure/Axes/AxesImage."""
    st = _STATE
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")
    image = ax.imshow(
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    for f in frame_indices:
        tau = f / LOOP_FRAMES
        image.set_data(_compose(_disc_intensity(tau, st), st))
        fig.savefig(st["frames_path"] / f"frame{f:04d}.png", facecolor=BG_COLOR)
    plt.close(fig)
    gc.collect()


def generate(settings: ImageProcessingSettings = None):
    """
    Gargantua -- a thin accretion disc around a Schwarzschild black hole,
    ray traced through the curved spacetime and painted with a looping
    turbulent pattern, as an exactly looping 9:16 clip.

    One photon per pixel is traced backwards from a distant camera. In the
    photon's orbital plane the Schwarzschild null geodesic (M = 1) satisfies
    u'' + u = 3u^2 with u = 1/r, integrated with RK4 in the swept angle. The
    disc plane is crossed at fixed angles phi_1 + k*pi, so the first crossing
    that lands between the ISCO at r = 6 and the outer radius is the pixel's
    source. Rays passing over the hole reach the far side of the disc,
    lifting it into the arch above the shadow; rays passing below reach the
    disc's underside, drawn as the arc beneath; rays skimming the photon
    sphere at r = 3 wind round more than once and paint the thin photon
    ring. Each hit also stores the redshift factor of a Keplerian emitter,
    g = sqrt(1 - 3/r) / (1 - Omega L_z/E), whose power brightens the side
    of the disc rushing towards the camera.

    The geodesic maps are computed once. Every frame then paints the disc
    with an emissivity falling as a power of radius, times a turbulent
    texture: cubic value noise on a 4-D lattice whose last two axes are
    walked around a circle by the loop phase, so it is exactly periodic,
    sheared into trailing spirals and spun one full turn per loop.
    Brightness is tone-mapped and sent through a blackbody-style heat ramp,
    then bloomed. Frames are independent, so rendering is spread across
    worker processes and one period is copied REPEATS times.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    cam = dict(
        distance=300.0,  # camera radius in M; large = near-orthographic
        inclination_deg=82.0,  # angle from the disc axis; 90 is edge-on
        r_in=6.0,  # ISCO
        r_out=15.0,
        px_per_m=32.0,  # image scale; 1080 px = 33.75 M, so the r_out = 15 disc fits
        dphi=0.01,  # RK4 step in swept angle
        phi_max=3.0 * np.pi,  # give up on photons that wind further than this
    )
    emissivity_power = 1.6  # radial falloff of the disc brightness
    outer_fade = 2.0  # M over which the outer edge fades out
    beaming_power = 2.2  # exponent on the redshift factor g
    turbulence = 0.55  # log-amplitude of the texture
    spiral_shear = 4.0  # radians of azimuth shear per e-fold of radius
    exposure_pct = 99.3  # intensity percentile driven to full brightness
    bloom_weight = 0.35
    bloom_sigma = 6.0

    # --- geodesics, once, across workers ---
    logger.info("tracing geodesics")
    _init_worker(dict(cam=cam))
    row_chunks = np.array_split(np.arange(HEIGHT), N_WORKERS * 3)
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=_init_worker, initargs=(dict(cam=cam),)
    ) as pool:
        parts = list(pool.map(_trace_rows, row_chunks))
    hit_r = np.concatenate([p[0] for p in parts])
    hit_psi = np.concatenate([p[1] for p in parts])
    hit_g = np.concatenate([p[2] for p in parts])
    hit_mask = hit_r > 0
    logger.info(
        f"disc covers {hit_mask.mean():.1%} of the frame, "
        f"g in [{hit_g[hit_mask].min():.2f}, {hit_g[hit_mask].max():.2f}]"
    )

    grids = [rng.uniform(-1.0, 1.0, (10 * 2**o, 6 * 2**o, 6, 6)) for o in range(3)]
    lut = LinearSegmentedColormap.from_list("heat", PALETTE, N=1024)(
        np.linspace(0, 1, 1024)
    )[:, :3].astype(np.float32)

    state = dict(
        cam=cam,
        hit_r=hit_r[hit_mask].astype(np.float64),
        hit_psi=hit_psi[hit_mask].astype(np.float64),
        hit_g=hit_g[hit_mask].astype(np.float64),
        hit_mask=hit_mask,
        grids=grids,
        noise_mean=0.0,
        noise_std=1.0,
        emissivity_power=emissivity_power,
        outer_fade=outer_fade,
        beaming_power=beaming_power,
        turbulence=turbulence,
        spiral_shear=spiral_shear,
        lut=lut,
        bloom_weight=bloom_weight,
        bloom_sigma=bloom_sigma,
        frames_path=str(settings.frames_path),
    )

    # --- normalise the noise and fix one exposure for the whole clip ---
    _init_worker(state)
    probe = _loop_noise(grids, rng.uniform(0, 1, 4000), rng.uniform(0, 1, 4000), 0.0)
    state["noise_mean"], state["noise_std"] = float(probe.mean()), float(probe.std())
    _STATE.update(noise_mean=state["noise_mean"], noise_std=state["noise_std"])
    highs = [
        np.percentile(_disc_intensity(tau, _STATE), exposure_pct)
        for tau in np.arange(3) / 3
    ]
    state["gain"] = 3.5 / max(float(np.max(highs)), 1e-9)
    logger.info(f"gain={state['gain']:.3f}")

    frames_path = settings.frames_path
    n_workers = min(N_WORKERS, LOOP_FRAMES)
    chunks = np.array_split(np.arange(LOOP_FRAMES), n_workers * 3)
    logger.info(f"rendering {LOOP_FRAMES} frames across {n_workers} workers")
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(state,)
    ) as pool:
        list(pool.map(_render_chunk, chunks))

    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png",
            )

    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()
