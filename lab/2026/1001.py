"""
Reuleaux Drift

A circle is not the only shape that rolls. Take an equilateral triangle
and bulge each side into an arc drawn from the opposite corner: the
Reuleaux triangle. Whichever way you turn it, it is exactly the same
width, so it rolls between two rails as smoothly as a wheel, and it can
turn inside a square while touching all four sides at once. That is how
a drill can bore a very nearly square hole.

Here two of them turn inside their squares, one each way. The triangle
carries a lattice of marked points, and the path each point sweeps over
one full turn is drawn behind it: the corners trace the rounded square,
the centre a small four-lobed clover, and everything between them
something in between. One whole turn fills the clip, so it ends exactly
as it began.

Made with Python, PyTorch & NumPy.
#generativeart #reuleaux #constantwidth #kinematics #geometry
#mathematics #randomplots #creativecoding #algorithmicart #mathisbeautiful
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common.image_processing import ImageProcessingSettings

# --- Video -------------------------------------------------------------------
# RP_FPS in the environment gives a quick low-frame-rate test render.
FPS = int(os.environ.get("RP_FPS", "30"))
DURATION = 20.0
FRAME_COUNT = max(2, int(FPS * DURATION))
WIDTH, HEIGHT = 1080, 1920  # 9:16 portrait for Reels / Stories
PNG_WRITERS = 6
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

# Frames rendered before frame 0 so the motion trail is at steady state.
WARMUP_FRAMES = 16

# --- Mechanism ---------------------------------------------------------------
SQUARE = 0.86  # side of each square as a fraction of the width
GAP = 0.03  # gap between the two squares, as a fraction of the width
TURNS = 1  # whole turns of each triangle over the loop (integer -> exact)
MARK_ROWS = 9  # triangular lattice of marked points, rows from a vertex
OUTLINE_SAMPLES = 900  # per arc of the triangle outline
TRACK_SAMPLES = 1800  # samples per marked point's track over the loop

# --- Appearance --------------------------------------------------------------
BACKGROUND = np.array([6, 6, 12], dtype=np.float32)  # RGB
PALETTE = np.array(  # marked points from centre -> corners
    [
        [0.95, 0.95, 1.00],
        [1.00, 0.75, 0.35],
        [0.95, 0.35, 0.40],
        [0.55, 0.20, 0.65],
        [0.15, 0.40, 0.85],
    ],
    dtype=np.float32,
)
PALETTE_SIZE = 256
TRACK_ENERGY = 70.0  # brightness of the static tracks, per pixel of track
OUTLINE_ENERGY = 220.0
SQUARE_ENERGY = 110.0
SQUARE_COLOUR = np.array([0.45, 0.50, 0.65], dtype=np.float32)
OUTLINE_COLOUR = np.array([1.00, 0.92, 0.80], dtype=np.float32)
MARK_RADIUS = 6.0
MARK_ENERGY = 255.0  # per pixel of disc
TRAIL_DECAY = 0.78
BLOOM = 0.45
BLOOM_SIGMA = 3.0
WIDE_BLOOM = 0.12
WIDE_BLOOM_SIGMA = 20.0
VIGNETTE = 0.18


def select_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("no CUDA device, rendering on CPU")
    return torch.device("cpu")


def palette_lut(device: torch.device) -> torch.Tensor:
    positions = np.linspace(0.0, 1.0, len(PALETTE))
    t = np.linspace(0.0, 1.0, PALETTE_SIZE)
    lut = np.stack([np.interp(t, positions, PALETTE[:, c]) for c in range(3)], axis=1)
    return torch.tensor(lut, dtype=torch.float32, device=device)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    return kernel / kernel.sum()


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    planes = image.permute(2, 0, 1)[None]
    channels = planes.shape[1]
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.conv2d(
        F.pad(planes, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    out = F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
    )
    return out[0].permute(1, 2, 0)


def splat(
    flat: torch.Tensor, px: torch.Tensor, py: torch.Tensor, colours: torch.Tensor
):
    """Bilinear point splatting into a flat (H*W, 3) buffer; deposits add."""
    x0 = torch.floor(px)
    y0 = torch.floor(py)
    fx = px - x0
    fy = py - y0
    x0 = x0.to(torch.long)
    y0 = y0.to(torch.long)
    for ox, oy, corner in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        ix = x0 + ox
        iy = y0 + oy
        inside = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)
        index = iy.clamp(0, HEIGHT - 1) * WIDTH + ix.clamp(0, WIDTH - 1)
        flat.index_add_(0, index, colours * (corner * inside)[:, None])


def sample_lengths(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    steps = torch.hypot(px[:, 1:] - px[:, :-1], py[:, 1:] - py[:, :-1])
    lengths = torch.empty_like(px)
    lengths[:, 1:-1] = 0.5 * (steps[:, :-1] + steps[:, 1:])
    lengths[:, 0] = steps[:, 0]
    lengths[:, -1] = steps[:, -1]
    return lengths


def support(phi: np.ndarray, theta: float, r: float, w: float) -> np.ndarray:
    """
    Support function of a Reuleaux triangle: distance from its centroid to the
    tangent line with outward normal at angle phi.

    Vertices sit at angles theta + 2 pi i / 3, at distance r from the
    centroid; the width is w = r sqrt 3. Within 30 degrees of a vertex
    direction the vertex is the support point and h is its projection;
    otherwise the support lies on the arc centred at the vertex facing away,
    and h is w plus that vertex's (negative) projection. The two regimes meet
    continuously at r cos 30, and h(phi) + h(phi + pi) = w throughout.
    """
    alphas = theta + 2.0 * np.pi * np.arange(3) / 3.0
    proj = r * np.cos(phi[..., None] - alphas)
    vertex = proj.max(-1)
    return np.where(vertex >= r * np.cos(np.pi / 6.0) - 1e-12, vertex, w + proj.min(-1))


def centroid_in_square(theta: float, r: float, w: float) -> np.ndarray:
    """
    Centroid position, in a square [0, w]^2, of a Reuleaux triangle at
    rotation theta that touches all four sides.

    The left wall is the tangent with outward normal pi, so the centroid sits
    h(pi) from it; likewise the bottom wall with normal -pi/2. Constant width
    guarantees the opposite walls are touched too.
    """
    phi = np.array([np.pi, -np.pi / 2.0])
    return support(phi, theta, r, w)


def triangle_marks(rows: int) -> np.ndarray:
    """A triangular lattice inside the unit-circumradius equilateral triangle."""
    # Vertices at 0, 120 and 240 degrees, matching the outline at theta = 0.
    verts = np.array(
        [[np.cos(a), np.sin(a)] for a in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)]
    )
    points = []
    for i in range(rows):
        for j in range(rows - i):
            k = rows - 1 - i - j
            bary = np.array([i, j, k]) / (rows - 1)
            points.append(bary @ verts)
    return np.array(points)


def outline(theta: float, r: float, w: float, device: torch.device) -> torch.Tensor:
    """The three arcs of the Reuleaux triangle, as (3, OUTLINE_SAMPLES, 2) in centroid units."""
    alphas = theta + 2.0 * np.pi * np.arange(3) / 3.0
    verts = np.stack([r * np.cos(alphas), r * np.sin(alphas)], axis=-1)
    arcs = []
    for i in range(3):
        a = verts[i]
        b = verts[(i + 1) % 3]
        c = verts[(i + 2) % 3]
        start = np.arctan2(b[1] - a[1], b[0] - a[0])
        end = np.arctan2(c[1] - a[1], c[0] - a[0])
        # Shortest way round: the arc spans 60 degrees.
        delta = (end - start + np.pi) % (2.0 * np.pi) - np.pi
        psi = start + delta * np.linspace(0.0, 1.0, OUTLINE_SAMPLES)
        arcs.append(np.stack([a[0] + w * np.cos(psi), a[1] + w * np.sin(psi)], axis=-1))
    return torch.tensor(np.stack(arcs), dtype=torch.float32, device=device)


def generate(settings: ImageProcessingSettings = None):
    """
    Two Reuleaux triangles turning inside squares, with their point tracks.

    A Reuleaux triangle of width w turning inside a square of side w touches
    all four sides at every angle; its centroid is placed from the support
    function h(phi) = max(max_i r cos(phi - a_i), w + min_i r cos(phi - a_i)),
    which for a body of constant width gives the distance to each wall
    directly. The rotation covers TURNS whole turns over the loop, and every
    point fixed to the triangle then traces a closed curve.

    A triangular lattice of marked points is attached to each triangle. Their
    tracks over the whole loop are computed once and splatted into a static
    backdrop, coloured by the point's distance from the centroid. Each frame
    then adds the square, the triangle's three arcs and the marks themselves,
    the latter as small discs into a decaying buffer so they leave short
    trails. Two-scale bloom and a vignette finish the frame. The seed sets
    the starting phase and which triangle turns which way.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    seed_phase = rng.uniform(0.0, 1.0)
    spin = 1.0 if rng.uniform() < 0.5 else -1.0

    logger.info(f"Rendering Reuleaux Drift: {FRAME_COUNT} frames @ {FPS} fps")
    frames_path = settings.frames_path
    lut = palette_lut(device)
    bloom_kernel = gaussian_kernel(BLOOM_SIGMA, device)
    wide_kernel = gaussian_kernel(WIDE_BLOOM_SIGMA, device)

    yy = torch.linspace(-1.0, 1.0, HEIGHT, device=device)[:, None]
    xx = torch.linspace(-1.0, 1.0, WIDTH, device=device)[None, :]
    vignette = torch.clamp(1.0 - VIGNETTE * (xx**2 + yy**2), 0.6, 1.0)[..., None]

    side = SQUARE * WIDTH  # pixels
    w = 1.0  # work in units of the width; r = w / sqrt 3
    r = w / np.sqrt(3.0)
    gap = GAP * WIDTH
    # Square origins (top-left corner in screen pixels), stacked and centred.
    total = 2.0 * side + gap
    origins = [
        (WIDTH / 2.0 - side / 2.0, HEIGHT / 2.0 - total / 2.0),
        (WIDTH / 2.0 - side / 2.0, HEIGHT / 2.0 - total / 2.0 + side + gap),
    ]
    spins = [spin, -spin]

    marks = triangle_marks(MARK_ROWS) * r  # (M, 2), centroid units
    mark_shade = torch.tensor(
        np.linalg.norm(marks, axis=1) / r, dtype=torch.float32, device=device
    )
    mark_colours = lut[(mark_shade * (PALETTE_SIZE - 1)).to(torch.long)]  # (M, 3)
    marks_t = torch.tensor(marks, dtype=torch.float32, device=device)

    def to_screen(points: torch.Tensor, theta: float, origin) -> torch.Tensor:
        """Rotate centroid-frame points by theta, add the centroid, map to pixels."""
        c = np.cos(theta)
        s = np.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device)
        centre = centroid_in_square(theta, r, w)
        p = points @ rot.T + torch.tensor(centre, dtype=torch.float32, device=device)
        px = origin[0] + p[..., 0] * side
        py = origin[1] + side - p[..., 1] * side  # y up in the mechanism
        return px, py

    # Static backdrop: the closed tracks of every mark over one loop.
    logger.info("tracing the marks' closed tracks")
    flat = torch.zeros(HEIGHT * WIDTH, 3, device=device)
    for origin, direction in zip(origins, spins):
        track_x = []
        track_y = []
        for u in np.linspace(0.0, 1.0, TRACK_SAMPLES + 1):
            theta = direction * 2.0 * np.pi * TURNS * u
            px, py = to_screen(marks_t, theta, origin)
            track_x.append(px)
            track_y.append(py)
        px = torch.stack(track_x, dim=1)  # (M, T+1)
        py = torch.stack(track_y, dim=1)
        weight = (sample_lengths(px, py) * TRACK_ENERGY).reshape(-1)
        colours = (
            mark_colours[:, None, :].expand(-1, TRACK_SAMPLES + 1, -1).reshape(-1, 3)
        )
        splat(flat, px.reshape(-1), py.reshape(-1), colours * weight[:, None])

        # The square itself.
        edge = torch.linspace(0.0, 1.0, 2000, device=device)
        corners = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], device=device
        )
        segs = (
            corners[:-1, None, :]
            + edge[None, :, None] * (corners[1:] - corners[:-1])[:, None, :]
        )
        sx = origin[0] + segs[..., 0] * side
        sy = origin[1] + side - segs[..., 1] * side
        weight = (sample_lengths(sx, sy) * SQUARE_ENERGY).reshape(-1)
        square_colour = torch.tensor(SQUARE_COLOUR, device=device)
        splat(
            flat,
            sx.reshape(-1),
            sy.reshape(-1),
            square_colour[None, :] * weight[:, None],
        )
    backdrop = flat.view(HEIGHT, WIDTH, 3).clone()

    # Disc stencil for the marks.
    span = int(np.ceil(MARK_RADIUS)) + 1
    grid = torch.arange(-span, span + 1, device=device, dtype=torch.float32)
    oy, ox = torch.meshgrid(grid, grid, indexing="ij")
    stencil = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)
    stencil_cover = torch.clamp(
        MARK_RADIUS + 0.5 - torch.hypot(stencil[:, 0], stencil[:, 1]), 0.0, 1.0
    )

    outline_colour = torch.tensor(OUTLINE_COLOUR, device=device)
    background = torch.tensor(BACKGROUND, device=device)
    trail = torch.zeros(HEIGHT, WIDTH, 3, device=device)

    with ThreadPoolExecutor(max_workers=PNG_WRITERS) as pool:
        pending = []
        for frame in range(-WARMUP_FRAMES, FRAME_COUNT):
            t = (frame / FRAME_COUNT + seed_phase) % 1.0
            trail.mul_(TRAIL_DECAY)
            flat.zero_()

            for origin, direction in zip(origins, spins):
                theta = direction * 2.0 * np.pi * TURNS * t
                # Marks, as discs, into the trail buffer.
                px, py = to_screen(marks_t, theta, origin)
                sx = (px[:, None] + stencil[None, :, 0]).reshape(-1)
                sy = (py[:, None] + stencil[None, :, 1]).reshape(-1)
                deposit = (
                    mark_colours[:, None, :]
                    * stencil_cover[None, :, None]
                    * MARK_ENERGY
                ).reshape(-1, 3)
                splat(flat, sx, sy, deposit)
            trail.add_(flat.view(HEIGHT, WIDTH, 3))

            flat.zero_()
            for origin, direction in zip(origins, spins):
                theta = direction * 2.0 * np.pi * TURNS * t
                arcs = outline(theta, r, w, device).reshape(-1, 2)
                # The outline is already in the rotated frame; only translate and map.
                centre = centroid_in_square(theta, r, w)
                ax = origin[0] + (arcs[:, 0] + centre[0]) * side
                ay = origin[1] + side - (arcs[:, 1] + centre[1]) * side
                ax = ax.view(3, OUTLINE_SAMPLES)
                ay = ay.view(3, OUTLINE_SAMPLES)
                weight = (sample_lengths(ax, ay) * OUTLINE_ENERGY).reshape(-1)
                splat(
                    flat,
                    ax.reshape(-1),
                    ay.reshape(-1),
                    outline_colour[None, :] * weight[:, None],
                )
            outline_layer = flat.view(HEIGHT, WIDTH, 3)

            if frame < 0:
                continue

            image = background + backdrop + trail + outline_layer
            highlights = torch.clamp(image - 120.0, min=0.0)
            composite = (
                image
                + BLOOM * blur(image, bloom_kernel)
                + WIDE_BLOOM * blur(highlights, wide_kernel)
            )
            composite = composite * vignette

            out = composite.clamp_(0.0, 255.0).to(torch.uint8).flip(-1).cpu().numpy()
            pending.append(
                pool.submit(
                    cv2.imwrite,
                    str(frames_path / f"frame{frame:04d}.png"),
                    out,
                    PNG_PARAMS,
                )
            )
            while len(pending) > 2 * PNG_WRITERS:
                if not pending.pop(0).result():
                    raise OSError("Could not write a frame")
            if frame % FPS == 0:
                logger.info(f"Frame {frame}/{FRAME_COUNT}")

        for task in pending:
            if not task.result():
                raise OSError("Could not write a frame")

    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
