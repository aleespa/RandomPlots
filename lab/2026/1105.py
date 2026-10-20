"""
Five Squares at Every Corner

On a flat floor, four squares meet at every corner, no more. Here five do, again and
again, and every square in this hyperbolic tiling is exactly the same size.

The trick is the floor: the hyperbolic plane, a surface that curves away from itself
everywhere, so there is more room around each point than flat space allows. It cannot
be drawn without distortion, so this is a map of it, squeezed into a strip like a world
map squeezes the globe. Squares near the middle look big, those near the edges shrink
to dust, but walk there and they would be just as large, and the edge infinitely far
away. It is Escher's Circle Limit, unrolled into a river.

A bowed chord changes with every square that glides past, and each corner plucks a
string as it crosses the middle: higher towards the edges, where the squares look
smaller.

Made with Python & PyTorch. Sound synthesized in NumPy.
#hyperbolictiling #noneuclideangeometry #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

P, Q = 4, 5  # Schläfli symbol: P-gons, Q at each vertex. P even (see generate).
FOLDS = 48  # reflection rounds per pixel; far more than the band ever needs

BACKGROUND = "#020807"
DEEP = "#04161a"  # tile rim
RIM = "#0a2a4a"  # tiles towards the band's edges drift to sea blue
JADE = "#1a5e52"  # tile centre
LINE = "#d8f3e8"  # edges
VERTEX = "#f6fff9"  # the corners where Q tiles meet

KNOBS = dict(
    tiles=6,  # tiles that glide past the centre per loop (a whole number)
    sway=0.18,  # sideways hyperbolic drift, distance units
    line=0.011,  # edge half-width, hyperbolic units
    dot=0.05,  # vertex glow radius, hyperbolic units
)


def _rgb(colour: str) -> list[float]:
    return [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]


def geometry(p: int, q: int) -> dict:
    """
    The fundamental triangle of {p, q} in the Poincaré disk: angle pi/p at the tile
    centre (the origin, between the real axis and the line at angle pi/p), pi/q at the
    vertex, pi/2 at the edge midpoint. The third side is the edge circle |z - c| = r,
    orthogonal to the unit circle.
    """
    alpha, beta = np.pi / p, np.pi / q
    c = np.cos(beta) / np.sqrt(np.cos(beta) ** 2 - np.sin(alpha) ** 2)
    r = c * np.sin(alpha) / np.cos(beta)
    midpoint = c - r  # edge midpoint on the real axis
    rho = c * np.cos(alpha) - np.sqrt((c * np.cos(alpha)) ** 2 - 1)
    return dict(
        alpha=alpha,
        c=c,
        r=r,
        vertex=rho * np.exp(1j * alpha),
        # Reflection in the imaginary axis then in the edge: a translation along the
        # real axis by twice the centre-to-edge distance, a symmetry when p is even.
        translation=4 * np.arctanh(midpoint),
        circumradius=2 * np.arctanh(rho),
    )


# Soundtrack: bowed strings, one chord per tile glided; pizzicato on corner crossings.
MODE = "dorian"
ROOTS = range(50, 56)  # MIDI of the chords' root (D3 .. G3)
CHORDS = (0, 5, 3, 4, 0, 6)  # scale degrees, one per tile; len == KNOBS["tiles"]
INNER = 0.95  # corners count only within |v| < INNER (the rim's dust stays silent)
SAMPLES = 6000  # time samples per loop for finding crossings


def vertices(geo: dict, radius: float) -> np.ndarray:
    """Every corner of the tiling within hyperbolic `radius` of a tile centre (disk)."""
    alpha, c, r = geo["alpha"], geo["c"], geo["r"]
    limit = np.tanh(radius / 2)
    rotations = np.exp(2j * alpha * np.arange(round(np.pi / alpha)))
    found = {}
    frontier = [geo["vertex"]]
    while frontier:
        fresh = []
        for z in frontier:
            for w in (z * rotations).tolist() + [np.conj(z)]:
                d = w - c
                for x in (w, c + r * r * d / abs(d) ** 2):
                    key = (round(x.real, 9), round(x.imag, 9))
                    if abs(x) < limit and key not in found:
                        found[key] = x
                        fresh.append(x)
        frontier = fresh
    return np.array(list(found.values()))


def crossings(geo: dict, step: float, knobs: dict) -> list:
    """
    (time s, v) for every corner within |v| < INNER crossing the screen's horizontal
    centre line, found by mapping the corners through the inverse of the camera in
    generate() at SAMPLES times per loop.
    """
    corners = vertices(geo, 7.0)

    def screen(t, glide):
        b = 1j * np.tanh(knobs["sway"] * np.sin(2 * np.pi * t) / 2)
        a = np.tanh(np.pi / 4 * step * glide)
        z = (corners + a) / (1 + a * corners)
        z = (z - b) / (1 - np.conj(b) * z)
        w = 4 / np.pi * np.arctanh(z)
        return w.real, w.imag  # u (0 on the centre line), v

    events = []
    for i in range(SAMPLES):
        t0, t1 = i / SAMPLES, (i + 1) / SAMPLES
        g0 = (knobs["tiles"] * t0) % 1.0
        g1 = g0 + knobs["tiles"] / SAMPLES  # unwrapped across a tile boundary
        u0, v0 = screen(t0, g0)
        u1, v1 = screen(t1, g1)
        for k in np.flatnonzero(((u0 > 0) != (u1 > 0)) & (np.abs(v0) < INNER)):
            share = u0[k] / (u0[k] - u1[k])
            events.append(((t0 + share / SAMPLES) * DURATION, float(v0[k])))
    return sorted({(round(t, 4), round(v, 4)) for t, v in events})


def soundtrack(events: list, rng: np.random.Generator) -> np.ndarray:
    """
    A loop: a bowed string section holds one chord of CHORDS per tile glided (the
    loop's six symmetry steps), with the bass an octave under. Every corner crossing
    the centre line is plucked (pizzicato), panned by its place across the frame, two
    scale steps higher and a little quieter per unit of hyperbolic distance from the
    axis (where the tiles look smaller).
    """
    scale = Scale(int(rng.choice(ROOTS)), MODE)
    track = audio.silence(DURATION)
    span = DURATION / len(CHORDS)
    for i, degree in enumerate(CHORDS):
        for n in scale.chord(degree, 3):
            hz = float(audio.midi_to_hz(n))
            audio.add(track, audio.strings(hz, span, 0.35, 0.9, 1.4, 0.45), i * span)
        bass = scale.hz(degree, -1)
        audio.add(track, audio.strings(bass, span, 0.4, 0.9, 1.4, 0.3), i * span)
    for when, v in events:
        # Hyperbolic distance from the axis: how much smaller the tile looks.
        far = np.log(np.tan(np.pi / 4 * (1 + abs(v))))
        degree = 7 + int(round(2 * far))
        velocity = 0.5 * np.exp(-0.45 * far)
        note = audio.pluck(scale.hz(degree), velocity, 1.6, 0.12)
        audio.add(track, audio.pan(note, 0.8 * v / INNER), when)
    logger.info(f"Soundtrack: root MIDI {scale.root}, {len(events)} corner plucks")
    track = audio.reverb(track, rng, rt60=2.6, wet=0.3, tone=4500.0)
    return audio.master(track)


def generate(settings: ImageProcessingSettings = None):
    """
    The hyperbolic tiling {4,5} in the band model, gliding along its axis.

    Pixels map to the band w = u + iv, |v| < 1 (v across the frame, u down it), then to
    the Poincaré disk by z = tanh(pi w / 4), which sends the band's axis to the disk's
    real axis with hyperbolic arc length (pi / 2) du. The camera is two Möbius
    translations: a sideways sway about the screen centre,
    z -> (z + b) / (1 + conj(b) z), b = i tanh(s/2), s = sway sin(2 pi t), then the
    glide along the axis, z -> (z - A) / (1 - A z), A = tanh(pi g / 4), where the band
    shift g is taken modulo one symmetry translation. Then each pixel is folded into the fundamental triangle (wedge reflections about the
    origin plus inversion in the edge circle, repeated until nothing moves). The fold
    keeps track of the local scale (every map is conformal), so distances in the
    triangle convert to pixels: edges are drawn with a fixed hyperbolic width and analytic
    anti-aliasing (thinning to lace near the band's rim), vertices as glows, and tiles
    shaded by hyperbolic distance from their centre, with a faint two-tone of the
    triangle parity (the kaleidoscope's mirror count).

    Loop: the camera moves `tiles` symmetry translations (reflection in the imaginary
    axis then in an edge, which exists because P is even) along u over the loop, and the
    sway completes one whole cycle, so t = 1 is t = 0 exactly. Each frame is a pure
    function of t, computed per pixel with torch on the GPU and rendered in order.
    Nothing is random: --seed does not change the picture.

    Sound (see `soundtrack`): strings, one chord per symmetry step, and a pizzicato
    pluck whenever a corner crosses the centre line (from `crossings`, the corners
    mapped through the inverse camera). The key comes from settings.audio_rng; the
    posted soundtrack uses --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS
    device = select_device()
    geo = geometry(P, Q)
    step = 2 * geo["translation"] / np.pi  # one symmetry translation, in band units u
    logger.info(
        f"{{{P},{Q}}}: translation {geo['translation']:.3f}, "
        f"{WIDTH / 2 * step:.0f} px per tile on the axis"
    )
    f32 = dict(device=device, dtype=torch.float32)
    half = WIDTH / 2
    px = torch.arange(WIDTH, **f32) + 0.5
    py = torch.arange(HEIGHT, **f32) + 0.5
    v = ((px - half) / half)[None, :].expand(HEIGHT, WIDTH)
    u0 = ((HEIGHT / 2 - py) / half)[:, None].expand(HEIGHT, WIDTH)
    alpha, c, r = geo["alpha"], geo["c"], geo["r"]
    vertex = torch.tensor(geo["vertex"], device=device, dtype=torch.complex64)
    background, deep, jade, rim, line, glow = (
        torch.tensor(_rgb(x), **f32)
        for x in (BACKGROUND, DEEP, JADE, RIM, LINE, VERTEX)
    )
    outer = (v.abs() ** 2)[..., None]
    w0 = torch.complex(u0, v)
    z0 = torch.tanh(np.pi / 4 * w0)
    scale0 = (np.pi / 4 / half) * torch.cosh(np.pi / 4 * w0).abs() ** -2

    def render(frame):
        t = frame / FRAME_COUNT
        # Sway about the screen centre, then glide along the axis. The glide is taken
        # modulo one symmetry translation (exact, and keeps float32 away from |z| = 1).
        b = 1j * np.tanh(knobs["sway"] * np.sin(2 * np.pi * t) / 2)
        denom = 1 + np.conj(b) * z0
        z = (z0 + b) / denom
        scale = scale0 * (1 - abs(b) ** 2) / denom.abs() ** 2
        a = np.tanh(np.pi / 4 * step * ((knobs["tiles"] * t) % 1.0))
        denom = 1 - a * z
        z = (z - a) / denom
        scale = scale * (1 - a * a) / denom.abs() ** 2
        parity = torch.zeros(HEIGHT, WIDTH, **f32)
        for _ in range(FOLDS):
            # Into the wedge 0 <= arg z <= alpha: rotate by multiples of 2 alpha (even
            # reflections), then reflect once if past alpha.
            angle = torch.remainder(torch.angle(z), 2 * alpha)
            past = angle > alpha
            angle = torch.where(past, 2 * alpha - angle, angle)
            parity += past
            z = torch.polar(z.abs(), angle)
            # Across the edge: invert in the circle.
            d = z - c
            d2 = d.real**2 + d.imag**2
            inside = d2 < r * r
            if not bool(inside.any()):
                break
            z = torch.where(inside, c + r * r * d / d2, z)
            scale = torch.where(inside, scale * r * r / d2, scale)
            parity += inside
        # Local pixels per hyperbolic unit at the folded point.
        rho2 = (z.real**2 + z.imag**2).clamp(max=1 - 1e-6)
        density = 2 / (1 - rho2)  # hyperbolic length per disk length
        px_per_h = 1 / (density * scale)
        # Edge: distance to the circle, in pixels.
        edge_px = ((z - c).abs() - r).clamp(min=0) / scale
        width = knobs["line"] * px_per_h
        edge = (width + 0.5 - edge_px).clamp(0, 1) * (width / 0.6).clamp(max=1)
        # Vertex glow and centre shading, in hyperbolic units.
        to_vertex = 2 * torch.atanh(
            ((z - vertex).abs() / (1 - vertex.conj() * z).abs()).clamp(max=1 - 1e-6)
        )
        dot = torch.exp(-((to_vertex / knobs["dot"]) ** 2))
        centre = 1 - (2 * torch.atanh(rho2.sqrt()) / geo["circumradius"]).clamp(0, 1)
        tint = jade + (rim - jade) * outer
        fill = deep + (tint - deep) * centre[..., None] ** 1.5
        fill = fill * (1 + 0.04 * (2 * (parity % 2) - 1))[..., None]
        # Tiles smaller than a few pixels melt into the background.
        tile_px = geo["translation"] * px_per_h
        detail = ((tile_px - 2) / 30).clamp(0, 1)[..., None]
        image = background + (fill - background) * detail
        light = line * edge[..., None] + glow * (dot * detail[..., 0])[..., None]
        image = image + 0.9 * light + 0.5 * wide_blur(light, 8.0, 4)
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    track = soundtrack(crossings(geo, step, knobs), settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
