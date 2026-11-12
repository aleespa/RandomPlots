"""
The Room You Can't Light

A room of mirrors that one candle can never fully light. Light bounces off
the walls forever, yet the two side wings stay pitch black, though they open
wide onto the glowing domes.

The trick is Roger Penrose's, from 1958. Each dome is half an ellipse, and
the thin walls end exactly at its two focal points. In an elliptical mirror, a
ray passing between the foci always bounces back between them; a ray passing
outside them stays outside. Light from the hall can only fall back into the
hall.

Then a blue candle is lit in one wing. It lights both wings, even the far
one, but never the hall between them. Two lights in one room that never meet.

Every bell is a ray striking a mirror: high for the warm candle, low for the
blue, higher notes higher up the room, and a gong as each candle is lit.

Made with Python & PyTorch. Sound synthesized in NumPy.
#penroseunilluminableroom #optics #billiards #mathart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The room (px, y up, origin at the centre of the frame) ------------------------
SEMI_A, SEMI_B = 470.0, 400.0  # each dome is the half-ellipse x^2/a^2 + y'^2/b^2 = 1
HALF_H = 400.0  # the domes' flat sides sit at y = +-HALF_H
FOCUS = float(np.sqrt(SEMI_A**2 - SEMI_B**2))  # the partitions stand at x = +-FOCUS
CENTRE = (WIDTH / 2, HEIGHT / 2)

# --- The candles ----------------------------------------------------------------------
# (position, lit at s, colour in linear RGB). Rays leave each one at evenly spaced
# angles, with a random phase.
CANDLES = (
    ((-70.0, -170.0), 0.0, (1.0, 0.52, 0.16)),  # warm, in the hall
    ((-(SEMI_A + FOCUS) / 2, 140.0), 6.5, (0.22, 0.52, 1.0)),  # blue, in the left wing
)
RAYS = 4000  # per candle
SPEED = 900.0  # px of path per second
HEAD_START = 140.0  # px of path the first candle's light has already run at t = 0
STEP = 1.5  # px between samples along each ray

# --- Look -----------------------------------------------------------------------------
BACKGROUND = (0.012, 0.010, 0.016)
EXPOSURE, GAMMA = 0.55, 0.8  # light = 1 - exp(-EXPOSURE * density^GAMMA)
SETTLE = 2200.0  # px of path: the exposure stops adapting once light ran this far
SOFT = 1.0  # px blur of the ray density
SPARK_TRAIL, SPARK_GAIN = 28, 0.06  # samples of comet trail at each ray's head
GLOWS = ((10.0, 2, 0.10), (40.0, 8, 0.06))  # (sigma px, shrink, gain) halos
WALL_COLOUR, WALL_ALPHA, WALL_WIDTH = (0.86, 0.80, 0.70), 0.7, 3
FLAME_SIGMA = 7.0  # px

# --- Sound ----------------------------------------------------------------------------
STYLE = "gamelan"  # common.music.STYLES; an --audio-only tag naming one overrides it
LISTENED = ((2, 1), (2, 2))  # rays sounded per candle: before / after the blue candle
DEGREES = 9  # pelog degrees from the bottom of the room to the top
MIN_GAP = 0.14  # s: a ray's taps closer than this to its previous one are skipped


def wall_hit(p: np.ndarray, d: np.ndarray):
    """Distance along d from each p (inside the room) to the next wall, and its normal."""
    best = np.full(len(p), np.inf)
    normal = np.zeros_like(p)
    for side in (1.0, -1.0):  # the two domes, centred at (0, +-HALF_H)
        q = p - np.array([0.0, side * HALF_H])
        qa = d[:, 0] ** 2 / SEMI_A**2 + d[:, 1] ** 2 / SEMI_B**2
        qb = 2 * (q[:, 0] * d[:, 0] / SEMI_A**2 + q[:, 1] * d[:, 1] / SEMI_B**2)
        qc = q[:, 0] ** 2 / SEMI_A**2 + q[:, 1] ** 2 / SEMI_B**2 - 1
        disc = qb * qb - 4 * qa * qc
        root = np.sqrt(np.maximum(disc, 0))
        for s in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)):
            y = q[:, 1] + s * d[:, 1]
            ok = (disc > 0) & (s > 1e-6) & (side * y >= -1e-9) & (s < best)
            best = np.where(ok, s, best)
            hit = q + s[:, None] * d
            n = np.stack([hit[:, 0] / SEMI_A**2, hit[:, 1] / SEMI_B**2], 1)
            normal[ok] = n[ok] / np.linalg.norm(n[ok], axis=1, keepdims=True)
    for x in (SEMI_A, -SEMI_A, FOCUS, -FOCUS):  # outer walls and the two partitions
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (x - p[:, 0]) / d[:, 0]
        y = p[:, 1] + s * d[:, 1]
        ok = (s > 1e-6) & (np.abs(y) <= HALF_H) & (s < best)
        best = np.where(ok, s, best)
        normal[ok] = (1.0, 0.0)
    return best, normal


def trace(source, angles, length: float) -> dict:
    """
    Bounce rays from `source` until each has run `length` px: the arc length at
    which each straight segment starts, its start point and direction,
    (rays, segments) arrays. Rays that are done keep 'bouncing' at zero cost.
    """
    p = np.tile(np.array(source, float), (len(angles), 1))
    d = np.stack([np.cos(angles), np.sin(angles)], 1)
    s = np.zeros(len(angles))
    keep = [(s.copy(), p.copy(), d.copy())]
    while s.min() <= length:
        active = s <= length
        step, n = wall_hit(p[active], d[active])
        pa, da = p[active] + step[:, None] * d[active], d[active]
        da = da - 2 * (da * n).sum(1, keepdims=True) * n
        da /= np.linalg.norm(da, axis=1, keepdims=True)
        p[active], d[active] = pa, da
        s[active] += step
        s[~active] = np.inf
        keep.append((s.copy(), p.copy(), d.copy()))
        if len(keep) > 20_000:
            raise RuntimeError("a ray is stuck bouncing")
    out = {key: np.stack([k[i] for k in keep], 1) for i, key in enumerate("spd")}
    out["s"][:, 0] = 0.0
    return out


def path_length(seconds, lit: float, head: float = 0.0):
    """How far a candle's light has run at `seconds` (0 before it is lit)."""
    return np.where(seconds >= lit, head + SPEED * (seconds - lit), 0.0)


def in_wing(x, y):
    return (abs(x) > FOCUS) & (abs(x) < SEMI_A) & (abs(y) < HALF_H)


def in_hall(x, y):
    return (abs(x) < FOCUS) & (abs(y) < HALF_H)


def sample(room: dict, count: int, device) -> torch.Tensor:
    """
    Pixel index of every ray every STEP px along its path, (count, rays) int32,
    sample-major so any prefix is 'all the light up to that length'.
    """
    to = dict(device=device, dtype=torch.float64)
    starts = torch.tensor(room["s"], **to)
    points = torch.tensor(room["p"], **to)
    dirs = torch.tensor(room["d"], **to)
    starts = torch.where(torch.isinf(starts), torch.full_like(starts, 1e30), starts)
    rays = starts.shape[0]
    out = torch.empty(count, rays, dtype=torch.int32, device=device)
    leaks = 0
    for j0 in range(0, count, 1000):
        j = torch.arange(j0, min(count, j0 + 1000), device=device, dtype=torch.float64)
        query = (j * STEP)[None, :].expand(rays, -1).contiguous()
        k = (torch.searchsorted(starts, query, right=True) - 1).clamp_min(0)
        s0 = starts.gather(1, k)
        x = points[..., 0].gather(1, k) + (query - s0) * dirs[..., 0].gather(1, k)
        y = points[..., 1].gather(1, k) + (query - s0) * dirs[..., 1].gather(1, k)
        leaks += int(in_wing(x, y).sum()) if room["home"] == "hall" else 0
        leaks += int(in_hall(x, y).sum()) if room["home"] == "wing" else 0
        px = torch.round(CENTRE[0] + x).clamp(0, WIDTH - 1).long()
        py = torch.round(CENTRE[1] - y).clamp(0, HEIGHT - 1).long()
        out[j0 : j0 + len(j)] = (py * WIDTH + px).T.to(torch.int32)
    logger.info(f"Sampled {count:,} x {rays:,} points; {leaks} forbidden samples")
    return out


def walls() -> np.ndarray:
    """The mirrors as an alpha mask (H, W), drawn at 4x and shrunk."""
    big = np.zeros((HEIGHT * 4, WIDTH * 4), np.uint8)
    u = np.linspace(0, np.pi, 1500)
    dome = np.stack([SEMI_A * np.cos(u), HALF_H + SEMI_B * np.sin(u)], 1)
    outline = np.concatenate([dome, dome[::-1] * [1, -1]])
    lines = [outline]
    for x in (FOCUS, -FOCUS):
        lines.append(np.array([[x, HALF_H], [x, -HALF_H]]))
    for pts, closed in zip(lines, (True, False, False)):
        pix = np.stack([CENTRE[0] + pts[:, 0], CENTRE[1] - pts[:, 1]], 1)
        cv2.polylines(
            big,
            [np.round(pix * 16 * 4).astype(np.int32)],
            closed,
            255,
            WALL_WIDTH * 4,
            cv2.LINE_AA,
            4,
        )
    for x in (FOCUS, -FOCUS):  # the foci, where the partitions end
        for y in (HALF_H, -HALF_H):
            centre = (int(4 * (CENTRE[0] + x)), int(4 * (CENTRE[1] - y)))
            cv2.circle(big, centre, 4 * WALL_WIDTH * 2, 255, -1, cv2.LINE_AA)
    mask = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32) / 255.0


def soundtrack(rooms: list, style_name: str, rng: np.random.Generator):
    """
    A one-shot in pelog bells.

    - A gong (the style's bass, two octaves down) and a bell on the root as
      each candle is lit, at 0 s and at the blue candle's time.
    - Wall strikes: LISTENED rays of each candle (chosen with the audio rng)
      sound every bounce at its exact time, (arc length) / SPEED after the
      candle was lit. Pitch is the height of the bounce in the room (DEGREES
      pelog steps bottom to top), the warm candle an octave above the blue
      one; pan is x. Taps closer than MIN_GAP to the ray's previous one are
      skipped (grazing hops in the domes), and fewer warm rays sound once the
      blue candle is lit, to keep the density near 5 notes/s.
    - A quiet pad on the root and fifth, joined by the third when the blue
      candle is lit.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    blue_lit = CANDLES[1][1]
    top = HALF_H + SEMI_B

    def low(hz):
        while hz < 65:
            hz *= 2
        return hz

    for lit in (CANDLES[0][1], blue_lit):
        audio.add(track, style.bass(low(scale.hz(0, -2)), 0.6, length=6.0), lit, **kw)
        audio.add(track, style.voice(scale.hz(0, 0), 0.45), lit + 0.02, **kw)

    taps = 0
    for c, (room, (before, after), octave) in enumerate(zip(rooms, LISTENED, (0, -1))):
        lit = CANDLES[c][1]
        head = HEAD_START if c == 0 else 0.0
        chosen = rng.choice(room["s"].shape[0], max(before, after), replace=False)
        for r, ray in enumerate(chosen):
            last = -1.0
            for s, (x, y) in zip(room["s"][ray, 1:], room["p"][ray, 1:]):
                when = lit + (s - head) / SPEED
                if not np.isfinite(when) or when >= DURATION - 0.3:
                    break
                if when < 0.05 or when - last < MIN_GAP:
                    continue
                if c == 0 and r >= after and when >= blue_lit:
                    break
                last = when
                degree = int(round((y + top) / (2 * top) * DEGREES))
                fade = 1 - 0.6 * float(smoothstep((when - 13.0) / 2.0))
                velocity = fade * rng.uniform(0.26, 0.4)
                note = style.voice(scale.hz(degree, octave), velocity)
                audio.add(track, audio.pan(note, 0.75 * x / SEMI_A), when, **kw)
                taps += 1

    for degree, start in ((0, 0.0), (3, 0.0), (2, blue_lit)):
        hz = low(scale.hz(degree, -1))
        audio.add(
            track,
            audio.pad(hz, DURATION - start - 1.5, 0.08, 2.0, 1.5, 0.3),
            start,
            **kw,
        )
    logger.info(f"Soundtrack: {style_name}, root MIDI {scale.root}, {taps} wall taps")
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Penrose's unilluminable room, lit by two candles.

    The room: two half-ellipses (semi-axes SEMI_A, SEMI_B) with flat sides at
    y = +-HALF_H, joined by vertical outer walls at x = +-SEMI_A, plus two
    two-sided partitions at x = +-FOCUS = +-sqrt(a^2 - b^2) from one dome's
    focus to the other's. A chord of an ellipse that crosses the major axis
    between the foci reflects into another such chord (its caustic is a
    confocal hyperbola); one that crosses outside them stays outside (a
    confocal ellipse). Reflection in the flat side preserves this, so the
    domes return light from the hall (|x| < FOCUS) to the hall and light from
    the wings (FOCUS < |x| < SEMI_A) to the wings: a candle in the hall never
    lights the wings and vice versa (`sample` counts forbidden samples; 0).

    Each candle sends RAYS rays at evenly spaced angles plus one random phase
    from settings.rng; `trace` reflects them exactly (quadratic for the domes,
    linear for the walls) in float64 until each has run the whole clip. Every
    ray is then sampled every STEP px of path into a pixel index, sample-major,
    so the light a candle has shed by path length L is a prefix of that array.

    A frame is a pure function of t: L = HEAD_START + SPEED t for the warm
    candle, SPEED (t - 6.5) for the blue one; each prefix is bincounted on the
    GPU into a ray-density image, normalised by min(L, SETTLE) so the exposure
    adapts as light spreads, then lightly blurred. The heads of the rays (the
    last SPARK_TRAIL samples) are added as sparks. Both colours are tone-mapped
    together as 1 - exp(-EXPOSURE * density), glowed, and the mirrors and foci
    laid on top. Caustics of the domes show in the hall as bright envelopes.

    The posted render uses --seed 0 (the default), with the gamelan soundtrack.
    """
    settings = settings or ImageProcessingSettings(1)
    lengths = []
    rooms = []
    for c, (source, lit, _) in enumerate(CANDLES):
        head = HEAD_START if c == 0 else 0.0
        length = float(path_length(DURATION, lit, head)) + 2 * STEP
        phase = settings.rng.uniform(0, 2 * np.pi / RAYS)
        angles = phase + 2 * np.pi * np.arange(RAYS) / RAYS
        room = trace(source, angles, length)
        room["home"] = "hall" if c == 0 else "wing"
        rooms.append(room)
        lengths.append(length)
        logger.info(
            f"Candle {c}: {RAYS} rays traced to {length:.0f} px, "
            f"{room['s'].shape[1] - 1} bounces"
        )

    if not settings.audio_only:
        device = select_device()
        to = dict(device=device, dtype=torch.float32)
        samples = [
            sample(room, int(length / STEP), device)
            for room, length in zip(rooms, lengths)
        ]
        colours = [torch.tensor(c[2], **to) for c in CANDLES]
        area = (4 * SEMI_A * HALF_H) + np.pi * SEMI_A * SEMI_B  # px^2
        soft = gaussian_kernel(SOFT, device)
        wall = torch.tensor(walls(), **to)[..., None]
        wall_rgb = torch.tensor(WALL_COLOUR, **to)
        background = torch.tensor(BACKGROUND, **to)
        yy, xx = torch.meshgrid(
            torch.arange(HEIGHT, **to), torch.arange(WIDTH, **to), indexing="ij"
        )
        flames = []
        for (x, y), _, _ in CANDLES:
            r2 = (xx - CENTRE[0] - x) ** 2 + (yy - CENTRE[1] + y) ** 2
            flames.append(torch.exp(-r2 / (2 * FLAME_SIGMA**2))[..., None])
        trail = torch.linspace(0.15, 1.0, SPARK_TRAIL, **to)[:, None]
        heads = torch.ones(SPARK_TRAIL, RAYS, **to) * trail

    def render(frame: int) -> torch.Tensor:
        seconds = frame / (FRAME_COUNT - 1) * DURATION
        linear = torch.zeros(HEIGHT * WIDTH, 3, **to)
        sparks = torch.zeros(HEIGHT * WIDTH, 3, **to)
        for c, (points, colour) in enumerate(zip(samples, colours)):
            lit = CANDLES[c][1]
            head = HEAD_START if c == 0 else 0.0
            length = float(path_length(seconds, lit, head))
            n = min(int(length / STEP) + 1, points.shape[0]) if seconds >= lit else 0
            if n == 0:
                continue
            density = torch.bincount(points[:n].reshape(-1), minlength=HEIGHT * WIDTH)
            scale = STEP * area / (RAYS * max(length, SETTLE))
            linear += (density.to(torch.float32) * scale)[:, None] * colour
            m = min(n, SPARK_TRAIL)
            index = points[n - m : n].reshape(-1).long()
            weight = heads[SPARK_TRAIL - m :].reshape(-1)
            sparks.index_add_(0, index, weight[:, None] * colour[None, :])
        linear = blur(linear.view(HEIGHT, WIDTH, 3), soft)
        linear = linear + SPARK_GAIN * blur(sparks.view(HEIGHT, WIDTH, 3), soft) * 40
        for c, flame in enumerate(flames):
            lit = CANDLES[c][1]
            if seconds >= lit:
                on = float(smoothstep((seconds - lit) / 0.3))
                flicker = 1 + 0.12 * np.sin(11.3 * seconds + 2 * c) * np.sin(
                    3.7 * seconds
                )
                linear = linear + 6.0 * on * flicker * flame * colours[c]
        glow = sum(g * wide_blur(linear, sigma, f) for sigma, f, g in GLOWS)
        light = 1 - torch.exp(-EXPOSURE * (linear.clamp_min(0) ** GAMMA + glow))
        image = background + (1 - background) * light
        image = image * (1 - WALL_ALPHA * wall) + WALL_ALPHA * wall * wall_rgb
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    track = soundtrack(rooms, tag if tag in STYLES else STYLE, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
