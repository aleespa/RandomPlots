"""
Flame Metamorphosis

A handful of simple rules: shrink, turn, shift, then bend the plane with a
twist. Pick one at random, apply it to a point, and repeat. Do this millions
of times and the points never wander off. They settle on a single shape, the
attractor of the rules, glowing wherever they land most often.

These are fractal flames, invented by Scott Draves in 1992. Each rule is an
affine map z → Az + b followed by a nonlinear "variation" such as a swirl, a
sphere inversion or a horseshoe. Here the rules themselves drift slowly
through a loop, so the attractor melts from one creature into the next, and
copies of itself fade in and out as a hidden rotation joins the game.

The music plays the same game. On every beat a rule is picked with the same
odds the picture uses, and each rule strikes its own gamelan bell; when the
rotation is picked, the last bell echoes an octave higher. Under it, each
flame holds its own chord, fading into the next as the shape melts.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#fractalflame #fractal #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from numba import njit

from colors.palettes import EMBER_FURNACE
from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale, beat_times
from common.video import FPS, HEIGHT, WIDTH, frame_count, palette_lut, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# Orange haze -> white -> cyan spine, indexed by map colour.
PALETTE = EMBER_FURNACE[3:8]
SUPERSAMPLE = 2
POINTS = 600_000  # chaos-game trajectories per frame
ITERATIONS = 40  # steps per trajectory
BURN_IN = 8  # steps before a trajectory is plotted
KEYFRAMES = 3  # flames the loop passes through
BASE_MAPS = 3  # random maps per flame (plus one rotation map)
DWELL = 2.0  # how long the loop lingers near each keyframe (softmax sharpness)
WOBBLE = 0.12  # radians each map's angle sways
ROTATION_TURNS = 0  # whole turns of the camera per loop
GAMMA = 2.2
CAMERA_STEPS = 120  # camera samples around the loop
CAMERA_POINTS = 8_000
BLOOM = 0.3

STYLE = "gamelan"  # common.music.STYLES
BPM = 96  # 6 bars per loop
PROGRESSION = (0, 5, 3)  # chord (scale degree) under each keyframe: I, vi, IV

VARIATIONS = ("linear", "sinusoidal", "spherical", "swirl", "horseshoe", "disc")


@njit(inline="always")
def _hash(key):
    """lowbias32: a counter-based uniform in [0, 1), identical on every frame."""
    x = key & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x * (1.0 / 4294967296.0)


@njit(inline="always")
def _apply(k, x, y, affine, weights):
    """Map k: affine part, then the weighted sum of variations."""
    u = affine[k, 0] * x + affine[k, 1] * y + affine[k, 4]
    v = affine[k, 2] * x + affine[k, 3] * y + affine[k, 5]
    r2 = u * u + v * v + 1e-12
    r = np.sqrt(r2)
    ox = 0.0
    oy = 0.0
    w = weights[k, 0]
    if w != 0.0:
        ox += w * u
        oy += w * v
    w = weights[k, 1]
    if w != 0.0:
        ox += w * np.sin(u)
        oy += w * np.sin(v)
    w = weights[k, 2]
    if w != 0.0:
        ox += w * u / r2
        oy += w * v / r2
    w = weights[k, 3]
    if w != 0.0:
        s = np.sin(r2)
        c = np.cos(r2)
        ox += w * (u * s - v * c)
        oy += w * (u * c + v * s)
    w = weights[k, 4]
    if w != 0.0:
        ox += w * (u - v) * (u + v) / r
        oy += w * 2.0 * u * v / r
    w = weights[k, 5]
    if w != 0.0:
        th = np.arctan2(u, v) / np.pi
        ox += w * th * np.sin(np.pi * r)
        oy += w * th * np.cos(np.pi * r)
    return ox, oy


@njit(nogil=True, fastmath=True, cache=True)
def _chaos_game(
    affine,
    weights,
    colour,
    speed,
    cumulative,
    points,
    seed,
    view,
    density,
    hue,
    samples,
):
    """
    Run `points` trajectories of ITERATIONS steps. With `samples` of shape
    (points, 2) the last position of each trajectory is stored there (camera
    pass); otherwise plotted positions are binned into density/hue through the
    camera `view` = (cx, cy, cos, sin, scale).
    """
    maps = cumulative.shape[0]
    height, width = density.shape
    cx, cy, cr, sr, scale = view[0], view[1], view[2], view[3], view[4]
    store = samples.shape[0] > 0
    for p in range(points):
        base = seed + p * 64
        x = 2.0 * _hash(base + 61) - 1.0
        y = 2.0 * _hash(base + 62) - 1.0
        c = _hash(base + 63)
        for it in range(ITERATIONS):
            u = _hash(base + it)
            k = 0
            while k < maps - 1 and u > cumulative[k]:
                k += 1
            x, y = _apply(k, x, y, affine, weights)
            c += speed[k] * (colour[k] - c)
            if not (np.abs(x) < 1e6 and np.abs(y) < 1e6):
                x = 2.0 * _hash(base + 61 + it) - 1.0
                y = 2.0 * _hash(base + 62 + it) - 1.0
                continue
            if store or it < BURN_IN:
                continue
            dx = x - cx
            dy = y - cy
            px = 0.5 * width + scale * (cr * dx - sr * dy)
            py = 0.5 * height - scale * (sr * dx + cr * dy)
            if 0.0 <= px < width and 0.0 <= py < height:
                iy = int(py)
                ix = int(px)
                density[iy, ix] += 1.0
                hue[iy, ix] += c
        if store:
            samples[p, 0] = x
            samples[p, 1] = y


def _keyframes(rng):
    """Random flames to loop through, as interpolable parameter arrays."""
    keys = []
    for _ in range(KEYFRAMES):
        theta = rng.uniform(0, 2 * np.pi, BASE_MAPS)
        sx = rng.uniform(0.45, 0.9, BASE_MAPS) * rng.choice([-1, 1], BASE_MAPS)
        sy = rng.uniform(0.45, 0.9, BASE_MAPS) * rng.choice([-1, 1], BASE_MAPS)
        shear = rng.uniform(-0.3, 0.3, BASE_MAPS)
        tx, ty = rng.uniform(-0.9, 0.9, (2, BASE_MAPS))
        weights = np.zeros((BASE_MAPS, len(VARIATIONS)))
        for k in range(BASE_MAPS):
            chosen = rng.choice(len(VARIATIONS), 2, replace=False)
            weights[k, chosen] = rng.dirichlet([1.5, 1.5])
        prob = rng.uniform(0.5, 1.5, BASE_MAPS)
        keys.append(
            dict(
                theta=theta,
                sx=sx,
                sy=sy,
                shear=shear,
                tx=tx,
                ty=ty,
                weights=weights,
                prob=prob,
            )
        )
    # The rotation map: present in some keyframes, absent in others.
    symmetry = rng.permutation([1.0, 0.0] + list(rng.integers(0, 2, KEYFRAMES - 2)))
    for key, on in zip(keys, symmetry):
        key["sym_prob"] = on * rng.uniform(0.6, 1.2)
    return keys


def _blend(t):
    """Periodic softmax weights of the keyframes at loop time t (sums to 1)."""
    phase = np.cos(2 * np.pi * (t - np.arange(KEYFRAMES) / KEYFRAMES))
    blend = np.exp(DWELL * phase)
    return blend / blend.sum()


def _frame_params(t, keys, order, wobble):
    """
    Blend the keyframes with periodic softmax weights, and sway each map's
    angle by a whole number of cycles: an exact loop in t.
    """
    blend = _blend(t)

    def mix(name):
        return sum(b * key[name] for b, key in zip(blend, keys))

    theta, sx, sy, shear = mix("theta"), mix("sx"), mix("sy"), mix("shear")
    amplitude, cycles, offset = wobble
    theta = theta + amplitude * np.sin(2 * np.pi * cycles * t + offset)
    c, s = np.cos(theta), np.sin(theta)
    affine = np.zeros((BASE_MAPS + 1, 6))
    # R(theta) @ [[sx, shear], [0, sy]]
    affine[:BASE_MAPS, 0] = c * sx
    affine[:BASE_MAPS, 1] = c * shear - s * sy
    affine[:BASE_MAPS, 2] = s * sx
    affine[:BASE_MAPS, 3] = s * shear + c * sy
    affine[:BASE_MAPS, 4] = mix("tx")
    affine[:BASE_MAPS, 5] = mix("ty")
    angle = 2 * np.pi / order
    affine[BASE_MAPS] = [
        np.cos(angle),
        -np.sin(angle),
        np.sin(angle),
        np.cos(angle),
        0,
        0,
    ]

    weights = np.zeros((BASE_MAPS + 1, len(VARIATIONS)))
    weights[:BASE_MAPS] = mix("weights")
    weights[BASE_MAPS, 0] = 1.0

    colour = np.append(np.linspace(0.0, 1.0, BASE_MAPS), 0.0)
    speed = np.append(np.full(BASE_MAPS, 0.5), 0.0)  # rotation keeps the colour
    prob = np.append(mix("prob"), mix("sym_prob") * mix("prob").sum() / BASE_MAPS)
    cumulative = np.cumsum(prob) / prob.sum()
    return affine, weights, colour, speed, cumulative


def make_renderer(rng):
    """
    Build the flame loop from `rng` and return render(t) -> (HEIGHT, WIDTH, 3)
    float RGB in 0..255, plus the flame itself (keyframes, rotation order,
    wobble, camera track) for the soundtrack. Separate from `generate` so seeds
    can be auditioned with single frames.
    """
    keys = _keyframes(rng)
    order = int(rng.integers(3, 8))
    seed = int(rng.integers(0, 2**20)) * 2**26
    wobble = (
        np.full(BASE_MAPS, WOBBLE),
        rng.integers(1, 4, BASE_MAPS),
        rng.uniform(0, 2 * np.pi, BASE_MAPS),
    )
    lut = palette_lut(PALETTE, 1024) * 255.0
    height, width = HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE
    no_samples = np.zeros((0, 2))
    no_view = np.zeros(5)
    logger.info(f"{KEYFRAMES} keyframes, {BASE_MAPS} maps + {order}-fold rotation")

    # Camera pass: centre and radius of the attractor at CAMERA_STEPS times
    # around the loop, from a small chaos game, smoothed circularly; frames
    # interpolate between them.
    centre = np.zeros((CAMERA_STEPS, 2))
    radius = np.zeros(CAMERA_STEPS)
    scratch = np.zeros((1, 1))
    for step in range(CAMERA_STEPS):
        params = _frame_params(step / CAMERA_STEPS, keys, order, wobble)
        samples = np.zeros((CAMERA_POINTS, 2))
        _chaos_game(*params, CAMERA_POINTS, seed, no_view, scratch, scratch, samples)
        centre[step] = np.median(samples, axis=0)
        radius[step] = np.percentile(np.hypot(*(samples - centre[step]).T), 97)
    sigma = 0.03 * CAMERA_STEPS  # 3% of the loop
    offsets = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    idx = (np.arange(CAMERA_STEPS)[:, None] + offsets[None, :]) % CAMERA_STEPS
    track = np.column_stack([centre, np.log(radius)])
    track = np.tensordot(track[idx], kernel, axes=([1], [0]))

    def render(t):
        f = t * CAMERA_STEPS
        i0 = int(np.floor(f)) % CAMERA_STEPS
        a = f - np.floor(f)
        cx, cy, log_r = (1 - a) * track[i0] + a * track[(i0 + 1) % CAMERA_STEPS]
        turn = 2 * np.pi * ROTATION_TURNS * t
        view = np.array(
            [cx, cy, np.cos(turn), np.sin(turn), 0.47 * width / np.exp(log_r)]
        )

        density = np.zeros((height, width))
        hue = np.zeros((height, width))
        params = _frame_params(t, keys, order, wobble)
        _chaos_game(*params, POINTS, seed, view, density, hue, no_samples)

        density = cv2.resize(density, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        hue = cv2.resize(hue, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        hit = density > 0
        hue = np.divide(hue, density, out=np.zeros_like(hue), where=hit)
        reference = np.percentile(density[hit], 99.0) if hit.any() else 1.0
        scale = SUPERSAMPLE**2  # density counts per output pixel
        alpha = np.log1p(density * scale) / np.log1p(reference * scale)
        bright = np.clip(alpha, 0.0, 1.0) ** (1.0 / GAMMA)

        colour = lut[np.clip((hue * 1023).astype(np.int32), 0, 1023)]
        image = colour * bright[..., None]
        image += (255.0 - colour) * (bright**8)[..., None] * 0.35  # white-hot cores
        image = image.astype(np.float32)
        glow = cv2.GaussianBlur(cv2.resize(image, (WIDTH // 4, HEIGHT // 4)), (0, 0), 6)
        image += BLOOM * cv2.resize(glow, (WIDTH, HEIGHT))
        return image

    flame = dict(keys=keys, order=order, wobble=wobble, camera=track)
    return render, flame


def soundtrack(flame, rng):
    """
    The chaos game made audible, over a pad that follows the morph.

    On every eighth note at BPM the soundtrack plays one step of the chaos
    game: it picks a map with the same probabilities the picture uses at that
    moment (uniforms from `rng`, so --audio-seed re-rolls the melody), applies
    it to a single point, and strikes that map's note of the current chord
    (map k -> chord tone k; the chord is PROGRESSION[j] for the dominant
    keyframe j). The point's position, normalised by the camera, sets pan (x)
    and register (y, an octave up in the top of the frame). Picking the
    rotation map plays no new pitch: the previous note echoes an octave higher
    on the other side, like a rotated copy. A bass note marks every bar, and a
    pad holds each keyframe's chord, faded in and out by the same softmax
    blend as the picture. Everything is circular, so the sound loops.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    keys, order, wobble, camera = (
        flame["keys"],
        flame["order"],
        flame["wobble"],
        flame["camera"],
    )
    track = audio.silence(DURATION)

    def chord_root(t):
        return PROGRESSION[int(np.argmax(_blend(t)))]

    # The chaos game, one step per eighth note.
    times = beat_times(DURATION, BPM, subdivision=2)
    choices = rng.uniform(size=len(times))
    x, y = rng.uniform(-1.0, 1.0, 2)
    affine, weights, _, _, cumulative = _frame_params(0.0, keys, order, wobble)
    for u in rng.uniform(size=20):  # burn in onto the attractor
        k = min(int(np.searchsorted(cumulative, u)), BASE_MAPS)
        x, y = _apply(k, x, y, affine, weights)
    previous = None
    notes = echoes = 0
    for i, when in enumerate(times):
        t = when / DURATION
        affine, weights, _, _, cumulative = _frame_params(t, keys, order, wobble)
        k = min(int(np.searchsorted(cumulative, choices[i])), BASE_MAPS)
        x, y = _apply(k, x, y, affine, weights)
        if not (abs(x) < 1e6 and abs(y) < 1e6):
            x, y = rng.uniform(-1.0, 1.0, 2)
        f = t * CAMERA_STEPS
        i0 = int(f) % CAMERA_STEPS
        a = f - int(f)
        cx, cy, log_r = (1 - a) * camera[i0] + a * camera[(i0 + 1) % CAMERA_STEPS]
        r = np.exp(log_r)
        position = float(np.clip(0.8 * (x - cx) / r, -0.9, 0.9))
        accent = 0.1 if i % 8 == 0 else 0.0
        if k == BASE_MAPS:  # the rotation map: an echo of the last note
            if previous is not None:
                freq, side = previous
                note = style.voice(2.0 * freq, 0.22)
                audio.add(track, audio.pan(note, -side), when)
                echoes += 1
            continue
        lift = len(scale.steps) if (y - cy) / r > 0.15 else 0
        freq = scale.hz(chord_root(t) + 2 * k + lift)
        velocity = (
            0.32 + 0.25 * (cumulative[k] - (cumulative[k - 1] if k else 0.0)) + accent
        )
        audio.add(track, audio.pan(style.voice(freq, velocity), position), when)
        previous = (freq, position)
        notes += 1

    # A bass note on every bar.
    for when in beat_times(DURATION, BPM / 4):
        bass = scale.hz(chord_root(when / DURATION), octave=-2)
        audio.add(track, style.bass(bass, 0.45, length=3.0), when)

    # The pad: each keyframe's chord, faded by its blend weight. Held for the
    # whole loop with matching attack and release, so the wrapped tail
    # crossfades into the start.
    n = len(track)
    blend = np.array([_blend(i / n) for i in range(0, n, 480)])
    envelope = np.stack(
        [
            np.interp(np.arange(n), np.arange(0, n, 480), blend[:, j])
            for j in range(KEYFRAMES)
        ],
        axis=1,
    )
    for j, degree in enumerate(PROGRESSION):
        layer = audio.silence(DURATION)
        for midi in scale.chord(degree, 3, octave=-1):
            tone = audio.pad(
                float(audio.midi_to_hz(midi)), DURATION, 0.3, 3.0, 3.0, 0.3
            )
            audio.add(layer, tone, 0.0)
        track += envelope[:, j : j + 1] * layer

    logger.info(
        f"Soundtrack: {STYLE}, root MIDI {scale.root}, {notes} notes, {echoes} echoes"
    )
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Fractal flame (Draves & Reckase) whose parameters loop through KEYFRAMES
    random flames.

    Each flame has BASE_MAPS maps F_k(z) = Σ_v w_kv V_v(A_k z + b_k): a random
    contractive affine map (rotation θ, scales sx, sy, shear, translation)
    followed by a weighted mix of two of six variations (linear, sinusoidal,
    spherical, swirl, horseshoe, disc). A fourth map, a rotation by 2π/n, is
    switched on in some keyframes and off in others, so n-fold symmetry fades
    in and out. Every parameter is a convex blend of the keyframes with
    periodic softmax weights exp(DWELL·cos 2π(t − j/K)), so the loop is exact
    and lingers near each keyframe; each map's angle also sways by WOBBLE
    radians over 1–3 whole cycles, so the holds keep moving.

    A frame is a pure function of t: POINTS trajectories of ITERATIONS steps run
    a chaos game (numba, nogil) with a counter-based hash, so every frame uses
    the same random choices and the image morphs coherently instead of
    flickering. Plotted points are binned at SUPERSAMPLE× into a density and an
    average map-colour buffer; brightness is log-density (normalised to the
    99th percentile) with gamma, hue comes from the colour index through the
    palette, the densest cores bleach to white, and a quarter-resolution blur
    adds bloom. The camera follows the attractor's median and 97th-percentile
    radius, measured with a cheap pass over all frames up front and smoothed
    circularly. Frames render in parallel on the thread pool.

    Posted with --seed 1.
    """
    settings = settings or ImageProcessingSettings(1)
    render_t, flame = make_renderer(settings.rng)

    def render(frame):
        return render_t(frame / FRAME_COUNT)

    render_frames(settings, render, FRAME_COUNT, rgb=True)
    track = soundtrack(flame, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
