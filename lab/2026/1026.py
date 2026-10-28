"""
Jacobi's Star

Geodesics on an ellipsoid: hundreds of paths leave one point of an
egg-shaped surface in every direction, each as straight as a path on a curved
surface can be. Where neighbouring paths meet again they fold into a glowing
star with exactly four points. Then they keep going, and fold again, and again.

These "straight lines" of a curved surface are called geodesics. On a sphere,
all of them from one point meet again at the opposite pole, then back at the
start, over and over. On an ellipsoid with three different axes each meeting
point spreads out into a curve. Carl Jacobi claimed around 1842 that the first
always has exactly four cusps; it was proved only in 2004. Every later curve
here has four as well.

Listen: each cusp strikes a vibraphone note as it lights, the four notes of
one chord per curve, while soft glints follow each curve as it draws itself.
Every new curve brings a new chord over swelling strings, until the paths
stop and the music comes home.

Made with Python, NumPy & PyTorch. Sound synthesized in NumPy.
#geodesic #differentialgeometry #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
from loguru import logger
from scipy.interpolate import PchipInterpolator

from common import audio
from common.gpu import select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The surface and the paths ---------------------------------------------------
# Semi-axes along x (screen right), y (up) and z (towards the viewer at the start).
AXES = np.array([0.75, 1.0, 0.85])
LAUNCH = np.radians(
    30.0
)  # the start point, this far above the equator in the y-z plane
THREADS = 480  # geodesics, evenly spaced in launch direction
STEP = 1.25e-3  # arc-length step of the integration, about one screen pixel
GENERATIONS = 4  # conjugate loci shown: the star, then each later refocusing
LIMIT = 13.0  # integrate this far: every thread reaches its last conjugate point
OVERSHOOT = 1.02  # the fronts run this far past the last conjugate point
STAR_SPACING = 0.75  # screen pixels between the points that draw a locus

# --- Timing (s) --------------------------------------------------------------------
FRONT_END = 13.0  # the fronts stop here...
ACCEL = 1.0  # ...speeding up to (1 + ACCEL) times their starting speed...
BRAKE = 1.5  # ...and braking to a stop over the last BRAKE seconds
HOLD = 0.8  # the camera holds on the burst, then turns to face each locus as it lights
DRIFT = np.radians(12.0)  # and turns on this much more after the last, by REST
REST = 14.0
FADE = (12.0, 13.5)  # the threads dim to THREAD_REST once the fronts stop
THREAD_REST = 0.5

# --- Look --------------------------------------------------------------------------
SCALE = 600.0  # screen pixels per unit length
BACKGROUND = "#05070f"
THREAD_COLOURS = ("#ffb347", "#ff6b6b", "#8a6bff", "#ffb347")  # round the directions
STAR_COLOURS = ("#ffd27a", "#7fd0ff", "#ff8fc8", "#7dffc4")  # one per generation
SURFACE_COLOUR = "#2a3f6e"
SURFACE_FILL, SURFACE_RIM = 0.10, 0.55  # face-on and silhouette brightness
THREAD_WEIGHT = 0.12  # light per path sample
HEAD_GAIN, HEAD_LENGTH = 6.0, 0.06  # the bright front of each thread
TRAIL_FLOOR, TRAIL_LENGTH = 0.15, 0.8  # behind it, the thread dims to this
BACK_SIDE = 0.3  # brightness of what is behind the surface
STAR_WEIGHT = 1.2
FLASH_GAIN, FLASH_LENGTH = 4.0, 0.05  # a locus point flares as the front reaches it
EXPOSURE = 1.6
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 18.0, 4, 0.6

# --- Sound -------------------------------------------------------------------------
STYLE = "vibes"  # common.music.STYLES; an --audio-only tag naming one overrides it
PROGRESSION = (0, 5, 1, 4)  # chord degree under each locus: I, vi, ii, V, then home
GLINT_RATE = 4.0  # quiet notes per second while the loci light fastest


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def start_point() -> np.ndarray:
    """The launch point: LAUNCH above the equator, in the y-z plane, facing the viewer."""
    return np.array([0.0, AXES[1] * np.sin(LAUNCH), AXES[2] * np.cos(LAUNCH)])


def geodesics(start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit-speed geodesics from `start` in THREADS evenly spaced directions, by RK4
    in arc length up to LIMIT, with the scalar Jacobi equation m'' + K m = 0
    (m(0) = 0, m'(0) = 1) integrated alongside. Returns the positions
    (THREADS, steps + 1, 3) and m (THREADS, steps + 1).
    """
    inv2 = 1.0 / AXES**2
    inv4 = inv2**2
    product = np.prod(AXES**2)

    normal = start * inv2
    normal /= np.linalg.norm(normal)
    e1 = np.array([1.0, 0.0, 0.0])  # tangent: the normal has no x part
    e2 = np.cross(normal, e1)
    theta = 2 * np.pi * np.arange(THREADS) / THREADS
    x = np.repeat(start[None], THREADS, 0)
    v = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    m = np.zeros(THREADS)
    w = np.ones(THREADS)

    def deriv(x, v, m, w):
        g = (x**2 * inv4).sum(1)  # |grad F|^2 / 4
        acc = -((v**2 * inv2).sum(1) / g)[:, None] * x * inv2  # normal to the surface
        curvature = 1.0 / (product * g**2)  # Gaussian curvature K
        return v, acc, w, -curvature * m

    steps = int(round(LIMIT / STEP))
    positions = np.empty((THREADS, steps + 1, 3), np.float32)
    jacobi = np.empty((THREADS, steps + 1))
    positions[:, 0], jacobi[:, 0] = x, m
    h = STEP
    for j in range(1, steps + 1):
        k1 = deriv(x, v, m, w)
        k2 = deriv(*(s + 0.5 * h * d for s, d in zip((x, v, m, w), k1)))
        k3 = deriv(*(s + 0.5 * h * d for s, d in zip((x, v, m, w), k2)))
        k4 = deriv(*(s + h * d for s, d in zip((x, v, m, w), k3)))
        x, v, m, w = (
            s + h / 6 * (a + 2 * b + 2 * c + d)
            for s, a, b, c, d in zip((x, v, m, w), k1, k2, k3, k4)
        )
        # back onto the surface, unit speed along it
        x = x / np.sqrt((x**2 * inv2).sum(1))[:, None]
        n = x * inv2
        n /= np.linalg.norm(n, axis=1)[:, None]
        v = v - (v * n).sum(1)[:, None] * n
        v /= np.linalg.norm(v, axis=1)[:, None]
        positions[:, j], jacobi[:, j] = x, m
    return positions, jacobi


def conjugate_points(positions: np.ndarray, jacobi: np.ndarray, k: int):
    """The k-th zero of m along each geodesic: its arc length and position."""
    rows = np.arange(len(jacobi))
    changes = np.cumsum(
        np.signbit(jacobi[:, 1:-1]) != np.signbit(jacobi[:, 2:]), axis=1
    )
    if (changes[:, -1] < k).any():
        raise ValueError(f"raise LIMIT: not every geodesic reaches conjugate point {k}")
    j = 1 + np.argmax(changes >= k, axis=1)
    fraction = jacobi[rows, j] / (jacobi[rows, j] - jacobi[rows, j + 1])
    distance = (j + fraction) * STEP
    point = positions[rows, j] + fraction[:, None] * (
        positions[rows, j + 1] - positions[rows, j]
    )
    return distance, point


def cusps(distance: np.ndarray) -> np.ndarray:
    """Threads whose conjugate distance is a local extremum: the locus's cusps."""
    before, after = np.roll(distance, 1), np.roll(distance, -1)
    extremum = ((distance > before) & (distance > after)) | (
        (distance < before) & (distance < after)
    )
    return np.flatnonzero(extremum)


def densify(point: np.ndarray, distance: np.ndarray):
    """A closed locus resampled every STAR_SPACING pixels, with its distances."""
    following, further = np.roll(point, -1, 0), np.roll(distance, -1)
    length = np.linalg.norm(following - point, axis=1) * SCALE
    count = np.maximum(1, np.ceil(length / STAR_SPACING)).astype(int)
    segment = np.repeat(np.arange(len(point)), count)
    f = (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)) / count[
        segment
    ]
    dense = point[segment] * (1 - f)[:, None] + following[segment] * f[:, None]
    dense /= np.sqrt((dense**2 / AXES**2).sum(1))[:, None]  # back onto the surface
    return dense, distance[segment] * (1 - f) + further[segment] * f


def front_table(end: float) -> tuple[np.ndarray, np.ndarray]:
    """
    The arc length reached by the fronts, sampled finely in seconds: speeding
    up by (1 + ACCEL) until FRONT_END, braking over the last BRAKE seconds, and
    reaching `end` exactly as they stop.
    """
    t = np.linspace(0.0, FRONT_END, 4001)
    speed = (1 + ACCEL * t / FRONT_END) * (
        1 - smoothstep((t - (FRONT_END - BRAKE)) / BRAKE)
    )
    reached = np.concatenate(
        [[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(t))]
    )
    return t, end * reached / reached[-1]


def facing_angles(normal: np.ndarray) -> tuple[float, float]:
    """Tilt and spin of `_rotation` that turn `normal` straight at the viewer."""
    normal = normal / np.linalg.norm(normal)
    return float(np.arcsin(normal[1])), float(np.arctan2(-normal[0], normal[2]))


def _rotation(tilt: float, spin: float) -> np.ndarray:
    """World -> camera: a spin about the vertical axis, then a tilt."""
    a, b = tilt, spin
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rx @ ry


def sound_events(loci, front_s, front_t, view) -> list[dict]:
    """
    For the soundtrack, from the same closed forms as the frames: for each
    locus, when each of its points lights (the fronts reach its distance), where
    it is on screen then (x, -1 to 1), and which of its threads are cusps.
    """
    events = []
    half_width = WIDTH / 2 / SCALE
    for distance, point in loci:
        when = np.interp(distance, front_s, front_t)
        x = np.array([(view(t) @ q)[0] for t, q in zip(when, point)]) / half_width
        events.append(dict(when=when, x=np.clip(x, -1, 1), cusps=cusps(distance)))
    return events


def soundtrack(events: list[dict], style_name: str, rng: np.random.Generator):
    """
    A one-shot following the loci.

    - The burst: the root over a low root.
    - Strings hold a chord under each locus (PROGRESSION: I, vi, ii, V),
      changing as its first point lights, louder and brighter each time, with
      a low note on each change.
    - Each cusp strikes a note as it lights: the four notes of that locus's
      seventh chord, low to high in the order its cusps light, panned to the
      cusp on screen, louder generation by generation.
    - Quiet glints while the loci light, their rate following how many locus
      points light per second against a typical moment (capped at GLINT_RATE),
      on the locus's chord tones an
      octave up, panned to a point lighting then.
    - As the fronts stop (FRONT_END), the home chord: low root, rolled chord
      and strings that ring out.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def register(chord):  # whole octaves so the chord sits in MIDI 57..84
        chord = list(chord)
        while max(chord) > 84:
            chord = [n - 12 for n in chord]
        while min(chord) < 57:
            chord = [n + 12 for n in chord]
        return chord

    def struck(midi, velocity, x=0.0):
        return audio.pan(style.voice(float(audio.midi_to_hz(midi)), velocity), x)

    def low(degree):  # the chord root folded into MIDI 36..47
        return float(audio.midi_to_hz(36 + (scale.note(degree) - 36) % 12))

    # The burst.
    audio.add(track, struck(register([scale.note(0)])[0], 0.45), 0.0, **kw)
    audio.add(track, style.bass(low(0), 0.45, length=3.0), 0.0, **kw)

    # Strings under each locus, and a low note as each begins.
    bounds = [0.0] + [float(e["when"].min()) for e in events] + [FRONT_END]
    degrees = (0,) + tuple(PROGRESSION)
    for i, degree in enumerate(degrees):
        begin, finish = bounds[i], bounds[i + 1]
        swell = (i + 1) / len(degrees)
        for midi in scale.chord(degree, 3):
            f = float(audio.midi_to_hz(48 + (midi - 48) % 12))
            tone = audio.strings(
                f, finish - begin + 0.3, 0.2 + 0.3 * swell, 0.6, 0.8, 0.25 + 0.5 * swell
            )
            audio.add(track, tone, max(begin - 0.3, 0.0), **kw)
        if i:
            audio.add(track, style.bass(low(degree), 0.5, length=3.0), begin, **kw)

    # The cusps, and the glints.
    step = 0.05
    bins = np.arange(0.0, DURATION + step, step)
    counts = [np.histogram(e["when"], bins)[0] for e in events]
    # near a cusp hundreds of points light at once: scale by a typical moment
    typical = np.median(np.concatenate([c[c > 0] for c in counts]))
    glints = 0
    for k, (e, count) in enumerate(zip(events, counts)):
        chord = register(scale.chord(PROGRESSION[k % len(PROGRESSION)], 4))
        order = e["cusps"][np.argsort(e["when"][e["cusps"]])]
        for rank, c in enumerate(order):
            note = struck(chord[rank % 4], 0.32 + 0.06 * k, 0.8 * e["x"][c])
            audio.add(track, note, e["when"][c], **kw)
        for b in np.flatnonzero(count):
            for when in bins[b] + step * rng.uniform(
                size=rng.poisson(GLINT_RATE * step * min(1.0, count[b] / typical))
            ):
                lighting = np.flatnonzero(
                    (e["when"] >= bins[b]) & (e["when"] < bins[b + 1])
                )
                x = e["x"][rng.choice(lighting)]
                midi = chord[rng.integers(4)] + 12
                midi = midi - 12 if midi > 84 else midi
                audio.add(
                    track, struck(midi, rng.uniform(0.08, 0.16), 0.8 * x), when, **kw
                )
                glints += 1

    # Home, as the fronts stop.
    audio.add(track, style.bass(low(0), 0.6, length=4.0), FRONT_END, **kw)
    for i, midi in enumerate(register(scale.chord(0, 4))):
        audio.add(
            track, struck(midi, 0.42, 0.4 * (i - 1.5)), FRONT_END + 0.07 * i, **kw
        )
    for midi in scale.chord(0, 3):
        f = float(audio.midi_to_hz(48 + (midi - 48) % 12))
        tone = audio.strings(f, DURATION - FRONT_END - 1.0, 0.5, 0.6, 1.2, 0.75)
        audio.add(track, tone, FRONT_END - 0.3, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, "
        f"{sum(len(e['cusps']) for e in events)} cusp notes, {glints} glints"
    )
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.5)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Geodesics from one point of a triaxial ellipsoid and their successive
    conjugate loci, starting with Jacobi's four-cusped star.

    The ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 (AXES) is the level set F = 1.
    A unit-speed geodesic has acceleration normal to it, x'' = -mu grad F, and
    keeping x' tangent fixes mu = (x'^T H x') / |grad F|^2, which here is
    x'' = -(sum x_i'^2/a_i^2) / (sum x_i^2/a_i^4) * x/a^2. THREADS of them leave
    the start point (LAUNCH above the equator in the y-z plane, which holds no
    umbilic) in evenly spaced directions and are integrated together by RK4 in
    arc length (STEP), re-projected onto the surface each step. Alongside each,
    the scalar Jacobi equation m'' + K m = 0 with the Gaussian curvature
    K = 1 / (a^2 b^2 c^2 (sum x_i^2/a_i^4)^2) measures how far neighbouring
    geodesics are apart; its k-th zero is the k-th conjugate point, where they
    meet again. Those points, in launch order, trace the k-th conjugate locus
    (GENERATIONS of them, resampled every STAR_SPACING pixels), and its cusps
    are the directions where the k-th conjugate distance is stationary (logged:
    Jacobi's statement says four for the first; here each later one has four
    too).

    Every frame is a closed form in t. The fronts sit at arc length front(t),
    a table integrating a speed that grows by (1 + ACCEL) and brakes to a stop
    at FRONT_END, OVERSHOOT past the last conjugate point. Each thread is
    splatted on the GPU up to its front, brightest at the front (HEAD_GAIN,
    HEAD_LENGTH) and dimming behind it to TRAIL_FLOOR, coloured by launch
    direction, and dimmed where the surface faces away (BACK_SIDE); the bright
    part fades as the fronts brake. A locus point lights once the fronts pass
    its distance, with a short flare, in its generation's colour, so each locus
    grows from its nearest cusps to its farthest. The camera (orthographic,
    SCALE px per unit) is a spin about the vertical axis then a tilt; it holds
    on the burst for HOLD, then turns, always the same way, to face each
    locus's mean normal at the moment the fronts are halfway through it, and
    drifts on by DRIFT to rest at REST (a monotone cubic through those
    keyframes). The ellipsoid itself is a faint per-pixel ray cast with a
    Fresnel-like rim. Light is tone-mapped (1 - exp(-EXPOSURE x)) and bloomed
    with a wide, cheap blur.

    `sound_events` reads, for each locus, when its points light and where
    they are on screen; `soundtrack` turns its cusps into the notes of a chord
    and the lighting into glints over a string progression.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    inv2 = 1.0 / AXES**2

    start = start_point()
    positions, jacobi = geodesics(start)
    loci = [conjugate_points(positions, jacobi, k) for k in range(1, GENERATIONS + 1)]
    end = float(loci[-1][0].max() * OVERSHOOT)
    table_t, table_s = front_table(end)

    def front(seconds: float) -> float:
        return float(np.interp(seconds, table_t, table_s))

    # Camera keyframes: the start point, then each locus as the fronts cross it.
    tilt, spin = facing_angles(start * inv2)
    keys = [(0.0, tilt, spin), (HOLD, tilt, spin)]
    for k, (distance, point) in enumerate(loci, 1):
        lights = np.interp([distance.min(), distance.max()], table_s, table_t)
        when = float(np.interp(np.median(distance), table_s, table_t))
        a, b = facing_angles(
            (point * inv2 / np.linalg.norm(point * inv2, axis=1)[:, None]).sum(0)
        )
        b = keys[-1][2] + (b - keys[-1][2]) % (2 * np.pi)  # always turning the same way
        keys.append((when, a, b))
        logger.info(
            f"locus {k}: distances {distance.min():.2f}..{distance.max():.2f}, "
            f"{len(cusps(distance))} cusps, lights {lights[0]:.1f}..{lights[1]:.1f} s"
        )
    keys += [
        (REST, keys[-1][1], keys[-1][2] + DRIFT),
        (DURATION, keys[-1][1], keys[-1][2] + DRIFT),
    ]
    times, tilts, spins = map(np.array, zip(*keys))
    camera_tilt, camera_spin = PchipInterpolator(times, tilts), PchipInterpolator(
        times, spins
    )

    # The paths up to the end, as tensors.
    samples = int(np.ceil(end / STEP)) + 1
    to = dict(device=device, dtype=torch.float32)
    path = torch.tensor(positions[:, :samples], **to)
    normals = path * torch.tensor(inv2, **to)
    normals /= normals.norm(dim=-1, keepdim=True)
    arc = torch.arange(samples, **to) * STEP
    lut = palette_lut(THREAD_COLOURS, 1024)
    hue = (np.arange(THREADS) / THREADS * 1023).astype(int)
    thread_rgb = torch.tensor(lut[hue], **to)
    del positions, jacobi
    gc.collect()

    # The loci, resampled, with each point's distance and colour.
    dense = [densify(point, distance) for distance, point in loci]
    star = np.concatenate([p for p, _ in dense])
    star_normals = star * inv2
    star_normals /= np.linalg.norm(star_normals, axis=1)[:, None]
    star_rgb = np.concatenate(
        [
            np.tile(_rgb(STAR_COLOURS[k % len(STAR_COLOURS)]), (len(p), 1))
            for k, (p, _) in enumerate(dense)
        ]
    )
    star, star_normals = torch.tensor(star, **to), torch.tensor(star_normals, **to)
    star_distance = torch.tensor(np.concatenate([d for _, d in dense]), **to)
    star_rgb = torch.tensor(star_rgb, **to)

    # The screen in camera units, for the surface.
    gx = (torch.arange(WIDTH, **to) + 0.5 - WIDTH / 2) / SCALE
    gy = (HEIGHT / 2 - torch.arange(HEIGHT, **to) - 0.5) / SCALE
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
    inv2_t = torch.tensor(inv2, **to)
    surface_rgb = torch.tensor(_rgb(SURFACE_COLOUR), **to)
    background = torch.tensor(_rgb(BACKGROUND), **to)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **to)

    def facing_weight(z: torch.Tensor) -> torch.Tensor:
        u = ((z + 0.15) / 0.3).clamp(0.0, 1.0)
        return BACK_SIDE + (1 - BACK_SIDE) * u * u * (3 - 2 * u)

    def surface(rotation: torch.Tensor) -> torch.Tensor:
        """Ray cast of the ellipsoid along the view axis: a faint fill and rim."""
        o = gx[..., None] * rotation[0] + gy[..., None] * rotation[1]
        d = rotation[2]
        a = (d**2 * inv2_t).sum()
        b = (o * d * inv2_t).sum(-1)
        c = (o**2 * inv2_t).sum(-1) - 1
        disc = b * b - a * c
        root = disc.clamp_min(0).sqrt()
        hit = o + ((-b + root) / a)[..., None] * d
        n = hit * inv2_t
        facing = (n @ rotation[2]) / n.norm(dim=-1)
        cover = (root / a * SCALE / 2).clamp(0, 1)  # soft silhouette, about a pixel
        shade = SURFACE_FILL + SURFACE_RIM * (1 - facing.clamp(0, 1)) ** 3
        return (cover * shade)[..., None] * surface_rgb

    def render(frame: int) -> torch.Tensor:
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        rotation = torch.tensor(
            _rotation(float(camera_tilt(seconds)), float(camera_spin(seconds))), **to
        )
        h = front(seconds)
        level = 1 - (1 - THREAD_REST) * smoothstep(
            (seconds - FADE[0]) / (FADE[1] - FADE[0])
        )
        moving = 1 - smoothstep((seconds - (FRONT_END - BRAKE)) / BRAKE)
        canvas.zero_()

        reached = min(samples, int(h / STEP) + 2)
        cam = path[:, :reached] @ rotation.T
        behind = (h - arc[:reached]).clamp_min(0)
        weight = (
            THREAD_WEIGHT
            * level
            * (arc[:reached] <= h)
            * (
                TRAIL_FLOOR
                + moving * (1 - TRAIL_FLOOR) * torch.exp(-behind / TRAIL_LENGTH)
                + moving * HEAD_GAIN * torch.exp(-behind / HEAD_LENGTH)
            )
        )
        weight = weight * facing_weight(normals[:, :reached] @ rotation[2])
        splat(
            canvas,
            WIDTH / 2 + SCALE * cam[..., 0],
            HEIGHT / 2 - SCALE * cam[..., 1],
            (thread_rgb[:, None, :] * weight[..., None]).reshape(-1, 3),
        )

        past = h - star_distance
        lit = past >= 0
        if bool(lit.any()):
            cam = star[lit] @ rotation.T
            glow = STAR_WEIGHT * (1 + FLASH_GAIN * torch.exp(-past[lit] / FLASH_LENGTH))
            glow = glow * facing_weight(star_normals[lit] @ rotation[2])
            splat(
                canvas,
                WIDTH / 2 + SCALE * cam[:, 0],
                HEIGHT / 2 - SCALE * cam[:, 1],
                glow[:, None] * star_rgb[lit],
            )

        light = 1 - torch.exp(-EXPOSURE * canvas)
        image = background + surface(rotation) + light
        image = image + BLOOM_GAIN * wide_blur(light, BLOOM_SIGMA, BLOOM_FACTOR)
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)

    def view(seconds: float) -> np.ndarray:
        return _rotation(float(camera_tilt(seconds)), float(camera_spin(seconds)))

    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    events = sound_events(loci, table_s, table_t, view)
    track = soundtrack(events, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
