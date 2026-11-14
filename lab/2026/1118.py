"""
The Third Root

A Newton's method fractal. Every point on screen is a first guess for
Newton's method, the classic recipe for solving an equation, and is coloured
by the root its guesses settle on.

With two roots it is easy: you reach the nearer one, and the plane splits
along a straight line. Arthur Cayley proved that in 1879, then tried three
roots and called it a problem of "considerable difficulty". Watch the third
root fly in: the line erupts into chains of beads, and wherever two colours
meet, the third is there too. It is a fractal: zoom into the centre and the
beads repeat forever.

A harp climbs as the third root nears, and the chord is complete when it
lands. In the zoom, each new generation of beads plucks one more step of a
scale that keeps rising without ever getting higher.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#newtonfractal #fractal #mathart #generativeart #randomplots
"""

import gc

import numpy as np
from loguru import logger
from numba import njit

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Story (seconds) ------------------------------------------------------------------
ARRIVE = 5.0  # the third root lands
FAR = 30.0  # its distance at t = 0, in units of the final root circle
LANDING = 2.5  # ease-out power of its approach: fast at first, a gentle landing
SWOOP = 0.9  # rad: how far round it curls on the way in (sign from the seed)
ZOOM_START, ZOOM_END = 5.8, 13.4  # zoom into the centre
GENERATIONS = 24  # bead generations passed by the zoom: depth (3/2)^(G/2)
TWIST = np.pi / 3  # rad the view turns during the zoom (the flower is 6-fold)
FADE = 0.4  # s of fade to black at the very end

# --- Maths ----------------------------------------------------------------------------
HALF_WIDTH = 1.55  # complex units across half the screen, before the zoom
MAX_ITER = 80
EPS = 1e-3  # converged: within this of a root
FAR_FIELD = 20.0  # beyond this |z|, act 2 jumps z -> (2/3)^k z in one go
SS = 2  # supersampling per axis, on boundary pixels
AA_TOL = 10.0  # 0..255: corners closer than this need no supersampling

# --- Look -----------------------------------------------------------------------------
PALETTE = ["#ff5a3c", "#20d6a8", "#a77bff"]  # ember, jade, lilac: one per root
GLOW_BASE = 3.0  # steps every pixel gets for free: below this it stays dark
GLOW_SCALE = 8.0  # steps more for the boundary glow to saturate
DOT = 7.0  # px: root marker core

# --- Sound ----------------------------------------------------------------------------
STYLE = "harp"  # common.music.STYLES; an --audio-only tag naming one overrides it
APPROACH_STEP = 2 ** (
    1 / 3
)  # a note each time the third root's distance shrinks by this
SHEPARD_CENTRE, SHEPARD_WIDTH = 74.0, 9.0  # MIDI: where the endless climb is loudest
RISER = 1.4  # s of build-up into the landing


def root_lut(colour, size=256):
    """A (size, 3) BGR ramp 0..255: near black -> the root's colour -> pale."""
    rgb = np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])
    u = np.linspace(0.0, 1.0, size)[:, None]
    dark = 0.07 * rgb
    pale = 0.35 * rgb + 0.65
    lo = dark + (rgb - dark) * np.clip(u / 0.65, 0, 1)
    hi = rgb + (pale - rgb) * np.clip((u - 0.65) / 0.35, 0, 1)
    return (np.where(u < 0.65, lo, hi)[:, ::-1] * 255).astype(np.float32)


@njit(nogil=True, fastmath=True, cache=True, inline="always")
def sample(zr, zi, rre, rim, rs, tre, tim, cubic):
    """
    Newton's method from z = zr + i zi: (index of the root reached, smooth step
    count v = sum over steps of clip(log(d / EPS) / log(1 / EPS), 0, 1), d the
    distance to the nearest root).

    General case: p(z) = prod (z - r_j / s_j) up to a constant, step
    z <- z - 1 / sum_j s_j / (s_j z - r_j), so a root with s_j -> 0 recedes to
    infinity continuously. cubic (exactly z^3 - 1 turned by a quarter turn,
    worked in the rotated frame w = -i z, whose roots are the cube roots of
    unity): N(w) = 2w/3 + 1/(3 w^2), and any
    |w| > FAR_FIELD jumps to (2/3)^n w at once (n steps counted), since there
    N(w) = 2w/3 to 1e-4.
    """
    inv_log = 1.0 / np.log(1.0 / EPS)
    nr = rre.shape[0]
    v = 0.0
    if cubic:
        wr, wi = zi, -zr  # w = -i z: the up-pointing root goes to w = 1
        for it in range(MAX_ITER):
            r2 = wr * wr + wi * wi
            if r2 > FAR_FIELD * FAR_FIELD:
                n = int((0.5 * np.log(r2) - np.log(FAR_FIELD)) / np.log(1.5)) + 1
                f = (2.0 / 3.0) ** n
                wr *= f
                wi *= f
                v += n
                continue
            sr = wr * wr - wi * wi  # w^2
            si = 2.0 * wr * wi
            dx = wr + 0.5  # squared distance to the nearest cube root of unity
            dy = abs(wi) - 0.8660254037844386
            d2 = min((wr - 1.0) * (wr - 1.0) + wi * wi, dx * dx + dy * dy)
            if d2 < EPS * EPS:
                break
            if d2 < 1.0:
                v += 0.5 * np.log(d2 / (EPS * EPS)) * inv_log
            else:
                v += 1.0
            q = 1.0 / (3.0 * (sr * sr + si * si) + 1e-300)
            wr = 2.0 / 3.0 * wr + q * sr
            wi = 2.0 / 3.0 * wi - q * si
        # nearest cube root of unity in w: 1 (k 0), e^{2 pi i/3} (k 1), e^{-2 pi i/3} (k 2)
        if wr >= 0.0 and abs(wi) * 0.5773502691896258 <= wr:
            return 0, v
        return (1 if wi > 0.0 else 2), v
    k = 0
    for it in range(MAX_ITER):
        dmin = 1e300
        k = 0
        for j in range(nr):
            dx = zr - tre[j]
            dy = zi - tim[j]
            d = dx * dx + dy * dy
            if d < dmin:
                dmin = d
                k = j
        if dmin < EPS * EPS:
            break
        if dmin < 1.0:
            v += 0.5 * np.log(dmin / (EPS * EPS)) * inv_log
        else:
            v += 1.0
        sre = 0.0
        sim = 0.0
        for j in range(nr):
            ure = rs[j] * zr - rre[j]
            uim = rs[j] * zi - rim[j]
            q = rs[j] / (ure * ure + uim * uim + 1e-300)
            sre += q * ure
            sim -= q * uim
        q = 1.0 / (sre * sre + sim * sim + 1e-300)
        zr -= q * sre
        zi += q * sim
    return k, v


@njit(nogil=True, fastmath=True, cache=True)
def newton_frame(scale, rot, rre, rim, rs, luts, offset, cubic, out):
    """
    One frame into out (H, W, 3) float32 BGR. The point at p (pixels from the
    centre, y up) is z = scale e^{i rot} p. Every pixel corner is sampled and
    coloured (lut of its root at 1 - exp(-(v - offset) / GLOW_SCALE)); a pixel
    whose four corners agree (same root, colours within AA_TOL) is their mean,
    any other gets SS x SS interior samples: adaptive anti-aliasing, so the
    flat basins cost one sample per pixel and only the boundaries pay for more.
    """
    h, w = out.shape[0], out.shape[1]
    nr = rre.shape[0]
    cr, sr = np.cos(rot) * scale, np.sin(rot) * scale
    tre = np.empty(nr)
    tim = np.empty(nr)
    for j in range(nr):
        tre[j] = rre[j] / rs[j]
        tim[j] = rim[j] / rs[j]
    corner = np.empty((h + 1, w + 1, 3), np.float32)
    root = np.empty((h + 1, w + 1), np.int8)
    for i in range(h + 1):
        py = 0.5 * h - i
        for jx in range(w + 1):
            px = jx - 0.5 * w
            k, v = sample(
                cr * px - sr * py, sr * px + cr * py, rre, rim, rs, tre, tim, cubic
            )
            v = max(v - offset, 0.0)
            idx = int((1.0 - np.exp(-v / GLOW_SCALE)) * 255.0)
            root[i, jx] = k
            for c in range(3):
                corner[i, jx, c] = luts[k, idx, c]
    inv_ss = 1.0 / SS
    for i in range(h):
        for jx in range(w):
            k = root[i, jx]
            same = (
                root[i, jx + 1] == k
                and root[i + 1, jx] == k
                and root[i + 1, jx + 1] == k
            )
            if same:
                for c in range(3):
                    lo = min(
                        corner[i, jx, c],
                        corner[i, jx + 1, c],
                        corner[i + 1, jx, c],
                        corner[i + 1, jx + 1, c],
                    )
                    hi = max(
                        corner[i, jx, c],
                        corner[i, jx + 1, c],
                        corner[i + 1, jx, c],
                        corner[i + 1, jx + 1, c],
                    )
                    if hi - lo > AA_TOL:
                        same = False
            if same:
                for c in range(3):
                    out[i, jx, c] = 0.25 * (
                        corner[i, jx, c]
                        + corner[i, jx + 1, c]
                        + corner[i + 1, jx, c]
                        + corner[i + 1, jx + 1, c]
                    )
                continue
            br = 0.0
            bg = 0.0
            bb = 0.0
            for a in range(SS):
                for b in range(SS):
                    px = jx + (b + 0.5) * inv_ss - 0.5 * w
                    py = 0.5 * h - i - (a + 0.5) * inv_ss
                    k, v = sample(
                        cr * px - sr * py,
                        sr * px + cr * py,
                        rre,
                        rim,
                        rs,
                        tre,
                        tim,
                        cubic,
                    )
                    v = max(v - offset, 0.0)
                    idx = int((1.0 - np.exp(-v / GLOW_SCALE)) * 255.0)
                    br += luts[k, idx, 0]
                    bg += luts[k, idx, 1]
                    bb += luts[k, idx, 2]
            n = SS * SS
            out[i, jx, 0] = br / n
            out[i, jx, 1] = bg / n
            out[i, jx, 2] = bb / n


def story(tau, swoop_sign):
    """
    The state at time tau (s): roots (re, im) and weights s (root j sits at
    r_j / s_j), the view's complex units per pixel, its rotation, the zoom depth.
    """
    angles = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    a = 1 - (1 - min(tau / ARRIVE, 1.0)) ** LANDING
    distance = np.exp(np.log(FAR) * (1 - a))  # log-distance eases from FAR to 1
    angles[0] += swoop_sign * SWOOP * (1 - a) ** 2
    rre, rim = np.cos(angles), np.sin(angles)
    rs = np.array([1.0 / distance, 1.0, 1.0])
    z = smoothstep((tau - ZOOM_START) / (ZOOM_END - ZOOM_START))
    depth = 1.5 ** (GENERATIONS / 2 * z)
    scale = 2 * HALF_WIDTH / WIDTH / depth
    return rre, rim, rs, scale, swoop_sign * TWIST * z, depth


def add_marker(image, x, y, colour, strength):
    """A root: a soft halo in its colour round a white core, at pixel (x, y)."""
    r = int(6 * DOT)
    x0, x1 = max(0, int(x) - r), min(WIDTH, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(HEIGHT, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    halo = np.exp(-d2 / (2 * (2.2 * DOT) ** 2))[..., None] * colour * 0.9
    core = np.exp(-((np.sqrt(d2) / DOT) ** 6))[..., None] * 255.0
    patch = image[y0:y1, x0:x1]
    patch += strength * (halo + core - patch * np.minimum(1, core / 255.0))


def soundtrack(swoop_sign, style_name, rng):
    """
    Cinematic swell with a struck voice on top, every onset from the closed
    forms of the picture (one-shot: circular=False, master(loop=False)).

    Two roots: a bare fifth (root and fifth, strings) from the first frame,
    struck once at t = 0. The approach: a note each time the third root's
    distance FAR^((1-u)^LANDING) shrinks by APPROACH_STEP, climbing the scale
    (a step every two notes) to the chord's third an octave up, panned by the root's x on screen and louder as it nears,
    then a riser into ARRIVE, where the third lands: struck triad, strings and
    choir complete the chord. The zoom: one note per bead generation (depth
    crossing (3/2)^(g/2), solved through the smoothstep), each a Shepard tone
    (the same pitch class in several octaves, weighted by a bell curve round
    SHEPARD_CENTRE), stepping up the scale every generation: a climb that never
    gets higher, like beads that never get smaller. Louder where the zoom is
    fastest, panned round the twist. A low root and a high chord close it at
    ZOOM_END and ring out.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    root = scale.root
    third = scale.note(2) - root  # a major or minor third in every catalogue mode
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def hz(midi):
        return float(audio.midi_to_hz(midi))

    def strike(midi, velocity, when, x=0.0):
        velocity = round(float(np.clip(velocity, 0.05, 1.0)) * 20) / 20  # cache hits
        note = style.voice(hz(midi), velocity)
        audio.add(track, audio.pan(note, float(np.clip(x, -1, 1))), when, **kw)

    def screen_x(tau):
        rre, rim, rs, scale_, rot, _ = story(tau, swoop_sign)
        z = complex(rre[0], rim[0]) / rs[0] * np.exp(-1j * rot) / scale_
        return z.real / (WIDTH / 2)

    # Two roots: the open fifth, the easy case.
    for semitones, vel in ((-12, 0.4), (-5, 0.3)):
        voice = audio.strings(hz(root + semitones), ARRIVE + 0.6, vel, 0.25, 1.5, 0.35)
        audio.add(track, voice, 0.0, **kw)
    strike(root - 12, 0.55, 0.02, -0.35)
    strike(root + 7 - 12, 0.5, 0.02, 0.35)

    # The approach: log-distance log(FAR) (1-u)^LANDING crosses each step.
    levels = np.log(FAR) - np.log(APPROACH_STEP) * np.arange(1, 100)
    levels = levels[levels > 0.05]
    times = ARRIVE * (1 - (levels / np.log(FAR)) ** (1 / LANDING))
    n = len(times)
    for k, when in enumerate(times):
        # climbing a step every two notes to the third, an octave up, at the landing
        degree = 2 + len(scale.steps) - (n - k) // 2
        distance = np.exp(levels[k])
        strike(scale.note(degree), 0.25 + 0.35 / distance**0.5, when, screen_x(when))
    audio.add(track, audio.riser(RISER, rng, 220.0, 1800.0, 0.18), ARRIVE - RISER, **kw)
    logger.info(f"{n} approach notes, {times[0]:.2f}-{times[-1]:.2f} s")

    # The landing: the third completes the chord.
    for i, semitones in enumerate((0, 7, third + 12, 12 + 7)):
        strike(
            root + semitones, 0.6 - 0.07 * i, ARRIVE + 0.07 * i, 0.5 * (i % 2) - 0.25
        )
    strike(root - 24, 0.6, ARRIVE)
    hold = DURATION - ARRIVE
    for semitones in (-12, -5, third):
        voice = audio.strings(hz(root + semitones), hold - 0.8, 0.4, 0.4, 1.6, 0.45)
        audio.add(track, voice, ARRIVE, **kw)
    for semitones in (0, 7, third + 12):
        voice = audio.choir(hz(root + semitones), hold - 2.0, 0.22, 2.5, 1.6, "a")
        audio.add(track, voice, ARRIVE + 0.3, **kw)

    # The zoom: one Shepard step per bead generation.
    grid = np.linspace(0.0, 1.0, 4001)
    eased = smoothstep(grid)
    span = ZOOM_END - ZOOM_START
    for g in range(1, GENERATIONS + 1):
        when = ZOOM_START + span * float(np.interp(g / GENERATIONS, eased, grid))
        u = (when - ZOOM_START) / span
        speed = 6 * u * (1 - u)  # d(smoothstep)/du, 1.5 at the middle
        pitch_class = scale.note(g)
        x = 0.6 * np.sin(2 * np.pi * g / 6)
        for octave in range(-3, 4):
            midi = pitch_class + 12 * octave
            weight = np.exp(-0.5 * ((midi - SHEPARD_CENTRE) / SHEPARD_WIDTH) ** 2)
            if weight > 0.12 and 40 <= midi <= 96:
                strike(midi, (0.2 + 0.25 * speed) * weight, when, x)

    # The end: home, ringing out over the rest.
    strike(root - 24, 0.65, ZOOM_END)
    for i, semitones in enumerate((12, 12 + third, 19, 24)):
        strike(root + semitones, 0.45, ZOOM_END + 0.09 * i, 0.4 * (i % 2) - 0.2)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Newton's method on p(z) = (z - r0)(z - r1)(z - r2), the cube roots of unity
    turned so that r0 points up. Each pixel (2x2 supersampled) is iterated
    z <- z - p/p' (written as z - 1 / sum 1/(z - r_j)) until it is within EPS
    of a root, and coloured by that root: dark where it converges at once,
    glowing where it takes many steps, i.e. along the basin boundaries.

    Act 1 (0-5 s): r0 starts FAR = 30 units away and its log-distance eases
    down to 1 (ease-out, power LANDING), curling SWOOP rad round on the way (direction from
    the seed). With r0 far, only r1 and r2 matter and the boundary is their
    perpendicular bisector, Cayley's straight line; as r0 nears, beads of its
    basin grow along the line. The receding root enters through a weight s:
    sum s / (s z - r) is continuous as s -> 0.
    Act 2 (5.8-13.4 s): a smoothstep-in-log zoom into z = 0, the double
    critical point of z^3 - 1, where N(z) ~ 1/(3z^2): the picture there is the
    far field inverted, self-similar under zooming by sqrt(3/2), so the zoom
    passes GENERATIONS bead generations (depth (3/2)^12 ~ 130) while the view
    turns TWIST. A pixel near 0 takes one more step per generation, so that
    baseline is subtracted to keep the glow the same at every depth. Then the
    image rests and fades out.

    Each frame is a pure function of t, computed by a numba kernel (nogil),
    frames in parallel on the thread pool. The seed picks which colour goes to
    which root and which way the third root curls in.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    swoop_sign = float(rng.choice([-1.0, 1.0]))
    order = rng.permutation(3)
    colours = [PALETTE[k] for k in order]
    luts = np.stack([root_lut(c) for c in colours])
    marker_bgr = [luts[k, 166] for k in range(3)]  # the root's own colour

    def render(frame):
        tau = frame / FPS
        rre, rim, rs, scale, rot, depth = story(tau, swoop_sign)
        offset = GLOW_BASE + 2 * np.log(depth) / np.log(1.5)
        image = np.empty((HEIGHT, WIDTH, 3), np.float32)
        newton_frame(scale, rot, rre, rim, rs, luts, offset, tau >= ARRIVE, image)
        for j in range(3):
            z = complex(rre[j], rim[j]) / rs[j] * np.exp(-1j * rot) / scale
            add_marker(image, WIDTH / 2 + z.real, HEIGHT / 2 - z.imag, marker_bgr[j], 1)
        image *= smoothstep((DURATION - tau) / FADE)
        return image

    render_frames(settings, render, FRAME_COUNT)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(swoop_sign, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
