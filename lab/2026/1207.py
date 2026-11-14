"""
The Cat That Always Comes Back

This is Arnold's cat map, the textbook example of chaos. Take a 322 × 322 pixel
picture of a cat, shear it up, shear it sideways, and wrap whatever spills off
one edge back in at the other: (x, y) → (2x + y, x + y) mod 322. Neighbouring
pixels fly apart, and within a few steps the cat is pure static.

Yet nothing is lost. There are only so many pixels, so the shuffle must
repeat: after 12 steps the cat returns upside down, after 24 it is back
exactly. This chaos is not randomness, just order scrambled beyond recognition.

Listen: a slow chord brightens with the static, a soft piano note climbs with
every step, and each return lands on a rolled chord, minor upside down, home
when the cat is back.

Made with Python & PyTorch. Sound synthesized in NumPy.
#arnoldscatmap #chaostheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds, an exact loop
FRAME_COUNT = frame_count(DURATION)

# --- The map -----------------------------------------------------------------------
N = 322  # grid size; the cat map M = [[2, 1], [1, 1]] has period 24 mod 322
CAT = np.array([[2, 1], [1, 1]], dtype=np.int64)
HOLD = 1.0  # s of rest on the cat and on the upside-down cat
SLOW = 1.3  # steps near a return last up to 1 + SLOW times as long as the others
SHEAR = 0.44  # fraction of a step each of the two shears takes

# --- Layout & look -----------------------------------------------------------------
SS = 2  # supersampling
SIDE = 1008  # px, the torus square
LEFT, TOP = (WIDTH - SIDE) // 2, 380
BG_TOP, BG_BOTTOM = "#141022", "#06050c"
FRAME_LINE = "#3a3352"
TEXT, TEXT_DIM, ACCENT = "#efe8dc", "#9a93b0", "#f2a03d"

# --- Sound -------------------------------------------------------------------------
LEADS = {"felt_piano": audio.piano, "harp": audio.pluck}  # --audio-only tag picks one
LEAD = "felt_piano"
ROOTS = (50, 52, 53)  # MIDI roots (D3, E3, F3) of the chords; the lead is an octave up


def period(n: int = N) -> int:
    """Smallest k with M^k = I mod n."""
    power = np.eye(2, dtype=np.int64)
    for k in range(1, 10 * n):
        power = CAT @ power % n
        if (power == np.eye(2, dtype=np.int64)).all():
            return k
    raise ValueError(n)


STEPS = period()  # 24
HALF = STEPS // 2  # M^12 = -I mod 322: the cat upside down


def schedule():
    """
    Start time (s) of every step and its length. Time 0 is the middle of the rest on
    the cat; the steps slow down near the two returns (k = 0, HALF, STEPS) and a rest
    of HOLD sits at each.
    """
    centres = np.arange(STEPS) + 0.5
    near = np.minimum.reduce([abs(centres - r) for r in (0, HALF, STEPS)])
    weight = 1 + SLOW * np.exp(-((near / 2.2) ** 2))
    lengths = weight / weight.sum() * (DURATION - 2 * HOLD)
    starts = np.empty(STEPS)
    clock = HOLD / 2
    for k in range(STEPS):
        if k == HALF:
            clock += HOLD
        starts[k] = clock
        clock += lengths[k]
    return starts, lengths


def state(seconds, starts, lengths):
    """(completed steps k, first shear s1, second shear s2) at a time in the loop."""
    k = int(np.searchsorted(starts, seconds, side="right")) - 1
    if k < 0:
        return 0, 0.0, 0.0
    tau = (seconds - starts[k]) / lengths[k]
    if tau >= 1:  # resting after step k
        return k + 1, 0.0, 0.0
    s1 = float(smoothstep(tau / SHEAR))
    s2 = float(smoothstep((tau - (1 - SHEAR)) / SHEAR))
    return k, s1, s2


def inverse_powers():
    """M^-k mod N for k = 0 .. STEPS, as integer 2x2 matrices."""
    inverse = np.array([[1, -1], [-1, 2]], dtype=np.int64) % N
    out = [np.eye(2, dtype=np.int64)]
    for _ in range(STEPS):
        out.append(inverse @ out[-1] % N)
    return np.array(out)


def _rgb(colour: str):
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def draw_cat(n: int = N) -> np.ndarray:
    """The picture: a ginger cat's face on a dusk gradient, (n, n, 3) RGB uint8."""
    big = 4  # draw at 4x and shrink, for clean edges at grid resolution
    s = n * big / 322.0

    def p(x, y):
        return int(round(x * s)), int(round(y * s))

    def r(a):
        return int(round(a * s))

    yy, xx = np.mgrid[0 : n * big, 0 : n * big] / (n * big)
    top, bottom = np.array(_rgb("#2a7f8f")), np.array(_rgb("#3b1f5e"))
    g = np.clip(0.65 * yy + 0.35 * xx, 0, 1)[..., None]
    image = ((1 - g) * top + g * bottom).astype(np.uint8)
    image = np.ascontiguousarray(image)
    aa = cv2.LINE_AA

    # A low moon behind, and a few stars.
    cv2.circle(image, p(258, 62), r(40), _rgb("#f6e7b8"), -1, aa)
    for x, y, a in (
        (40, 40, 3),
        (90, 22, 2),
        (30, 120, 2),
        (296, 150, 2),
        (220, 20, 2),
    ):
        cv2.circle(image, p(x, y), r(a), _rgb("#fdf6dc"), -1, aa)

    fur, dark, light = _rgb("#f08a24"), _rgb("#b4520f"), _rgb("#fbf1df")
    pink, ink = _rgb("#f4a1ae"), _rgb("#1a1020")
    # Ears.
    for sign in (-1, 1):
        outer = np.array(
            [p(161 + sign * 112, 150), p(161 + sign * 92, 32), p(161 + sign * 36, 104)]
        )
        inner = np.array(
            [p(161 + sign * 98, 132), p(161 + sign * 88, 56), p(161 + sign * 54, 106)]
        )
        cv2.fillPoly(image, [outer], fur, aa)
        cv2.fillPoly(image, [inner], pink, aa)
    # Head, cheeks and chin.
    cv2.ellipse(image, p(161, 190), (r(118), r(100)), 0, 0, 360, fur, -1, aa)
    cv2.ellipse(image, p(161, 248), (r(96), r(62)), 0, 0, 360, fur, -1, aa)
    # Forehead stripes.
    for dx, h in ((-26, 44), (0, 56), (26, 44)):
        poly = np.array([p(161 + dx - 9, 96), p(161 + dx + 9, 96), p(161 + dx, 96 + h)])
        cv2.fillPoly(image, [poly], dark, aa)
    for sign in (-1, 1):  # cheek stripes
        for dy in (0, 20):
            poly = np.array(
                [
                    p(161 + sign * 118, 196 + dy),
                    p(161 + sign * 116, 208 + dy),
                    p(161 + sign * 78, 204 + dy),
                ]
            )
            cv2.fillPoly(image, [poly], dark, aa)
    # Muzzle.
    for sign in (-1, 1):
        cv2.ellipse(
            image, p(161 + sign * 24, 238), (r(32), r(24)), 0, 0, 360, light, -1, aa
        )
    cv2.ellipse(image, p(161, 262), (r(30), r(18)), 0, 0, 360, light, -1, aa)
    # Eyes: green, slit pupils, a highlight.
    for sign in (-1, 1):
        c = (161 + sign * 46, 172)
        cv2.ellipse(image, p(*c), (r(25), r(29)), 0, 0, 360, ink, -1, aa)
        cv2.ellipse(image, p(*c), (r(22), r(26)), 0, 0, 360, _rgb("#a6e05a"), -1, aa)
        cv2.ellipse(image, p(*c), (r(5), r(22)), 0, 0, 360, ink, -1, aa)
        cv2.circle(image, p(c[0] + 8, c[1] - 10), r(5), _rgb("#ffffff"), -1, aa)
    # Nose and mouth.
    nose = np.array([p(147, 218), p(175, 218), p(161, 234)])
    cv2.fillPoly(image, [nose], pink, aa)
    cv2.line(image, p(161, 234), p(161, 246), ink, r(2.5), aa)
    cv2.ellipse(image, p(149, 246), (r(12), r(9)), 0, 0, 180, ink, r(2.5), aa)
    cv2.ellipse(image, p(173, 246), (r(12), r(9)), 0, 0, 180, ink, r(2.5), aa)
    # Whiskers.
    for sign in (-1, 1):
        for dy, ey in ((-6, -18), (2, 2), (10, 22)):
            cv2.line(
                image,
                p(161 + sign * 50, 240 + dy),
                p(161 + sign * 150, 240 + ey),
                light,
                r(2),
                aa,
            )
    return cv2.resize(image, (n, n), interpolation=cv2.INTER_AREA)


def generate(settings: ImageProcessingSettings = None):
    """
    Arnold's cat map on a 322 x 322 torus, as an exact loop. The picture is a grid of
    N = 322 cells (a cat's face drawn with cv2). One step of the map is M = [[2, 1],
    [1, 1]] = A·B with the shears B: (x, y) -> (x, y + x) and A: (x, y) -> (x + y, y),
    mod N. M has period 24 mod 322 (Pisano-type period; 322 is the Lucas number L12)
    and M^12 = -I, so the cat returns rotated by 180° halfway and exactly at the end.

    Time: the loop's 15 s hold HOLD = 1 s on each return and spread the 24 steps over
    the rest, longer near the returns (weight 1 + SLOW·exp(-(d/2.2)²), d = steps to
    the nearest return). Within a step the two shears are animated continuously:
    column i slides by s1·i cells, then row j by s2·j cells (smoothstep over SHEAR
    of the step each), wrapping round the torus. A display point (u, v) in cell units
    shows the cell x1 = ⌊u − s2⌊v⌋⌋, y1 = ⌊v − s1·x1⌋ of the image after k steps,
    which is the cell M^-k (x1, y1) mod N of the original. Integer arithmetic, so
    the map is exact and the loop closes bit for bit.

    Rendering: one integer gather per 2x supersampled pixel of the square on the GPU
    (torch), average-pooled to 1x and composited over a gradient with LaTeX text (a
    step counter n, the map, a note at the halfway return), render_frames(parallel=
    False). Nothing is random, so --seed does not change the picture.

    Soundtrack (`soundtrack`, ambient Drone & pad with a felt-piano lead): pad and
    "u" choir chords I, IV, vi, V on the step schedule, lowpass opening with the
    scramble, one piano note per completed step (pitch by distance from the nearest
    return), rolled vi / I chords plus a bell at the two returns, delay and a 5 s
    reverb, all circular. `--audio-only harp` swaps the lead for a harp pluck. The
    posted soundtrack is LEAD = "felt_piano" with --seed 0 (settings.audio_rng
    picks root MIDI 52 and the room).
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    starts, lengths = schedule()
    powers = torch.tensor(inverse_powers(), device=device)
    logger.info(f"Cat map period mod {N}: {STEPS} steps, M^{HALF} = -I")

    def tensor(a, dtype=torch.float32):
        return torch.tensor(np.asarray(a), device=device, dtype=dtype)

    cat = tensor(draw_cat().reshape(-1, 3))
    px = SIDE * SS
    centres = (torch.arange(px, device=device, dtype=torch.float64) + 0.5) * N / px
    v, u = torch.meshgrid(centres, centres, indexing="ij")
    v_cell = torch.floor(v)

    # Background with the static text, RGB 0..255.
    rows = np.linspace(0, 1, HEIGHT)[:, None, None]
    top, bottom = np.array(_rgb(BG_TOP)), np.array(_rgb(BG_BOTTOM))
    background = ((1 - rows) * top + rows * bottom) * np.ones((1, WIDTH, 1))
    background = background.astype(np.float32)
    line = np.array(_rgb(FRAME_LINE), np.float32)
    background[TOP - 3 : TOP + SIDE + 3, LEFT - 3 : LEFT + SIDE + 3] = line
    text, dim, accent = (
        np.array(_rgb(c), np.float32) for c in (TEXT, TEXT_DIM, ACCENT)
    )
    title = r"$\mathrm{Arnold}\text{'}\mathrm{s\ cat\ map}$"
    for tex, size, y, colour in (
        (title, 74, 92, text),
        (r"$(x{,}\,y)\ \mapsto\ (2x+y{,}\ x+y)\ \mathrm{mod}\ 322$", 54, 192, accent),
        (
            r"$\mathrm{on\ a}\ 322\times 322\ \mathrm{grid{,}\ the\ cat\ returns\ after}\ 24\ \mathrm{steps}$",
            42,
            284,
            dim,
        ),
    ):
        blit_mask(
            background, latex_mask(tex, size), WIDTH // 2, y, colour, anchor="centre"
        )
    background = tensor(background)

    def layer(tex, size, y, colour):
        """A text mask as (alpha, premultiplied colour) tensors and its box."""
        mask = latex_mask(tex, size)
        h, w = mask.shape
        x = WIDTH // 2 - w // 2
        alpha = tensor(mask[..., None])
        return alpha, alpha * tensor(colour), (y, y + h, x, x + w)

    counters = [layer(rf"$n = {k}$", 76, 1432, text) for k in range(STEPS + 1)]
    upside = layer(r"$\mathrm{halfway{,}\ upside\ down}$", 46, 1540, accent)
    half_time = starts[HALF] - HOLD / 2  # middle of the halfway rest

    def put(image, overlay, a=1.0):
        alpha, colour, (y0, y1, x0, x1) = overlay
        region = image[y0:y1, x0:x1]
        region.mul_(1 - a * alpha).add_(a * colour)

    def render(frame):
        seconds = frame / FRAME_COUNT * DURATION
        k, s1, s2 = state(seconds, starts, lengths)
        x1 = torch.remainder(torch.floor(u - s2 * v_cell), N).long()
        y1 = torch.remainder(torch.floor(v - s1 * x1), N).long()
        m = powers[k % STEPS]
        x0 = (m[0, 0] * x1 + m[0, 1] * y1) % N
        y0 = (m[1, 0] * x1 + m[1, 1] * y1) % N
        square = cat[(y0 * N + x0).reshape(-1)].reshape(px, px, 3)
        square = F.avg_pool2d(square.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
        image = background.clone()
        image[TOP : TOP + SIDE, LEFT : LEFT + SIDE] = square
        put(image, counters[k])
        a = float(smoothstep((1.6 - abs(seconds - half_time)) / 0.3))
        if a > 0:
            put(image, upside, a)
        return image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    lead = settings.audio_tag if settings.audio_tag in LEADS else LEAD
    track = soundtrack(lead, starts, lengths, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def scramble(seconds, starts, lengths):
    """How far the picture is from a return, 0 (a cat) .. 1 (deep static), smooth in t."""
    k, s1, s2 = state(seconds, starts, lengths)
    steps = k + 0.5 * (s1 + s2)  # continuous count of steps done
    near = min(steps, abs(steps - HALF), STEPS - steps)
    return float(smoothstep(near / 5.0))


def soundtrack(lead, starts, lengths, rng):
    """
    Ambient, from the same schedule as the picture; an exact loop (everything wraps
    circularly). A bed of pad and "u" choir chords: I around the cat, IV as it
    scrambles, vi around the upside-down cat, V as it scrambles again, back to I.
    The bed's lowpass opens with the scramble (crossfade of a dark and a bright
    filtered copy, weighted by `scramble(t)`). One soft lead note (felt piano or
    harp) as each step completes, its pitch climbing the scale with the distance
    from the nearest return, so it rises into the static and falls back out; at
    the two returns a rolled chord instead (I for the cat, vi for the upside-down
    cat) with a quiet bell an octave up. Delay, long reverb, master(loop=True).
    """
    voice = LEADS[lead]
    root = int(rng.choice(ROOTS))
    scale = Scale(root, "major")
    ends = starts + lengths  # when each step completes
    cat_back, upside_back = ends[-1], ends[HALF - 1]

    # The bed.
    bed = audio.silence(DURATION)
    sections = (  # (degree, start, end)
        (0, cat_back - 1.6, DURATION + 2.6),
        (3, 2.4, 5.6),
        (5, upside_back - 1.6, upside_back + 2.6),
        (4, upside_back + 2.4, cat_back - 1.4),
    )
    for degree, start, end in sections:
        for i, midi in enumerate(scale.chord(degree, 3)):
            hz = float(audio.midi_to_hz(midi))
            held = audio.pad(hz, end - start, 0.26, attack=1.4, release=2.2)
            audio.add(bed, audio.pan(held[:, 0], 0.45 * (i - 1)), start)
            sung = audio.choir(
                hz, end - start, 0.11, attack=1.6, release=2.2, vowel="u"
            )
            audio.add(bed, sung, start)
    bed = audio.highpass(bed, 120.0)
    weight = np.array(
        [scramble(s, starts, lengths) for s in np.arange(0, DURATION, 0.01)]
    )
    weight = np.interp(
        np.arange(len(bed)) / audio.SAMPLE_RATE, np.arange(len(weight)) * 0.01, weight
    )[:, None]
    track = (1 - weight) * audio.lowpass(bed, 700.0) + weight * audio.lowpass(
        bed, 2600.0
    )

    # The lead: one note per completed step, a rolled chord at each return.
    for k, when in enumerate(ends):
        done = k + 1
        if done in (HALF, STEPS):
            degree = 0 if done == STEPS else 5
            for j, midi in enumerate(scale.chord(degree, 4, octave=1)):
                note = voice(float(audio.midi_to_hz(midi)), 0.42 - 0.04 * j)
                audio.add(track, audio.pan(note, -0.3 + 0.2 * j), when + 0.07 * j)
            shine = audio.bell(scale.hz(degree, 2), 0.16, length=5.0)
            audio.add(track, audio.pan(shine, 0.2), when + 0.3)
            continue
        near = min(done, abs(done - HALF), STEPS - done)  # 1 .. 5
        degree = 2 * near - 1 if near > 1 else 1  # 1, 3, 5, 7, 9: up the scale
        note = voice(scale.hz(degree, 1), 0.30 + 0.03 * near)
        audio.add(track, audio.pan(note, 0.35 if k % 2 else -0.35), when)

    track = audio.delay(track, 0.42, feedback=0.35, wet=0.2)
    track = audio.reverb(track, rng, rt60=5.0, wet=0.42, tone=4200.0)
    logger.info(
        f"Soundtrack ({lead}): root {root}, cat back at {cat_back:.2f} s, "
        f"upside down at {upside_back:.2f} s"
    )
    return audio.master(track)


if __name__ == "__main__":
    generate()
