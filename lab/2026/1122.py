"""
The Riemann Hypothesis in One Picture

The Riemann hypothesis in one picture. Part 2 of 3.

Feed the zeta function ζ(s) = 1 + 1/2ˢ + 1/3ˢ + 1/4ˢ + … the numbers s = ½ + it, one after
another as t grows, and draw where they land. The curve loops round and round, and every
so often it passes exactly through the centre. Each pass is a zero of the zeta function:
the first at t = 14.1347…, then 21.0220…, 25.0109…

Now slide the line a little, to Re(s) = 0.4 or 0.6. Every loop misses the centre. Only on
the line Re(s) = ½ does the curve ever touch zero, at every zero, forever. That is the
Riemann hypothesis: posed in 1859, checked by computer for more than ten trillion zeros,
and proved in November 2026 with a computer-checked proof in Lean.

Listen: whenever the line drifts off ½, the strings hang on an unresolved chord, and they
come home each time every loop passes through zero.

Send this to someone who has heard of the Riemann hypothesis but never seen it.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#riemannhypothesis #zetafunction #numbertheory #mathart #randomplots
"""

import gc
from math import factorial

import cv2
import mpmath
import numpy as np
from loguru import logger
from scipy.special import bernoulli

from common import audio
from common.image_processing import ImageProcessingSettings
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    hook_masks,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 30.0  # seconds
FRAME_COUNT = frame_count(DURATION)

HOOK = "Only one line ever hits zero"
HOOK_Y = 270
HOOK_COLOUR = "#f4f0e7"
COVER_SECONDS = 15.0  # grid cover (tools.post_reel): every loop through the centre

BACKGROUND = "#07060a"
INK = "#f4f0e7"
GOLD = "#ffd37a"
MUTED = "#8a8494"
# the curve, coloured by t from 0 to T_END
PALETTE = ["#ffd37a", "#ffa463", "#ff6f91", "#c86bfa", "#7b8cff"]

ORIGIN = (370.0, 1150.0)  # screen position of 0 in the ζ-plane
SCALE = 150.0  # px per unit of ζ
T_START = 8.0  # the curve is already this far along at frame 0
T_FIRST = 50.5  # the trace phase ends past the 10th zero (49.77)
T_END = 100.0  # the run-on phase: 29 zeros
SIGMA_LO, SIGMA_HI = 0.30, 0.70  # the table's range of Re(s)
SWING = 0.15  # Re(s) swings between 1/2 - SWING and 1/2 + SWING
# phases (seconds)
TRACE = (0.0, 9.0)  # the pen draws ζ(1/2 + it) from T_START to T_FIRST
SWEEP = (9.0, 21.0)  # Re(s) swings twice; every loop hits 0 only at Re(s) = 1/2
ONWARD = (21.0, 26.0)  # the pen runs on to T_END, all zeros through the centre
HOOK_END = 20.8
FLASH = 0.45  # seconds a hit flash lasts
SUPERSAMPLE = 2  # the curve is drawn at this multiple of the frame size
LINE_PX = 2.5  # curve width in output pixels
# the strip of the critical strip at the top: t across, Re(s) from 0 (bottom) to 1 (top)
STRIP = (90, 960, 470, 580)  # x0, x1, y0, y1


def zeta(s: np.ndarray, terms: int = 40, corrections: int = 10) -> np.ndarray:
    """ζ(s) by Euler-Maclaurin summation, vectorised; ~1e-12 for |Im s| <= 100."""
    s = np.asarray(s, complex)
    n = terms
    total = sum(k ** (-s) for k in range(1, n))
    total = total + n ** (1 - s) / (s - 1) + 0.5 * n ** (-s)
    b = bernoulli(2 * corrections)
    rising = s.copy()  # s (s+1) ... (s+2j-2)
    for j in range(1, corrections + 1):
        total = total + b[2 * j] / factorial(2 * j) * rising * n ** (-s - 2 * j + 1)
        rising = rising * (s + 2 * j - 1) * (s + 2 * j)
    return total


def halo(mask, sigma=6, gain=2.0):
    pad = int(2 * sigma)
    return np.clip(cv2.GaussianBlur(np.pad(mask, pad), (0, 0), sigma) * gain, 0, 1)


def window(seconds, start, end, fade=0.4):
    rise = smoothstep(np.clip((seconds - start) / fade, 0, 1))
    fall = 1 - smoothstep(np.clip((seconds - end + fade) / fade, 0, 1))
    return float(rise * fall)


def state(seconds: float) -> tuple[float, float]:
    """(σ, pen t) at a moment of the video: the whole timeline of the picture."""
    if seconds < TRACE[1]:
        u = seconds / TRACE[1]
        return 0.5, T_START + (T_FIRST - T_START) * u
    if seconds < SWEEP[1]:
        v = seconds - SWEEP[0]
        span = SWEEP[1] - SWEEP[0]
        envelope = smoothstep(np.clip(v / 1.5, 0, 1)) * smoothstep(
            np.clip((span - v) / 1.5, 0, 1)
        )
        return 0.5 + SWING * envelope * np.sin(2 * np.pi * v / 6.0), T_FIRST
    if seconds < ONWARD[1]:
        u = smoothstep((seconds - ONWARD[0]) / (ONWARD[1] - ONWARD[0]))
        return 0.5, T_FIRST + (T_END - T_FIRST) * u
    return 0.5, T_END


def zero_heights() -> np.ndarray:
    """Heights of the zeros of ζ on the critical line below T_END (29 of them)."""
    zeros = np.array([float(mpmath.zetazero(k).imag) for k in range(1, 30)])
    return zeros[zeros < T_END]


def hit_times(zeros: np.ndarray) -> np.ndarray:
    """When the pen passes each zero (trace and run-on phases), from the timeline."""
    fine = np.linspace(0, DURATION, 30001)
    pen = np.array([state(x)[1] for x in fine])
    return np.array([fine[int(np.argmax(pen >= g))] for g in zeros if g >= T_START])


# --- soundtrack: an ambient string bed in D, like Part 1, breathing with the sweep ---
MUSIC = "strings"  # default; `--audio-only <tag>` picks "bells" or "choir"
KEY = 50  # D3
QUALITY = {"maj": (0, 4, 7), "min": (0, 3, 7), "sus4": (0, 5, 7)}
ON_LINE = [SWEEP[0] + 3.0 * k for k in range(5)]  # σ = 1/2: every loop through 0
OFF_LINE = [SWEEP[0] + 1.5 + 3.0 * k for k in range(4)]  # σ at its farthest
RESOLVE = ONWARD[1]  # "proved in November 2026"
# (start s, root above D, quality, loudness): Bm G | D <-> Asus4, breathing with σ |
# Bm G | D, held to the end
CHORDS = (
    [(0.0, 9, "min", 0.32), (4.5, 7, "maj", 0.36)]
    + sorted(
        [(t, 0, "maj", 0.42) for t in ON_LINE[:-1]]
        + [(t, 7, "sus4", 0.40) for t in OFF_LINE]
    )
    + [(ON_LINE[-1], 9, "min", 0.44), (23.5, 7, "maj", 0.48), (RESOLVE, 0, "maj", 0.62)]
)
BELL_SCALE = (0, 2, 4, 7, 9)  # D major pentatonic, for the zero bells


def voice_lead(previous, pitch_classes, low=57, high=76):
    """Each upper voice moves to the nearest chord tone, avoiding doubled notes."""
    options = [n for n in range(low, high + 1) if n % 12 in pitch_classes]
    voices, used = [], set()
    for prev in previous:
        ranked = sorted(options, key=lambda n: (n % 12 in used, abs(n - prev)))
        voices.append(ranked[0])
        used.add(ranked[0] % 12)
    return voices


def soundtrack(variant: str, rng: np.random.Generator) -> np.ndarray:
    """
    Ambient, one-shot. A string bed (bass on the root, three voice-led upper voices,
    slow bows) follows CHORDS: B minor and G while the curve is traced; during the
    sweep a suspended A chord whenever σ is far from 1/2 and D major at each moment
    σ = 1/2, when every loop passes through 0; B minor and G as the zeros run on; D
    major, with a high octave, on "proved". "bells": a soft glass bell each time the
    pen passes a zero, climbing a D-major pentatonic with the zero's index, and a
    two-note bell at each σ = 1/2. "choir": a quiet "oo" choir doubles the upper voices.
    """
    track = audio.silence(DURATION)
    upper = [59, 62, 66]  # B3 D4 F#4, re-voiced for each chord
    for i, (start, root, quality, loud) in enumerate(CHORDS):
        end = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else DURATION
        last = i + 1 == len(CHORDS)
        pcs = {(KEY + root + q) % 12 for q in QUALITY[quality]}
        upper = voice_lead(upper, pcs)
        bass = KEY - 12 + root if root <= 7 else KEY - 24 + root
        notes = [(bass, 1.0)] + [(n, 0.75) for n in upper]
        if last:
            notes.append((upper[-1] + 12, 0.4))
        length = (DURATION - start) if last else (end - start + 0.8)
        for midi, weight in notes:
            hz = float(audio.midi_to_hz(midi))
            sound = audio.strings(
                hz,
                length,
                velocity=loud * weight,
                attack=1.8 if start == 0 else (0.6 if last else 1.0),
                release=1.6 if not last else 2.5,
                brightness=0.35 + (0.2 if last else 0.0),
            )
            audio.add(track, sound, start, circular=False)
            if variant == "choir" and midi != bass:
                voice = audio.choir(
                    hz, length, velocity=0.35 * loud * weight, vowel="u"
                )
                audio.add(track, voice, start, circular=False)
    if variant == "bells":
        for k, when in enumerate(hit_times(zero_heights())):
            degree = k // 2
            midi = KEY + 24 + BELL_SCALE[degree % 5] + 12 * (degree // 5)
            note = audio.bell(float(audio.midi_to_hz(min(midi, 96))), velocity=0.22)
            audio.add(
                track, audio.pan(note, 0.25 * np.sin(1.7 * k)), when, circular=False
            )
        for when in ON_LINE[1:]:
            for midi in (KEY + 24, KEY + 31):
                note = audio.bell(float(audio.midi_to_hz(midi)), velocity=0.18)
                audio.add(track, audio.pan(note, 0.0), when, circular=False)
    track = audio.reverb(track, rng, rt60=3.0, wet=0.35, circular=False)
    audio.fade(track, 0.01, 1.0)  # out with the picture's fade
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The curve t -> ζ(σ + it), t in [0, T], drawn in the complex plane; its passes through
    0 are the zeros of ζ on the vertical line Re(s) = σ. ζ is tabulated once by
    Euler-Maclaurin (40 terms, 10 Bernoulli corrections) on a grid of 81 values of σ in
    [0.3, 0.7] by 20,001 values of t in [0, 100]; a frame interpolates the rows linearly.

    0-9 s: σ = 1/2 and the pen's t rises steadily from 8 to 50.5; each zero γ_k
    (mpmath.zetazero) flashes the origin as the pen passes it and lights its dot on a
    strip of the critical strip drawn across the top (t horizontal, Re(s) vertical).
    9-21 s: σ(t) = 1/2 + 0.15·sin(2π(t - 9)/6), the amplitude eased in and out, so σ
    crosses 1/2 at 9, 12, 15, 18 and 21 s; the whole curve t <= 50.5 is drawn, a ghost
    of the σ = 1/2 curve stays behind it, and min_t |ζ(σ + it)| over the loops is shown:
    0 only when σ = 1/2. 21-26 s: σ = 1/2 again and the pen runs on to t = 100 (29
    zeros). 26-30 s: closing text.

    Rendering: cv2 on a thread pool. The curve is drawn as anti-aliased polylines in
    48 colour bands (colour by t), plus a glow from a blurred quarter-size copy; text is
    LaTeX masks built once (every number shown is precomputed per frame). Frames are pure
    functions of t.
    """
    settings = settings or ImageProcessingSettings(1)
    if settings.audio_only:  # the soundtrack needs none of the picture's arrays
        track = soundtrack(settings.audio_tag or MUSIC, settings.audio_rng)
        settings.save_video(fps=FPS, crf=18, audio=track)
        return

    # --- the table ---
    sigmas = np.linspace(SIGMA_LO, SIGMA_HI, 81)
    ts = np.linspace(0, T_END, 20001)
    table = zeta(sigmas[:, None] + 1j * ts[None, :])
    zeros = zero_heights()
    logger.info(f"{zeros.size} zeros below t = {T_END:g}; table {table.shape}")

    def curve(sigma):
        x = (sigma - SIGMA_LO) / (SIGMA_HI - SIGMA_LO) * (len(sigmas) - 1)
        i = min(int(x), len(sigmas) - 2)
        f = x - i
        return (1 - f) * table[i] + f * table[i + 1]

    # --- the timeline, frame by frame (precomputed: every number on screen) ---
    seconds_all = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION

    states = [state(s) for s in seconds_all]
    sweep_rows = [i for i, s in enumerate(seconds_all) if SWEEP[0] <= s < SWEEP[1]]
    miss = {}  # frame -> closest approach to 0 over the loops (t from 10 to T_FIRST)
    loops = ts >= 10.0
    first = ts <= T_FIRST
    for i in sweep_rows:
        sigma = states[i][0]
        miss[i] = float(np.abs(curve(sigma)[loops & first]).min())

    # --- palette and screen geometry ---
    lut = palette_lut(PALETTE, 256)  # RGB 0..1
    bands = 48
    band_colour = [
        tuple(float(255 * c) for c in lut[int(k / (bands - 1) * 255)][::-1])
        for k in range(bands)
    ]
    ink = np.array(hex_to_bgr(INK), np.float32)
    gold = np.array(hex_to_bgr(GOLD), np.float32)
    muted = np.array(hex_to_bgr(MUTED), np.float32)
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    ox, oy = ORIGIN
    sx0, sx1, sy0, sy1 = STRIP

    def to_screen(z):
        return np.stack([ox + SCALE * z.real, oy - SCALE * z.imag], -1)

    def strip_x(t, t_view):
        return sx0 + (sx1 - sx0) * t / t_view

    def strip_y(sigma):
        return sy1 - (sy1 - sy0) * sigma

    # --- text, built once ---
    hook_lines = hook_masks(HOOK)
    hook_halos = [
        np.clip(cv2.GaussianBlur(np.pad(m, 14), (0, 0), 7) * 2.2, 0, 1)
        for m in hook_lines
    ]

    def readout(label_tex, values):
        """Each 'label value' typeset as one LaTeX line (one baseline, true spacing),
        split at the end of the label so the number can take its own colour."""
        label = latex_mask(rf"${label_tex}$", 38)
        cut = label.shape[1]  # the label's ink ends 2 px before this; then a space
        out = {}
        for v in values:
            full = latex_mask(rf"${label_tex}\ \mathrm{{{v}}}$", 38)
            out[v] = (full[:, :cut], full[:, cut:])
        return out

    sigma_values = sorted({f"{s:.3f}" for s, _ in states})
    sigma_masks = readout(r"\mathrm{Re}(s)\ =", sigma_values)
    count_masks = readout(
        r"\mathrm{zeros\ hit{:}}", [str(k) for k in range(zeros.size + 1)]
    )
    miss_values = sorted({f"{m:.3f}" for m in miss.values()})
    miss_masks = readout(r"\mathrm{closest\ to\ 0{:}}", miss_values)
    hit_mask = latex_mask(r"$\zeta(s)=0$", 40)
    hit_halo = halo(hit_mask)
    zero_labels = [latex_mask(rf"$\mathrm{{{g:.2f}}}$", 27) for g in zeros]
    strip_title = latex_mask(
        r"$\mathrm{the\ critical\ strip{,}\ }0<\mathrm{Re}(s)<1$", 30
    )
    line_label = latex_mask(r"$\mathrm{Re}(s)=\frac{1}{2}$", 24)
    t_label = latex_mask(r"$t\ \to$", 32)
    onward_text = latex_mask(r"$\mathrm{every\ zero{,}\ on\ the\ same\ line}$", 50)
    end_lines = [
        (latex_mask(r"$\mathrm{Riemann\ hypothesis{,}\ 1859}$", 46), ink),
        (latex_mask(r"$\mathrm{proved\ in\ November\ 2026}$", 56), gold),
    ]
    next_lines = [
        latex_mask(r"$\mathrm{Part\ 3\ tomorrow}$", 38),
        latex_mask(r"$\mathrm{the\ primes\ are\ made\ of\ waves}$", 38),
    ]
    credit = latex_mask(r"$\mathrm{@random\_plot}$", 36)
    halos = {
        id(m): halo(m)
        for m in [onward_text, credit] + [m for m, _ in end_lines] + next_lines
    }

    def text(image, mask, x, y, colour, alpha, anchor="centre", backing=0.7):
        if alpha <= 0:
            return
        if id(mask) in halos:
            g = halos[id(mask)]
            pad = (g.shape[0] - mask.shape[0]) // 2
            gx = x - pad if anchor == "left" else x
            blit_mask(image, g, gx, y - pad, (0, 0, 0), backing * alpha, anchor)
        blit_mask(image, mask, x, y, colour, alpha, anchor)

    background = np.empty((HEIGHT, WIDTH, 3), np.float32)
    background[:] = bg

    def render(frame):
        seconds = seconds_all[frame]
        sigma, t_pen = states[frame]
        fade = 1 - smoothstep(np.clip((seconds - (DURATION - 0.5)) / 0.5, 0, 1))
        image = background.copy()

        # axes through 0
        axis = tuple(float(v) * 0.35 for v in muted)
        cv2.line(image, (40, int(oy)), (WIDTH - 40, int(oy)), axis, 1, cv2.LINE_AA)
        cv2.line(image, (int(ox), 700), (int(ox), 1600), axis, 1, cv2.LINE_AA)

        # the curve, coloured by t; a ghost of the σ = 1/2 curve during the sweep.
        # Drawn at SUPERSAMPLE x the size and averaged down, for smooth lines.
        n = int(np.searchsorted(ts, t_pen))
        big = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE, 3), np.float32)
        fixed = 4 * SUPERSAMPLE  # cv2 shift=2: coordinates in quarter pixels
        in_sweep = SWEEP[0] <= seconds < SWEEP[1]
        if in_sweep:
            ghost = to_screen(curve(0.5)[:n])
            pts = np.round(ghost * fixed).astype(np.int32)
            cv2.polylines(
                big,
                [pts],
                False,
                tuple(0.25 * float(v) for v in gold),
                SUPERSAMPLE,
                cv2.LINE_AA,
                2,
            )
        points = to_screen(curve(sigma)[:n])
        edges = np.linspace(0, n, bands + 1).astype(int)
        for k in range(bands):
            a, b = edges[k], min(edges[k + 1] + 1, n)
            if b - a < 2:
                continue
            band = int(np.mean(ts[a:b]) / T_END * (bands - 1))
            pts = np.round(points[a:b] * fixed).astype(np.int32)
            cv2.polylines(
                big,
                [pts],
                False,
                band_colour[band],
                round(LINE_PX * SUPERSAMPLE),
                cv2.LINE_AA,
                2,
            )
        head = np.round(points[-1] * fixed).astype(np.int32)
        cv2.circle(
            big,
            (int(head[0]), int(head[1])),
            7 * fixed,
            tuple(float(v) for v in ink),
            -1,
            cv2.LINE_AA,
            2,
        )
        layer = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        del big
        small = cv2.resize(
            layer, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA
        )
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (WIDTH, HEIGHT))
        image += layer + 1.6 * glow

        # hits: the pen passing a zero (trace and onward), or σ crossing 1/2 (sweep)
        passed = zeros[zeros <= t_pen]
        flash = 0.0
        if not in_sweep and passed.size:
            since = []
            for g in passed:  # seconds since the pen passed γ
                lo, hi = 0, frame
                while lo < hi:  # first frame whose pen t reaches γ
                    mid = (lo + hi) // 2
                    if states[mid][1] >= g:
                        hi = mid
                    else:
                        lo = mid + 1
                since.append(seconds - seconds_all[lo])
            recent = min(since)
            flash = max(0.0, 1 - recent / FLASH)
        if in_sweep:
            phase = (seconds - SWEEP[0]) / 3.0  # σ = 1/2 every 3 s
            recent = min(phase % 1, 1 - phase % 1) * 3.0
            flash = max(0.0, 1 - recent / FLASH) * (
                miss.get(frame, 1) < 0.02 or recent < 0.1
            )
        if flash > 0:
            ring = np.zeros((HEIGHT, WIDTH), np.float32)
            radius = int(10 + 50 * (1 - flash))
            cv2.circle(ring, (int(ox), int(oy)), radius, 1.0, 3, cv2.LINE_AA)
            cv2.circle(ring, (int(ox), int(oy)), 6, 1.0, -1, cv2.LINE_AA)
            ring = ring + 2 * cv2.GaussianBlur(ring, (0, 0), 8)
            image += flash * ring[..., None] * gold
            text(image, hit_mask, int(ox) - 20, int(oy) - 110, gold, flash, "left")
        else:
            cv2.circle(
                image,
                (int(ox), int(oy)),
                4,
                tuple(float(v) * 0.7 for v in ink),
                -1,
                cv2.LINE_AA,
            )

        # the critical strip across the top
        t_view = 52.0 + (T_END + 2 - 52.0) * smoothstep(
            np.clip((seconds - ONWARD[0]) / (ONWARD[1] - ONWARD[0]), 0, 1)
        )
        cv2.rectangle(
            image,
            (sx0, sy0),
            (sx1, sy1),
            tuple(float(v) * 0.5 for v in muted),
            1,
            cv2.LINE_AA,
        )
        y_half = int(strip_y(0.5))
        for x in range(sx0, sx1, 14):  # the critical line, dashed
            cv2.line(
                image,
                (x, y_half),
                (min(x + 7, sx1), y_half),
                tuple(float(v) * 0.6 for v in gold),
                1,
                cv2.LINE_AA,
            )
        y_sigma = int(round(strip_y(sigma)))
        cv2.line(
            image,
            (sx0, y_sigma),
            (int(strip_x(t_pen, t_view)), y_sigma),
            tuple(float(v) * 0.8 for v in ink),
            2,
            cv2.LINE_AA,
        )
        shown = zeros if in_sweep or seconds >= SWEEP[1] else passed
        for k, g in enumerate(zeros):
            if g > t_pen and not in_sweep:
                continue
            x = int(strip_x(g, t_view))
            if x > sx1:
                continue
            cv2.circle(
                image, (x, y_half), 5, tuple(float(v) for v in gold), -1, cv2.LINE_AA
            )
            if t_view < 60 and k < 10:  # heights of the first ten, alternating sides
                m = zero_labels[k]
                y = y_half + 12 if k % 2 == 0 else y_half - 12 - m.shape[0]
                blit_mask(image, m, x, y, gold * 0.9, 0.9, "centre")
        px = int(strip_x(t_pen, t_view))
        cv2.circle(
            image, (px, y_sigma), 7, tuple(float(v) for v in ink), -1, cv2.LINE_AA
        )
        blit_mask(image, strip_title, sx0, sy0 - strip_title.shape[0] - 10, muted, 1.0)
        blit_mask(image, t_label, sx1 + 12, y_half - t_label.shape[0] // 2, muted, 1.0)

        # readouts under the strip: Re(s) on the left, a count or a distance on the right
        y = sy1 + 40

        def put(parts, x, colour, alpha, right=False):
            label, number = parts
            if right:
                x -= label.shape[1] + number.shape[1]
            blit_mask(image, label, x, y, ink, 0.9 * alpha)
            blit_mask(image, number, x + label.shape[1], y, colour, alpha)

        on_line = abs(sigma - 0.5) < 5e-4
        put(sigma_masks[f"{sigma:.3f}"], sx0, gold if on_line else ink, 1.0)
        a_miss = (
            window(seconds, SWEEP[0] + 0.3, SWEEP[1] - 0.1, 0.4)
            if frame in miss
            else 0.0
        )
        if a_miss < 1:  # zeros hit, on the right
            k = int(passed.size if not in_sweep else (zeros < T_FIRST).sum())
            put(count_masks[str(k)], sx1, ink, 1 - a_miss, right=True)
        if a_miss > 0:  # during the sweep: how close the loops come to 0
            hit = miss[frame] < 0.0015
            put(
                miss_masks[f"{miss[frame]:.3f}"],
                sx1,
                gold if hit else ink,
                a_miss,
                right=True,
            )

        # top text: the hook, then the run-on line, then the close
        a_hook = window(seconds, -1.0, HOOK_END, 0.6)
        if a_hook > 0:
            y = HOOK_Y
            for mask, g in zip(hook_lines, hook_halos):
                blit_mask(
                    image, g, WIDTH // 2, y - 14, (0, 0, 0), 0.6 * a_hook, "centre"
                )
                blit_mask(image, mask, WIDTH // 2, y, ink, a_hook, "centre")
                y += int(mask.shape[0] * 1.25)
        text(
            image,
            onward_text,
            WIDTH // 2,
            HOOK_Y + 20,
            ink,
            window(seconds, ONWARD[0] + 0.3, ONWARD[1] + 0.2, 0.5),
        )
        a_end = window(seconds, ONWARD[1], 99, 0.6)
        if a_end > 0:
            y = HOOK_Y
            for mask, colour in end_lines:
                text(image, mask, WIDTH // 2, y, colour, a_end)
                y += mask.shape[0] + 20
            a_next = window(seconds, ONWARD[1] + 1.2, 99, 0.6)
            y = 1560
            for mask in next_lines:
                text(
                    image,
                    mask,
                    WIDTH // 2,
                    y,
                    ink if mask is next_lines[1] else gold,
                    a_next,
                )
                y += mask.shape[0] + 14
            text(image, credit, WIDTH // 2, y + 20, ink, 0.85 * a_next)
        return np.clip(image * fade + bg * (1 - fade) * 0, 0, 255)

    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(MUSIC, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
