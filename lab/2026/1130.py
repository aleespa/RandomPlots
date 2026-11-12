"""
The 9% That Never Goes Away

A square wave built from sine waves, one Fourier series term at a time. At the
top are the ingredients: sin x, then sin 3x a third as tall, sin 5x a fifth as
tall, and so on. In the middle is their sum, which straightens into a square
wave as more and more sine waves are added, up to 256 of them.

But look at the corners. Every sum overshoots the jump and rings, and the
bottom panel zooms in on the jump as the waves pile up. The ripples squeeze
closer to the corner, yet the first one never gets lower: it always overshoots
by about 9% of the jump. Add a million sine waves and it is still there. This
is the Gibbs phenomenon, and it is why sharp edges in compressed images and
audio come with a faint ring.

Listen: a pure tone grows into a square wave's hollow buzz as its overtones
join, each new one struck on a bell at its own pitch. Every time the number of
waves doubles, the loudest three ring again, and the overshoot is still there.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#gibbsphenomenon #fourierseries #mathart #generativeart #randomplots
"""

import gc
import io

import cv2
import numpy as np
from loguru import logger
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    hex_to_bgr,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The series --------------------------------------------------------------------
TERMS = 256  # odd harmonics k = 1, 3, ..., 511 by the end
BUILD = 13.0  # s: terms arrive until here, then the picture rests
GROWTH = 2.5  # M(s) = (1 + s/τ)^GROWTH terms at time s, reaching TERMS at BUILD
TAU = BUILD / (TERMS ** (1 / GROWTH) - 1)
GIBBS = 2 / np.pi * 1.851937051982466  # (2/π) Si(π) = 1.17898: the overshoot peak
FADE = 0.4  # s: fade to the background at the end

# --- Layout (px, 1x) ---------------------------------------------------------------
SS = 2  # supersampling
X_RANGE = (-0.6 * np.pi, 2.6 * np.pi)  # x shown by the fan and the sum
FAN_Y, FAN_SCALE = 410, 210  # centre line and px per unit
SUM_Y, SUM_SCALE = 1010, 190
LENS_TOP, LENS_BOTTOM = 1330, 1700  # px rows of the magnified jump
LENS_S = (0.3, 1.25)  # values of the sum shown in the lens
LENS_XI = (-2 * np.pi, 6 * np.pi)  # lens shows x = ξ / (2 M^ZOOM)
ZOOM = 0.6  # < 1: the zoom lags the ripples, so they squeeze towards the jump
LENS_X = (60, 1020)
LABEL_Y, COUNTER_Y, LENS_LABEL_Y = 62, 708, 1288

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#0a0710"
FAN_PALETTE = ["#4b2bb0", "#a23ac0", "#ff5e7e", "#ffb347", "#fff1b8"]
SUM_CORE, SUM_GLOW = "#fff6e6", "#ff9a5c"
GHOST = "#4a3d5c"  # the target square wave, the lens frame
OVERSHOOT = "#ff6f61"
TEXT, TEXT_DIM = "#f3e9ff", "#9a8bb0"
GLOW_SIGMA, GLOW_GAIN = 6.0, 0.9

# --- Sound -------------------------------------------------------------------------
ROOTS = (45, 47, 48, 50)  # MIDI fundamentals f0 to choose from (A2, B2, C3, D3)
STRUCK = 16  # the first STRUCK terms are struck on a bell as they arrive
DRONE_TOP = 7000.0  # Hz: overtones of the drone stop here
CHIME = (1, 3, 5)  # harmonics struck again whenever the term count doubles


def terms_at(s):
    """M(s): how many terms are in at time s (seconds), continuous, 1 .. TERMS."""
    s = np.clip(np.asarray(s, float), 0.0, BUILD)
    return np.minimum((1 + s / TAU) ** GROWTH, TERMS)


def time_of(m):
    """The inverse of `terms_at`: when M reaches m."""
    return TAU * (np.asarray(m, float) ** (1 / GROWTH) - 1)


def weights(m):
    """Weight of each term j = 0 .. TERMS-1 (k = 2j + 1) when M = m: 0 -> 1 as M passes j."""
    return smoothstep(m - np.arange(TERMS))


def text_mask(tex, size):
    """LaTeX (mathtext, Computer Modern) at `size` px per em, as a float32 alpha crop."""
    buffer = io.BytesIO()
    prop = FontProperties(size=size, math_fontfamily="cm")
    mathtext.math_to_image(tex, buffer, prop=prop, dpi=72, format="png")
    grey = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    mask = 1 - grey.astype(np.float32) / 255  # black ink on white
    rows = np.flatnonzero(mask.max(1) > 0)
    cols = np.flatnonzero(mask.max(0) > 0)
    return np.pad(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], 2)


def blit(image, mask, x, y, colour, alpha=1.0, anchor="left"):
    """Composite a text mask onto a float BGR image at (x, y) = top-left (or top-centre)."""
    h, w = mask.shape
    if anchor == "centre":
        x -= w // 2
    region = image[y : y + h, x : x + w]
    a = alpha * mask[..., None]
    region *= 1 - a
    region += a * np.array(colour, np.float32)


def soundtrack(voice, rng):
    """
    The Fourier series made audible, from the same M(s) as the picture.

    - Drone: additive synthesis of the very same partial sum, Σ w_j(s) sin(2π f0 k τ)/k
      for k = 2j + 1, with w_j(s) the weights the frames use, so the tone starts as a pure
      sine at f0 and becomes a square wave's buzz. Overtones above DRONE_TOP are left out.
    - Strikes: the first STRUCK terms are struck on `voice` (a bell) at f0·k as they reach
      half weight, louder for larger amplitude 1/k, panned by k.
    - Chimes: each time M doubles from 16 to 256, harmonics CHIME (f0, 3f0, 5f0) are
      struck together; the last lands as the build finishes and rings over the rest.
    One-shot: nothing wraps (circular=False) and `master(loop=False)`.
    """
    f0 = float(audio.midi_to_hz(int(rng.choice(ROOTS))))
    sr = audio.SAMPLE_RATE
    track = audio.silence(DURATION)
    n = len(track)
    tau = np.arange(n) / sr
    m = terms_at(tau)
    drone = np.zeros(n)
    for j in range(TERMS):
        k = 2 * j + 1
        if k * f0 > DRONE_TOP:
            break
        w = smoothstep(m - j)
        drone += w * np.sin(2 * np.pi * f0 * k * tau) / k
    fade = np.clip((DURATION - tau) / 1.5, 0, 1)
    drone *= 0.4 * (0.5 - 0.5 * np.cos(np.pi * fade))
    drone = audio.lowpass(drone[:, None] * np.ones(2), 4500.0, circular=False)
    track += drone

    for j in range(STRUCK):
        k = 2 * j + 1
        when = 0.0 if j == 0 else float(time_of(j + 0.5))
        note = voice(f0 * k, float(np.clip(0.75 / np.sqrt(k), 0.12, 0.75)))
        audio.add(
            track, audio.pan(note, 0.7 * np.tanh((k - 8) / 10)), when, circular=False
        )
    doublings = [2**p for p in range(4, 9)]
    for i, count in enumerate(doublings):
        when = float(time_of(count))
        last = count == TERMS
        for k in CHIME:
            velocity = (0.8 if last else 0.5) / np.sqrt(k)
            audio.add(track, voice(f0 * k, velocity), when, circular=False)
    logger.info(
        f"Soundtrack: f0 {f0:.1f} Hz, drone up to k = {int(DRONE_TOP // f0)}, "
        f"{STRUCK} strikes, chimes at {[round(float(time_of(c)), 2) for c in doublings]} s"
    )
    track = audio.reverb(track, rng, rt60=2.6, wet=0.3, tone=5000.0, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The Gibbs phenomenon, as a one-shot. The Fourier series of the square wave
    sign(sin x) is (4/π) Σ sin(kx)/k over odd k. At time s the series holds
    M(s) = (1 + s/τ)^GROWTH terms (continuous; term j = 0, 1, ... has weight
    smoothstep(M − j), so each new sine grows in from flat), from 1 at s = 0 to
    TERMS = 256 at s = BUILD, then the picture rests and fades out.

    Three panels share that one closed form. Top: every weighted term
    w_j (4/π) sin(kx)/k drawn over the same x, coloured by log k, the newest
    brighter. Middle: the partial sum, a glowing curve over the dim target
    square wave; a box at the jump x = 0 marks the lens. Bottom (the lens): the
    sum near the jump, x = ξ / (2 M^ZOOM), values LENS_S. Near the jump
    S_M(x) → (2/π) Si(2Mx), so the ripples are 1/M wide; the lens zooms only as
    M^ZOOM (ZOOM = 0.6), so they visibly squeeze into the wall while the first
    peak stays at (2/π) Si(π) ≈ 1.179, an overshoot of 9% of the jump of 2 (the
    red line, which pulses each time M doubles).

    Rendering: sin(kx) for every k is precomputed once on the 2x-supersampled
    column grid; each frame is a weighted sum, 256 cv2 polylines, a downsample,
    a Gaussian glow and text masks typeset once in
    Computer Modern with matplotlib mathtext, all a pure function of the frame
    index on the render_frames thread pool. Nothing is random: --seed does not
    change the picture. The soundtrack (`soundtrack`, Harmonic series) uses
    settings.audio_rng only to pick f0 and the reverb; the posted one used --seed 0.
    An --audio-only tag naming a common.music style swaps the bell for its voice.
    """
    settings = settings or ImageProcessingSettings(1)
    W2, H2 = WIDTH * SS, HEIGHT * SS
    shift = 2  # cv2 fractional bits
    one = 1 << shift

    def px(v):
        return np.round(v * SS * one).astype(np.int32)

    ks = 2 * np.arange(TERMS) + 1
    amps = 4 / (np.pi * ks)
    cols = (np.arange(W2) + 0.5) / SS
    xs = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * cols / WIDTH
    basis = np.sin(np.outer(ks, xs)).astype(np.float32)  # (TERMS, W2)
    lens_cols = np.linspace(LENS_X[0], LENS_X[1], (LENS_X[1] - LENS_X[0]) * SS)
    xi = LENS_XI[0] + (LENS_XI[1] - LENS_XI[0]) * (lens_cols - LENS_X[0]) / (
        LENS_X[1] - LENS_X[0]
    )
    lens_scale = (LENS_BOTTOM - LENS_TOP) / (LENS_S[1] - LENS_S[0])

    def lens_y(s):
        return LENS_BOTTOM - (s - LENS_S[0]) * lens_scale

    def screen_x(x):
        return (x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * WIDTH

    lut = palette_lut(FAN_PALETTE, 1024)[:, ::-1] * 255  # BGR
    fan_colour = lut[(np.clip(np.log(ks) / np.log(127), 0, 1) * 1023).astype(int)]
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)

    # The static layer at 2x: the target square wave and the lens frame.
    static = np.full((H2, W2, 3), hex_to_bgr(BACKGROUND), np.uint8)
    ghost = hex_to_bgr(GHOST)
    square = np.sign(np.sin(xs + 1e-9))
    pts = np.stack([px(cols), px(SUM_Y - SUM_SCALE * square)], 1)
    cv2.polylines(static, [pts], False, ghost, 2 * SS, cv2.LINE_AA, shift)
    cv2.line(
        static, (0, px(FAN_Y)), (W2 * one, px(FAN_Y)), ghost, SS, cv2.LINE_AA, shift
    )
    for s_value in (1.0,):
        y = px(lens_y(s_value))
        cv2.line(
            static,
            (px(LENS_X[0]), y),
            (px(LENS_X[1]), y),
            ghost,
            SS,
            cv2.LINE_AA,
            shift,
        )
    lens_square = np.where(xi < 0, -1.0, 1.0)
    lens_rows = slice(int(LENS_TOP - 30) * SS, int(LENS_BOTTOM + 30) * SS)
    lens_offset = (LENS_TOP - 30) * SS * one
    pts = np.stack([px(lens_cols), px(lens_y(lens_square)) - lens_offset], 1)
    cv2.polylines(static[lens_rows], [pts], False, ghost, 2 * SS, cv2.LINE_AA, shift)

    # Text, once.
    label = text_mask(
        r"$\sin x\, +\, \frac{\sin 3x}{3}\, +\, \frac{\sin 5x}{5}\, +\, "
        r"\frac{\sin 7x}{7}\, +\, \cdots$",
        40,
    )
    lens_label = text_mask(r"$\mathrm{the\ jump{,}\ magnified}$", 36)
    nine = text_mask(r"$+9\%$", 48)
    counters = [None] + [
        text_mask(
            rf"${c}\ \mathrm{{sine\ wave{'s' if c > 1 else ''}{{,}}\ added\ up}}$", 54
        )
        for c in range(1, TERMS + 1)
    ]
    text, text_dim = hex_to_bgr(TEXT), hex_to_bgr(TEXT_DIM)
    doubling_times = time_of(np.array([2**p for p in range(1, 9)]))
    core, halo = hex_to_bgr(SUM_CORE), hex_to_bgr(SUM_GLOW)
    overshoot = np.array(hex_to_bgr(OVERSHOOT), np.float32)
    logger.info(
        f"τ = {TAU:.3f} s; M doubles at {np.round(doubling_times, 2).tolist()} s"
    )

    def render(frame: int) -> np.ndarray:
        s = frame / (FRAME_COUNT - 1) * DURATION
        m = float(terms_at(s))
        zoom = m**ZOOM
        w = weights(m)
        live = int(np.count_nonzero(w))
        coeff = (w * amps).astype(np.float32)
        canvas = static.copy()

        # Top: the ingredients, high k first so the big ones sit on top.
        fan = coeff[:live, None] * basis[:live]
        ys = px(FAN_Y - FAN_SCALE * fan)
        xcol = px(cols)
        for j in range(live - 1, -1, -1):
            fresh = np.exp(-max(m - j, 0.0) / 1.5)
            colour = np.clip(fan_colour[j] * (0.55 + 0.45 * w[j] + 0.6 * fresh), 0, 255)
            thick = SS * (2 if j < 8 else 1)
            pts = np.stack([xcol, ys[j]], 1)
            cv2.polylines(
                canvas, [pts], False, colour.tolist(), thick, cv2.LINE_AA, shift
            )

        # Middle: the sum, and the lens box at the jump.
        total = coeff[:live] @ basis[:live]
        pts = np.stack([xcol, px(SUM_Y - SUM_SCALE * total)], 1)
        cv2.polylines(canvas, [pts], False, halo, 5 * SS, cv2.LINE_AA, shift)
        cv2.polylines(canvas, [pts], False, core, 2 * SS, cv2.LINE_AA, shift)
        x0, x1 = (screen_x(v / (2 * zoom)) for v in LENS_XI)
        x0, x1 = max(x0, 2), min(x1, WIDTH - 2)
        y0 = SUM_Y - SUM_SCALE * 1.24
        y1 = SUM_Y - SUM_SCALE * LENS_S[0]
        box_colour = hex_to_bgr(TEXT_DIM)
        cv2.rectangle(
            canvas,
            (px(x0), px(y0)),
            (px(x1), px(y1)),
            box_colour,
            SS,
            cv2.LINE_AA,
            shift,
        )
        for a, b in ((x0, LENS_X[0]), (x1, LENS_X[1])):
            cv2.line(
                canvas,
                (px(a), px(y1)),
                (px(b), px(LENS_TOP - 30)),
                ghost,
                SS,
                cv2.LINE_AA,
                shift,
            )

        # Bottom: the lens, S_M(ξ / 2M) against ξ.
        lens = coeff[:live] @ np.sin(np.outer(ks[:live], xi / (2 * zoom))).astype(
            np.float32
        )
        pulse = float(np.sum(np.exp(-(s - doubling_times[doubling_times <= s]) / 0.45)))
        pulse = min(pulse, 1.0)
        y = px(lens_y(GIBBS)) - lens_offset
        line_colour = (overshoot * (0.55 + 0.45 * pulse)).tolist()
        sub = canvas[lens_rows]
        cv2.line(
            sub,
            (px(LENS_X[0]), y),
            (px(LENS_X[1]), y),
            line_colour,
            SS * (2 if pulse > 0.5 else 1),
            cv2.LINE_AA,
            shift,
        )
        pts = np.stack([px(lens_cols), px(lens_y(lens)) - lens_offset], 1)
        cv2.polylines(sub, [pts], False, halo, 5 * SS, cv2.LINE_AA, shift)
        cv2.polylines(sub, [pts], False, core, 2 * SS, cv2.LINE_AA, shift)

        image = cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        light = np.clip(image - bg, 0, None)
        glow = cv2.GaussianBlur(light, (0, 0), GLOW_SIGMA)
        image += GLOW_GAIN * glow

        blit(image, label, WIDTH // 2, LABEL_Y, text_dim, anchor="centre")
        count = int(np.clip(round(m), 1, TERMS))
        blit(image, counters[count], WIDTH // 2, COUNTER_Y, text, anchor="centre")
        blit(image, lens_label, LENS_X[0], LENS_LABEL_Y, text_dim)
        blit(
            image,
            nine,
            LENS_X[1] - nine.shape[1],
            int(lens_y(GIBBS)) - nine.shape[0] - 10,
            overshoot.tolist(),
            alpha=0.7 + 0.3 * pulse,
        )
        out = float(smoothstep((DURATION - s) / FADE))
        if out < 1:
            image = bg + out * (image - bg)
        return image

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag
    voice = STYLES[tag].voice if tag in STYLES else audio.bell
    track = soundtrack(voice, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
