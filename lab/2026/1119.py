"""
The Most Irrational Number

Ford circles: over every fraction p/q on the number line sits a circle
q² times smaller than the one over 1. They never overlap; they only kiss.
Here the camera dives towards the golden ratio, φ = 1.618…, almost a
billion times closer.

The circles that reach nearest to φ are always 1/1, 2/1, 3/2, 5/3, 8/5,
13/8…: ratios of neighbouring Fibonacci numbers, closing in alternately
from the left and the right. The golden ratio is the hardest number of
all to approximate with fractions (Hurwitz, 1891), so the gaps never get
any easier: every zoom looks just like the last one, forever.

Each new Fibonacci fraction strikes a piano note: the Fibonacci numbers
themselves, counted round a seven-note scale.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#fordcircles #numbertheory #mathart #generativeart #randomplots
"""

import gc
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import PROGRESSIONS, STYLES, Scale
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

# Opening line over the first 3 s, drawn by render_frames.
HOOK = "The number fractions can't catch"
HOOK_Y = 400
HOOK_COLOUR = "#1c1b22"

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- The dive -------------------------------------------------------------------------
PHI = (1 + 5**0.5) / 2
VIEW_START, VIEW_END = 1.8, 2.0e-9  # number-line units across the screen
ZOOM_START, ZOOM_END = 0.0, 13.2  # s: the dive (eased in and out)
EASE_IN, EASE_OUT = 1.4, 2.2  # s of speed ramp at either end
FADE = 0.4  # s fade to the paper at the very end

# --- Layout ---------------------------------------------------------------------------
LINE_Y = 1420  # px: the number line; the circles stand on it, mirrored below
MIRROR = 0.32  # reflection strength just under the line, fading downwards
SS = 2  # supersampling of the circle layer
MIN_RADIUS = 0.3  # px: smaller circles are not drawn
LABEL_IN = (48.0, 78.0)  # px radius over which a circle's fraction fades in

# --- Look -----------------------------------------------------------------------------
PAPER = "#f4f0e7"
INK = "#1f2a55"  # indigo
ACCENT = "#d6452a"  # vermilion: the Fibonacci fractions
FILLS = ["#1f2a55", "#3d5a8a", "#7d9cb8", "#cdbf9f", "#e9dfc9"]  # tiny -> huge
CM = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # BaKoMa Computer Modern
PHI_GLYPH, TIMES = chr(193), chr(163)  # in cmmi10, cmsy10

# --- Sound ----------------------------------------------------------------------------
STYLE = "felt_piano"  # common.music.STYLES; an --audio-only tag naming one overrides it
EVENT_RADIUS = 78.0  # px: a Fibonacci circle sounds when its label is fully in
PROGRESSION = "drift"  # common.music.PROGRESSIONS, a chord per four Fibonacci notes


def dive_curve():
    """(times, log view width) on a fine grid: speed eased in and out, then rest."""
    tau = np.linspace(0.0, DURATION, 30001)
    speed = smoothstep((tau - ZOOM_START) / EASE_IN) * smoothstep(
        (ZOOM_END - tau) / EASE_OUT
    )
    progress = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    progress /= progress[-1]
    log_view = np.log(VIEW_START) + (np.log(VIEW_END) - np.log(VIEW_START)) * progress
    return tau, log_view


def fibonacci(count):
    """[(p, q)]: the Fibonacci ratios F(k+1)/F(k), k = 1 .. count."""
    pairs, a, b = [], 1, 1
    for _ in range(count):
        pairs.append((b, a))
        a, b = b, a + b
    return pairs


def ford_circles(view):
    """
    Every Ford circle that touches the screen when it is `view` units wide,
    centred on φ: arrays p, q (coprime) and the circle's centre x offset from φ
    and radius, in px. The circle over p/q has radius 1/(2q²) and touches the
    line at p/q; it is kept if its radius is at least MIN_RADIUS px and it
    meets the rectangle above the line that the screen shows.
    """
    scale = WIDTH / view  # px per unit
    half = 0.5 * WIDTH / scale + 8 / scale
    top = (LINE_Y + 8) / scale
    q_max = int(np.sqrt(scale / (2 * MIN_RADIUS)))
    q = np.arange(1, q_max + 1, dtype=np.int64)
    reach = 1.0 / (2.0 * q * q)  # the circle's radius in units
    lo = np.ceil((PHI - half - reach) * q).astype(np.int64)
    hi = np.floor((PHI + half + reach) * q).astype(np.int64)
    count = np.maximum(hi - lo + 1, 0)
    q = np.repeat(q, count)
    start = np.repeat(lo, count)
    offset = np.arange(len(q)) - np.repeat(np.cumsum(count) - count, count)
    p = start + offset
    keep = np.gcd(p, q) == 1
    p, q = p[keep], q[keep]
    radius = 1.0 / (2.0 * q * q)
    dx = (p - q * PHI) / q  # tangent point minus φ
    # Does the circle meet the box |x - φ| <= half, 0 <= y <= top? (no cancellation)
    gap = np.maximum(np.abs(dx) - half, 0.0)
    near = top * (2 * radius - top)
    meets = (gap * gap < near) | (top >= radius) & (gap < radius)
    p, q, dx, radius = p[meets], q[meets], dx[meets], radius[meets]
    return p, q, dx * scale, radius * scale


def circle_polygon(cx, cy, r, w, h):
    """Points of the part of a large circle near the box [0, w] x [0, h], in px."""
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) - (cx, cy)
    inside = abs(cx - w / 2) < w / 2 and abs(cy - h / 2) < h / 2
    if inside:
        a0, a1 = 0.0, 2 * np.pi
    else:
        mid = np.arctan2(h / 2 - cy, w / 2 - cx)
        angles = np.arctan2(corners[:, 1], corners[:, 0]) - mid
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        a0, a1 = mid + angles.min() - 0.02, mid + angles.max() + 0.02
    n = int(np.clip((a1 - a0) * r / 3.0, 16, 20000))
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


class Typesetter:
    """Computer Modern text onto an RGBA PIL layer."""

    def __init__(self):
        self.fonts = {}

    def font(self, face, size):
        key = face, int(size)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(CM / f"{face}.ttf"), int(size))
        return self.fonts[key]


def generate(settings: ImageProcessingSettings = None):
    """
    Ford circles (L. R. Ford, 1938): over each reduced fraction p/q, the circle
    of radius 1/(2q²) tangent to the real line at p/q. Two are tangent exactly
    when |p q' - p' q| = 1 (Farey neighbours) and never overlap otherwise; they
    are the horoballs of the modular group, so the packing is invariant under
    SL(2, Z).

    The camera is centred on φ and its width W(τ) shrinks from VIEW_START to
    VIEW_END, exponentially with an eased speed (smoothstep ramps of EASE_IN
    and EASE_OUT s), then rests. Per frame, every circle with radius >= MIN_RADIUS
    px that meets the screen is enumerated directly: q <= sqrt(S / 2 r_min),
    p in the window widened by the radius, gcd(p, q) = 1, then an exact
    box-circle test written without cancellation (the huge circles of small q
    only graze the top of the screen deep in the dive). Since φ is badly
    approximable (|φ - p/q| > ~1/(√5 q²)), the visible circles never exceed a
    few thousand px, and their count stays near a thousand at every depth.

    Drawing: cv2 at SS x supersampling, fills coloured by on-screen size
    (FILLS, tiny -> huge) mixed into the paper, ink outlines; circles larger
    than ~200 px are drawn as clipped polygons of their visible arc (cv2's own
    circles are 72-gons at that size). The Fibonacci ratios F(k+1)/F(k), the
    continued-fraction convergents of φ = [1; 1, 1, ...], are filled and
    outlined in ACCENT. Fractions are typeset in each circle above LABEL_IN px
    radius. The layer is mirrored under the line, faded (MIRROR). On top: φ's
    decimals as far as the view resolves them and the zoom factor.

    Each frame is a pure function of τ; frames render in parallel on the
    thread pool. Nothing is random in the picture: the seed only picks the
    soundtrack's key (through the audio generator).
    """
    settings = settings or ImageProcessingSettings(1)
    curve_tau, curve_log = dive_curve()
    convergents = {pair for pair in fibonacci(40)}
    phi_text = f"{PHI:.15f}"

    bg = np.array(hex_to_bgr(PAPER), np.float32)
    ink = hex_to_bgr(INK)
    accent = np.array(hex_to_bgr(ACCENT), np.float32)
    fill_lut = palette_lut(FILLS, 256)[:, ::-1] * 255.0  # BGR
    type_ = Typesetter()
    ink_rgb = tuple(int(INK[i : i + 2], 16) for i in (1, 3, 5))
    accent_rgb = tuple(int(ACCENT[i : i + 2], 16) for i in (1, 3, 5))

    def fill_colour(r):
        """Fill for a circle of r px: tiny circles dark, huge ones pale; strength by size."""
        u = np.clip((np.log2(max(r, 0.25)) + 1) / 12.0, 0, 1)
        colour = fill_lut[int(u * 255)]
        alpha = 0.95 - 0.55 * u
        return tuple(float(c) for c in alpha * colour + (1 - alpha) * bg)

    def render(frame):
        tau = frame / FPS
        view = float(np.exp(np.interp(tau, curve_tau, curve_log)))
        p, q, cx, r = ford_circles(view)
        h, w = LINE_Y * SS, WIDTH * SS
        layer = np.empty((h, w, 3), np.uint8)
        layer[:] = bg.astype(np.uint8)
        order = np.argsort(-r)  # big first (they never overlap, but outlines do)
        labels = []
        for i in order:
            pi_, qi = int(p[i]), int(q[i])
            rad = float(r[i])
            x = WIDTH / 2 + float(cx[i])
            y = LINE_Y - rad
            fib = (pi_, qi) in convergents
            if fib:
                base = np.array(fill_colour(rad), np.float32)
                fill = tuple(float(c) for c in 0.3 * accent + 0.7 * base)
                line = tuple(float(c) for c in accent)
            else:
                fill = fill_colour(rad)
                line = ink
            X, Y, R = x * SS, y * SS, rad * SS
            thick = int(np.clip(0.035 * R, 1.0, 5.0 if fib else 3.0))
            if R < 180:
                c = (int(round(X * 16)), int(round(Y * 16)))
                cv2.circle(layer, c, max(int(R * 16), 1), fill, -1, cv2.LINE_AA, 4)
                if R > 3:
                    cv2.circle(layer, c, int(R * 16), line, thick, cv2.LINE_AA, 4)
            else:
                poly = circle_polygon(X, Y, R, w, h)
                pts = np.round(poly * 16).astype(np.int64)
                pts = np.clip(pts, -(2**28), 2**28).astype(np.int32)
                cv2.fillPoly(layer, [pts], fill, cv2.LINE_AA, 4)
                closed = R < 1.2 * max(w, h) and abs(X - w / 2) < w and Y > -R
                cv2.polylines(layer, [pts], closed, line, thick, cv2.LINE_AA, 4)
            if rad > LABEL_IN[0] and -rad < y < LINE_Y and 0 < x < WIDTH:
                labels.append((pi_, qi, x, y, rad, fib))
        layer = cv2.resize(layer, (WIDTH, LINE_Y), interpolation=cv2.INTER_AREA)

        image = np.empty((HEIGHT, WIDTH, 3), np.float32)
        image[:] = bg
        image[:LINE_Y] = layer
        below = HEIGHT - LINE_Y
        mirror = layer[::-1][:below].astype(np.float32)
        fade = MIRROR * (1 - np.linspace(0, 1, below)) ** 1.6
        image[LINE_Y:] = bg + (mirror - bg) * fade[:, None, None]
        cv2.line(image, (0, LINE_Y), (WIDTH, LINE_Y), ink, 2, cv2.LINE_AA)
        # φ marker: a small vermilion triangle under the line
        tri = np.array([[540, LINE_Y + 6], [528, LINE_Y + 28], [552, LINE_Y + 28]])
        cv2.fillPoly(image, [tri.astype(np.int32)], tuple(float(c) for c in accent))

        # Paper band behind the heading
        band = np.clip((330 - np.arange(330)) / 160.0, 0, 1) ** 1.5
        image[:330] = image[:330] + (bg - image[:330]) * band[:, None, None]

        text = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        pen = ImageDraw.Draw(text)
        for pi_, qi, x, y, rad, fib in labels:
            fade_in = smoothstep((rad - LABEL_IN[0]) / (LABEL_IN[1] - LABEL_IN[0]))
            digits = max(len(str(pi_)), len(str(qi)))
            size = min(0.62 * rad, 1.25 * rad / (0.5 * digits + 0.3), 120.0)
            font = type_.font("cmr10", size)
            colour = accent_rgb if fib else ink_rgb
            fill = colour + (int(235 * fade_in),)
            gap = 0.16 * size
            pen.text((x, y - gap), str(pi_), font=font, fill=fill, anchor="ms")
            pen.text((x, y + gap), str(qi), font=font, fill=fill, anchor="mt")
            bar = 0.55 * size * digits * 0.95 / 2 + 0.1 * size
            pen.line(
                [(x - bar, y), (x + bar, y)], fill=fill, width=max(1, int(0.05 * size))
            )

        # Heading: φ's decimals as far as the view resolves them, and the zoom.
        decimals = int(np.clip(np.floor(np.log10(2.0 / view)) + 1, 1, 12))
        size = 92
        f_math, f_rm = type_.font("cmmi10", size), type_.font("cmr10", size)
        shown = phi_text[: 2 + decimals]
        head = [(PHI_GLYPH, f_math), (" = ", f_rm), (shown, f_rm), ("...", f_rm)]
        width = sum(f.getlength(s) for s, f in head)
        left = (WIDTH - width) / 2
        for s, f in head:
            pen.text((left, 180), s, font=f, fill=ink_rgb + (255,), anchor="ls")
            left += f.getlength(s)
        small = type_.font("cmr10", 40)
        zoom = f"{VIEW_START / view:,.0f}"
        times = type_.font("cmsy10", 40)
        parts = [("zoom ", small), (TIMES, times), (" " + zoom, small)]
        width = sum(f.getlength(s) for s, f in parts)
        left = (WIDTH - width) / 2
        for s, f in parts:
            pen.text((left, 250), s, font=f, fill=ink_rgb + (150,), anchor="ls")
            left += f.getlength(s)

        overlay = np.asarray(text, np.float32)
        alpha = overlay[..., 3:] / 255.0
        image += (overlay[..., 2::-1] - image) * alpha
        end = smoothstep((DURATION - tau) / FADE)
        image = bg + (image - bg) * end
        return image

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(curve_tau, curve_log, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def soundtrack(curve_tau, curve_log, style_name, rng):
    """
    Felt piano over a soft pad, one-shot (circular=False, master(loop=False)).

    Melody: one note per Fibonacci ratio F(k+1)/F(k), at the moment its circle
    grows past EVENT_RADIUS px (its label fully in), solved from the same dive
    curve: radius = S / (2 F(k)²) with S = WIDTH / W(τ). The pitch is F(k+1)
    mod 7, a degree of the style's scale (Fibonacci numbers mod 7 repeat every
    16: the Pisano period), panned to the side of φ the fraction lies on
    (they alternate). Fractions already that big at τ = 0 are struck together
    as the opening chord. Velocity follows the dive speed.
    Under it: a pad chord from PROGRESSION, one per four Fibonacci notes, and
    a bass note on every second one. When the dive stops, the home chord and
    a low root ring out over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    progression = PROGRESSIONS[PROGRESSION]
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    size = len(scale.steps)

    def strike(degree, velocity, when, x=0.0, octave=0, voice=None):
        velocity = round(float(np.clip(velocity, 0.05, 1.0)) * 20) / 20  # cache hits
        note = (voice or style.voice)(scale.hz(degree, octave), velocity)
        audio.add(track, audio.pan(note, float(np.clip(x, -1, 1))), when, **kw)

    # τ when each Fibonacci circle reaches EVENT_RADIUS px: log W = log(WIDTH / (2 q² r))
    log_speed = -np.gradient(curve_log, curve_tau)
    peak_speed = log_speed.max()
    events = []
    for p, q in fibonacci(40):
        target = np.log(WIDTH / (2.0 * q * q * EVENT_RADIUS))
        if target >= curve_log[0]:
            events.append((0.02, p, q))
        elif target >= curve_log[-1]:
            when = float(np.interp(-target, -curve_log, curve_tau))
            events.append((when, p, q))
    opening = [e for e in events if e[0] == 0.02]
    rest = [e for e in events if e[0] > 0.02]
    logger.info(
        f"{len(opening)} opening, {len(rest)} Fibonacci notes "
        f"{rest[0][0]:.2f}-{rest[-1][0]:.2f} s"
    )

    for i, (when, p, q) in enumerate(opening):
        strike(p % 7, 0.45, when + 0.05 * i, 0.4 * np.sign(p / q - PHI))
    for k, (when, p, q) in enumerate(rest):
        speed = float(np.interp(when, curve_tau, log_speed)) / peak_speed
        strike(p % 7, 0.32 + 0.3 * speed, when, 0.55 * np.sign(p / q - PHI))
        if k % 2 == 0:
            chord = progression[(k // 4) % len(progression)]
            strike(chord, 0.42, when, 0.0, octave=-2, voice=style.bass)

    # Pad: one chord per four Fibonacci notes, from the first note to the end.
    starts = [0.0] + [rest[k][0] for k in range(4, len(rest), 4)]
    ends = starts[1:] + [DURATION]
    for j, (a, b) in enumerate(zip(starts, ends)):
        chord = scale.chord(progression[j % len(progression)], 3, octave=-1)
        for midi in chord:
            voice = audio.pad(
                float(audio.midi_to_hz(midi)), b - a + 0.3, 0.16, 0.8, 1.6, 0.35
            )
            audio.add(track, voice, a, **kw)

    # The dive stops: home.
    strike(0, 0.6, ZOOM_END - 0.6, 0.0, octave=-2, voice=style.bass)
    for i, degree in enumerate((0, 2, 4, 7)):
        strike(degree, 0.42, ZOOM_END - 0.6 + 0.08 * i, 0.4 * (i % 2) - 0.2)
    return audio.master(style.room(track, rng, circular=False), loop=False)


if __name__ == "__main__":
    generate()
