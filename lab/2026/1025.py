"""
Buffon's Needles

Estimating pi by dropping needles: every needle that lands across a line
flashes gold, and twice the number of throws divided by the number of crossings
closes in on π, the number of circles, with no circle anywhere in sight.

This is Buffon's needle problem, from 1777. On a floor ruled with lines one
needle-length apart, a needle dropped at random crosses a line with
probability exactly 2/π. So counting crossings measures π, ever more
precisely as the throws pile up, though slowly: 200,000 throws get within
about a hundredth, and each further digit takes a hundred times more needles.
Watch the gold gather along the lines: the closer a needle lands to
one, the likelier it is to cross it.

Made with Python, NumPy & PyTorch.
#buffonsneedle #probability #mathart #generativeart #randomplots
"""

import gc
from pathlib import Path

import matplotlib
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import lfilter

from common import audio
from common.gpu import select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The throws (lengths in line spacings; a needle is exactly one long) ----------
NEEDLES = 200_000
FIRST, LAST = 0.45, 11.0  # s: the first and the last landing
FALL_FIRST, FALL_LAST = 0.45, 0.18  # s in the air: the first needles slowly, then fast
RISE, SPIN = (
    1.4,
    2.5,
)  # in the air at height h (1..0): (1 + RISE h^2) larger, SPIN h^2 turned
NEEDLE_WIDTH = 0.012  # drawn width, at least a pixel

# --- Camera --------------------------------------------------------------------------
SCALE_START, SCALE_END = 480.0, 16.0  # screen pixels per line spacing
ZOOM_START = 1.0  # the pull-back runs from here to LAST (exponential in scale, eased),
PUSH = 1.08  # then a slow push-in by this much to the end; centred on a line

# --- Look ----------------------------------------------------------------------------
BACKGROUND = "#0b0b0f"
LINE_COLOUR, LINE_ALPHA = "#4fd6ff", 0.7  # the ruled lines, drawn over the needles
LINE_WIDTH = 0.004  # in line spacings, at least 0.8 px
MISS_COLOUR, MISS_LIGHT = "#8fa3b8", 0.4  # misses dimmer, so the gold bands show
CROSS_COLOUR = "#ffc857"
NEEDLE_LIGHT = 0.9  # light per pixel of needle
CHUNK = 2_000_000  # points per splat call, to bound GPU memory
FLASH_GAIN, FLASH_TIME = (
    3.0,
    0.25,
)  # a landing crossing flares, decaying over FLASH_TIME s
EXPOSURE, COVER = (
    1.5,
    0.6,
)  # exposure halves each time the needles cover COVER more floor
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 14.0, 4, 0.5

# --- Sound -------------------------------------------------------------------------
STYLE = "kalimba"  # common.music.STYLES; an --audio-only tag naming one overrides it
NOTE_GAP = 0.14  # s: at most one single-needle note this often
NOTE_SPAN = 6  # scale degrees from a flat crossing (low) to a steep one (high)
SHIMMER = 30.0  # ticks per second at the heaviest rain
GLIDE = 0.08  # s: the tuning tone glides to each new estimate
DRONE = 0.16  # loudness of the two tuning tones

# --- Counter (LaTeX-style, in Computer Modern) ---------------------------------------
TEXT_TOP, TEXT_HEIGHT = 190, 460  # px: the band the counter is drawn in
DIGITS = 4
REVEAL = (11.8, 12.6)  # s: the true value fades in under the estimate
TEXT_COLOUR = "#f2efe8"
DIM_COLOUR = "#c9cdd6"
CARD_ALPHA, CARD_RADIUS = 0.82, 36  # a dark card behind the counter...
CARD_COVER = 0.25  # ...fading in as the needles cover this much of the floor
CM = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # matplotlib's BaKoMa fonts
PI, APPROX, TIMES = chr(188), chr(188), chr(163)  # in cmmi10, cmsy10, cmsy10
THICK, MEDIUM, THIN, WORD = 5 / 18, 4 / 18, 3 / 18, 1 / 3  # TeX spaces, in em


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def view_scale(seconds):
    """Screen pixels per line spacing: the eased pull-back to LAST, then the push-in."""
    t = np.asarray(seconds, float)
    out = smoothstep((t - ZOOM_START) / (LAST - ZOOM_START))
    push = smoothstep((t - LAST) / (DURATION - LAST))
    return SCALE_START * (SCALE_END / SCALE_START) ** out * PUSH**push


def throws(rng: np.random.Generator) -> dict:
    """
    Every needle, drawn up front. Needle i lands at FIRST + ln(1 + i) / k, an
    exponentially growing rain (k puts the last at LAST), at a uniformly random
    spot of the view at that moment: x uniform, y a uniformly chosen line row
    plus a uniform fraction, the angle uniform. Needle 0 is staged: at the
    centre, across the line y = 0.
    """
    i = np.arange(NEEDLES)
    land = FIRST + np.log1p(i) / (np.log(NEEDLES) / (LAST - FIRST))
    scale = view_scale(land)
    half_w, half_h = WIDTH / 2 / scale, HEIGHT / 2 / scale
    x = half_w * rng.uniform(-1, 1, NEEDLES)
    y = np.floor(half_h * rng.uniform(-1, 1, NEEDLES)) + rng.uniform(0, 1, NEEDLES)
    angle = rng.uniform(0, np.pi, NEEDLES)
    reach = 0.5 * np.sin(angle)
    x[0], y[0] = 0.0, reach[0] * rng.uniform(-1, 1)  # the first one crosses
    crosses = np.floor(y - reach) != np.floor(y + reach)
    fall = FALL_LAST + (FALL_FIRST - FALL_LAST) * np.exp(-i / 3)
    return dict(land=land, x=x, y=y, angle=angle, crosses=crosses, fall=fall)


class Counter:
    """
    The running estimate set like LaTeX in Computer Modern:
    π ≈ 3.1416 (the digits that agree with π in gold) over the display fraction
    = (2 × N needles) / (C crossings), and π = 3.14159… fading in at the end.
    A run is a list of (text, font, colour) pieces, or (em, None, None) spaces.
    """

    def __init__(self):
        self.fonts = {}
        self.pi = f"{np.pi:.{DIGITS}f}"

    def font(self, face: str, size: int) -> ImageFont.FreeTypeFont:
        if (face, size) not in self.fonts:
            self.fonts[face, size] = ImageFont.truetype(str(CM / f"{face}.ttf"), size)
        return self.fonts[face, size]

    def width(self, run, size: int) -> float:
        return sum(
            piece * size if face is None else self.font(face, size).getlength(piece)
            for piece, face, _ in run
        )

    def put(self, pens, run, size, left, baseline, opacity=1.0):
        pen, halo = pens
        for piece, face, colour in run:
            if face is None:
                left += piece * size
                continue
            font = self.font(face, size)
            fill = tuple(int(255 * c) for c in _rgb(colour)) + (int(255 * opacity),)
            pen.text((left, baseline), piece, font=font, fill=fill, anchor="ls")
            halo.text(
                (left, baseline), piece, font=font, fill=int(255 * opacity), anchor="ls"
            )
            left += font.getlength(piece)

    def draw(self, thrown: int, crossed: int, reveal: float, card: float) -> np.ndarray:
        """An RGBA float image (TEXT_HEIGHT, WIDTH, 4) of the counter, on a card of opacity `card`."""
        text = Image.new("RGBA", (WIDTH, TEXT_HEIGHT), (0, 0, 0, 0))
        halo = Image.new("L", (WIDTH, TEXT_HEIGHT), 0)
        pens = ImageDraw.Draw(text), ImageDraw.Draw(halo)

        estimate = f"{2 * thrown / crossed:.{DIGITS}f}"
        matched = 0
        while matched < len(estimate) and estimate[matched] == self.pi[matched]:
            matched += 1
        size = 120
        line = [
            (PI, "cmmi10", TEXT_COLOUR),
            (THICK, None, None),
            (APPROX, "cmsy10", TEXT_COLOUR),
            (THICK, None, None),
            (estimate[:matched], "cmr10", CROSS_COLOUR),
            (estimate[matched:], "cmr10", TEXT_COLOUR),
        ]
        self.put(pens, line, size, (WIDTH - self.width(line, size)) / 2, 150)

        # = (2 × N needles) / (C crossings), a display fraction on the math axis
        size, axis = 46, 262
        noun = "needle" if thrown == 1 else "needles"
        top = [
            ("2", "cmr10", DIM_COLOUR),
            (MEDIUM, None, None),
            (TIMES, "cmsy10", DIM_COLOUR),
            (MEDIUM, None, None),
            (f"{thrown:,}", "cmr10", DIM_COLOUR),
            (WORD, None, None),
            (noun, "cmr10", DIM_COLOUR),
        ]
        bottom = [
            (f"{crossed:,}", "cmr10", DIM_COLOUR),
            (WORD, None, None),
            ("crossing" if crossed == 1 else "crossings", "cmr10", DIM_COLOUR),
        ]
        equals = [("=", "cmr10", DIM_COLOUR), (THICK, None, None)]
        bar = max(self.width(top, size), self.width(bottom, size)) + 0.3 * size
        left = (WIDTH - self.width(equals, size) - bar) / 2
        self.put(pens, equals, size, left, axis + 0.25 * size)
        left += self.width(equals, size)
        rule = max(2, round(0.045 * size))
        ink = tuple(int(255 * c) for c in _rgb(DIM_COLOUR)) + (255,)
        for pen, fill in zip(pens, (ink, 255)):
            pen.rectangle(
                (left, axis - rule / 2, left + bar, axis + rule / 2), fill=fill
            )
        middle = left + bar / 2
        self.put(
            pens, top, size, middle - self.width(top, size) / 2, axis - 0.42 * size
        )
        self.put(
            pens,
            bottom,
            size,
            middle - self.width(bottom, size) / 2,
            axis + 0.95 * size,
        )

        if reveal > 0:
            size = 54
            dots = [(".", "cmr10", DIM_COLOUR), (THIN, None, None)] * 3
            line = [
                (PI, "cmmi10", DIM_COLOUR),
                (THICK, None, None),
                ("=", "cmr10", DIM_COLOUR),
                (THICK, None, None),
                (f"{np.pi:.5f}", "cmr10", DIM_COLOUR),
                (THIN, None, None),
                *dots[:-1],
            ]
            left = (WIDTH - self.width(line, size)) / 2
            self.put(pens, line, size, left, 400, reveal)

        halo = np.asarray(halo.filter(ImageFilter.GaussianBlur(10)), np.float32) / 255
        backing = Image.new("L", (WIDTH, TEXT_HEIGHT), 0)
        bottom_edge = 330 + 100 * reveal
        ImageDraw.Draw(backing).rounded_rectangle(
            (110, 20, WIDTH - 110, bottom_edge),
            CARD_RADIUS,
            fill=int(255 * CARD_ALPHA * card),
        )
        backing = (
            np.asarray(backing.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
        )
        text = np.asarray(text, np.float32) / 255
        shade = np.maximum(backing, 0.75 * np.minimum(1.0, 2.5 * halo))
        alpha = text[..., 3:] + (1 - text[..., 3:]) * shade[..., None]
        rgb = text[..., :3] * text[..., 3:] + (1 - text[..., 3:]) * shade[
            ..., None
        ] * _rgb(BACKGROUND)
        return np.concatenate([rgb, alpha], -1).astype(np.float32)


def soundtrack(
    data: dict, crossed: np.ndarray, style_name: str, rng: np.random.Generator
):
    """
    A one-shot following the throws.

    - Single needles, at most one every NOTE_GAP seconds: a crossing strikes
      the style's voice, higher the steeper it lies (NOTE_SPAN degrees), a miss
      a dry wooden click; panned to where it lands, quieter as the rain grows.
    - The rain: soft high ticks whose rate follows the log of how many needles
      land per moment, up to SHIMMER per second.
    - The tuning: two soft tones, one on the root and one at root x estimate/π,
      gliding (GLIDE) to each new estimate. Early estimates put them far apart;
      as 2N/C converges they close to a unison, the beating slowing down.
    - As the rain stops (LAST), the home chord: a low root and a rolled chord.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    land = data["land"]
    pan = np.clip(data["x"] * view_scale(land) / (WIDTH / 2), -1, 1)
    steep = np.sin(data["angle"])

    # Single needles.
    notes, i = 0, 0
    while i < NEEDLES and land[i] <= LAST:
        when = land[i]
        quiet = 1 - 0.6 * float(smoothstep(when / LAST))
        if data["crosses"][i]:
            degree = int(round(steep[i] * NOTE_SPAN))
            sound = style.voice(scale.hz(degree), 0.45 * quiet)
        else:
            sound = audio.tick(rng, 0.35 * quiet, 0.02, 2500.0)
        audio.add(track, audio.pan(sound, 0.8 * pan[i]), when, **kw)
        notes += 1
        i = max(i + 1, int(np.searchsorted(land, when + NOTE_GAP)))

    # The rain.
    step = 0.05
    bins = np.arange(0.0, DURATION + step, step)
    counts = np.histogram(land, bins)[0]
    rate = SHIMMER * np.log1p(counts) / np.log1p(counts.max())
    ticks = 0
    for b in np.flatnonzero(counts > 8):
        for when in bins[b] + step * rng.uniform(size=rng.poisson(rate[b] * step)):
            tick = audio.tick(rng, rng.uniform(0.03, 0.07), 0.015, 7000.0)
            audio.add(track, audio.pan(tick, rng.uniform(-0.7, 0.7)), when, **kw)
            ticks += 1

    # The tuning: root, and root x estimate / π.
    sr = audio.SAMPLE_RATE
    seconds = np.arange(len(track)) / sr
    thrown = np.searchsorted(land, seconds, side="right")
    estimate = 2 * thrown / np.maximum(crossed[np.maximum(thrown - 1, 0)], 1)
    ratio = np.log(np.clip(np.where(thrown > 0, estimate / np.pi, 1.0), 0.5, 2.0))
    a = np.exp(-1 / (GLIDE * sr))
    ratio = lfilter([1 - a], [1, -a], ratio)
    root = scale.hz(0, -1)
    swell = smoothstep((seconds - FIRST) / 2.0) * (
        1 - smoothstep((seconds - 13.5) / 1.4)
    )
    for frequency, side in (
        (np.full_like(seconds, root), -0.3),
        (root * np.exp(ratio), 0.3),
    ):
        phase = 2 * np.pi * np.cumsum(frequency) / sr
        tone = sum(np.sin(k * phase) / k**2 for k in (1, 2, 3, 4))
        audio.add(track, audio.pan(DRONE * swell * tone, side), 0.0, **kw)

    # Home, as the rain stops.
    bass = scale.hz(0, -2)
    bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
    audio.add(track, style.bass(bass, 0.55, length=4.0), LAST, **kw)
    for i, degree in enumerate((0, 2, 4, 7)):
        note = style.voice(scale.hz(degree), 0.4)
        audio.add(track, audio.pan(note, 0.3 * (i - 1.5)), LAST + 0.08 * i, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} needle notes, "
        f"{ticks} rain ticks"
    )
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.5)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Buffon's needle, thrown NEEDLES times, with the running estimate of π.

    Lines y = k (k integer) rule the floor; needles have length 1, the line
    spacing. A needle with centre height y and angle a covers heights
    y +- sin(a) / 2 and crosses a line when those fall in different unit rows;
    for uniform y and a that happens with probability 2/π, so 2N/C estimates
    π, with relative error about sqrt((1 - p) / (p N)) for p = 2/π. All throws
    are drawn up front from settings.rng (`throws`): needle i lands at
    FIRST + ln(1 + i)/k, a rain growing exponentially until LAST, uniformly
    inside the view at that moment. Needle 0 is staged across the centre line
    so the first second shows a crossing; it shifts the final count by one.
    The posted render uses --seed 33, a typical run (it ends at 3.14431, an
    error of +0.0027; over seeds 0-39 the median error is 0.0037).

    Every frame is a closed form in t. The view is centred on a line; it pulls
    back exponentially in scale (eased) from SCALE_START to SCALE_END pixels
    per spacing by LAST, so every needle lands on screen and the last view is
    evenly covered, then pushes in by PUSH to the end. Landed needles are
    splatted on the GPU as runs of about one point per screen pixel, in
    parallel rows for their width (NEEDLE_WIDTH, at least a pixel), steel for
    misses (MISS_LIGHT dimmer) and gold for crossings, a crossing flaring as it lands. Needles in
    the air (for FALL_FIRST down to FALL_LAST seconds) are drawn at height
    h = time left / fall time, enlarged by 1 + RISE h^2 and turned by SPIN h^2.
    Exposure drops as the needles cover the floor (EXPOSURE / (1 + coverage /
    COVER)), so the final weave shows gold bands along the lines rather than
    white; light is tone-mapped (1 - exp(-x)) and bloomed. The ruled lines
    are drawn over everything, a Gaussian of the distance to the nearest
    integer height. The counter (`Counter`, PIL text in Computer Modern with a
    soft dark halo, on a card that fades in as the floor fills) shows 2N/C over the landed needles, the digits that agree
    with π in gold, and the true value fades in at REVEAL.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    data = throws(settings.rng)
    land = data["land"]
    crossed_total = np.cumsum(data["crosses"])
    final = 2 * NEEDLES / crossed_total[-1]
    logger.info(
        f"{NEEDLES:,} needles, {crossed_total[-1]:,} crossings: "
        f"pi ~ {final:.5f} (error {final - np.pi:+.5f})"
    )

    to = dict(device=device, dtype=torch.float32)
    x, y = torch.tensor(data["x"], **to), torch.tensor(data["y"], **to)
    angle = torch.tensor(data["angle"], **to)
    cos, sin = torch.cos(angle), torch.sin(angle)
    crosses = torch.tensor(data["crosses"], device=device)
    land_t, fall_t = torch.tensor(land, **to), torch.tensor(data["fall"], **to)
    miss_rgb = torch.tensor(_rgb(MISS_COLOUR), **to)
    cross_rgb = torch.tensor(_rgb(CROSS_COLOUR), **to)
    colours = torch.where(crosses[:, None], cross_rgb, MISS_LIGHT * miss_rgb)
    rows = torch.arange(HEIGHT, **to) + 0.5
    line_rgb = torch.tensor(_rgb(LINE_COLOUR), **to)
    background = torch.tensor(_rgb(BACKGROUND), **to)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **to)
    counter = Counter()

    def needles(scale, cx, cy, c, s, size, weight, rgb):
        """
        Splat needles (centres cx, cy, direction c, s, length size) as runs of
        about one point per screen pixel, in parallel rows for their width,
        CHUNK points at a time.
        """
        count = max(4, int(np.ceil(scale * float(size.max()))))
        along = (torch.arange(count, **to) + 0.5) / count - 0.5
        rows_across = round(max(1.0, NEEDLE_WIDTH * scale))
        across = (torch.arange(rows_across, **to) - (rows_across - 1) / 2) / scale
        light = (weight * size * scale / count)[:, None] * rgb
        step = max(1, CHUNK // (count * rows_across))
        for i in range(0, len(cx), step):
            part = slice(i, i + step)
            offset = along[None, :] * size[part, None]
            for a in across:
                px = cx[part, None] + offset * c[part, None] - a * s[part, None]
                py = cy[part, None] + offset * s[part, None] + a * c[part, None]
                splat(
                    canvas,
                    WIDTH / 2 + scale * px,
                    HEIGHT / 2 - scale * py,
                    light[part].repeat_interleave(count, 0),
                )

    def render(frame: int) -> torch.Tensor:
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        scale = float(view_scale(seconds))
        landed = int(np.searchsorted(land, seconds, side="right"))
        airborne = int(np.searchsorted(land, seconds + FALL_FIRST, side="right"))
        coverage = landed * scale / (WIDTH * HEIGHT)
        exposure = EXPOSURE / (1 + coverage / COVER)
        canvas.zero_()

        if landed:
            on = slice(0, landed)
            age = seconds - land_t[on]
            flare = 1 + FLASH_GAIN * torch.exp(-age / FLASH_TIME) * torch.where(
                crosses[on], 1.0, 0.3
            )
            size = torch.ones(landed, **to)
            needles(
                scale,
                x[on],
                y[on],
                cos[on],
                sin[on],
                size,
                NEEDLE_LIGHT * flare,
                colours[on],
            )

        if airborne > landed:
            on = slice(landed, airborne)
            h = (land_t[on] - seconds) / fall_t[on]
            up = h < 1
            if bool(up.any()):
                h = h[up]
                turn = angle[on][up] + SPIN * h * h
                needles(
                    scale,
                    x[on][up],
                    y[on][up],
                    torch.cos(turn),
                    torch.sin(turn),
                    1 + RISE * h * h,
                    NEEDLE_LIGHT * (0.25 + 0.75 * (1 - h)),
                    miss_rgb.expand(int(up.sum()), 3),
                )

        light = 1 - torch.exp(-exposure * canvas)
        image = (
            background
            + light
            + BLOOM_GAIN * wide_blur(light, BLOOM_SIGMA, BLOOM_FACTOR)
        )
        height = (HEIGHT / 2 - rows) / scale
        distance = (
            height - torch.round(height)
        ).abs() * scale  # px to the nearest line
        width = max(0.8, LINE_WIDTH * scale)
        alpha = (LINE_ALPHA * torch.exp(-((distance / width) ** 2)))[:, None, None]
        image = (image.clamp(0, 1) * (1 - alpha) + alpha * line_rgb).clamp(0, 1)

        if landed:
            reveal = float(smoothstep((seconds - REVEAL[0]) / (REVEAL[1] - REVEAL[0])))
            card = float(smoothstep(coverage / CARD_COVER))
            text = counter.draw(landed, int(crossed_total[landed - 1]), reveal, card)
            text = torch.tensor(text, **to)
            band = image[TEXT_TOP : TEXT_TOP + TEXT_HEIGHT]
            band.mul_(1 - text[..., 3:]).add_(text[..., :3])
        return image * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(data, crossed_total, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
