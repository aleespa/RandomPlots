"""
A Square Wheel That Rolls Smoothly

A square wheel can roll as smoothly as a round one, if the road is right. Its
road is a row of upside-down catenaries, the curve a hanging chain makes:
y = −r·cosh(x/r). Watch the dashed lines: every axle glides perfectly level.

Every regular polygon has a road of its own. More sides mean smaller bumps,
until the circle's road is flat. Only the triangle fails: its corners would dig
into the next bump. Square-wheeled tricycles really ride on roads like this,
at New York's Museum of Mathematics.

Listen: each wheel plucks a harp whenever a corner settles into a dip, the
square lowest and slowest, over a slow chord that moves with the words. The
circle never plucks.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#squarewheel #catenary #mathart #generativeart #randomplots
"""

import gc
import math

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds, a one-shot
FRAME_COUNT = frame_count(DURATION)
SS = 2  # supersampling
SHIFT = 2  # cv2 fixed-point bits for sub-pixel drawing

BACKGROUND = "#f4f0e7"
INK = "#2a2723"
GROUND = "#e4dccb"
SIDES = (4, 5, 6, 8, 0)  # 0 = the circle
COLOURS = ("#c8553d", "#d99a2b", "#4f8a83", "#3d5a80", "#7b4b94")

APOTHEM = 104.0  # px: every wheel's inradius r, so all roads share y = -r cosh(x/r)
AXLE_X = 600.0  # px: the camera follows the wheels, the road scrolls
GROUND_DEPTH = 28.0  # px of ground under the deepest dip
LANE_GAP = 50.0
TEXT_BAND = 300  # px at the top kept for text
TRAIL = 560.0  # px of road behind each wheel over which its marked corner is traced

SQUARE_SIDES_ROLLED = 7.5  # the square rolls this many sides from frame 0 to rest
CRUISE_END, REST = 12.2, 13.8  # s: constant speed, then ease to rest
FADE = 0.35  # s fade to the background at the very end

PHASES = (  # (start, end, words) of the top line
    (0.0, 5.0, r"$\mathrm{A\ square\ wheel\ can\ roll\ smoothly}$"),
    (5.0, 10.0, r"$\mathrm{on\ a\ road\ of\ upturned\ hanging\ chains}$"),
    (10.0, 99.0, r"$\mathrm{Every\ polygon\ has\ its\ own\ road}$"),
)
FORMULA = r"$y = -r\,\cosh\frac{x}{r}$"

BEDS = ("strings", "choir")  # --audio-only tag picks one
BED = "strings"
ROOTS = range(55, 61)  # G3 .. C4, major key
CHORDS = ((0, 0.0), (5, 5.0), (3, 10.0), (0, REST))  # (degree, start s): I vi IV I


def lane_period(n: int) -> float:
    """Road period of an n-gon of inradius r: 2 r asinh(tan(pi/n)), 0 for the circle."""
    return 2.0 * APOTHEM * math.asinh(math.tan(math.pi / n)) if n else 0.0


def circumradius(n: int) -> float:
    return APOTHEM / math.cos(math.pi / n) if n else APOTHEM


def position(seconds):
    """Distance rolled (px) at time `seconds`: constant speed, then a smooth stop."""
    period = lane_period(4)
    speed = SQUARE_SIDES_ROLLED * period / (0.5 * (CRUISE_END + REST))
    end = 2 * period * 10  # rest with the square flat on top of a bump
    start = end - speed * 0.5 * (CRUISE_END + REST)
    s = np.asarray(seconds, float)
    u = np.clip((s - CRUISE_END) / (REST - CRUISE_END), 0.0, 1.0)
    eased = CRUISE_END + (REST - CRUISE_END) * (u - u**3 + 0.5 * u**4)
    return start + speed * np.where(s < CRUISE_END, s, eased)


def wheel_angle(x, n):
    """
    Rotation (radians, counter-clockwise, y up) of an n-gon rolled a distance x on its
    road: whole sides done plus the road's tangent angle at the contact point.
    """
    if not n:
        return -x / APOTHEM
    period = lane_period(n)
    k = np.round(x / period)
    u = x - k * period
    return -2.0 * np.pi * k / n - np.arctan(np.sinh(u / APOTHEM))


def road_depth(x, n):
    """How far below the axle the road lies at road position x (px, down positive)."""
    if not n:
        return np.full_like(x, APOTHEM)
    period = lane_period(n)
    u = x - np.round(x / period) * period
    return APOTHEM * np.cosh(u / APOTHEM)


def dip_times(n):
    """Seconds at which a corner of the n-gon settles into a dip of its road."""
    if not n:
        return np.array([])
    period = lane_period(n)
    grid = np.linspace(0.0, DURATION, 30001)
    x = position(grid)
    first = math.ceil(x[0] / period - 0.5)
    last = math.floor(x[-1] / period - 0.5)
    cusps = (np.arange(first, last + 1) + 0.5) * period
    return np.interp(cusps, x, grid)


def lanes_layout():
    """Axle heights of the lanes, stacked under the text band and centred."""
    heights = [2 * circumradius(n) + GROUND_DEPTH for n in SIDES]
    total = sum(heights) + LANE_GAP * (len(SIDES) - 1)
    top = TEXT_BAND + (HEIGHT - 40 - TEXT_BAND - total) / 2
    axles = []
    for n, h in zip(SIDES, heights):
        axles.append(top + circumradius(n))
        top += h + LANE_GAP
    return axles


def generate(settings: ImageProcessingSettings = None):
    """
    Roads and wheels (Hall & Wagon, 1992), as a 15 s one-shot. A regular n-gon of
    inradius r rolls without slipping on a road of upside-down catenary arcs
    y = -r cosh(u/r), |u| <= r asinh(tan(pi/n)), repeated with period
    L_n = 2 r asinh(tan(pi/n)); the arc length of half an arc, r tan(pi/n), is half a
    side, and the axle stays exactly at height 0, directly above the contact point.
    The wheel's angle after rolling a distance x is -(2 pi k / n) - atan(sinh(u/r))
    with k = round(x / L_n), u = x - k L_n: the side lies along the road's tangent.

    Five lanes share r = 104 px and the same rolled distance x(t): square, pentagon,
    hexagon, octagon and the circle on a flat road. The camera follows the wheels
    (the road scrolls); x(t) is constant speed until 12.2 s, then a C1 ease to rest at
    13.8 s with the square flat on a bump, a 1.2 s rest and a 0.35 s fade. Each frame
    is closed form in t: dashed axle lines anchored to the road, and the trace of one
    marked corner, evaluated at the distances rolled over the last 560 px. Drawn with
    cv2 at 2x and downsampled, on a thread pool. The top line of LaTeX text changes at
    5 s and 10 s; the formula and the lane labels hold. The picture uses no randomness.

    Soundtrack (ambient, one-shot, from the same x(t)): a slow bed, I-vi-IV-I with
    the home chord landing as the wheels stop, of strings (posted, --seed 0) or a "o"
    choir with a pad (the "choir" tag). A harp pluck each time a corner of a polygon
    settles into a dip (x = (k + 1/2) L_n), a chord tone of the current chord, lower for
    fewer sides and quieter for shallower dips; the lanes join the music one by one
    with the words (square from the start, pentagon at 5 s, hexagon and octagon at
    10 s), a rolled harp chord on the rest. The circle never plucks.
    """
    settings = settings or ImageProcessingSettings(1)
    background = np.array(hex_to_bgr(BACKGROUND), np.float32)
    ink, ground = hex_to_bgr(INK), hex_to_bgr(GROUND)
    axles = lanes_layout()

    # Text, typeset once.
    words = [(a, b, latex_mask(tex, 50)) for a, b, tex in PHASES]
    formula = latex_mask(FORMULA, 58)
    labels = [
        latex_mask(r"$n = %d$" % n if n else r"$n \to \infty$", 40) for n in SIDES
    ]

    sx = np.arange(0, WIDTH + 2, 2, dtype=np.float64)  # road samples, screen px
    scale = SS * (1 << SHIFT)

    def pts(x, y):
        return np.round(np.stack([x, y], -1) * scale).astype(np.int32)

    def blend(colour, a):
        """colour faded towards the background by 1 - a."""
        return tuple(float(c * a + b * (1 - a)) for c, b in zip(colour, background))

    def draw_lane(image, n, axle, colour, x):
        aa, sh = cv2.LINE_AA, SHIFT
        radius = circumradius(n)
        # Road: ground below the arcs, ink along them.
        road_y = axle + road_depth(x + sx - AXLE_X, n)
        bottom = axle + circumradius(n) + GROUND_DEPTH
        outline = pts(sx, road_y)
        poly = np.vstack([outline, pts(np.array([sx[-1], 0.0]), np.full(2, bottom))])
        cv2.fillPoly(image, [poly], ground, aa, sh)
        cv2.polylines(image, [outline], False, ink, 2 * SS, aa, sh)
        # Axle line: dashes anchored to the road, so they scroll with it.
        dash = 26.0
        first = math.floor((x - AXLE_X) / dash)
        for m in range(first, first + int(AXLE_X / dash) + 2):
            if m % 2:
                continue
            a = max(m * dash - x + AXLE_X, 0.0)
            b = min((m + 1) * dash - x + AXLE_X, AXLE_X)
            if b > a:
                cv2.line(
                    image,
                    tuple(pts(np.array(a), np.array(axle))),
                    tuple(pts(np.array(b), np.array(axle))),
                    blend(colour, 0.85),
                    2 * SS,
                    aa,
                    sh,
                )
        # Trace of the marked corner over the last TRAIL px rolled.
        offset = -np.pi / 2 + (np.pi / n if n else 0.0)
        back = np.linspace(x - TRAIL, x, 240)
        angle = wheel_angle(back, n) + offset
        trace = pts(
            AXLE_X + back - x + radius * np.cos(angle), axle - radius * np.sin(angle)
        )
        chunks = 12
        for c in range(chunks):
            part = trace[c * 20 : (c + 1) * 20 + 1]
            cv2.polylines(
                image, [part], False, blend(colour, (c + 1) / chunks), 2 * SS, aa, sh
            )
        # The wheel.
        angle = float(wheel_angle(np.array(x), n)) + offset
        centre = pts(np.array(AXLE_X), np.array(axle))
        if n:
            a = angle + 2 * np.pi * np.arange(n) / n
            corners = pts(AXLE_X + radius * np.cos(a), axle - radius * np.sin(a))
            cv2.fillPoly(image, [corners], blend(colour, 0.8), aa, sh)
            for corner in corners:
                cv2.line(image, tuple(centre), tuple(corner), colour, SS, aa, sh)
            cv2.polylines(image, [corners], True, colour, 3 * SS, aa, sh)
        else:
            cv2.circle(
                image,
                tuple(centre),
                int(radius * scale),
                blend(colour, 0.8),
                -1,
                aa,
                sh,
            )
            for j in range(6):
                a = angle + np.pi * j / 3
                end = pts(AXLE_X + radius * np.cos(a), axle - radius * np.sin(a))
                cv2.line(image, tuple(centre), tuple(end), colour, SS, aa, sh)
            cv2.circle(
                image, tuple(centre), int(radius * scale), colour, 3 * SS, aa, sh
            )
        mark = pts(AXLE_X + radius * np.cos(angle), axle - radius * np.sin(angle))
        cv2.circle(image, tuple(mark), 9 * scale, ink, -1, aa, sh)
        cv2.circle(image, tuple(centre), 7 * scale, ink, -1, aa, sh)

    canvas = np.empty((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
    canvas[:] = hex_to_bgr(BACKGROUND)
    ink_f = np.array(ink, np.float32)

    def render(frame):
        seconds = frame / (FRAME_COUNT - 1) * DURATION
        x = float(position(seconds))
        big = canvas.copy()
        for n, axle, colour in zip(SIDES, axles, COLOURS):
            draw_lane(big, n, axle, hex_to_bgr(colour), x)
        image = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        for a, b, mask in words:
            alpha = float(
                smoothstep((seconds - a) / 0.4) * smoothstep((b - seconds) / 0.4)
            )
            if a == 0.0:
                alpha = float(smoothstep((b - seconds) / 0.4))
            if alpha > 0:
                blit_mask(image, mask, WIDTH // 2, 92, ink_f, alpha, "centre")
        blit_mask(image, formula, WIDTH // 2, 178, ink_f, 1.0, "centre")
        for label, axle, colour in zip(labels, axles, COLOURS):
            y = int(axle - label.shape[0] / 2)
            blit_mask(image, label, 945, y, hex_to_bgr(colour), 1.0, "centre")
        fade = float(smoothstep((DURATION - seconds) / FADE))
        if fade < 1:
            image = fade * image + (1 - fade) * background
        return image

    render_frames(settings, render, FRAME_COUNT)
    bed = settings.audio_tag if settings.audio_tag in BEDS else BED
    track = soundtrack(bed, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def soundtrack(bed, rng):
    """
    Ambient one-shot (nothing wraps). The bed holds I, vi, IV, then the home chord I
    from the moment the wheels stop, with a low root under each; strings, or a soft
    "o" choir over a pad. On top, the harp: one pluck per corner settling into a dip,
    a tone of the chord sounding at that moment, the square lowest. Lanes join with
    the words; a rolled harp chord as the wheels come to rest.
    """
    root = int(rng.choice(ROOTS))
    scale = Scale(root, "major")
    sr = audio.SAMPLE_RATE
    track = audio.silence(DURATION)

    def chord_at(seconds):
        return [d for d, start in CHORDS if start <= seconds][-1]

    # The bed.
    bounds = [start for _, start in CHORDS] + [DURATION]
    for (degree, start), end in zip(CHORDS, bounds[1:]):
        length = end - start + (0.9 if end < DURATION else 0.0)
        tones = scale.chord(degree, 3)
        for i, midi in enumerate(tones):
            hz = float(audio.midi_to_hz(midi))
            if bed == "strings":
                held = audio.strings(hz, length, 0.30, attack=1.2, release=1.8)
            else:
                held = audio.choir(hz, length, 0.26, attack=1.4, release=2.0)
                held = held + audio.pad(hz, length, 0.16, attack=1.4, release=2.0)
            audio.add(track, held, start - (0.0 if start == 0 else 0.6), False)
        low = float(audio.midi_to_hz(tones[0] - 12))
        audio.add(
            track,
            audio.strings(low, length, 0.18, attack=1.5, release=2.0),
            max(start - 0.6, 0.0),
            False,
        )
    track = audio.highpass(track, 85.0, circular=False)
    track = audio.lowpass(track, 3200.0, circular=False)

    # The harp.
    joins = {4: 0.0, 5: 5.0, 6: 10.0, 8: 10.0}
    lane = {4: 0, 5: 1, 6: 2, 8: 3}
    pans = {4: -0.35, 5: 0.35, 6: -0.15, 8: 0.2}
    plucks = 0
    for n in SIDES[:-1]:
        depth = circumradius(n) - APOTHEM  # how deep its dips are
        velocity = 0.18 + 0.9 * depth / (circumradius(4) - APOTHEM) * 0.32
        for when in dip_times(n):
            if when < joins[n] or when > REST + 0.05:
                continue
            tones = scale.chord(chord_at(when), 4, octave=1)
            midi = tones[lane[n]] - (12 if n == 4 else 0)
            note = audio.pluck(float(audio.midi_to_hz(midi)), round(velocity, 3), 3.0)
            audio.add(track, audio.pan(note, pans[n]), when, False)
            plucks += 1
    for j, midi in enumerate(scale.chord(0, 5, octave=0)):
        note = audio.pluck(float(audio.midi_to_hz(midi + 12)), 0.42 - 0.04 * j, 4.0)
        audio.add(track, audio.pan(note, -0.4 + 0.2 * j), REST + 0.09 * j, False)

    track = audio.delay(track, 0.48, feedback=0.3, wet=0.16, circular=False)
    track = audio.reverb(track, rng, rt60=4.5, wet=0.4, tone=4200.0, circular=False)
    logger.info(f"Soundtrack ({bed}): root {root}, {plucks} harp plucks, {sr} Hz")
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
