"""
Chaos From a Single Cell

One coloured square, and one rule. This is Rule 30, a cellular automaton: each row of
cells is drawn from the row above it, and every new cell looks only at the three cells
just above it and follows a table of eight cases (shown at the start). That's all.
Repeat it row after row while the camera pulls back over 650 generations.

Nothing here is random, yet one side settles into neat diagonal stripes while the rest
boils into chaos that never repeats. The middle column, the teal lane, is so
unpredictable that Stephen Wolfram's Mathematica once used it to make random numbers,
and the cone snail Conus textile paints its shell with a rule much like this one.
The four colours are the four cases that switch a cell on.

Listen: an 8-bit tune read off the middle column. Each note's pitch comes from the last
three bits it produced, and in the final bars the lead plays the column itself: a note
for every 1, a rest for every 0.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#rule30 #cellularautomata #wolfram #mathart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import PROGRESSIONS, STYLES, Scale, beat
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

RULE = 30  # bit k of RULE is the new cell for the neighbourhood code k = 4l + 2c + r

# --- Look: printed on paper, inked like a cone shell ------------------------------
PAPER = "#f4f0e7"
CASE_INK = (
    {  # the four neighbourhoods that switch a cell on (Rule 30: 001, 010, 011, 100)
        1: "#24140d",  # 001: dark chocolate
        2: "#6e2410",  # 010: sienna
        3: "#a8460f",  # 011: copper
        4: "#c98418",  # 100: ochre
    }
)
LANE_ON, LANE_OFF = "#0e6a73", "#d7e4df"  # the middle column: on / off
PARENT = "#4a3a30"  # the legend's "above" cells
GAP = 0.09  # paper seam between cells, as a fraction of a cell (big cells only)
GRAIN = 0.02  # paper grain, relative

# --- Camera and growth --------------------------------------------------------------
BPM = 128.0  # 8 bars in 15 s; the zoom lands on the downbeat of bar 8
APEX_Y = 150.0  # px: the first cell's top edge
FRONT_Y = 0.92 * HEIGHT  # px: where the newest row settles
CELL_START, CELL_END = 72.0, 2.5  # px per cell
HOLD = 0.6  # s at the start before the camera starts pulling back
FRONT_TIME = 4.0  # s for the newest row to travel down to FRONT_Y
ZOOM_END = 7 * 4 * 60.0 / BPM  # 13.125 s: the zoom stops, the picture rests
SPOT = (11.6, 12.6)  # s: everything but the middle column dims
SPOT_DIM = 0.2
LANE_HALF = (
    2.5,
    6.0,
)  # px: half-width of the middle-column lane, before / in the spotlight
LEGEND_FADE = (2.0, 2.6)  # s
FADE_OUT = 0.4  # s
LEVELS = (128, 91, 64, 45, 32, 23, 16, 11, 8, 6, 4, 3, 2, 1)  # px per cell, textures

# --- Sound -------------------------------------------------------------------------
STYLE = "chiptune"  # common.music.STYLES; an --audio-only tag naming one overrides it
MODE, CHORDS, DUTY = "dorian", "drift", 0.125  # Game Boy thin pulse
MELODY = (0, 2, 4, 6, 7, 9, 11, 14)  # degrees above the bar's chord root, by 3 bits


def cell_size(seconds):
    """px per cell: held, then a smooth log-zoom from CELL_START to CELL_END."""
    u = np.clip((np.asarray(seconds) - HOLD) / (ZOOM_END - HOLD), 0.0, 1.0)
    eased = u * u * (3 - 2 * u)
    return np.exp(np.log(CELL_START) + (np.log(CELL_END) - np.log(CELL_START)) * eased)


def generations(seconds):
    """Rows grown by `seconds` (continuous, >= 1): the front's depth over the cell size."""
    seconds = np.minimum(np.asarray(seconds, float), ZOOM_END)
    u = np.clip(seconds / FRONT_TIME, 0.0, 1.0)
    front = CELL_START + (FRONT_Y - APEX_Y - CELL_START) * (1 - (1 - u) ** 2)
    return front / cell_size(seconds)


def automaton(rows, half):
    """
    Rule 30 from one live cell: (rows, 2 half + 1) uint8 holding, for each live cell,
    the neighbourhood code (1..7) that switched it on, 0 where dead. The seed gets 2.
    """
    codes = np.zeros((rows, 2 * half + 1), np.uint8)
    live = np.zeros(2 * half + 3, np.uint8)  # one dead cell of padding on each side
    live[half + 1] = 1
    codes[0, half] = 2
    for g in range(1, rows):
        code = 4 * live[:-2] + 2 * live[1:-1] + live[2:]
        on = (RULE >> code) & 1
        codes[g] = code * on
        live[1:-1] = on
    return codes


def bgr(colour):
    return np.array([int(colour[i : i + 2], 16) for i in (5, 3, 1)], np.uint8)


def build_levels(codes, half):
    """
    One texture per entry of LEVELS: the cells needed while that level is in use,
    drawn p x p pixels each, coloured by case. A frame uses
    the smallest p >= its cell size, so it is never shrunk by more than ~1.5x.
    """
    lut = np.zeros((9, 3), np.uint8)
    lut[0] = bgr(PAPER)
    for code, colour in CASE_INK.items():
        lut[code] = bgr(colour)
    lut[7], lut[8] = bgr(LANE_ON), bgr(LANE_OFF)
    span = FRONT_Y - APEX_Y
    levels = {}
    for i, p in enumerate(LEVELS):
        smaller = LEVELS[i + 1] if i + 1 < len(LEVELS) else CELL_END
        rows = min(codes.shape[0], int(np.ceil(span / smaller)) + 2)
        cols = min(rows, int(np.ceil(WIDTH / 2 / smaller)) + 2)
        index = codes[:rows, half - cols : half + cols + 1].copy()
        lane = index[:, cols]
        index[:, cols] = np.where(lane > 0, 7, 8)
        texture = np.repeat(np.repeat(lut[index], p, axis=0), p, axis=1)
        levels[p] = (texture, cols)
    return levels


def build_legend():
    """The rule table, Wolfram style, with its binary digits: BGR image + alpha."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = bgr(PAPER)
    alpha = np.zeros((HEIGHT, WIDTH), np.float32)
    cell, gap = 33, 20
    left = (WIDTH - (8 * 3 * cell + 7 * gap)) // 2
    top = int(0.845 * HEIGHT)
    ink = tuple(int(v) for v in bgr(PARENT))
    edge = tuple(int(v) for v in (0.6 * bgr(PARENT) + 0.4 * bgr(PAPER)))
    font = cv2.FONT_HERSHEY_DUPLEX
    for i, code in enumerate(range(7, -1, -1)):
        x0 = left + i * (3 * cell + gap)
        for j in range(3):
            on = (code >> (2 - j)) & 1
            box = (x0 + j * cell + 2, top + 2), (
                x0 + (j + 1) * cell - 2,
                top + cell - 2,
            )
            cv2.rectangle(image, *box, ink if on else edge, -1 if on else 2)
        out = (RULE >> code) & 1
        box = (x0 + cell + 2, top + cell + 8), (x0 + 2 * cell - 2, top + 2 * cell + 4)
        colour = tuple(int(v) for v in bgr(CASE_INK[code])) if out else edge
        cv2.rectangle(image, *box, colour, -1 if out else 2)
        digit = str(out)
        (w, _), _ = cv2.getTextSize(digit, font, 0.9, 2)
        cv2.putText(
            image,
            digit,
            (x0 + (3 * cell - w) // 2, top + 2 * cell + 48),
            font,
            0.9,
            ink,
            2,
            cv2.LINE_AA,
        )
    title = "RULE 30"
    (w, _), _ = cv2.getTextSize(title, font, 1.4, 2)
    cv2.putText(
        image, title, ((WIDTH - w) // 2, top - 34), font, 1.4, ink, 2, cv2.LINE_AA
    )
    diff = np.abs(image.astype(np.int16) - bgr(PAPER).astype(np.int16)).max(axis=2)
    alpha[:] = (diff > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    alpha = np.clip(alpha * 1.5, 0, 1)
    y0, y1 = top - 90, top + 2 * cell + 70
    return image[y0:y1].astype(np.float32), alpha[y0:y1, :, None], (y0, y1)


def soundtrack(centre, style_name, rng):
    """
    A one-shot chiptune at BPM (8 bars), read off the middle column `centre` (its bits
    by generation). At each step time tau, g = the newest row then (generations(tau));
    the three newest middle bits c[g], c[g-1], c[g-2] form a number 0..7 that picks
    the lead's degree above the bar's chord (MELODY), louder when c[g] = 1. Bars 1-4
    in 8ths, 5-6 in 16ths as the growth races. Bar 7 (the spotlight on the column)
    plays the column itself in 16ths: a note for each 1, a rest for each 0, from the
    rows reached then. Triangle bass from bar 2, noise hi-hat whose level follows
    the birth rate (log rows per second), kick and noise snare from bar 3. Bar 8, when
    the zoom stops, is the home chord held with the bass, ringing to the end.
    """
    style = STYLES[style_name]
    chip = style_name == "chiptune"
    scale = Scale(int(rng.choice(style.roots)), MODE if chip else style.mode)
    chords = PROGRESSIONS[CHORDS]
    kw = dict(circular=False)
    bar = 4 * beat(BPM)
    track = audio.silence(DURATION)
    lead = audio.silence(DURATION)

    def lead_note(midi, velocity, length):
        freq = float(audio.midi_to_hz(midi))
        if chip:
            return audio.chip(freq, velocity, length, DUTY)
        return style.voice(freq, velocity)

    def bass_note(freq, velocity, length):
        if chip:
            return audio.chip_bass(freq, velocity, length)
        return style.bass(freq, velocity)

    def bit(g):
        return int(centre[g]) if 0 <= g < len(centre) else 0

    rate = np.gradient(generations(np.linspace(0, ZOOM_END, 400)), ZOOM_END / 399)
    pour = np.log1p(np.maximum(rate, 0))
    pour = (pour - pour.min()) / max(np.ptp(pour), 1e-9)

    notes = 0
    for b in range(7):
        sub = 2 if b < 4 else 4
        step = beat(BPM) / sub
        root = chords[b % len(chords)]
        for k in range(4 * sub):
            when = b * bar + k * step
            g = int(generations(when))
            if b < 6:
                value = 4 * bit(g) + 2 * bit(g - 1) + bit(g - 2)
                degree = root + MELODY[value]
                velocity = 0.26 + 0.14 * bit(g)
                play = True
            else:  # the column itself, one bit per 16th, newest rows first
                value = bit(g - k)
                degree = root + MELODY[(k * 3) % 8]
                velocity = 0.36
                play = value == 1
            if play:
                midi = scale.note(degree) + (12 if b >= 4 else 0)
                note = lead_note(midi, velocity, 0.8 * step)
                audio.add(lead, audio.pan(note, 0.2 * (-1) ** k), when, **kw)
                notes += 1
            level = float(np.interp(when, np.linspace(0, ZOOM_END, 400), pour))
            hat = audio.chip_noise(0.06 + 0.22 * level, 0.025, 9000.0)
            audio.add(track, audio.pan(hat, 0.3), when, **kw)
        for q in range(8 if b >= 1 else 0):  # 8th-note bass, root / octave
            when = b * bar + q * beat(BPM) / 2
            octave = -2 if q % 2 == 0 else -1
            bass = bass_note(scale.hz(root, octave), 0.55, 0.4 * beat(BPM))
            audio.add(track, bass, when, **kw)
        for q in range(4 if b >= 2 else 0):
            when = b * bar + q * beat(BPM)
            if q % 2 == 0:
                audio.add(track, audio.kick(0.45, 72.0), when, **kw)
            else:
                audio.add(track, audio.chip_noise(0.45, 0.09, 2200.0), when, **kw)
    lead = audio.delay(lead, 0.75 * beat(BPM), 0.25, 0.2, circular=False)
    track += lead

    # Bar 8: the zoom stops and the home chord rings.
    land = 7 * bar
    ring = DURATION - land - 0.15
    for degree, side in ((0, 0.0), (2, -0.3), (4, 0.3), (7, 0.0)):
        note = lead_note(scale.note(degree, 1), 0.3, ring)
        audio.add(track, audio.pan(note, side), land, **kw)
    audio.add(track, bass_note(scale.hz(0, -2), 0.65, ring), land, **kw)
    audio.add(track, audio.kick(0.5, 60.0), land, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} lead notes, "
        f"home chord at {land:.3f} s"
    )
    track = audio.lowpass(
        style.room(track, rng, circular=False), 9000.0, circular=False
    )
    audio.fade(track, 0.0, FADE_OUT)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Wolfram's elementary cellular automaton Rule 30 grown from a single live cell.

    Row g + 1 is computed from row g alone: the new cell at x is bit k of 30 (binary
    00011110), where k = 4 l + 2 c + r reads the three cells above it. The whole
    history, about 650 generations, is precomputed once (a few ms of NumPy) and each live
    cell keeps the code k that switched it on (1, 2, 3 or 4): its colour (CASE_INK).
    The middle column, the sequence Wolfram used as a random-number generator, is the
    teal lane.

    Time drives a camera, not a simulation. The newest row's depth below the apex
    eases from one cell to FRONT_Y - APEX_Y px over FRONT_TIME s while the cell size
    holds at CELL_START for HOLD s, then log-zooms (smoothstep) to CELL_END by
    ZOOM_END (the downbeat of bar 8); the generation count G(t) is depth / cell size.
    Each frame is a pure function of t (one-shot, t = frame / (FRAME_COUNT - 1)):
    pick the mip texture with the smallest p px/cell >= the cell size (LEVELS,
    prebuilt), cv2.warpAffine it into place (bilinear), draw paper seams
    between big cells in screen space (so they fade smoothly), reveal rows
    up to G (the newest row fades in), and overlay the middle column as a 5 px lane (12 px in the spotlight)
    once cells shrink below 6 px (its bits averaged per pixel row). The rule
    table is shown at the bottom for the first 2.6 s; from SPOT everything but the
    lane dims; a static paper grain (settings.rng) is multiplied in; the last FADE_OUT
    s fade to paper. Rendered on the render_frames thread pool.

    Sound: `soundtrack`, from the same middle-column bits and G(t). The seed only sets
    the paper grain and (through settings.audio_rng) the key; the posted soundtrack
    uses --seed 0, the default.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    rows = int(np.ceil(generations(ZOOM_END))) + 3
    half = rows + 2
    codes = automaton(rows, half)
    centre = (codes[:, half] > 0).astype(np.uint8)
    logger.info(
        f"Rule {RULE}: {rows} generations, middle column {centre.mean():.3f} ones"
    )

    grain = rng.normal(0.0, 1.0, (HEIGHT, WIDTH)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.9)
    grain = 1.0 + GRAIN * grain / grain.std()
    grain = grain[..., None]

    if not settings.audio_only:
        levels = build_levels(codes, half)
        legend, legend_alpha, (ly0, ly1) = build_legend()
        logger.info(
            f"Textures: {sum(t.nbytes for t, _ in levels.values()) / 2**20:.0f} MB"
        )
    paper = bgr(PAPER).astype(np.float32)
    lane_on = bgr(LANE_ON).astype(np.float32)
    lane_off = bgr(LANE_OFF).astype(np.float32)
    ones = np.concatenate([[0.0], np.cumsum(centre, dtype=np.float64)])
    ys = np.arange(HEIGHT, dtype=np.float64)
    cx = WIDTH / 2

    def render(frame):
        seconds = frame / (FRAME_COUNT - 1) * DURATION
        s = float(cell_size(seconds))
        grown = float(generations(seconds))
        p = next((q for q in reversed(LEVELS) if q >= s), LEVELS[0])
        texture, cols = levels[p]
        k = s / p
        matrix = np.array(
            [
                [k, 0, cx - (cols + 0.5) * s + 0.5 * k - 0.5],
                [0, k, APEX_Y + 0.5 * k - 0.5],
            ]
        )
        image = cv2.warpAffine(
            texture,
            matrix,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(float(v) for v in paper),
        ).astype(np.float32)

        # Paper seams between cells while they are big (antialiased, fading out).
        strength = np.clip((GAP * s - 1.0) / 2.0, 0.0, 1.0)
        if strength > 0:
            f = ((np.arange(WIDTH) + 0.5 - cx) / s + 0.5) % 1.0
            inside_x = np.clip((np.minimum(f, 1 - f) - GAP / 2) * s + 0.5, 0, 1)
            f = ((ys + 0.5 - APEX_Y) / s) % 1.0
            inside_y = np.clip((np.minimum(f, 1 - f) - GAP / 2) * s + 0.5, 0, 1)
            keep = 1 - strength * (1 - np.outer(inside_y, inside_x).astype(np.float32))
            image = paper + (image - paper) * keep[..., None]

        # The spotlight: everything dims towards paper, then the lane is drawn on top.
        spot = np.clip((seconds - SPOT[0]) / (SPOT[1] - SPOT[0]), 0, 1)
        spot = spot * spot * (3 - 2 * spot)
        if spot > 0:
            image = paper + (image - paper) * (1 - SPOT_DIM * spot)

        # The middle column as a lane ~5 px wide, bits averaged per pixel row.
        lane = np.clip((6.0 - s) / 2.0, 0.0, 1.0)
        if lane > 0 or spot > 0:
            top = np.clip((ys - APEX_Y) / s, 0, grown)
            bottom = np.clip((ys + 1 - APEX_Y) / s, 0, grown)
            filled = np.interp(bottom, np.arange(len(ones)), ones) - np.interp(
                top, np.arange(len(ones)), ones
            )
            value = filled / np.maximum(bottom - top, 1e-9)
            colour = lane_off + (lane_on - lane_off) * value[:, None]
            weight = max(lane, spot) * (ys >= APEX_Y)[:, None]
            reach = LANE_HALF[0] + (LANE_HALF[1] - LANE_HALF[0]) * spot  # px
            for x in range(int(cx - reach) - 1, int(cx + reach) + 2):
                cover = np.clip(min(x + 1, cx + reach) - max(x, cx - reach), 0, 1)
                if cover > 0:
                    image[:, x] += weight * cover * (colour - image[:, x])

        # Reveal rows up to the front; the newest one fades in.
        r = (ys + 0.5 - APEX_Y) / s
        if s >= 2.0:
            alpha = np.clip(grown - np.floor(r), 0.0, 1.0)
        else:
            alpha = np.clip((grown - r) * s, 0.0, 1.0)
        image = paper + (image - paper) * alpha[:, None, None].astype(np.float32)

        show = 1.0 - np.clip(
            (seconds - LEGEND_FADE[0]) / (LEGEND_FADE[1] - LEGEND_FADE[0]), 0, 1
        )
        if show > 0:
            region = image[ly0:ly1]
            region += show * legend_alpha * (legend - region)

        image *= grain
        fade = np.clip((DURATION - seconds) / FADE_OUT, 0.0, 1.0)
        if fade < 1:
            image = paper + (image - paper) * fade
        return np.clip(image, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag
    track = soundtrack(centre, tag if tag in STYLES else STYLE, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
