"""
Langton's Ant

Langton's ant is a cellular automaton with one rule: on an empty cell turn right,
on a filled cell turn left, then flip the cell and step forward.

For a few hundred steps it draws neat symmetric patterns. Then chaos: a
shapeless blob grows for about 10,000 steps. Then, out of nowhere, the ant
builds a highway, the same 104 steps repeated forever, marching off to
infinity. Every starting pattern ever tried ends this way, yet nobody has
proved it always must.

Listen: a pad and a soft choir hold the bed while gamelan bells strike whenever
the ant gets farther from home than ever before; once the highway begins, the
bells fall into its steady beat.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#langtonsant #cellularautomata #emergence #mathart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from scipy.interpolate import PchipInterpolator

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'One ant. One rule. Wait for step 10,000.'
HOOK_Y = 440
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds, a one-shot
FRAME_COUNT = frame_count(DURATION)

# --- The ant -----------------------------------------------------------------------
G = 256  # grid side, cells; the ant starts in the middle facing down
DY, DX = (-1, 0, 1, 0), (0, 1, 0, -1)  # headings: up, right, down, left
START_HEADING = 2  # facing down, so the highway heads up and to the right
PERIOD = 104  # steps in one cycle of the highway
# Pacing: (time in s, steps taken). Symmetric patterns, chaos, the highway, rest.
KEYS = ((0.0, 0), (1.0, 150), (2.4, 600), (9.4, 9976), (13.8, 13400), (15.0, 13400))
STEPS = KEYS[-1][1]
SYMMETRIC_END, CHAOS_END = 2.4, 9.4  # s

# --- Layout (px) -------------------------------------------------------------------
TITLE_Y, RULE1_Y, RULE2_Y = 200, 306, 366
VIEW_Y0, VIEW_Y1 = 450, 1450  # the band the grid lives in; text stays outside it
VIEW_H = VIEW_Y1 - VIEW_Y0
EDGE = 36  # px of soft fade at the top and bottom of the band
COUNT_Y, LABEL_Y = 1484, 1556
MAX_CELL = 56.0  # px, the closest zoom
MARGIN = 4  # cells of room around everything the ant has visited
CAMERA_SIGMA = 0.35  # s, smoothing of the camera

# --- Look --------------------------------------------------------------------------
BG_TOP, BG_BOTTOM = "#081212", "#030707"
EMPTY = "#0c1919"  # an unvisited (or empty) cell, a faint lattice
FILLED = ["#e9ffd8", "#9ff0c0", "#3fbf9f", "#1f7f78", "#1b5a60"]  # fresh -> settled
SETTLE = 1.6  # s for a filled cell to settle into its old colour
EMBER = "#2e9c88"  # a cell flipped back to empty glows briefly
EMBER_TIME = 0.22  # s
ANT = "#ff7a55"
GLOW_SCALE, GLOW_SIGMA, GLOW = 4, 5.0, 0.9
TEXT, TEXT_DIM, ACCENT = "#f1eee6", "#a9bdb9", "#ff9a78"
FADE_OUT = 0.3  # s to the background at the very end

# --- Sound -------------------------------------------------------------------------
STYLE = "gamelan"  # bells in pelog over a pad bed
BEDS = ("choir", "strings")  # an --audio-only tag naming one picks the upper bed voice
BED = "choir"
MIN_GAP = 0.45  # s between two record bells
PULSE = 8  # highway cycles per bell once the highway runs
FIGURE = (0, 2, 4, 3, 1, 3)  # pelog degrees the highway bells cycle through


def walk():
    """The ant's cell before each step: positions (STEPS + 1, 2) as (row, col)."""
    grid = np.zeros((G, G), np.uint8)
    y = x = G // 2
    heading = START_HEADING
    pos = np.empty((STEPS + 1, 2), np.int64)
    pos[0] = y, x
    for k in range(STEPS):
        heading = (heading + (1 if grid[y, x] == 0 else -1)) % 4
        grid[y, x] ^= 1
        y, x = y + DY[heading], x + DX[heading]
        pos[k + 1] = y, x
    return pos


def highway_start(pos):
    """First step from which the walk repeats every PERIOD steps with a fixed shift."""
    shift = pos[PERIOD:] - pos[:-PERIOD]
    irregular = np.flatnonzero(np.any(shift != shift[-1], axis=1))
    return int(irregular[-1]) + 1


def pacing():
    """steps(t): a smooth monotone map from seconds to steps taken."""
    times, steps = zip(*KEYS)
    return PchipInterpolator(times, steps)


def counter_glyphs(size):
    """LaTeX masks for 'n =' and the digits and comma, top-aligned to the digits."""
    zero = latex_mask(r"$0$", size)
    w0 = zero.shape[1]
    lead = latex_mask(r"$n\ =\ 0$", size)
    prefix = lead[:, : lead.shape[1] - w0 + 2]
    digits = [latex_mask(f"${d}$", size) for d in range(10)]
    comma = latex_mask(r"$0{,}$", size)[:, w0 - 2 :]
    return prefix, digits, comma


def counter_mask(n, glyphs):
    """'n = 12,345' as one mask, digits on a fixed advance so nothing jitters."""
    prefix, digits, comma = glyphs
    advance = max(d.shape[1] for d in digits) - 3
    text = f"{n:,}"
    height = max(prefix.shape[0], comma.shape[0], max(d.shape[0] for d in digits))
    width = (
        prefix.shape[1]
        + 8
        + sum(comma.shape[1] + 2 if c == "," else advance for c in text)
    )
    mask = np.zeros((height, width + 4), np.float32)
    mask[: prefix.shape[0], : prefix.shape[1]] = prefix
    x = prefix.shape[1] + 8
    for c in text:
        glyph = comma if c == "," else digits[int(c)]
        h, w = glyph.shape
        off = 0 if c == "," else (advance - w) // 2
        x0 = max(x + off, 0)
        region = mask[:h, x0 : x0 + w]
        np.maximum(region, glyph[:, : region.shape[1]], out=region)
        x += comma.shape[1] + 2 if c == "," else advance
    return mask


def generate(settings: ImageProcessingSettings = None):
    """
    Langton's ant from a blank grid, 13,400 steps in 15 s. A one-shot.

    The rule (Langton 1986): on an empty cell turn right, on a filled cell turn left,
    flip the cell, step forward. Starting on an empty 256^2 grid facing down, the walk
    (precomputed once, 13,400 steps, a few ms) shows symmetric patterns for ~500
    steps, an irregular blob up to step 9,976, then the highway: a cycle of 104 steps
    that shifts the ant two cells diagonally (up-right here), forever. The highway's
    start is found from the walk (the last step where pos[k + 104] - pos[k] differs
    from its final value) and only logged; the pacing is fixed.

    Time: steps(t) is a monotone PCHIP through the keys (0 s, 0), (1 s, 150),
    (2.4 s, 600), (9.4 s, 9,976), (13.8 s, 13,400), then flat to 15 s (a ~1.2 s rest,
    fading to the background over the last 0.3 s). Each frame is a pure function of
    t: n = floor(steps(t)) flips, cell parity = bincount(flipped cells[:n]) mod 2,
    last visit = max step index per cell, its age in seconds from the inverse
    pacing. Filled cells run through a teal LUT from pale (fresh) to settled over
    1.6 s; cells flipped back to empty glow briefly. The camera (precomputed on a
    fine time grid, then Gaussian-smoothed over 0.35 s in log zoom) frames the
    bounding box of every visited cell plus 4 cells, zooming out monotonically from
    56 px per cell, inside the band y = 450..1450 so the LaTeX text above and below
    is never crossed. The ant is a coral disc, interpolated between cells.

    Render: per frame, the grid's colours are gathered to pixels with separable
    row/column cell indices and an analytic per-pixel coverage (a thin gap between
    cells, anti-aliased), plus a quarter-resolution blurred glow of the fresh cells
    and the ant, on the render_frames thread pool. Every seed gives the same picture
    (the rule is deterministic); the seed only reaches the sound.

    Sound (posted: --seed 0, no --audio-only tag, i.e. the "choir" bed, audio seed
    = seed): gamelan bells in pelog over an ambient bed of pad chords with a quiet
    "u" choir (the "strings" tag swaps in strings), one-shot. Open fifth during the
    symmetric phase, i - VI - bII - i under the chaos (filter dark), the home chord
    with an octave and minor third once the highway runs (filter opening). A bell
    strikes whenever the ant's distance from its start beats its record (at least
    0.45 s apart): pitch by the direction it reached, pan by x. A low gong marks the
    highway's start; then a bell every 8 highway cycles cycling a six-note pelog
    figure, a gong every other one. Long reverb and an echo, circular=False,
    master(loop=False).
    """
    settings = settings or ImageProcessingSettings(1)

    pos = walk()
    hw = highway_start(pos)
    logger.info(f"Langton's ant: {STEPS} steps, highway from step {hw}")
    steps_of = pacing()
    rest = KEYS[-2][0]
    t_fine = np.linspace(0.0, rest, 20001)
    s_fine = np.maximum.accumulate(steps_of(t_fine))
    # time_of[k]: when the k-th step is taken (count passes k).
    time_of = np.interp(np.arange(STEPS + 1), s_fine, t_fine)
    flat = pos[:, 0] * G + pos[:, 1]

    # --- Camera on a fine time grid ---------------------------------------------
    lo = np.minimum.accumulate(pos, axis=0)
    hi = np.maximum.accumulate(pos, axis=0)
    cam_t = np.linspace(0.0, DURATION, 1501)
    cam_n = np.minimum(np.floor(steps_of(cam_t)).astype(int), STEPS)
    span = hi[cam_n] - lo[cam_n] + 1 + 2 * MARGIN  # (rows, cols)
    cell = np.minimum(MAX_CELL, np.minimum(VIEW_H / span[:, 0], WIDTH / span[:, 1]))
    log_cell = np.minimum.accumulate(np.log(cell))
    centre = (lo[cam_n] + hi[cam_n] + 1) / 2.0  # cell coordinates (row, col)
    sigma = CAMERA_SIGMA / (cam_t[1] - cam_t[0])
    kernel = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1) / sigma) ** 2)
    kernel /= kernel.sum()

    def smooth(values):
        pad = len(kernel) // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    cam_cell = np.exp(smooth(log_cell))
    cam_row, cam_col = smooth(centre[:, 0]), smooth(centre[:, 1])

    # --- Static background and text ----------------------------------------------
    top, bottom = np.array(hex_to_bgr(BG_TOP)), np.array(hex_to_bgr(BG_BOTTOM))
    ramp = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    background = np.broadcast_to(
        (1 - ramp) * top + ramp * bottom, (HEIGHT, WIDTH, 3)
    ).astype(np.float32)
    for tex, size, y, colour in (
        (r"$\mathrm{Langton\text{'}s\ ant}$", 76, TITLE_Y, TEXT),
        (
            r"$\mathrm{empty\ cell}\ \rightarrow\ \mathrm{turn\ right}\qquad\mathrm{filled\ cell}\ \rightarrow\ \mathrm{turn\ left}$",
            44,
            RULE1_Y,
            TEXT_DIM,
        ),
        (
            r"$\mathrm{then\ flip\ the\ cell\ and\ step\ forward}$",
            44,
            RULE2_Y,
            TEXT_DIM,
        ),
    ):
        mask = latex_mask(tex, size)
        blit_mask(background, mask, WIDTH // 2, y, hex_to_bgr(colour), anchor="centre")
    band_bg = background[VIEW_Y0:VIEW_Y1].copy()
    labels = [
        (latex_mask(r"$\mathrm{simple{,}\ symmetric\ patterns}$", 44), 0.0, 2.6),
        (latex_mask(r"$\mathrm{then\ chaos}$", 44), 2.6, 9.9),
        (latex_mask(r"$\mathrm{then\ a\ highway{,}\ forever}$", 44), 9.9, 13.2),
        (latex_mask(r"$\mathrm{always?\ Nobody\ has\ proved\ it}$", 44), 13.2, 99.0),
    ]
    glyphs = counter_glyphs(46)
    count_x = WIDTH // 2 - counter_mask(STEPS, glyphs).shape[1] // 2

    # Edge fade of the band, and colours.
    rows = np.arange(VIEW_H, dtype=np.float32)
    band_fade = smoothstep(np.minimum(rows, VIEW_H - 1 - rows) / EDGE)[:, None, None]
    lut = palette_lut(FILLED, 256)[:, ::-1] * 255  # BGR
    empty = np.array(hex_to_bgr(EMPTY), np.float32)
    ember = np.array(hex_to_bgr(EMBER), np.float32)
    ant_c = np.array(hex_to_bgr(ANT), np.float32)
    text_c, accent_c = hex_to_bgr(TEXT), hex_to_bgr(ACCENT)
    cols_px = np.arange(WIDTH, dtype=np.float32) + 0.5
    rows_px = rows + 0.5
    small_w, small_h = WIDTH // GLOW_SCALE, VIEW_H // GLOW_SCALE

    def camera(t):
        return (
            float(np.interp(t, cam_t, cam_cell)),
            float(np.interp(t, cam_t, cam_row)),
            float(np.interp(t, cam_t, cam_col)),
        )

    def gather(values, cs, row0, col0, xs, ys):
        """values (G, G, 3) gathered to pixel centres xs, ys of the band."""
        u = col0 + (xs - WIDTH / 2) / cs
        v = row0 + (ys - VIEW_H / 2) / cs
        iu = np.clip(np.floor(u).astype(np.int64), 0, G - 1)
        iv = np.clip(np.floor(v).astype(np.int64), 0, G - 1)
        return values[iv[:, None], iu[None, :]], u - np.floor(u), v - np.floor(v)

    def render(frame):
        t = frame / (FRAME_COUNT - 1)
        now = t * DURATION
        s = float(steps_of(min(now, rest)))
        n = min(int(s), STEPS)
        visited = flat[:n]
        parity = np.bincount(visited, minlength=G * G) & 1
        last = np.full(G * G, -1, np.int64)
        np.maximum.at(last, visited, np.arange(n))
        seen = last >= 0
        age = np.where(seen, now - time_of[np.maximum(last, 0) + 1], 1e9)

        filled = (parity == 1)[:, None]
        heat = np.clip(age / SETTLE, 0.0, 1.0)
        filled_c = lut[(255 * heat).astype(np.int64)]
        flash = np.exp(-np.maximum(age, 0.0) / EMBER_TIME)[:, None]
        cells = np.where(filled, filled_c, empty + flash * (ember - empty))
        cells = cells.reshape(G, G, 3).astype(np.float32)
        fresh = np.where(filled, np.exp(-age / 0.5)[:, None] * filled_c, flash * ember)
        fresh = fresh.reshape(G, G, 3).astype(np.float32)

        cs, row0, col0 = camera(now)
        colours, fu, fv = gather(cells, cs, row0, col0, cols_px, rows_px)
        half_gap = 0.04 * cs + 0.35
        cov_u = np.clip((0.5 - np.abs(fu - 0.5)) * cs - half_gap + 0.5, 0.0, 1.0)
        cov_v = np.clip((0.5 - np.abs(fv - 0.5)) * cs - half_gap + 0.5, 0.0, 1.0)
        coverage = (cov_v[:, None] * cov_u[None, :])[..., None].astype(np.float32)
        band = band_bg + coverage * (colours - band_bg)

        # Glow of the fresh cells and the ant, at quarter resolution.
        sx = (np.arange(small_w, dtype=np.float32) + 0.5) * GLOW_SCALE
        sy = (np.arange(small_h, dtype=np.float32) + 0.5) * GLOW_SCALE
        light, _, _ = gather(fresh, cs, row0, col0, sx, sy)
        light = np.ascontiguousarray(light * 0.6)
        frac = s - n if n < STEPS else 0.0
        a, b = pos[n], pos[min(n + 1, STEPS)]
        ant = a + frac * (b - a) + 0.5
        ax = WIDTH / 2 + (ant[1] - col0) * cs
        ay = VIEW_H / 2 + (ant[0] - row0) * cs
        radius = max(7.0, 0.34 * cs)
        cv2.circle(
            light,
            (int(16 * ax / GLOW_SCALE), int(16 * ay / GLOW_SCALE)),
            int(16 * 2.2 * radius / GLOW_SCALE),
            tuple(float(c) for c in ant_c),
            -1,
            cv2.LINE_AA,
            4,
        )
        light = cv2.GaussianBlur(light, (0, 0), GLOW_SIGMA)
        band += GLOW * cv2.resize(
            light, (WIDTH, VIEW_H), interpolation=cv2.INTER_LINEAR
        )
        centre_px = (int(16 * ax), int(16 * ay))
        cv2.circle(
            band,
            centre_px,
            int(16 * radius),
            tuple(float(c) for c in ant_c),
            -1,
            cv2.LINE_AA,
            4,
        )
        cv2.circle(
            band,
            centre_px,
            int(16 * radius * 0.45),
            (235.0, 245.0, 255.0),
            -1,
            cv2.LINE_AA,
            4,
        )
        band = band_bg + band_fade * (band - band_bg)

        image = background.copy()
        image[VIEW_Y0:VIEW_Y1] = band
        blit_mask(image, counter_mask(n, glyphs), count_x, COUNT_Y, text_c)
        for mask, start, end in labels:
            alpha = float(
                smoothstep((now - start) / 0.3) * smoothstep((end - now) / 0.3)
            )
            if start == 0.0:
                alpha = float(smoothstep((end - now) / 0.3))
            if alpha > 0:
                colour = accent_c if start > 13.0 else text_c
                blit_mask(image, mask, WIDTH // 2, LABEL_Y, colour, alpha, "centre")
        fade = float(smoothstep((DURATION - now) / FADE_OUT))
        if fade < 1:
            image = background + fade * (image - background)
        return image

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    bed = settings.audio_tag if settings.audio_tag in BEDS else BED
    track = soundtrack(pos, hw, time_of, bed, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def soundtrack(pos, hw, time_of, bed, rng):
    """
    Ambient bed with sparse gamelan bells, on the timeline of the walk (one-shot).
    See generate() for the mapping.
    """
    style = STYLES[STYLE]
    root = int(rng.choice(style.roots))
    scale = Scale(root, style.mode)  # pelog bells
    track = audio.silence(DURATION)
    hw_time = float(time_of[hw])

    # Bed: (start, end, semitones above root - 12, lowpass from, to, velocity)
    sections = [
        (0.0, SYMMETRIC_END, (0, 7, 12), 700.0, 1000.0, 0.26),
        (SYMMETRIC_END, 4.75, (0, 3, 7), 800.0, 900.0, 0.28),
        (4.75, 7.1, (-4, 0, 3), 800.0, 900.0, 0.28),
        (7.1, hw_time, (1, 5, 8), 800.0, 1100.0, 0.28),
        (hw_time, DURATION, (0, 7, 12, 15), 1100.0, 2400.0, 0.32),
    ]
    n = len(track)
    for start, end, chord, cut0, cut1, velocity in sections:
        layer = audio.silence(DURATION)
        length = end - start + 0.4
        attack = 0.5 if start == 0.0 else 1.2
        for i, semis in enumerate(chord):
            midi = root - 12 + semis
            hz = float(audio.midi_to_hz(midi))
            held = audio.pad(
                hz, length, velocity, attack=attack, release=2.2, brightness=0.35
            )
            audio.add(layer, held, start - 0.2, circular=False)
            upper = float(audio.midi_to_hz(midi + 12))
            if bed == "strings":
                voice = audio.strings(
                    upper,
                    length,
                    0.13,
                    attack=attack + 0.3,
                    release=2.2,
                    brightness=0.3,
                )
            else:
                voice = audio.choir(
                    upper, length, 0.15, attack=attack + 0.3, release=2.2, vowel="u"
                )
            audio.add(layer, voice, start - 0.2, circular=False)
        tt = np.clip((np.arange(n) / audio.SAMPLE_RATE - start) / (end - start), 0, 1)
        dark = audio.lowpass(layer, cut0, circular=False)
        bright = audio.lowpass(layer, cut1, circular=False)
        track += dark * (1 - tt)[:, None] + bright * tt[:, None]

    # Bells: the ant's distance from its start beats its record.
    bells = audio.silence(DURATION)
    origin = pos[0]
    offset = pos - origin
    dist = np.hypot(offset[:, 0], offset[:, 1])
    record = np.flatnonzero(dist[1:] > np.maximum.accumulate(dist)[:-1]) + 1
    last_bell = -1.0
    count = 0
    for k in record[record < hw]:
        when = float(time_of[k])
        if when - last_bell < MIN_GAP:
            continue
        last_bell = when
        angle = np.arctan2(-offset[k, 0], offset[k, 1]) % (2 * np.pi)
        degree = int(angle / (2 * np.pi) * 5) % 5
        octave = 1 if when < SYMMETRIC_END else 0
        x = np.clip(offset[k, 1] / 30.0, -1, 1)
        tone = style.voice(scale.hz(degree, octave), 0.32 + 0.1 * (octave == 0))
        audio.add(bells, audio.pan(tone, 0.6 * x), when, circular=False)
        count += 1
    first = style.voice(scale.hz(0, 1), 0.3)
    audio.add(bells, first, 0.05, circular=False)

    # The highway: a low gong at its start, then its steady beat.
    gong = audio.bell(float(audio.midi_to_hz(root - 12)), 0.5, length=6.0)
    low = style.bass(float(audio.midi_to_hz(root - 12)), 0.4)
    audio.add(bells, gong, hw_time, circular=False)
    audio.add(bells, low, hw_time, circular=False)
    beats = np.arange(hw + PULSE * PERIOD, STEPS, PULSE * PERIOD)
    for i, k in enumerate(beats):
        when = float(time_of[k])
        degree = FIGURE[i % len(FIGURE)]
        tone = style.voice(scale.hz(degree, 1), 0.3)
        audio.add(bells, audio.pan(tone, 0.25 * (-1) ** i), when, circular=False)
        if i % 2 == 1:
            audio.add(bells, style.bass(scale.hz(0, -1), 0.3), when, circular=False)
    bells = audio.lowpass(bells, 5500.0, circular=False)
    bells = audio.delay(bells, 0.42, feedback=0.35, wet=0.25, circular=False)
    track += bells
    track = audio.highpass(track, 80.0, circular=False)
    track = audio.reverb(track, rng, rt60=5.0, wet=0.42, tone=3800.0, circular=False)
    logger.info(
        f"Soundtrack ({bed}): root {root}, {count} record bells, "
        f"{len(beats)} highway bells, highway at {hw_time:.2f} s"
    )
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
