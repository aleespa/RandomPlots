"""
Where the Bell Curve Comes From

A Galton board: 1,600 beads fall through 24 rows of pegs, and at every
peg each bead goes left or right on the toss of a fair coin. Each bead is
coloured by the bin it will end up in, so the streams braid through the pegs
and sort themselves into a spectrum at the bottom.

No bead is aiming anywhere. Its bin is just the number of times it went
right, a coin-flip count. Yet the piles grow into the bell curve, the normal
distribution, e^(−x²/2): the central limit theorem, first glimpsed by de Moivre
in 1733 and built into this machine by Francis Galton in 1889. Add up enough
small random nudges and the bell curve always appears.

Every few landings ring a bell, pitched by the bin: low on the left, high on
the right. When the curve is drawn it plays itself, rising to the middle
and falling away again: the bell curve, as bells.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#galtonboard #probability #mathart #generativeart #randomplots
"""

import gc
from math import comb

import cv2
import numpy as np

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
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

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

ROWS = 24  # peg rows: 25 bins
BEADS = 1600
P_RIGHT = 0.5  # the coin
SPREAD = np.sqrt(
    ROWS * P_RIGHT * (1 - P_RIGHT)
)  # sd of the bins: colours span +-2.5 sd
RELEASE_START, RELEASE_END = -1.2, 9.0  # s; negative: beads already falling at 0
RELEASE_POWER = 2.0  # release rate grows like t^(POWER - 1): trickle to torrent
CURVE_START, CURVE_END = 12.0, 13.5  # s: the bell curve traces in

CX = WIDTH / 2
DX = 40.0  # peg spacing = bin width, px
DY = 36.0  # row spacing, px
Y_PEG = 260.0  # top peg
Y_FLOOR = 1800.0
DROP = 0.25  # s from the top of the screen to the first peg
ROW_TIME = 0.09  # s per row (each bead varies a little)
HOP = 7.0  # px: the little bounce off each peg
GRAVITY = 3000.0  # px/s^2 below the pegs

BEAD_R = 4.5
PEG_R = 3.0
SLOT_DX, SLOT_DY = 10.0, 10.0 * np.sqrt(3) / 2  # hexagonal stacking in a bin

STYLE = "music_box"  # common.music.STYLES: glassy bells, lydian
BELL_GAP = 0.16  # s: at most one landing bell per gap
CURVE_BINS = range(4, ROWS - 3)  # bins the traced curve plays

SS = 2
SHIFT = 4
PALETTE = ["#1d3b8f", "#1fa3b8", "#f2f2ee", "#f07a5c", "#c2263b"]
PEG = np.array([70, 64, 60], np.float32)  # BGR, dim
PEG_FLASH = np.array([230, 230, 255], np.float32)
DIVIDER = (48, 44, 42)
GOLD = hex_to_bgr("#f5c04a")


def slot_offsets(m):
    """Position of the m-th bead in a bin: rows of 4 and 3, hexagonally packed."""
    pair, rem = np.divmod(m, 7)
    row = 2 * pair + (rem >= 4)
    col = np.where(rem < 4, 5.0 + SLOT_DX * rem, 10.0 + SLOT_DX * (rem - 4))
    return col - DX / 2, Y_FLOOR - BEAD_R - row * SLOT_DY


def soundtrack(land_time, peg_time, bins, rng):
    """
    Bells. Before the first landing, beads striking the top peg tick a soft
    high bell (at most one per 2 BELL_GAP). Landings in time order ring a bell
    (a landing is skipped if one rang less than BELL_GAP s before, so the
    density follows the stream up to
    ~6 notes/s), at scale degree bin - ROWS/2, panned by bin, softer in the
    sparse opening. While the curve is traced, each bin in CURVE_BINS rings as
    the trace passes its centre (inverting the smoothstep), pitched by the
    curve's height there, so the melody rises to the middle and falls back.
    A soft pad holds the home chord under it all and a bell chord ends it.
    """
    style = STYLES[STYLE]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    centre = ROWS * P_RIGHT

    first = land_time.min()
    last = -1.0
    for when in np.sort(peg_time):  # the opening: beads striking the top peg
        if 0 <= when < first - BELL_GAP and when - last >= 2 * BELL_GAP:
            last = when
            audio.add(track, style.voice(scale.hz(7), 0.18), when, **kw)

    last = -1.0
    for i in np.argsort(land_time):
        when = land_time[i]
        if when - last < BELL_GAP or when >= DURATION:
            continue
        last = when
        degree = int(np.clip(bins[i] - round(centre), -8, 8))
        velocity = 0.25 + 0.25 * smoothstep(when / 6.0)
        note = audio.pan(
            style.voice(scale.hz(degree, -1), velocity), (bins[i] - centre) / 8
        )
        audio.add(track, note, when, **kw)

    grid = np.linspace(0, 1, 2001)
    traced = smoothstep(grid)
    sd = np.sqrt(ROWS * P_RIGHT * (1 - P_RIGHT))
    for k in CURVE_BINS:
        fraction = (k + 1) / (ROWS + 2)  # the bin centre along the traced curve
        when = CURVE_START + (CURVE_END - CURVE_START) * np.interp(
            fraction, traced, grid
        )
        height = np.exp(-((k - centre) ** 2) / (2 * sd * sd))
        degree = int(round(10 * height)) - 2
        note = audio.pan(
            style.voice(scale.hz(degree), 0.35 + 0.3 * height), (k - centre) / 8
        )
        audio.add(track, note, when, **kw)

    for degree in (0, 2, 4):  # the home chord underneath, swelling with the stream
        voice = audio.pad(scale.hz(degree, -2), CURVE_END - 0.5, 0.12, 1.0, 2.5, 0.3)
        audio.add(track, voice, 0.0, **kw)
    for degree, delay in ((0, 0.0), (4, 0.06), (7, 0.12), (9, 0.18)):
        audio.add(
            track, style.voice(scale.hz(degree), 0.5), CURVE_END + 0.1 + delay, **kw
        )
    audio.add(track, style.bass(scale.hz(0, -2), 0.6), CURVE_END + 0.1, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Galton board, every bead on a closed-form path. All randomness is drawn up
    front: 24 Bernoulli(P_RIGHT) flips per bead, a release time (cumulative
    release ~ u^RELEASE_POWER, so the stream thickens) and a +-10% row time. A
    bead drops from the top for DROP s, then crosses one peg row per row time:
    x interpolates linearly between pegs (cx + (j - r/2) DX for j rights after r
    rows), y = y_r + DY s - HOP 4 s(1 - s). Leaving the last row it falls under
    GRAVITY to its slot; slots in each bin are given in order of exit time and
    packed hexagonally (rows of 4 and 3). Pegs flash for 0.2 of a row time after
    each hit. Moving beads glow (blurred layer); resting ones don't.

    From CURVE_START the expected pile height, BEADS C(24, k) / 2^24 per bin
    (with the Gaussian of the same mean and variance between bins), is traced in
    gold. Frames are drawn with cv2 at 2x on the thread pool.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    flips = (rng.random((BEADS, ROWS)) < P_RIGHT).astype(np.int64)
    u = np.sort(rng.random(BEADS))
    release = RELEASE_START + (RELEASE_END - RELEASE_START) * u ** (1 / RELEASE_POWER)
    row_time = ROW_TIME * rng.uniform(0.9, 1.1, BEADS)

    rights = np.concatenate([np.zeros((BEADS, 1), np.int64), flips.cumsum(1)], 1)
    rows = np.arange(ROWS + 1)
    peg_x = CX + (rights - rows / 2) * DX  # (BEADS, ROWS + 1)
    peg_y = Y_PEG + rows * DY
    bins = rights[:, -1]
    exit_time = release + DROP + ROWS * row_time

    # Slots in each bin, in order of arrival.
    order = np.argsort(exit_time)
    slot = np.zeros(BEADS, np.int64)
    filled = np.zeros(ROWS + 1, np.int64)
    for i in order:
        slot[i] = filled[bins[i]]
        filled[bins[i]] += 1
    dx_slot, slot_y = slot_offsets(slot)
    slot_x = CX + (bins - ROWS / 2) * DX + dx_slot
    v0 = DY / row_time
    fall_d = slot_y - peg_y[-1]
    fall_time = (-v0 + np.sqrt(v0 * v0 + 2 * GRAVITY * fall_d)) / GRAVITY
    land_time = exit_time + fall_time

    lut = (palette_lut(PALETTE) * 255)[:, ::-1]
    colours = [
        tuple(
            float(c)
            for c in lut[
                int(
                    round(
                        255 * np.clip(0.5 + (k - ROWS * P_RIGHT) / (5 * SPREAD), 0, 1)
                    )
                )
            ]
        )
        for k in range(ROWS + 1)
    ]

    # The static board: dividers and floor.
    board = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
    for k in range(ROWS + 2):
        x = int(round((CX + (k - ROWS / 2 - 0.5) * DX) * SS * (1 << SHIFT)))
        top = int((peg_y[-1] + 18) * SS * (1 << SHIFT))
        bottom = int(Y_FLOOR * SS * (1 << SHIFT))
        cv2.line(board, (x, top), (x, bottom), DIVIDER, SS, cv2.LINE_AA, SHIFT)
    peg_r = [(r, j) for r in range(ROWS) for j in range(r + 1)]
    peg_xy = np.array([(CX + (j - r / 2) * DX, Y_PEG + r * DY) for r, j in peg_r])
    peg_index = {rj: n for n, rj in enumerate(peg_r)}
    peg_lookup = np.full((ROWS, ROWS + 1), -1)
    for (r, j), n in peg_index.items():
        peg_lookup[r, j] = n

    # The expected heights: binomial at bin centres, Gaussian in between.
    mean, sd = ROWS * P_RIGHT, np.sqrt(ROWS * P_RIGHT * (1 - P_RIGHT))
    xs = np.linspace(-0.5, ROWS + 0.5, 600)
    gauss = (
        BEADS / (np.sqrt(2 * np.pi) * sd) * np.exp(-((xs - mean) ** 2) / (2 * sd**2))
    )
    per_row = 7 / 2  # beads per stacked row, on average
    curve_y = Y_FLOOR - BEAD_R - (gauss / per_row) * SLOT_DY
    curve = np.stack([CX + (xs - ROWS / 2) * DX, curve_y], 1)
    curve_pts = np.round(curve * SS * (1 << SHIFT)).astype(np.int32)
    expected = [BEADS * comb(ROWS, k) / 2**ROWS for k in range(ROWS + 1)]
    assert max(filled) < 1.25 * max(expected)

    def to_px(v):
        return int(round(v * SS * (1 << SHIFT)))

    def render(frame):
        t = frame / FPS
        tau = t - release
        big = board.copy()
        glow = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # Pegs, flashing where a bead has just bounced.
        q = (tau - DROP) / row_time
        in_pegs = (q >= 0) & (q < ROWS)
        r = np.clip(np.floor(q).astype(int), 0, ROWS - 1)
        s = q - r
        hits = np.zeros(len(peg_r))
        just = in_pegs & (s < 0.2)
        idx = peg_lookup[r[just], rights[just, r[just]]]
        np.add.at(hits, idx, 1 - s[just] / 0.2)
        flash = np.clip(hits, 0, 1)[:, None]
        peg_cols = PEG * (1 - flash) + PEG_FLASH * flash
        for (x, y), c in zip(peg_xy, peg_cols):
            cv2.circle(
                big,
                (to_px(x), to_px(y)),
                to_px(PEG_R),
                tuple(map(float, c)),
                -1,
                cv2.LINE_AA,
                SHIFT,
            )

        # Bead positions.
        x = np.full(BEADS, np.nan)
        y = np.full(BEADS, np.nan)
        dropping = (tau >= 0) & (tau < DROP)
        a = tau[dropping] / DROP
        x[dropping] = CX
        y[dropping] = -10 + (Y_PEG + 10) * a * a
        ii = np.nonzero(in_pegs)[0]
        x[ii] = peg_x[ii, r[ii]] + (peg_x[ii, r[ii] + 1] - peg_x[ii, r[ii]]) * s[ii]
        y[ii] = peg_y[r[ii]] + DY * s[ii] - HOP * 4 * s[ii] * (1 - s[ii])
        falling = (t >= exit_time) & (t < land_time)
        f = t - exit_time[falling]
        x[falling] = slot_x[falling] - (slot_x[falling] - peg_x[falling, -1]) * np.exp(
            -f / 0.05
        )
        y[falling] = peg_y[-1] + v0[falling] * f + 0.5 * GRAVITY * f * f
        resting = t >= land_time
        x[resting] = slot_x[resting]
        y[resting] = slot_y[resting]
        moving = dropping | in_pegs | falling

        radius = to_px(BEAD_R)
        for i in np.nonzero(moving | resting)[0]:
            c = colours[bins[i]]
            centre = (to_px(x[i]), to_px(y[i]))
            cv2.circle(big, centre, radius, c, -1, cv2.LINE_AA, SHIFT)
            if moving[i]:
                small = (centre[0] // SS, centre[1] // SS)
                cv2.circle(glow, small, radius // SS + 16, c, -1, cv2.LINE_AA, SHIFT)

        # The bell curve, traced left to right.
        trace = smoothstep((t - CURVE_START) / (CURVE_END - CURVE_START))
        if trace > 0:
            n = max(2, int(trace * len(curve_pts)))
            cv2.polylines(big, [curve_pts[:n]], False, GOLD, 3 * SS, cv2.LINE_AA, SHIFT)
            small = (curve_pts[:n] // SS).astype(np.int32)
            cv2.polylines(glow, [small], False, GOLD, 8, cv2.LINE_AA, SHIFT)

        image = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        image += 0.9 * cv2.GaussianBlur(glow, (0, 0), 6).astype(np.float32)
        image *= smoothstep((DURATION - t) / FADE)
        return image

    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(land_time, release + DROP, bins, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
