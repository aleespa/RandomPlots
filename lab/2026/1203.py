"""
Every Circle Is a Whole Number

An Apollonian gasket, the fractal you get by packing circles into a circle, then into the
gaps between them, forever. Each circle is labelled with its curvature, one over its
radius, and every label is a whole number: 2, 3, 6, 11, 14, 15, 23, ...

The reason is a theorem by Descartes: four circles that all touch satisfy
(k₁ + k₂ + k₃ + k₄)² = 2(k₁² + k₂² + k₃² + k₄²). So the circle filling a gap is
k = 2(k₁ + k₂ + k₃) − k₄, built from its neighbours by adding and subtracting. Start
with whole numbers (−1 for the outside circle, 2, 2 and 3) and you can never leave them.
The zoom dives almost 200 times deeper into the fractal: the numbers reach the
thousands, still whole.

Listen: each circle strikes a vibraphone bar as it appears, big circles low, tiny ones high,
so the pitch climbs as the zoom goes deeper, then lands on one chord.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#apolloniangasket #fractal #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale, quantize
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

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- Timeline (s) ------------------------------------------------------------------
POP = 0.3  # a circle grows to full size over this long
BUILD = (0.1, 4.6)  # circles appear, by curvature, between these times
BUILD_POWER = 1.25  # > 1: the big circles arrive quickly, the dust later
TEXT_OUT = (4.9, 5.5)  # the opening text fades out
ZOOM = (5.0, 13.2)  # the dive; then the picture rests
END_TEXT_IN = (12.6, 13.2)

# --- The gasket --------------------------------------------------------------------
# The integral gasket (-1, 2, 2, 3): the unit circle, two of radius 1/2 above and
# below, two of radius 1/3 left and right. Every other circle follows from Descartes.
START_K = np.array([-1.0, 2.0, 2.0, 3.0, 3.0])
START_Z = np.array([0, 0.5j, -0.5j, 2 / 3, -2 / 3])
START_GAPS = ((0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0), (0, 1, 4, 2), (0, 2, 4, 1))
START_GAPS += ((1, 2, 4, 0),)  # each gap: three touching circles + the one opposite
TARGET_K = (120, 360)  # the zoom ends centred on a circle with k in this range
TARGET_RADIUS = 300.0  # px: its radius at the end of the zoom
MIN_RADIUS = 0.3  # px: circles smaller than this on screen are never generated

# --- Layout (px) -------------------------------------------------------------------
R0 = 520.0  # px radius of the outer circle before the zoom
CX, CY = WIDTH / 2, HEIGHT / 2
LABEL_R = 18.0  # px: a circle gets its number once its radius passes this
LABEL_FADE = 14.0  # px of radius over which the label fades in
DIGIT_SIZE = 150  # px per em of the digit masks (scaled down per label)

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#f4f0e7"
INK = "#1b2238"
# Flat swatches, cyclic in log k (k and CYCLE * k share a colour), so every zoom
# level is as rich: navy, teal, sage, ochre, terracotta, plum.
PALETTE = ["#1b2a4a", "#22707a", "#6f8f72", "#e3a53a", "#c8553d", "#6e2c5a"]
CYCLE = 4.0
COLOUR_PHASE = 0.1

# --- Sound -------------------------------------------------------------------------
STYLE = "vibes"  # common.music.STYLES; an --audio-only tag naming one overrides it
BPM = 96.0  # tempo_choices(15): six bars
TOP_DEGREE = 10  # the smallest circles reach this scale degree
TOP_K = 3.0e4


def view(s, target, zoom_end):
    """Zoom Z(s) and view centre c(s): Z = 1 until ZOOM[0], then a log-smooth dive to
    zoom_end while the target slides from where it sits to the middle of the screen."""
    u = smoothstep((np.asarray(s, float) - ZOOM[0]) / (ZOOM[1] - ZOOM[0]))
    z = np.exp(np.log(zoom_end) * u)
    return z, target - target * (1 - u) / z


def appear_time(k):
    """When a circle of curvature k pops in: increasing with k, BUILD[0] .. BUILD[1]."""
    k_top = R0 / MIN_RADIUS
    k = np.asarray(k, float)
    x = np.clip(np.log(np.abs(k) / 2) / np.log(k_top / 2), 0, 1)
    when = BUILD[0] + (BUILD[1] - BUILD[0]) * x**BUILD_POWER
    return np.where(k < 0, -POP, when)  # the outer circle is there from the start


def gasket(levels):
    """
    Every circle that is ever bigger than MIN_RADIUS on screen, by Descartes' theorem:
    a gap between three touching circles (k1, k2, k3) whose fourth neighbour is k4 is
    filled by k = 2(k1 + k2 + k3) - k4, centre kz = 2(k1z1 + k2z2 + k3z3) - k4z4 (the
    complex Descartes theorem). Breadth first over gaps; a gap is dropped once its new
    circle is too small for every view (`levels`: zoom, centre, curvature limit) whose
    window overlaps the gap's bounding box (from its three tangency points, padded).
    """
    zoom, centre, k_limit = levels
    half_x = (CX + 4) / (R0 * zoom)
    half_y = (CY + 4) / (R0 * zoom)
    w_start = START_K * START_Z
    gaps = np.array(START_GAPS)
    gk, gw = START_K[gaps], w_start[gaps]
    ks, ws = [START_K], [w_start]
    while len(gk):
        kn = 2 * gk[:, :3].sum(1) - gk[:, 3]
        wn = 2 * gw[:, :3].sum(1) - gw[:, 3]
        pairs = ((0, 1), (0, 2), (1, 2))
        touch = np.stack(
            [(gw[:, i] + gw[:, j]) / (gk[:, i] + gk[:, j]) for i, j in pairs], 1
        )
        lo_x, hi_x = touch.real.min(1), touch.real.max(1)
        lo_y, hi_y = touch.imag.min(1), touch.imag.max(1)
        pad = 0.3 * np.maximum(hi_x - lo_x, hi_y - lo_y)
        lo_x, hi_x, lo_y, hi_y = lo_x - pad, hi_x + pad, lo_y - pad, hi_y + pad
        overlap = (
            (lo_x[:, None] < centre.real + half_x)
            & (hi_x[:, None] > centre.real - half_x)
            & (lo_y[:, None] < centre.imag + half_y)
            & (hi_y[:, None] > centre.imag - half_y)
        )
        keep = kn <= np.where(overlap, k_limit, 0).max(1)
        gk, gw, kn, wn = gk[keep], gw[keep], kn[keep], wn[keep]
        ks.append(kn)
        ws.append(wn)
        a, b, c = gk[:, 0], gk[:, 1], gk[:, 2]
        wa, wb, wc = gw[:, 0], gw[:, 1], gw[:, 2]
        gk = np.concatenate(
            [np.stack(q, 1) for q in ((a, b, kn, c), (a, c, kn, b), (b, c, kn, a))]
        )
        gw = np.concatenate(
            [
                np.stack(q, 1)
                for q in ((wa, wb, wn, wc), (wa, wc, wn, wb), (wb, wc, wn, wa))
            ]
        )
    k = np.concatenate(ks)
    z = np.concatenate(ws) / k
    order = np.argsort(k, kind="stable")
    return k[order], z[order]


def screen(s, k, z, target, zoom_end):
    """Screen x, y and full-grown radius (px) of circles (k, z) at time s."""
    zoom, centre = view(s, target, zoom_end)
    scale = R0 * zoom
    x = CX + (z.real - centre.real) * scale
    y = CY - (z.imag - centre.imag) * scale
    return x, y, scale / np.abs(k)


def on_screen(x, y, r):
    return (x + r > -2) & (x - r < WIDTH + 2) & (y + r > -2) & (y - r < HEIGHT + 2)


def label_events(k, z, target, zoom_end, step=0.01):
    """
    (time, index) of every circle's label appearing: when its grown radius first passes
    LABEL_R while on screen, sampled on a fixed 10 ms grid (independent of FPS), and
    never before the circle has popped in. The outer circle (k = -1) has no label.
    """
    first = np.full(len(k), np.inf)
    born = appear_time(k) + 0.5 * POP
    for s in np.arange(0.0, DURATION, step):
        x, y, r = screen(s, k, z, target, zoom_end)
        hit = (r >= LABEL_R) & on_screen(x, y, r) & (k > 0) & (born <= s)
        hit &= np.isinf(first)
        first[hit] = s
    index = np.flatnonzero(np.isfinite(first))
    return first[index], index


def digit_masks():
    """Masks of 0..9 in Computer Modern, each in an equal-width cell on one baseline."""
    row = latex_mask(r"$0\,1\,2\,3\,4\,5\,6\,7\,8\,9$", DIGIT_SIZE)
    ink = row.max(0) > 0.02
    edges = np.flatnonzero(np.diff(ink.astype(int)))
    starts, ends = edges[0::2] + 1, edges[1::2] + 1
    cell = int(0.5 * DIGIT_SIZE) + 4
    digits = []
    for a, b in zip(starts, ends):
        glyph = row[:, a:b]
        box = np.zeros((row.shape[0], cell), np.float32)
        x = (cell - glyph.shape[1]) // 2
        box[:, x : x + glyph.shape[1]] = glyph
        digits.append(box)
    assert len(digits) == 10, len(digits)
    return digits


def soundtrack(k, z, target, zoom_end, style, rng):
    """
    The circles made audible, from the same closed forms as the picture.

    - Build: one bar per distinct curvature as its circles pop in (appear_time), for
      every k of the build; big circles low and loud, pitch rising with log k on the style's scale
      (degree 0 for k = 2 up to TOP_DEGREE at TOP_K).
    - Dive: one bar each time a circle's number appears (label_events), same mapping, so
      the pitch climbs as the zoom goes deeper; panned by its screen x.
    - Both are pulled onto a 16th-note grid at BPM, one note per slot (the biggest).
    - A bass note under the first circles (2 and 3), a soft pad on the home chord whose
      low-pass opens with log Z, and the home chord struck as the zoom lands.
    One-shot: circular=False everywhere, master(loop=False).
    """
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)

    def degree(kk):
        x = np.log(kk / 2) / np.log(TOP_K / 2)
        return int(round(TOP_DEGREE * np.clip(x, 0, 1)))

    events = []  # (time, k, x position -1..1)
    for kk in np.unique(k[(k > 0) & (k <= R0 / MIN_RADIUS)]):
        i = np.flatnonzero(k == kk)[0]
        events.append((float(appear_time(kk)), kk, float(z[i].real)))
    times, index = label_events(k, z, target, zoom_end)
    for when, i in zip(times, index):
        if when >= ZOOM[0] and when <= ZOOM[1] + 0.2:
            x, _, _ = screen(when, k[i : i + 1], z[i : i + 1], target, zoom_end)
            events.append((float(when), k[i], float((x[0] - CX) / CX)))
    slots = {}
    snapped = quantize([e[0] for e in events], DURATION, BPM, subdivision=4)
    for (when, kk, x), q in zip(events, snapped):
        key = int(round(q / (60 / BPM / 4)))
        if key not in slots or kk < slots[key][1]:
            slots[key] = (max(q, 0.0), kk, x)
    for when, kk, x in slots.values():
        size = 1 - np.log(kk / 2) / np.log(TOP_K / 2)
        note = style.voice(scale.hz(degree(kk)), 0.25 + 0.55 * np.clip(size, 0, 1))
        audio.add(track, audio.pan(note, 0.6 * np.clip(x, -1, 1)), when, circular=False)
    for kk in (2.0, 3.0):
        bass = style.bass(scale.hz(0 if kk == 2 else 4, octave=-1), 0.6)
        audio.add(track, audio.pan(bass, 0.0), float(appear_time(kk)), circular=False)

    # The pad: home chord the whole way, opening up as the zoom deepens.
    chord = [float(audio.midi_to_hz(n)) for n in scale.chord(0, 3, octave=-1)]
    pad = sum(audio.pad(f, DURATION)[: len(track)] for f in chord) * 0.22
    sr = audio.SAMPLE_RATE
    s = np.arange(len(pad)) / sr
    zoom, _ = view(s, target, zoom_end)
    opening = np.log(zoom) / np.log(zoom_end)
    dark = audio.lowpass(pad, 500.0, circular=False)
    bright = audio.lowpass(pad, 3500.0, circular=False)
    swell = 0.6 + 0.4 * opening
    pad = (dark * (1 - opening)[:, None] + bright * opening[:, None]) * swell[:, None]
    fade_in = np.clip(s / 0.4, 0, 1)
    track += pad * fade_in[:, None]

    # The landing chord, and its bass.
    land = ZOOM[1] - 0.15
    for n in scale.chord(0, 3, octave=1):
        audio.add(track, style.voice(float(audio.midi_to_hz(n)), 0.55), land, False)
    audio.add(track, style.bass(scale.hz(0, octave=-1), 0.7), land, circular=False)
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The integral Apollonian gasket (-1, 2, 2, 3), built and then zoomed, as a one-shot.

    Circles come from Descartes' circle theorem, (Σk)^2 = 2Σk^2: the circle filling a
    gap between touching circles k1, k2, k3 with fourth neighbour k4 is
    k = 2(k1 + k2 + k3) - k4, and by the complex Descartes theorem its centre satisfies
    kz = 2(k1z1 + k2z2 + k3z3) - k4z4, so all curvatures stay integers (exact in float64).
    `gasket` generates breadth first over gaps, keeping only circles that are ever at
    least MIN_RADIUS px on screen along the zoom path (sampled views), ~4 x 10^4 circles.

    Time: circles pop in by curvature (appear_time, a power of log k, over BUILD) and
    grow with an ease-out over POP; then from ZOOM[0] to ZOOM[1] the view zooms with
    log Z = smoothstep(u) log Z_end onto a target circle drawn by settings.rng with k in
    TARGET_K, which slides to the centre and ends TARGET_RADIUS px wide; then it rests.
    Fill colour is cyclic in log k (period CYCLE), so every depth looks alike. Labels:
    each circle wider than LABEL_R px shows k, composed from Computer Modern digit masks
    (latex_mask), scaled to fit. Opening text (Descartes' theorem with the first four
    curvatures) and the closing line are latex_mask too, built once.

    Render: per frame, a vectorised cull, cv2 anti-aliased filled discs (shift=4) for
    circles over 1 px, a NumPy coverage splat for the dust, cv2 thread pool.

    Sound: see `soundtrack` (vibes: vibraphone). The posted
    soundtrack is `--seed 0` with the default audio seed (= --seed 0).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # The zoom path: views sampled from the start of the dive to its end.
    k0, z0 = gasket((np.ones(1), np.zeros(1, complex), np.array([R0 / MIN_RADIUS])))
    candidates = np.flatnonzero((k0 >= TARGET_K[0]) & (k0 <= TARGET_K[1]))
    pick = int(rng.choice(candidates))
    target, k_target = complex(z0[pick]), float(k0[pick])
    zoom_end = TARGET_RADIUS * k_target / R0
    samples = np.concatenate([[0.0], np.linspace(ZOOM[0], DURATION, 240)])
    zoom, centre = view(samples, target, zoom_end)
    k, z = gasket((zoom, centre, R0 * zoom / MIN_RADIUS))
    logger.info(
        f"Gasket: {len(k)} circles, target k = {k_target:.0f}, zoom x{zoom_end:.0f}, "
        f"largest k = {k.max():.0f}"
    )

    born = appear_time(k)
    swatches = np.array([hex_to_bgr(c) for c in PALETTE], np.float32)
    hue = (np.log(np.abs(k)) / np.log(CYCLE) + COLOUR_PHASE) % 1.0
    fill = swatches[(hue * len(PALETTE)).astype(int) % len(PALETTE)]
    light = (fill @ np.array([0.114, 0.587, 0.299], np.float32)) / 255 > 0.55
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    ink = np.array(hex_to_bgr(INK), np.float32)
    background = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    background[:] = bg.astype(np.uint8)

    # Labels: digits composed per number (only numbers that ever show are built).
    digits = digit_masks()
    _, shown = label_events(k, z, target, zoom_end)
    labels = {
        int(kk): np.hstack([digits[int(d)] for d in str(int(kk))])
        for kk in np.unique(k[shown])
    }

    # Text (LaTeX), built once.
    top = [
        latex_mask(r"$\mathrm{Label\ every\ circle\ with\ its\ curvature}$", 50),
        latex_mask(
            r"$k = 1/r\mathrm{{,}\ \ and\ every\ one\ is\ a\ whole\ number}$", 50
        ),
    ]
    bottom = [
        latex_mask(r"$(k_1+k_2+k_3+k_4)^2 = 2\,(k_1^2+k_2^2+k_3^2+k_4^2)$", 46),
        latex_mask(r"$(-1+2+2+3)^2 = 2\,(1+4+4+9) = 36$", 46),
    ]
    ending = latex_mask(
        rf"$\times\,{int(round(zoom_end))}\ \mathrm{{deeper{{,}}\ still\ whole\ numbers}}$",
        50,
    )
    band = (1700, 1700 + ending.shape[0] + 48)

    def render(frame: int) -> np.ndarray:
        s = DURATION * frame / (FRAME_COUNT - 1)
        x, y, r_full = screen(s, k, z, target, zoom_end)
        grow = np.clip((s - born) / POP, 0, 1)
        r = r_full * (1 - (1 - grow) ** 3)
        visible = on_screen(x, y, r) & (r > 0.1)
        image = background.copy()

        # The outer circle, k = -1, as a ring (skipped once the view is inside it).
        zoom_s, centre_s = view(s, target, zoom_end)
        corners = np.array([-1, 1]) * CX / (R0 * zoom_s)
        far = max(
            abs(complex(centre_s.real + a, centre_s.imag + b))
            for a in corners
            for b in corners * CY / CX
        )
        if far > 1 - 1e-9:
            # A dense polyline: cv2.circle facets visibly at this size.
            radius = r_full[0]
            n = int(np.clip(2 * np.pi * radius / 3, 256, 200000))
            angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
            ring = np.stack(
                [x[0] + radius * np.cos(angle), y[0] + radius * np.sin(angle)]
            )
            ring = np.round(ring.T * 16).astype(np.int32)
            cv2.polylines(image, [ring], True, tuple(ink.tolist()), 3, cv2.LINE_AA, 4)

        # Dust: sub-pixel circles as coverage splats.
        dust = np.flatnonzero(visible & (r < 1.0) & (k > 0))
        if len(dust):
            ix = np.clip(np.round(x[dust]).astype(int), 0, WIDTH - 1)
            iy = np.clip(np.round(y[dust]).astype(int), 0, HEIGHT - 1)
            a = np.clip(np.pi * r[dust] ** 2 * 0.8, 0, 1)[:, None]
            pixels = image[iy, ix].astype(np.float32)
            image[iy, ix] = (pixels * (1 - a) + fill[dust] * a).astype(np.uint8)

        # Discs, each shrunk a little so a thin cream line separates neighbours.
        discs = np.flatnonzero(visible & (r >= 1.0) & (k > 0))
        gap = np.clip(0.035 * r[discs], 0.4, 2.5)
        for i, g in zip(discs, gap):
            cv2.circle(
                image,
                (int(round(x[i] * 16)), int(round(y[i] * 16))),
                int(round((r[i] - g) * 16)),
                tuple(fill[i].tolist()),
                -1,
                cv2.LINE_AA,
                4,
            )

        image = image.astype(np.float32)

        # Numbers.
        for i in discs:
            if r[i] < LABEL_R or int(k[i]) not in labels:
                continue
            mask = labels[int(k[i])]
            h, w = mask.shape
            fit = min(0.55 * r[i] / h, 1.2 * r[i] / w, 1.0)
            size = (max(1, int(w * fit)), max(1, int(h * fit)))
            m = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
            alpha = float(smoothstep((r[i] - LABEL_R) / LABEL_FADE))
            colour = ink if light[i] else bg
            blit_mask(
                image,
                m,
                int(round(x[i])),
                int(round(y[i] - m.shape[0] / 2)),
                colour,
                alpha=0.95 * alpha,
                anchor="centre",
            )

        # Opening text.
        text_in = float(smoothstep((s - 1.2) / 0.5))
        text_out = 1 - float(
            smoothstep((s - TEXT_OUT[0]) / (TEXT_OUT[1] - TEXT_OUT[0]))
        )
        if text_out > 0:
            for mask, ty in zip(top, (190, 290)):
                blit_mask(
                    image, mask, int(CX), ty, ink, alpha=text_out, anchor="centre"
                )
            for mask, ty in zip(bottom, (1580, 1680)):
                a = text_in * text_out
                blit_mask(image, mask, int(CX), ty, ink, alpha=a, anchor="centre")

        # Closing line on a cream band.
        end = float(
            smoothstep((s - END_TEXT_IN[0]) / (END_TEXT_IN[1] - END_TEXT_IN[0]))
        )
        if end > 0:
            region = image[band[0] : band[1]]
            region += 0.92 * end * (bg - region)
            blit_mask(
                image, ending, int(CX), band[0] + 24, ink, alpha=end, anchor="centre"
            )
        return image

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag
    style = STYLES[tag] if tag in STYLES else STYLES[STYLE]
    track = soundtrack(k, z, target, zoom_end, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
