"""
The Hole in Escher's Print Gallery

The Droste effect, Escher's way: a picture that contains itself, spiralling in
forever. Escher's 1956 Print Gallery twists a town into the gallery that shows
it, but he could not draw the centre and left a blank hole there.

In 2003 Hendrik Lenstra and Bart de Smit found the rule behind it. Take a
picture holding a copy of itself four times smaller, and bend it with a
conformal map from complex analysis, z ↦ z^α, α = 1 − i·ln 4 / 2π. Angles
survive, but once around the centre now takes you one picture deeper. The
frames never close: they form one endless spiral, and the hole fills itself in.

Listen: strings and a soft choir hold slow chords, and a music-box bell rings
each time a frame's corner crosses the same circle.

Made with Python & PyTorch. Sound synthesized in NumPy.
#drosteeffect #complexanalysis #mathart #generativeart #randomplots
"""

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    latex_mask,
    palette_lut,
    render_frames,
)

DURATION = 15.0  # seconds, one exact loop
FRAME_COUNT = frame_count(DURATION)

# --- The picture and the map ----------------------------------------------------
SCALE = 4.0  # each picture holds a copy SCALE times smaller
L = math.log(SCALE)
TWIST = L / (2.0 * math.pi)  # alpha = 1 - i * TWIST: one turn = one picture deeper
LEVELS = 2  # pictures zoomed through per loop (whole, so the loop closes)
SHIFT0 = 0.35  # where in the zoom the loop starts, in pictures
UNIT = WIDTH / 2  # output pixels per unit of z
FRAME_OUT = 1.3  # the frame runs from |z'|_inf = 1 (the hole) to FRAME_OUT
SS = 2  # supersampling

# --- Palette ----------------------------------------------------------------------
WALL = "#0f2c2e"  # bottle-green gallery wall
TRELLIS = "#2c5a52"
DOTS = "#8a6a32"
LAMP = "#f3c77a"  # warm light above each frame
GOLD = ("#2a1a08", "#6b4718", "#b9822f", "#e6b65a", "#fff0c4")
WIRE = "#3b2a14"
TEXT = "#f4ead2"
MEAN = "#2a3a2c"  # what the infinitely small centre averages to

# --- Text -------------------------------------------------------------------------
TITLE = r"$\mathrm{A\ picture\ inside\ itself}$"
FORMULA = rf"$z\ \mapsto\ z^{{\,1\,-\,\frac{{i\,\ln {SCALE:.0f}}}{{2\pi}}}}$"
TITLE_Y, FORMULA_Y = 150, 1660

# --- Soundtrack -------------------------------------------------------------------
SOUND = "swell"  # "swell" (strings, choir, music-box bells) or "lofi" (pad, Rhodes)
BED_ROOTS = range(52, 58)  # E3 .. A3: the chords' root
PROGRESSION = (0, 1, 5, 4)  # in lydian: I, II, vi, V, one chord per quarter loop
CORNER_RADIUS = 420  # px: a bell rings as a frame corner crosses this circle


def _rgb(colour):
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def _cover(d, aa):
    """Coverage of a shape with signed distance d (negative inside), edge width aa."""
    return (0.5 - d / aa).clamp(0.0, 1.0)


def _segment(px, py, ax, ay, bx, by):
    """Distance from points to the segment a-b."""
    dx, dy = bx - ax, by - ay
    h = (((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0)
    return torch.hypot(px - ax - h * dx, py - ay - h * dy)


def _text_layer():
    """Full-frame alpha of the on-screen LaTeX (title and formula), built once."""
    layer = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    blit_mask(layer, latex_mask(TITLE, 62), WIDTH // 2, TITLE_Y, (1, 1, 1), 1, "centre")
    blit_mask(
        layer, latex_mask(FORMULA, 96), WIDTH // 2, FORMULA_Y, (1, 1, 1), 1, "centre"
    )
    return layer[..., :1]


def corner_events():
    """
    (t, corner, x) for every frame corner crossing CORNER_RADIUS during the loop,
    t in [0, 1), from the same closed form as the picture. A corner of the frame,
    at log z' = log(FRAME_OUT·√2) + i(π/4 + jπ/2) + nL in the picture, sits at
    log z = (log z' + shift(t)) / α on screen, shift(t) = (LEVELS·t + SHIFT0)·L;
    the copies n of one corner cross the circle 1 / LEVELS of a loop apart.
    """
    alpha = complex(1.0, -TWIST)
    target = math.log(CORNER_RADIUS / UNIT)
    events = []
    for j in range(4):
        u0 = math.log(FRAME_OUT * math.sqrt(2.0)) + 1j * (math.pi / 4 + j * math.pi / 2)
        speed = (LEVELS * L / alpha).real  # d Re(log z) / dt
        t0 = (target - ((u0 + SHIFT0 * L) / alpha).real) / speed
        for n in range(-LEVELS - 2, LEVELS + 2):
            t = t0 - n / LEVELS
            if 0.0 <= t < 1.0:
                w = (u0 + n * L + (LEVELS * t + SHIFT0) * L) / alpha
                events.append((t, j, math.cos(w.imag)))
    return sorted(events)


def soundtrack(kind, rng):
    """
    Ambient, loop-periodic (everything circular). A bed of four chords in lydian,
    one per quarter loop (I, II, vi, V, so V leads back into I at the seam): "swell"
    holds them on strings with a soft "u" choir an octave up; "lofi" on a dark pad
    with a Rhodes stroke at each change. Over it, one struck note per frame corner
    crossing the circle CORNER_RADIUS (4 per picture, 8 a loop): a music-box bell
    ("swell") or a Rhodes note ("lofi"), the j-th tone of the chord sounding for
    corner j, panned by the corner's x. A tempo-free delay and a long room.
    """
    root = int(rng.choice(BED_ROOTS))
    scale = Scale(root, "lydian")
    track = audio.silence(DURATION)
    span = DURATION / len(PROGRESSION)
    hz = lambda midi: float(audio.midi_to_hz(midi))
    for i, degree in enumerate(PROGRESSION):
        start = i * span - 0.6  # swell in a little early, overlap the release
        chord = scale.chord(degree, 3, octave=1)
        bass = scale.note(degree, 0)
        if kind == "swell":
            for midi in chord:
                audio.add(
                    track, audio.strings(hz(midi), span, 0.32, 1.4, 2.2, 0.35), start
                )
            audio.add(track, audio.strings(hz(bass), span, 0.30, 1.4, 2.2, 0.2), start)
            for midi in chord:
                voice = audio.choir(hz(midi + 12), span, 0.12, 1.6, 2.4, vowel="u")
                audio.add(track, voice, start)
        else:
            for midi in chord + [bass]:
                audio.add(track, audio.pad(hz(midi), span, 0.28, 1.2, 2.2, 0.3), start)
            for k, midi in enumerate(chord):
                stroke = audio.rhodes(hz(midi), 0.35, length=3.5)
                audio.add(track, audio.pan(stroke, 0.3 * (k - 1)), i * span + 0.03 * k)
    events = corner_events()
    for t, j, x in events:
        when = t * DURATION
        degree = PROGRESSION[int(when // span) % len(PROGRESSION)]
        tones = scale.chord(degree, 4, octave=2)
        if kind == "swell":
            note = audio.bell(hz(tones[j]), 0.42, length=5.0)
        else:
            note = audio.rhodes(hz(tones[j] - 12), 0.5, length=4.0)
        audio.add(track, audio.pan(note, 0.6 * x), when)
    track = audio.highpass(track, 70.0)
    if kind == "lofi":
        track = audio.lowpass(track, 3800.0)
    track = audio.delay(track, DURATION / 24, feedback=0.35, wet=0.2)
    track = audio.reverb(track, rng, rt60=5.0, wet=0.45, tone=4500.0)
    logger.info(f"Soundtrack {kind}: root {root}, {len(events)} corner bells")
    return audio.master(track)


def generate(settings: ImageProcessingSettings = None):
    """
    The Droste effect of Escher's Print Gallery (Lenstra & de Smit, 2003), as a loop.

    The picture: in coordinates z' it is a gallery wall (bottle green, a diamond
    trellis, a warm lamp glow, the shadow of the outer frame) with a gold frame on
    the square ring 1 <= |z'|_inf <= FRAME_OUT (mitred mouldings, Lambert-shaded
    from the top left), a wire, a nail and a brass plate; the hole |z'|_inf < 1
    holds the same picture SCALE = 4 times smaller. So the picture is invariant
    under z' -> z'·4, and in u = log z' it is doubly periodic, with periods
    L = ln 4 and 2πi.

    The map: a screen point z goes to u = α·log z + shift(t), α = 1 − i·L/2π, then
    to the fundamental ring by subtracting kL, k = floor(log|e^u|_inf / L), and
    z' = e^u. α sends the screen's period 2πi to 2πi + L, a period of the picture,
    so the image is single-valued and conformal: once round the centre is one
    picture deeper, the frames are a single spiral. shift(t) = (LEVELS·t + SHIFT0)·L
    zooms through LEVELS = 2 pictures per loop, a whole number of periods, so the
    loop closes exactly. Edges are antialiased analytically with the pixel's
    footprint in z', |dz'/dz|·dz = |z'|·|α|/|z|·dz, and detail fades to the mean
    colour where it falls below a pixel (the centre); plus 2x supersampling.
    Per-pixel on the GPU (torch), render_frames(parallel=False). Static LaTeX:
    a title and the map.

    The picture uses no randomness, so every --seed draws the same frames.
    Soundtrack (`soundtrack`, see there): the posted one is "swell" with --seed 0
    (the default; settings.audio_rng picks the root and the room). An --audio-only
    tag "swell" or "lofi" picks that version; any other tag plays SOUND.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    logger.info(f"Droste: scale {SCALE}, twist {TWIST:.4f}, {LEVELS} pictures a loop")

    def colour(c):
        return torch.tensor(_rgb(c), device=device, dtype=torch.float32)

    wall, trellis, dots, lamp, wire, mean, text = (
        colour(c) for c in (WALL, TRELLIS, DOTS, LAMP, WIRE, MEAN, TEXT)
    )
    gold_lut = torch.tensor(palette_lut(GOLD, 512), device=device)

    # Screen grid, log z, and the parts of the map that don't depend on t.
    h, w = HEIGHT * SS, WIDTH * SS
    ys = (h / 2 - (torch.arange(h, device=device) + 0.5)) / (SS * UNIT)
    xs = ((torch.arange(w, device=device) + 0.5) - w / 2) / (SS * UNIT)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    log_r = 0.5 * torch.log(x * x + y * y + 1e-12)
    theta = torch.atan2(y, x)
    u_re0 = log_r + TWIST * theta  # alpha * log z, alpha = 1 - i TWIST
    u_im = theta - TWIST * log_r
    cos_u, sin_u = torch.cos(u_im), torch.sin(u_im)
    square = torch.log(torch.maximum(cos_u.abs(), sin_u.abs()))  # log of |.|_inf / |.|
    # Footprint of one supersample in z' per unit |z'|: |alpha| / (|z| SS UNIT).
    foot0 = math.hypot(1.0, TWIST) / (torch.exp(log_r) * SS * UNIT)
    del ys, xs, y, x, theta

    # Screen-space vignette and text bands, and the text itself.
    vy = torch.linspace(-1, 1, HEIGHT, device=device)[:, None]
    vx = torch.linspace(-1, 1, WIDTH, device=device)[None, :]
    shade = 1.0 - 0.35 * (vx**2 * 0.6 + vy**2 * 0.4)
    band = 1.0 - 0.55 * (
        torch.exp(-(((vy + 0.86) / 0.12) ** 2))
        + torch.exp(-(((vy - 0.76) / 0.16) ** 2))
    )
    shade = (shade * band)[..., None]
    text_alpha = torch.tensor(_text_layer(), device=device)

    light = np.array([-0.45, 0.55, 1.0])
    light /= np.linalg.norm(light)
    lx, ly, lz = (float(v) for v in light)

    def picture(X, Y, fw):
        """RGB of the picture at z' = X + iY (1 <= |z'|_inf < SCALE), edge width fw."""
        ax, ay = X.abs(), Y.abs()
        m = torch.maximum(ax, ay)
        detail = (1.0 - fw / 0.02).clamp(0.0, 1.0)[..., None]  # fades out sub-pixel
        # Wall: lamp glow above the frame, the outer frame's shadow near m = SCALE.
        glow = torch.exp(-((X / 1.1) ** 2) - ((Y - FRAME_OUT - 0.55) / 0.75) ** 2)
        lip = (m - (SCALE - 0.3)).clamp(0.0, 0.3) / 0.3
        img = wall * (0.8 + 0.45 * glow - 0.55 * lip**2)[..., None]
        img = img + 0.18 * glow[..., None] * lamp
        # Diamond trellis wallpaper with a dot in every diamond.
        P = 0.22
        a, c = (X + Y) / P, (X - Y) / P
        line = torch.minimum((a - a.round()).abs(), (c - c.round()).abs()) * P / 1.414
        cov = _cover(line - 0.0045, fw)[..., None] * detail
        img = img + (trellis - img) * cov * 0.55
        dot = torch.hypot(a - a.floor() - 0.5, c - c.floor() - 0.5) * P / 1.414
        cov = _cover(dot - 0.014, fw)[..., None] * detail
        img = img + (dots - img) * cov * 0.6
        # Drop shadow of this picture's frame, then the wire and the nail.
        sm = torch.maximum((X - 0.05).abs(), (Y + 0.07).abs())
        drop = 1.0 - ((sm - FRAME_OUT + 0.04) / 0.16).clamp(0.0, 1.0)
        img = img * (1.0 - 0.6 * drop)[..., None]
        top = FRAME_OUT - 0.04
        nail_y = FRAME_OUT + 0.5
        for sx in (-1.0, 1.0):
            d = _segment(X, Y, sx * 0.72 * FRAME_OUT, top, 0.0, nail_y)
            img = img + (wire - img) * _cover(d - 0.006, fw)[..., None]
        nail = torch.hypot(X, Y - nail_y)
        img = img + (gold_lut[400] - img) * _cover(nail - 0.022, fw)[..., None]
        # Brass plate under the frame.
        py = -(FRAME_OUT + 0.2)
        qx, qy = (X.abs() - 0.28).clamp(min=0), ((Y - py).abs() - 0.065).clamp(min=0)
        plate = torch.hypot(qx, qy) + torch.minimum(
            torch.maximum(X.abs() - 0.28, (Y - py).abs() - 0.065), torch.zeros_like(X)
        )
        brass = gold_lut[(330 + 120 * (Y - py) / 0.065).clamp(0, 511).long()]
        rim = _cover(-plate - 0.012, fw)[..., None]
        brass = brass * (0.7 + 0.3 * rim)
        img = img + (brass - img) * _cover(plate, fw)[..., None]
        # The frame: mouldings across q in [0, 1], a mitre on each diagonal.
        q = ((m - 1.0) / (FRAME_OUT - 1.0)).clamp(0.0, 1.0)
        bead = ((q - 0.06) / 0.38).clamp(0.0, 1.0)
        outer = ((q - 0.78) / 0.22).clamp(0.0, 1.0)
        slope = (
            1.7 * torch.cos(math.pi * bead) * ((q > 0.06) & (q < 0.44))
            + 0.25
            * torch.sin(2 * math.pi * (q - 0.44) / 0.34)
            * ((q >= 0.44) & (q < 0.78))
            + 1.4 * torch.cos(math.pi * outer) * (q >= 0.78)
            - 1.2 * (q <= 0.06)
        )
        horizontal = ay >= ax
        nx = torch.where(horizontal, torch.zeros_like(X), torch.sign(X))
        ny = torch.where(horizontal, torch.sign(Y), torch.zeros_like(Y))
        lambert = (-slope * (nx * lx + ny * ly) + lz) / torch.sqrt(1 + slope * slope)
        tone = (lambert.clamp(0.0, 1.0) ** 1.6) * 0.92 + 0.02
        tone = tone + 0.25 * lambert.clamp(0.0, 1.0) ** 18  # a little glint
        gold = gold_lut[(tone.clamp(0, 1) * 511).long()]
        mitre = ((ax - ay).abs() / 1.414 - 0.004) / fw
        gold = gold * (0.55 + 0.45 * mitre.clamp(0.0, 1.0))[..., None]
        inside = _cover(m - FRAME_OUT, fw)[..., None]
        img = img + (gold - img) * inside
        # Below a pixel, everything is the average colour.
        far = ((fw - 0.02) / 0.06).clamp(0.0, 1.0)[..., None]
        return img + (mean - img) * far

    def render(frame):
        t = frame / FRAME_COUNT  # loop: [0, 1)
        u_re = u_re0 + (LEVELS * t + SHIFT0) * L
        k = torch.floor((u_re + square) / L)
        radius = torch.exp(u_re - k * L)
        X, Y = radius * cos_u, radius * sin_u
        img = picture(X, Y, radius * foot0)
        img = F.avg_pool2d(img.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
        img = img * shade
        img = img + (text - img) * text_alpha
        return (img.clamp(0, 1) ** (1 / 1.0)) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = getattr(settings, "audio_tag", "")
    kind = tag if tag in ("swell", "lofi") else SOUND
    track = soundtrack(kind, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
