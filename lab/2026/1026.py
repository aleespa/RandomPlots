"""
You Can't Comb a Hairy Ball

The hairy ball theorem: you can never comb a hairy ball flat. However you
brush the fur on a sphere, a cowlick is left over. Here cowlicks come and
go, but never all of them.

At a whorl (pink) the hair swirls around a point; at a parting (teal) it
arrives from two sides and leaves by the other two. They are born and vanish
in pairs, one of each, so whorls minus partings never changes: on a sphere it
is always 2, its Euler characteristic.

Every cowlick on screen plays a note. When the last two whorls meet in one
double cowlick, the music comes home.

So somewhere on Earth, right now, the wind isn't blowing.

Made with Python, NumPy & PyTorch. Sound synthesized in NumPy.
#hairyballtheorem #topology #mathart #generativeart #randomplots
"""

import gc
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageFilter

from colors.palettes import TWLAGN
from common import audio
from common.gpu import select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 18.0
FRAME_COUNT = frame_count(DURATION)

# --- Cowlicks, as points of the stereographic plane (0 faces the camera) ------------
CENTRE = 0.02 + 0.05j  # the crown pair circles this point...
CROWN, CROWN_ANGLE = 0.36, 0.5  # ...this far out, and meets on it at t = 1/2
SEPARATION = 0.26  # the widest a transient whorl and parting get apart
# (birth, life, distance from CENTRE at birth and at death, angle off the crown
# axis, orientation of the pair, its turn over the life), times as loop fractions
PAIRS = [
    (0.78, 0.40, 0.42, 0.28, 1.75, 0.4, 1.4),
    (0.90, 0.38, 0.32, 0.45, -1.45, 2.2, -1.1),
    (0.10, 0.27, 0.26, 0.42, 1.25, 1.1, 0.9),
    (0.58, 0.30, 0.36, 0.22, -1.30, 0.6, 1.6),
    (0.66, 0.30, 0.46, 0.36, 2.15, 2.7, -1.2),
]
SWIRL, BRUSH = 1.05, 0.6  # rad: every hair turned by SWIRL + BRUSH sin(2 pi t)

# --- Fur -------------------------------------------------------------------------------
HAIRS = 110_000
HAIR_LENGTH = 0.085  # in sphere radii
STEPS, SAMPLES = 6, 6  # field evaluations along a hair, splatted points per step
FLAT, STAND = 0.16, 1.25  # rad: a hair's lift off the skin, extra lift in a whorl
TUFT = 0.075  # chord radius of a whorl's tuft
DROOP = 0.55  # a hair bends back towards the skin by this fraction of its lift
JITTER = 0.18  # rad: each hair's own deviation from the field
CULL = -0.3  # hairs rooted further behind the limb than this are skipped

# --- Camera & light ----------------------------------------------------------------
RADIUS_PX = 460
CENTRE_X, CENTRE_Y = WIDTH / 2, 1010
YAW, PITCH, TILT = 0.30, 0.14, -0.10  # rad: wobble amplitudes and a constant tilt
LIGHT = np.array([-0.45, 0.55, 0.70])  # in view space, towards the light

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#07060b"
BODY = "#1c1528"
FUR = TWLAGN + TWLAGN[:1]  # a cyclic gradient, coloured by hair direction
SHEEN_COLOUR = "#ffeef4"
HAIR_LIGHT, AMBIENT, DIFFUSE = 0.42, 0.2, 0.9
SHEEN, SHINE = 0.55, 40.0  # Kajiya-Kay highlight strength and sharpness
EXPOSURE = 1.6
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 12.0, 4, 0.35
WHORL_COLOUR, PARTING_COLOUR = "#f6b8de", "#6ee0e4"
RING_RADIUS, RING_WIDTH, RING_GAIN = 34.0, 2.2, 0.9  # px
RING_SHADE = 0.75  # a dark band under each ring, 2.5x as wide, darkens the fur
RING_POP = 0.1  # a transient pair's rings are at full strength this far apart
# (as a fraction of SEPARATION)

# --- Tally, typeset by LaTeX (needs latex and dvipng on PATH, e.g. MiKTeX) ---------
TEXT_TOP, TEXT_HEIGHT = 200, 240  # px: the band the tally is centred in
TEXT_DPI = 450  # the size: 10 pt type at this resolution
TEXT_COLOUR, DIM_COLOUR = "#f2efe8", "#b9bcc8"
TALLY = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{xcolor}
\definecolor{whorl}{HTML}{%s}
\definecolor{parting}{HTML}{%s}
\definecolor{ink}{HTML}{%s}
\definecolor{dim}{HTML}{%s}
\pagestyle{empty}
\begin{document}
\color{ink}$\displaystyle
\underset{\textcolor{dim}{\text{whorls}}}{\textcolor{whorl}{%d}}
-\underset{\textcolor{dim}{\text{partings}}}{\textcolor{parting}{%d}}
=\underset{\textcolor{dim}{\text{always}}}{2}$
\end{document}
"""

# --- Sound ---------------------------------------------------------------------------
STYLE = "handpan"  # common.music.STYLES; an --audio-only tag naming one overrides it
BARS = 6  # the loop is BARS bars of 4/4: 80 BPM at 18 s
PROGRESSION = (0, 3, 4, 0, 5, 4)  # a chord a bar; the home chord as the crowns meet
SLOTS = 8  # notes a bar at most: one per cowlick on screen, left to right
GROOVE = ("lofi_keys",)  # styles that also get a soft kick and shaker


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def cowlicks(t: float):
    """
    The zeros (whorls) and poles (partings) of the field at loop time t, as
    lists of (point of the plane, strength in [0, 1]).
    """
    axis = CROWN_ANGLE + np.pi * t
    a = CROWN * np.cos(np.pi * t) * np.exp(1j * axis)
    whorls, partings = [(CENTRE + a, 1.0), (CENTRE - a, 1.0)], []
    for pair in PAIRS:
        birth, life, angle, turn = pair[0], pair[1], pair[5], pair[6]
        u = ((t - birth) % 1.0) / life
        if not 0 < u < 1:
            continue
        gap = SEPARATION * np.sin(np.pi * u) ** 2
        half = 0.5 * gap * np.exp(1j * (angle + turn * u))
        strength = float(smoothstep(gap / (RING_POP * SEPARATION)))
        whorls.append((pair_centre(pair, u) + half, strength))
        partings.append((pair_centre(pair, u) - half, strength))
    return whorls, partings


def pair_centre(pair, u: float) -> complex:
    """Where a transient pair is centred at the fraction u of its life."""
    birth, life, start, end, side = pair[:5]
    # The crown axis turns by pi a loop, so measured at the pair's own
    # (unwrapped) time it is the same line on both sides of the seam.
    return CENTRE + (start + (end - start) * smoothstep(u)) * np.exp(
        1j * (CROWN_ANGLE + np.pi * (birth + u * life) + side)
    )


def to_sphere(z: complex) -> np.ndarray:
    """Inverse stereographic projection from the back pole (0, 0, -1)."""
    r2 = abs(z) ** 2
    return np.array([2 * z.real, 2 * z.imag, 1 - r2]) / (1 + r2)


def rotation(t: float) -> np.ndarray:
    """World to view: a gentle closed wobble of the ball."""
    yaw = YAW * np.sin(2 * np.pi * t)
    pitch = TILT + PITCH * np.sin(4 * np.pi * t + 0.7)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    turn_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    turn_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return turn_x @ turn_y


class Tally:
    """whorls - partings = 2, each number labelled underneath, typeset by LaTeX; cached per count."""

    def __init__(self):
        self.cache = {}

    def draw(self, whorls: int, partings: int) -> np.ndarray:
        """An RGBA float image (TEXT_HEIGHT, WIDTH, 4), premultiplied, with a dark halo."""
        key = whorls, partings
        if key in self.cache:
            return self.cache[key]
        colours = (WHORL_COLOUR, PARTING_COLOUR, TEXT_COLOUR, DIM_COLOUR)
        source = TALLY % (*(c[1:].upper() for c in colours), whorls, partings)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "tally.tex").write_text(source)
            for command in (
                ["latex", "-interaction=nonstopmode", "-halt-on-error", "tally.tex"],
                ["dvipng", "-D", str(TEXT_DPI), "-T", "tight", "-bg", "Transparent"]
                + ["--truecolor", "-o", "tally.png", "tally.dvi"],
            ):
                subprocess.run(command, cwd=folder, check=True, capture_output=True)
            glyphs = np.asarray(
                Image.open(Path(folder) / "tally.png").convert("RGBA"), np.float32
            )
        h, w = glyphs.shape[:2]
        top, left = (TEXT_HEIGHT - h) // 2, (WIDTH - w) // 2
        text = np.zeros((TEXT_HEIGHT, WIDTH, 4), np.float32)
        text[top : top + h, left : left + w] = glyphs / 255

        ink = Image.fromarray((255 * text[..., 3]).astype(np.uint8))
        halo = np.asarray(ink.filter(ImageFilter.GaussianBlur(12)), np.float32) / 255
        shade = 0.8 * np.minimum(1.0, 2.5 * halo)
        alpha = text[..., 3:] + (1 - text[..., 3:]) * shade[..., None]
        rgb = text[..., :3] * text[..., 3:] + (1 - text[..., 3:]) * shade[
            ..., None
        ] * _rgb(BACKGROUND)
        self.cache[key] = np.concatenate([rgb, alpha], -1)
        return self.cache[key]


def on_screen(t: float, z: complex):
    """Screen x in [-1, 1] (the ball's width) and depth (> 0 facing us) of a plane point."""
    p = rotation(t) @ to_sphere(z)
    return float(p[0]), float(p[2])


def soundtrack(style_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    A loop of BARS bars, one chord each (PROGRESSION), played by the cowlicks.

    - Bass and a quiet pad hold each bar's chord (plus a soft kick and
      shaker for the GROOVE styles).
    - The arpeggio: at each of SLOTS eighth notes in a bar, the cowlicks
      facing the camera, sorted left to right, play in turn: whorls climb the
      chord (root, third, fifth, ...), partings sit a step above a chord tone,
      an octave up, so tension grows with the number of pairs. A bar with
      fewer cowlicks than slots leaves the rest silent. Panned by screen x,
      quieter for a newborn pair.
    - A pair is born with its two notes struck together (a soft second) and
      dies on a single note an octave over the chord's root (the unison).
    - At t = 1/2 the crown pair meets in its double zero: the low root and
      the home chord rolled.
    Everything is circular; onsets come from the same closed forms as the
    frames.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    bar = DURATION / BARS
    slot = bar / SLOTS

    def chord_at(seconds):
        return PROGRESSION[int(seconds // bar) % BARS]

    for b, degree in enumerate(PROGRESSION):
        bass = scale.hz(degree, -2)
        bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
        audio.add(track, style.bass(bass, 0.5, length=bar + 1.0), b * bar)
        for note in scale.chord(degree, 3, octave=-1):
            pad = audio.pad(
                float(audio.midi_to_hz(note)), bar, 0.18, 0.8, 1.2, brightness=0.25
            )
            audio.add(track, pad, b * bar)
        if style_name in GROOVE:
            for beat in range(4):
                when = b * bar + beat * bar / 4
                if beat % 2 == 0:
                    audio.add(track, audio.kick(0.35), when)
                audio.add(track, audio.tick(rng, 0.12, 0.06), when + bar / 8)

    notes = 0
    for k in range(BARS * SLOTS):
        seconds = k * slot
        t = seconds / DURATION
        whorls, partings = cowlicks(t)
        voices = []
        for kind, marks in ((0, whorls), (1, partings)):
            for z, strength in marks:
                x, depth = on_screen(t, z)
                if depth > 0.1:
                    voices.append((x, kind, strength))
        voices.sort()
        j = k % SLOTS
        if j >= len(voices):
            continue
        x, kind, strength = voices[j]
        rank = sum(1 for v in voices[:j] if v[1] == kind)
        degree = chord_at(seconds) + 2 * rank + kind
        note = style.voice(scale.hz(degree, kind), 0.3 + 0.2 * strength)
        audio.add(track, audio.pan(note, 0.8 * x), seconds)
        notes += 1

    for pair in PAIRS:
        birth, life = pair[:2]
        for when, born in ((birth, True), ((birth + life) % 1.0, False)):
            seconds = when * DURATION
            x = on_screen(when, pair_centre(pair, 0.0 if born else 1.0))[0]
            degree = chord_at(seconds)
            if born:
                for step in (4, 5):
                    note = style.voice(scale.hz(degree + step), 0.32)
                    audio.add(track, audio.pan(note, 0.8 * x), seconds)
            else:
                note = style.voice(scale.hz(degree, 1), 0.36)
                audio.add(track, audio.pan(note, 0.8 * x), seconds)

    meet = DURATION / 2
    audio.add(track, style.bass(scale.hz(0, -1), 0.45, length=4.0), meet)
    for i, degree in enumerate((0, 2, 4, 7)):
        note = style.voice(scale.hz(degree), 0.36)
        audio.add(track, audio.pan(note, 0.3 * (i - 1.5)), meet + 0.06 * i)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} cowlick notes"
    )
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    The hairy ball theorem (Poincare-Hopf on the sphere), as fur combed by a
    tangent field whose zeros are placed by hand.

    The field lives on the stereographic plane z = (X + iY) / (1 + Z) (the
    pole facing the camera at 0, the far pole at infinity):
        V(z) = e^{i(SWIRL + BRUSH sin 2 pi t)} prod (z - a_i) / prod (z - b_j),
    pushed forward to the sphere by the (conformal) inverse projection. A
    zero a_i is an index +1 singularity (a whorl, spiralling by the turn), a
    pole b_j an index -1 one (a parting, saddle-shaped). There are always two
    more zeros than poles, so at the far pole V ~ z^2 and the pushed-forward
    field is smooth and non-zero there: the singularities on screen are all
    of them, and their indices sum to 2 = chi(S^2) by construction. Only the
    direction is used, computed as prod (z - a_i) prod conj(z - b_j), which
    has the argument of V without dividing by zero.

    Two zeros, the crown pair, sit at CENTRE +- CROWN cos(pi t) e^{i(.. + pi t)}:
    their product depends on a^2, so the pair closes as a set although each
    point swaps place, and they meet in a double zero (index 2) at t = 1/2.
    Each of PAIRS adds a zero and a pole that are born at one point, drift
    apart (up to SEPARATION, sin^2 in their life), and meet again, where the
    factors (z - a)/(z - b) cancel exactly: the tally changes only by these
    pair births and deaths. Every motion is periodic in t: an exact loop.
    The seed only places the hairs; the posted render uses the default --seed 0.

    Fur: HAIRS roots uniform on the unit sphere (settings.rng), each traced
    STEPS steps along the field (with its own JITTER rotation), lifted off the
    skin at FLAT, or up to FLAT + STAND inside a whorl's tuft, drooping back.
    Seen in a wobbling orthographic view, hairs behind the ball are dropped;
    the rest are splatted on the GPU as SAMPLES points per step, shaded with
    Kajiya-Kay (diffuse sqrt(1 - (T.L)^2), highlight sqrt(1 - (T.H)^2)^SHINE),
    darker at the root and on the ball's unlit side, coloured by the field's
    direction in the plane (FUR, cyclic), so the colours run once around a
    whorl and once backwards around a parting. Tone-mapped (1 - exp), bloomed,
    over a dark shaded ball. Rings mark the singularities (pink whorls, teal
    partings), growing with a transient pair's separation; the tally reads
    whorls - partings = 2, typeset by LaTeX once per count (`Tally`).

    Sound (`soundtrack`, STYLE handpan): the cowlicks on screen play the
    arpeggio over a six-bar progression, pair births and deaths are struck,
    and the home chord lands as the crown pair meets.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()
    to = dict(device=device, dtype=torch.float32)

    roots = rng.normal(size=(HAIRS, 3))
    roots /= np.linalg.norm(roots, axis=1, keepdims=True)
    roots = torch.tensor(roots, **to)
    lengths = torch.tensor(rng.uniform(0.75, 1.2, HAIRS), **to) * HAIR_LENGTH
    jitter = torch.polar(
        torch.ones(HAIRS, **to), torch.tensor(rng.normal(0, JITTER, HAIRS), **to)
    )
    brightness = torch.tensor(rng.uniform(0.6, 1.25, HAIRS), **to)

    light = torch.tensor(LIGHT / np.linalg.norm(LIGHT), **to)
    half = light + torch.tensor([0.0, 0.0, 1.0], **to)
    half = half / half.norm()
    fur = torch.tensor(palette_lut(FUR, 512), **to)
    sheen_rgb = torch.tensor(_rgb(SHEEN_COLOUR), **to)
    background = torch.tensor(_rgb(BACKGROUND), **to)
    frac = (torch.arange(SAMPLES, **to) + 0.5) / SAMPLES
    along = (torch.arange(STEPS, **to)[:, None] + frac[None, :]) / STEPS

    # The ball's skin, lit from LIGHT, and pixel coordinates for the rings.
    ys = (torch.arange(HEIGHT, **to)[:, None] + 0.5 - CENTRE_Y) / -RADIUS_PX
    xs = (torch.arange(WIDTH, **to)[None, :] + 0.5 - CENTRE_X) / RADIUS_PX
    r2 = xs * xs + ys * ys
    nz = torch.sqrt((1 - r2).clamp_min(0))
    lambert = (xs * light[0] + ys * light[1] + nz * light[2]).clamp_min(0)
    edge = ((1 - torch.sqrt(r2)) * RADIUS_PX + 0.5).clamp(0, 1)
    skin = _rgb(BODY)
    base = background + edge[..., None] * (
        torch.tensor(skin, **to) * (0.35 + 0.9 * lambert)[..., None] - background
    )
    px_x = torch.arange(WIDTH, **to)[None, :] + 0.5
    px_y = torch.arange(HEIGHT, **to)[:, None] + 0.5
    whorl_rgb = torch.tensor(_rgb(WHORL_COLOUR), **to)
    parting_rgb = torch.tensor(_rgb(PARTING_COLOUR), **to)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **to)
    tally = Tally()

    def field(q, whorls, partings, turn):
        """Unit tangent directions at sphere points q, and the field's unit value in the plane."""
        x, y, w = q.unbind(-1)
        z = torch.complex(x, y) / (1 + w).clamp_min(1e-4)
        v = turn
        for a, _ in whorls:
            v = v * (z - complex(a))
        for b, _ in partings:
            v = v * torch.conj(z - complex(b))
        v = v / v.abs().clamp_min(1e-30)
        x, y = z.real, z.imag
        r2 = x * x + y * y
        e = torch.stack([1 + r2 - 2 * x * x, -2 * x * y, -2 * x], -1)
        f = torch.stack([-2 * x * y, 1 + r2 - 2 * y * y, -2 * y], -1)
        tangent = v.real[:, None] * e + v.imag[:, None] * f
        return F.normalize(tangent, dim=-1), v

    def render(frame: int) -> torch.Tensor:
        t = frame / FRAME_COUNT
        whorls, partings = cowlicks(t)
        turn_np = rotation(t)
        turn = torch.tensor(turn_np, **to)
        normals = roots @ turn.T
        keep = normals[:, 2] > CULL
        q, normals = roots[keep], normals[keep]
        n = q.shape[0]

        # Tufts: hairs stand up near a whorl.
        stand = torch.zeros(n, **to)
        for a, strength in whorls:
            p = torch.tensor(to_sphere(a), **to)
            stand = stand + strength * torch.exp(-((q - p) ** 2).sum(-1) / TUFT**2)
        lift = FLAT + STAND * stand.clamp(max=1.0)

        spin = np.exp(1j * (SWIRL + BRUSH * np.sin(2 * np.pi * t)))
        spin = jitter[keep] * complex(spin)
        step = lengths[keep] / STEPS
        verts, height, value = [q], torch.zeros(n, **to), None
        for j in range(STEPS):
            direction, v = field(q, whorls, partings, spin)
            if value is None:
                value = v
            angle = lift * (1 - DROOP * (j + 0.5) / STEPS)
            q = F.normalize(q + (step * torch.cos(angle))[:, None] * direction, dim=-1)
            height = height + step * torch.sin(angle)
            verts.append(q * (1 + height)[:, None])
        view = torch.stack(verts, 1) @ turn.T  # (n, STEPS + 1, 3)

        # Points along each hair, and their shading.
        start, segment = view[:, :-1], view[:, 1:] - view[:, :-1]
        points = start[:, :, None] + segment[:, :, None] * frac[None, None, :, None]
        tangent = F.normalize(segment, dim=-1)
        tl, th = tangent @ light, tangent @ half
        diffuse = torch.sqrt((1 - tl * tl).clamp_min(0))
        highlight = torch.sqrt((1 - th * th).clamp_min(0)) ** SHINE
        shadow = (0.15 + 0.85 * (normals @ light)).clamp(0.04, 1.0) * brightness[keep]
        hue = (torch.angle(value) / (2 * np.pi)) % 1.0
        colour = fur[(hue * (len(fur) - 1)).long()]  # (n, 3)
        screen_len = RADIUS_PX * segment[..., :2].norm(dim=-1) / SAMPLES  # (n, STEPS)
        weight = HAIR_LIGHT * screen_len * shadow[:, None]
        ao = 0.15 + 0.85 * along  # (STEPS, SAMPLES)
        rgb = (
            colour[:, None, None, :]
            * (AMBIENT + DIFFUSE * diffuse)[:, :, None, None]
            * ao[None, :, :, None]
            + SHEEN * (highlight[:, :, None] * along[None])[..., None] * sheen_rgb
        ) * weight[:, :, None, None]

        px, py, pz = points.unbind(-1)
        hidden = (px * px + py * py < 1) & (pz < 0)
        rgb = rgb * (~hidden)[..., None]
        canvas.zero_()
        splat(
            canvas,
            CENTRE_X + RADIUS_PX * px,
            CENTRE_Y - RADIUS_PX * py,
            rgb.reshape(-1, 3),
        )

        light_img = 1 - torch.exp(-EXPOSURE * canvas)
        image = base * (1 - light_img) + light_img
        image = image + BLOOM_GAIN * wide_blur(light_img, BLOOM_SIGMA, BLOOM_FACTOR)

        # Rings around the singularities on the visible side, on a dark band.
        dark, glow = torch.zeros_like(image[..., 0]), torch.zeros_like(image)
        for marks, ring_rgb in ((whorls, whorl_rgb), (partings, parting_rgb)):
            for a, strength in marks:
                p = turn_np @ to_sphere(a)
                if p[2] <= 0:
                    continue
                cx, cy = CENTRE_X + RADIUS_PX * p[0], CENTRE_Y - RADIUS_PX * p[1]
                radius = RING_RADIUS * (0.5 + 0.5 * strength)
                d = torch.sqrt((px_x - cx) ** 2 + (px_y - cy) ** 2)
                gain = strength * float(smoothstep(p[2] / 0.3))
                ring = torch.exp(-(((d - radius) / RING_WIDTH) ** 2))
                band = torch.exp(-(((d - radius) / (2.5 * RING_WIDTH)) ** 2))
                dark = torch.maximum(dark, gain * band)
                glow = glow + RING_GAIN * gain * ring[..., None] * ring_rgb
        image = image * (1 - RING_SHADE * dark)[..., None] + glow

        text = torch.tensor(tally.draw(len(whorls), len(partings)), **to)
        band = image[TEXT_TOP : TEXT_TOP + TEXT_HEIGHT]
        band.mul_(1 - text[..., 3:]).add_(text[..., :3])
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    track = soundtrack(tag if tag in STYLES else STYLE, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
