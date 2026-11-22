"""
Lines Into Circles

Every Möbius transformation of the complex plane is secretly a ball moving above it.
Paint the plane's grid on a glass ball, the Riemann sphere, put a lamp on top, and the
light throws the grid onto the floor. Slide, spin, lift or roll the ball, and the grid
on the floor moves exactly as the formula w = (az + b)/(cz + d) says.

Whenever a painted line passes through the lamp, its image is straight; every other
line lands as a perfect circle. So a Möbius transformation turns lines into circles and
circles into lines, nothing else. Once per loop the ball's point at infinity passes
under the lamp, and every circle straightens at once.

Listen: each line plays a note as it straightens, rushing upward into the chord.

Made with Python & PyTorch. Sound synthesized in NumPy.
#mobiustransformation #complexanalysis #riemannsphere #mathart #randomplots
"""

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from common import audio
from common.gpu import select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'How a ball turns lines into circles'
HOOK_Y = 230
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds, one loop
FRAME_COUNT = frame_count(DURATION)
TAU = 2.0 * math.pi

# The ball (radius 1) and its motion, all whole cycles per loop.
GRID = 0.5  # spacing of the painted grid, in floor units at rest
FLIPS = 1  # full forward rolls of the ball per loop
SPINS = 1  # turns about the vertical per loop
LIFT = (1.0, 1.75)  # lowest and highest centre height (1 = touching the floor)
LIFTS = 2  # rises per loop
WANDER = (0.45, 0.7)  # x, y amplitude of the centre's figure-eight
PHASE = 0.12  # loop time of frame 0: the all-straight moment sits at t = 1 - PHASE

# Camera
EYE = (0.0, -5.0, 5.2)
TARGET = (0.0, 1.2, 0.3)
VFOV = math.radians(50.0)
SS = 2  # supersampling
FOG = 9.0  # floor units: distant lines fade into the dark

# Look (linear light is built from these)
BACKGROUND = "#08060f"
RE_LINES = "#34e6b4"  # lines Re z = const: mint
IM_LINES = "#ff4f86"  # lines Im z = const: rose
LAMP = "#fff1d6"
GLASS = "#6c5cff"
LINE_PX = 1.15  # line half-width, in output pixels
AXIS_GAIN = 2.2  # the two axes are brighter
CHECKER = 0.06  # faint tint of alternate grid cells
DENSE = 0.25  # brightness of lines too dense to draw, relative to their mean

# Sound
STYLE = "vibes"  # common.music.STYLES; an --audio-only tag naming one overrides it
SOUND_LINES = 7  # lines |k| <= this play a note when their shadow straightens
RUSH_LINES = 36  # farther lines, up to this, play a quick rush into the chord
RUSH_GAP = 0.07  # s, closest two rush notes may be


def _rgb(colour: str):
    return [(int(colour[i : i + 2], 16) / 255.0) ** 2.2 for i in (1, 3, 5)]


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def motion(t: float, wander_phase, spin_sign: int):
    """Centre C (3,) and orientation R (3, 3, body -> world) of the ball at loop time t."""
    u = (t + PHASE) % 1.0
    cx = WANDER[0] * math.sin(TAU * u + wander_phase[0])
    cy = WANDER[1] * math.sin(2 * TAU * u + wander_phase[1])
    lo, hi = LIFT
    h = lo + (hi - lo) * 0.5 * (1.0 - math.cos(LIFTS * TAU * u))
    rot = _rot_x(FLIPS * TAU * u) @ _rot_z(spin_sign * SPINS * TAU * u)
    return np.array([cx, cy, h]), rot


def lamp_point(rot):
    """Where the lamp sits on the painted plane: the body point under world-up, as z."""
    q = rot.T @ np.array([0.0, 0.0, 1.0])
    return 2.0 * complex(q[0], q[1]) / max(1.0 - q[2], 1e-12)


class Renderer:
    def __init__(self, device):
        self.device = device
        f32 = dict(device=device, dtype=torch.float32)
        self.h, self.w = HEIGHT * SS, WIDTH * SS
        eye = np.array(EYE)
        fwd = np.array(TARGET) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.focal = (self.h / 2) / math.tan(VFOV / 2)
        ys = torch.arange(self.h, **f32) + 0.5 - self.h / 2
        xs = torch.arange(self.w, **f32) + 0.5 - self.w / 2
        u = (xs / self.focal)[None, :, None]
        v = (-ys / self.focal)[:, None, None]
        d = (
            torch.tensor(fwd, **f32)
            + u * torch.tensor(right, **f32)
            + v * torch.tensor(up, **f32)
        )
        self.dirs = d / d.norm(dim=-1, keepdim=True)  # (h, w, 3)
        self.eye = eye
        self.cam = (eye, fwd, right, up)
        self.px = xs[None, :]
        self.py = ys[:, None]
        # The floor: fixed, since the camera is.
        dz = self.dirs[..., 2]
        self.floor_ok = dz < -1e-4
        tf = torch.where(self.floor_ok, -eye[2] / dz.clamp(max=-1e-4), 0.0)
        self.floor_p = torch.tensor(eye, **f32) + tf[..., None] * self.dirs
        self.fog = torch.exp(-tf / FOG) * self.floor_ok
        self.bg = torch.tensor(_rgb(BACKGROUND), **f32)
        self.re_c = torch.tensor(_rgb(RE_LINES), **f32)
        self.im_c = torch.tensor(_rgb(IM_LINES), **f32)
        self.lamp_c = torch.tensor(_rgb(LAMP), **f32)
        self.glass_c = torch.tensor(_rgb(GLASS), **f32)
        # A faint sky gradient above the horizon so the floor has an edge.
        sky = (1.0 - (self.py / (self.h / 2)).clamp(-1, 1).abs())[..., None]
        self.sky = (0.6 + 0.4 * sky) * self.bg

    @staticmethod
    def _grad(f):
        """|grad f| in supersampled pixels (forward differences, edge repeated)."""
        gx = torch.diff(f, dim=1, append=f[:, -1:])
        gy = torch.diff(f, dim=0, append=f[-1:, :])
        return torch.sqrt(gx * gx + gy * gy).clamp_min(1e-6)

    def _lines(self, f):
        """Anti-aliased glow of the lines f = integer, fading to their mean where dense."""
        g = self._grad(f)
        df = (f - torch.round(f)).abs()
        width = LINE_PX * SS
        core = torch.exp(-((df / g / width) ** 2))
        spacing = 1.0 / g
        sharp = ((spacing - 3.0 * SS) / (8.0 * SS)).clamp(0, 1)
        mean = (1.77 * width * g).clamp(max=1.0)
        axis = torch.where(torch.round(f) == 0, AXIS_GAIN, 1.0)
        return core * sharp * axis + DENSE * mean * (1 - sharp), sharp

    def _pattern(self, qx, qy, qz):
        """The painted grid at body points q: (Re-lines, Im-lines, checker) intensities."""
        den = (1.0 - qz).clamp_min(1e-6)
        f = (2.0 / GRID) * qx / den
        g = (2.0 / GRID) * qy / den
        f = f.clamp(-1e5, 1e5)
        g = g.clamp(-1e5, 1e5)
        lf, sf = self._lines(f)
        lg, sg = self._lines(g)
        parity = torch.remainder(torch.floor(f) + torch.floor(g), 2.0)
        checker = parity * sf * sg
        return lf, lg, checker

    def frame(self, t, centre, rot):
        f32 = dict(device=self.device, dtype=torch.float32)
        C = torch.tensor(centre, **f32)
        Rt = torch.tensor(rot.T.copy(), **f32)  # world -> body
        lamp = C + torch.tensor([0.0, 0.0, 1.0], **f32)

        # Floor: each point P is lit through the ball point on the segment lamp -> P.
        d = self.floor_p - lamp
        d2 = (d * d).sum(-1).clamp_min(1e-9)
        s = -2.0 * d[..., 2] / d2
        rel = s[..., None] * d
        rel[..., 2] += 1.0
        q = rel @ Rt.T
        lf, lg, chk = self._pattern(q[..., 0], q[..., 1], q[..., 2])
        r = torch.sqrt(d2)
        pool = ((centre[2] + 1.0) / r) ** 3
        gain = (0.28 + 0.72 * pool**0.6) * self.fog
        floor = (
            self.bg * (1.0 + 1.0 * pool[..., None] * self.fog[..., None])
            + 0.05 * pool[..., None] * self.lamp_c * self.fog[..., None]
            + gain[..., None]
            * (
                0.85 * lf[..., None] * self.re_c
                + 0.85 * lg[..., None] * self.im_c
                + CHECKER * chk[..., None] * self.glass_c
            )
        )
        image = torch.where(self.floor_ok[..., None], floor, self.sky)

        # The glass ball: lines painted on it, front bright, back dim.
        oc = torch.tensor(self.eye, **f32) - C
        eye, fwd, right, up = self.cam
        rel_c = centre - eye
        depth_c = rel_c @ fwd
        cx = self.focal * (rel_c @ right) / depth_c + self.w / 2
        cy = -self.focal * (rel_c @ up) / depth_c + self.h / 2
        reach = 1.4 * self.focal / math.sqrt(max(rel_c @ rel_c - 1.0, 1e-3)) + 8
        y0, y1 = max(0, int(cy - reach)), min(self.h, int(cy + reach))
        x0, x1 = max(0, int(cx - reach)), min(self.w, int(cx + reach))
        box = image[y0:y1, x0:x1]
        dirs = self.dirs[y0:y1, x0:x1]
        b = dirs @ oc
        disc = b * b - (float(oc @ oc) - 1.0)
        hit = disc > 0
        root = torch.sqrt(disc.clamp_min(0))
        glass = torch.zeros_like(box)
        for sign, weight in ((-1.0, 1.0), (1.0, 0.33)):
            dist = -b + sign * root
            p = oc + dist[..., None] * dirs  # relative to C
            q = p @ Rt.T
            lf, lg, chk = self._pattern(q[..., 0], q[..., 1], q[..., 2])
            glass += weight * (
                lf[..., None] * self.re_c
                + lg[..., None] * self.im_c
                + 0.6 * CHECKER * chk[..., None] * self.glass_c
            )
            if sign < 0:
                n = p
                facing = (n * dirs).sum(-1).abs()
        rim = (1.0 - facing).clamp(0, 1) ** 3
        glass += 0.55 * rim[..., None] * self.glass_c + 0.03 * self.glass_c
        # Soft edge over one pixel so the silhouette is anti-aliased.
        edge = (disc.clamp_min(0).sqrt() * self.focal / (2.0 * float(oc.norm()))).clamp(
            0, 1
        )
        edge = edge * hit
        image[y0:y1, x0:x1] = (
            box * (1.0 - 0.45 * edge[..., None]) + edge[..., None] * glass
        )

        # The lamp on top.
        rel_l = lamp.cpu().numpy() - eye
        depth = rel_l @ fwd
        lx = self.focal * (rel_l @ right) / depth
        ly = -self.focal * (rel_l @ up) / depth
        r2 = (self.px - lx) ** 2 + (self.py - ly) ** 2
        glow = 3.0 * torch.exp(-r2 / (2 * (5.0 * SS) ** 2)) + 0.35 * torch.exp(
            -r2 / (2 * (28.0 * SS) ** 2)
        )
        image = image + glow[..., None] * self.lamp_c

        image = F.avg_pool2d(image.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
        image = image + 0.45 * wide_blur(image, 14.0, 4)
        image = 1.0 - torch.exp(-1.15 * image)  # soft shoulder
        return image.clamp(0, 1) ** (1 / 2.2) * 255.0  # RGB


def straight_events(wander_phase, spin_sign, samples=300_000):
    """
    When each painted line passes through the lamp (its shadow is straight).
    Returns [(seconds, family 0 = Re / 1 = Im, k)] for |k| <= SOUND_LINES, and the
    times every line is straight at once (the ball's north pole under the lamp).
    """
    ts = np.arange(samples) / samples
    u = (ts + PHASE) % 1.0
    a, b = FLIPS * TAU * u, spin_sign * SPINS * TAU * u
    # rot.T @ e3 for rot = Rx(a) Rz(b): Rz(-b) Rx(-a) e3 = Rz(-b) (0, sin a, cos a)
    qx = np.sin(b) * np.sin(a)
    qy = np.cos(b) * np.sin(a)
    qz = np.cos(a)
    den = np.maximum(1.0 - qz, 1e-12)
    events = []
    for family, val in enumerate((2 * qx / den / GRID, 2 * qy / den / GRID)):
        val = np.clip(val, -1e6, 1e6)
        lo = np.floor(np.minimum(val, np.roll(val, -1)))
        hi = np.floor(np.maximum(val, np.roll(val, -1)))
        for i in np.nonzero((hi > lo) & (np.abs(val) < 4 * RUSH_LINES))[0]:
            k = int(hi[i])
            if abs(k) <= RUSH_LINES:
                events.append((ts[i] * DURATION, family, k))
    straight = [((j / FLIPS - PHASE) % 1.0) * DURATION for j in range(FLIPS)]
    return sorted(events), straight


def soundtrack(events, straight, style_name, rng):
    """
    A loop. Every line |k| <= SOUND_LINES whose shadow straightens strikes the
    style's voice: the mint (Re) lines at degree |k|, the rose (Im) lines three
    degrees lower, panned by side. Farther lines (up to RUSH_LINES) straighten ever
    faster as the ball's north pole nears the lamp: they play a soft, rising rush
    (at most one note per RUSH_GAP) into the moment every shadow is straight, where
    a held chord and a bass note land, and fall away after it. A quiet pad holds the
    home chord underneath.
    """
    style = STYLES[style_name]
    root = int(rng.choice(style.roots))
    scale = Scale(root, style.mode)
    track = audio.silence(DURATION)
    hz = lambda midi: float(audio.midi_to_hz(midi))
    last = {}
    struck = 0
    for when, family, k in events:
        if any(
            abs(when - s) < 0.12 or abs(abs(when - s) - DURATION) < 0.12
            for s in straight
        ):
            continue
        side = 1 if family == 0 else -1
        if abs(k) > SOUND_LINES:
            if when - last.get("rush", -9) < RUSH_GAP:
                continue
            last["rush"] = when
            degree = SOUND_LINES + 1 + (abs(k) - SOUND_LINES) // 3
            note = style.voice(scale.hz(min(degree, 16)), 0.22)
            audio.add(track, audio.pan(note, 0.5 * side * np.sign(k)), when)
            continue
        key = (family, k)
        if when - last.get(key, -9) < 0.15:
            continue
        last[key] = when
        degree = abs(k) - (0 if family == 0 else 3)
        velocity = 0.6 - 0.03 * abs(k)
        note = style.voice(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, 0.1 * k * side), when)
        struck += 1
    for s in straight:
        for i, note in enumerate(scale.chord(0, 4) + [scale.note(0, 1)]):
            audio.add(track, style.voice(hz(note), 0.5), s + 0.02 * i)
        audio.add(track, style.bass(scale.hz(0, octave=-2), 0.7), s)
    for note in Scale(root, style.mode).chord(0, 3, octave=-1):
        audio.add(track, audio.pad(hz(note), DURATION, 0.12, 2.0, 2.0, 0.25), 0.0)
    logger.info(
        f"Soundtrack: {style_name}, root MIDI {root}, {struck} line notes, "
        f"{len(straight)} all-straight chords"
    )
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Möbius transformations as rigid motions of the Riemann sphere (after Arnold &
    Rogness, "Möbius Transformations Revealed"). A unit glass ball carries the square
    grid of the plane, painted by inverse stereographic projection from its own north
    pole: body point q <-> z = 2(q_x + i q_y)/(1 - q_z), so at rest (resting on the
    floor, north pole up) a lamp at the top projects the grid back onto the floor
    exactly. Moving the ball (centre C(t), rotation R(t)) and projecting from the
    lamp at C + e_z again gives a floor map z -> w that is always Möbius: horizontal
    slides are translations, spinning about the vertical is rotation, lifting is
    scaling, and tilting the pole away from the lamp is the inversion part. Painted
    lines through the lamp project to straight lines; all others to circles.

    Loop: t in [0, 1), u = t + PHASE. C = (WANDER_x sin(2πu + φ1), WANDER_y sin(4πu + φ2), h), h rising
    LIFTS times between LIFT; R = Rx(2π FLIPS u) Rz(±2π SPINS u). All whole cycles.

    Render: torch, per pixel at 2x supersampling. Camera rays are fixed; the floor
    point P maps to the ball point lamp + s (P - lamp), s = -2 (P - lamp)_z / |P - lamp|²,
    then to body coordinates and z. Grid lines are drawn as Gaussian glows of width
    LINE_PX using |grad| of Re z / GRID and Im z / GRID from finite differences, faded
    to their mean where the lines are denser than a few pixels (near the image of
    infinity). The ball is ray-traced as glass: front and back grids, a Fresnel rim.
    Bloom, a soft shoulder, gamma.

    Sound: the times each painted line |k| <= SOUND_LINES passes the lamp, from the
    same rotation, sampled finely; see `soundtrack`. Rendered with the default seed 0
    unless noted in the schedule.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    wander_phase = rng.uniform(0, TAU, 2)
    spin_sign = int(rng.choice((-1, 1)))

    device = select_device()
    renderer = Renderer(device)

    def render(frame):
        t = frame / FRAME_COUNT
        centre, rot = motion(t, wander_phase, spin_sign)
        with torch.no_grad():
            return renderer.frame(t, centre, rot)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    del renderer
    torch.cuda.empty_cache() if device.type == "cuda" else None

    events, straight = straight_events(wander_phase, spin_sign)
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(events, straight, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
