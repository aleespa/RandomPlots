"""
Un-Stirring Ink

Five drops of ink in thick syrup, between two glass walls. The inner cylinder turns ten
times and the drops are stirred into fine spirals until the ink seems mixed for good. Then
it turns back ten times, and every drop comes home, round again.

In syrup, viscosity beats inertia. The flow has no memory of speed, only of how far the wall
has turned: each ring of fluid rotates by ω(r)·φ, with ω(r) ∝ 1/r² − 1/R². Turn the wall
back by φ and every bit of ink retraces its path exactly. G. I. Taylor filmed this in the 1960s
with dyed corn syrup. The ink never mixed: it was sheared into layers too thin to see.

Each drop strikes its own piano note every time it comes round. Wound back, the tune plays in
reverse, and the chord clears as the drops come home.

Made with Python & PyTorch. Sound synthesized in NumPy.
#taylorcouette #fluiddynamics #reversibility #mathart #randomplots
"""

import gc
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.special import betainc

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
TAU = 2 * np.pi

# Apparatus: radii in units of the outer wall.
R_INNER = 0.40
R_PX = 500.0  # outer wall radius on screen
CENTRE = (WIDTH / 2, HEIGHT * 0.535)
SHEAR = R_INNER**2 / (1 - R_INNER**2)  # omega(r) / Omega = SHEAR * (1/r^2 - 1)

# The crank: rest, wind TURNS forward, hold, unwind (the mirror image), rest.
TURNS = 10
CRANK_SHAPE = (3.2, 2.0)  # beta(a, b) speed profile: a > b lingers in the first turns
WIND = (0.15, 6.55)  # seconds
UNWIND = (8.05, 14.45)

# Ink. Drops: (radius, angle) are jittered by the seed.
DROP_RADII = (0.50, 0.59, 0.68, 0.78, 0.88)
DROP_SIZE = 0.062  # radius, outer radii
DROP_EDGE = 0.010  # softness of the rim
DROP_DENSITY = 1.5  # optical density at the centre, in units of the ink colour
SPOKE_WIDTH = 0.0045  # half-width of the line of ink across the gap
SPOKE_DENSITY = 1.4
INKS = ("#c8102e", "#1f3fb8", "#0b8a68", "#f0a400", "#7a2aa8")  # at density 1
SPOKE_INK = "#20202c"
SHUTTER = 0.5  # motion blur: fraction of a frame the shutter is open

# Polar texture of the ink at the start, with angular blur levels.
TEX_R, TEX_THETA = 640, 4096
LEVELS = 13  # blur sigma of level k: 2^k texels (level 0 unblurred)

# Light
BACKGROUND = "#0a0908"
SYRUP_CENTRE = "#fbf3e2"  # backlit syrup, brightest in the middle
SYRUP_EDGE = "#e8d3a8"
STEEL = ("#15171c", "#3a3f48", "#8c929c")  # inner cylinder: dark, body, highlight
NOTCHES = 6

# Turn counter above the apparatus
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf"
COUNTER_Y = 300  # px, centre of the number
COUNTER_SIZES = (118, 40)  # number, label
COUNTER_COLOUR = "#e9dcc0"

# Sound
STYLE = "felt_piano"  # common.music.STYLES; an --audio-only tag naming one overrides it
DROP_DEGREES = (7, 5, 4, 2, 0)  # innermost drop (fastest) highest


def ease(x):
    """
    Regularized incomplete beta I_x(CRANK_SHAPE): the crank's speed grows like
    x^(a-1) and dies like (1-x)^(b-1), so it starts moving at once but gently,
    lingers in the first few turns, and comes to rest smoothly.
    """
    return betainc(*CRANK_SHAPE, np.clip(x, 0.0, 1.0))


def crank(seconds):
    """
    Angle the inner cylinder has turned (radians) at video time `seconds`:
    eased, slow at first while the drops are still drops; the unwinding is the
    exact mirror image in time.
    """
    s = np.asarray(seconds, float)
    up = ease((s - WIND[0]) / (WIND[1] - WIND[0]))
    back = ease((UNWIND[1] - s) / (UNWIND[1] - UNWIND[0]))
    return TAU * TURNS * np.where(s < UNWIND[0], up, back)


def crank_rate(seconds, h=1e-3):
    return (crank(seconds + h) - crank(seconds - h)) / (2 * h)


def omega(r):
    """Couette flow: angular velocity of the ring at r per unit inner-wall rotation."""
    return SHEAR * (1.0 / r**2 - 1.0)


def drops(rng):
    """(radius, angle, ink, shape harmonics) per drop, and the spoke's angle."""
    order = rng.permutation(len(INKS))
    start = rng.uniform(0, TAU)
    out = []
    for i, r in enumerate(DROP_RADII):
        angle = start + TAU * (0.21 * i + 0.5 * (i % 2)) + rng.normal(0, 0.12)
        wobble = rng.normal(0, 0.035, 3), rng.uniform(0, TAU, 3)
        out.append((r + rng.normal(0, 0.008), angle, INKS[order[i]], wobble))
    spoke = start + TAU * 0.11
    return out, spoke


def _rgb(colour):
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def _absorbance(ink):
    """Beer-Lambert absorbance per channel that turns the syrup into `ink`."""
    return -np.log(np.clip(_rgb(ink) / _rgb(SYRUP_CENTRE), 0.02, 1))


def ink_texture(drop_list, spoke, device):
    """
    Optical density (RGB absorbance) of the ink at the start, on a polar grid
    r in [R_INNER, 1] x theta0 in [-pi, pi), then LEVELS copies blurred along
    theta by Gaussians of 2^k texels (exact on the circle, via FFT): the
    pixel footprint picks the level, so sub-pixel threads average correctly.
    Returned as (1, 3, LEVELS, TEX_R, TEX_THETA + 2), wrapped one column
    each side for grid_sample.
    """
    f = dict(device=device, dtype=torch.float32)
    r = torch.linspace(R_INNER, 1.0, TEX_R, **f)[:, None]
    theta = (-np.pi + (torch.arange(TEX_THETA, **f) + 0.5) * TAU / TEX_THETA)[None]
    x, y = r * torch.cos(theta), r * torch.sin(theta)
    density = torch.zeros(3, TEX_R, TEX_THETA, **f)
    for r0, a0, ink, (amp, phase) in drop_list:
        dx, dy = x - r0 * np.cos(a0), y - r0 * np.sin(a0)
        dist, angle = torch.sqrt(dx * dx + dy * dy), torch.atan2(dy, dx)
        wobble = sum(
            a * torch.cos((k + 2) * angle + p)
            for k, (a, p) in enumerate(zip(amp, phase))
        )
        edge = DROP_SIZE * (1 + wobble)
        inside = torch.sigmoid((edge - dist) / (0.5 * DROP_EDGE))
        rim = 0.35 * torch.exp(-(((dist - 0.85 * edge) / (0.12 * DROP_SIZE)) ** 2))
        d = DROP_DENSITY * inside * (1 + rim)
        density += torch.tensor(_absorbance(ink), **f)[:, None, None] * d
    # The spoke: a straight line of ink right across the gap.
    across = r * torch.sin(theta - spoke)
    along = r * torch.cos(theta - spoke)
    line = torch.sigmoid((SPOKE_WIDTH - across.abs()) / 0.0015)
    ends = torch.sigmoid((along - R_INNER - 0.012) / 0.004) * torch.sigmoid(
        (0.988 - along) / 0.004
    )
    density += torch.tensor(_absorbance(SPOKE_INK), **f)[:, None, None] * (
        SPOKE_DENSITY * line * ends
    )

    spectrum = torch.fft.rfft(density, dim=-1)
    n = torch.arange(spectrum.shape[-1], **f)
    levels = [density]
    for k in range(1, LEVELS):
        sigma = (2.0**k) * TAU / TEX_THETA
        gain = torch.exp(-0.5 * (n * sigma) ** 2)
        levels.append(torch.fft.irfft(spectrum * gain, n=TEX_THETA, dim=-1))
    tex = torch.stack(levels, 1)  # (3, LEVELS, TEX_R, TEX_THETA)
    tex = torch.cat((tex[..., -1:], tex, tex[..., :1]), -1)
    return tex[None].contiguous()


class Renderer:
    """Per pixel, closed form: back-rotate each pixel to the start, sample the ink."""

    def __init__(self, device, rng):
        self.device = device
        f = dict(device=device, dtype=torch.float32)
        self.f = f
        ys = (torch.arange(HEIGHT, **f) + 0.5 - CENTRE[1]) / R_PX
        xs = (torch.arange(WIDTH, **f) + 0.5 - CENTRE[0]) / R_PX
        y, x = torch.meshgrid(-ys, xs, indexing="ij")  # y up
        self.r = torch.sqrt(x * x + y * y)
        self.theta = torch.atan2(y, x)
        rc = self.r.clamp(R_INNER, 1.0)
        self.omega = SHEAR * (1 / rc**2 - 1)
        self.domega = 2 * SHEAR / rc**3
        self.s_pixel2 = (0.4 / (R_PX * rc)) ** 2
        self.grid_r = 2 * (rc - R_INNER) / (1 - R_INNER) - 1

        # Masks with analytic anti-aliasing.
        cover = lambda edge: ((self.r - edge) * R_PX + 0.5).clamp(0, 1)
        self.syrup = cover(R_INNER) * (1 - cover(1.0))
        self.steel = 1 - cover(R_INNER)
        self.glass = cover(1.0) * (1 - cover(1.035))

        # Backlight: bright centre, amber edge, shadows at both walls, a soft key.
        g = ((self.r - R_INNER) / (1 - R_INNER)).clamp(0, 1)[..., None]
        light = torch.tensor(_rgb(SYRUP_CENTRE), **f) * (1 - g**1.6) + torch.tensor(
            _rgb(SYRUP_EDGE), **f
        ) * (g**1.6)
        shade = (
            1
            - 0.22 * torch.exp(-(self.r - R_INNER) / 0.035)
            - 0.18 * torch.exp(-(1 - self.r) / 0.03)
        )
        key = 1 + 0.05 * (-x * 0.6 + y * 0.8)
        self.light = light * (shade * key)[..., None]

        # Background with the light spilling past the glass; the glass rim.
        bg = torch.tensor(_rgb(BACKGROUND), **f)
        spill = torch.exp(-((self.r - 1).clamp_min(0) / 0.09)) * 0.10
        self.background = bg + spill[..., None] * torch.tensor(_rgb(SYRUP_EDGE), **f)
        rim = torch.exp(-(((self.r - 1.012) / 0.006) ** 2))
        rim_hi = rim * (0.55 + 0.45 * torch.clamp(-x * 0.5 + y * 0.85, -1, 1))
        self.glass_colour = (
            self.background * 0.6
            + torch.tensor(_rgb("#d9e2e6"), **f) * (0.18 + 0.7 * rim_hi)[..., None]
        )

        # Steel: radial ramp, a lit edge, a dark hub.
        dark, body, hi = (torch.tensor(_rgb(c), **f) for c in STEEL)
        rr = (self.r / R_INNER).clamp(0, 1)[..., None]
        lit = 0.5 + 0.5 * torch.clamp(-x * 0.5 + y * 0.85, -1, 1) / R_INNER
        lit = lit.clamp(0, 1)[..., None]
        steel = dark + (body - dark) * (0.35 + 0.65 * lit) * (1 - rr**6)
        edge = torch.exp(-(((self.r - R_INNER * 0.985) / 0.006) ** 2))[..., None]
        steel = steel + (hi - body) * edge * lit
        hub = torch.exp(-((self.r / 0.05) ** 2))[..., None]
        self.steel_colour = steel * (1 - 0.6 * hub)
        self.hi = hi
        self.notch_band = torch.exp(-(((self.r - 0.86 * R_INNER) / 0.02) ** 2))

        # Ink.
        self.drops, spoke = drops(rng)
        self.tex = ink_texture(self.drops, spoke, device)

        # Counter text, cached per string.
        self.fonts = [ImageFont.truetype(str(FONT), size) for size in COUNTER_SIZES]
        self.counter_colour = torch.tensor(_rgb(COUNTER_COLOUR), **f)
        self.texts = {}

    def counter_alpha(self, number, label):
        key = number, label
        if key not in self.texts:
            height = 2 * COUNTER_SIZES[0]
            canvas = Image.new("L", (WIDTH, height), 0)
            draw = ImageDraw.Draw(canvas)
            big, small = self.fonts
            draw.text((WIDTH / 2, 0.42 * height), number, 255, big, "ms")
            draw.text((WIDTH / 2, 0.62 * height), label, 150, small, "mt")
            alpha = torch.tensor(np.asarray(canvas), **self.f) / 255.0
            self.texts[key] = alpha[..., None]
        return self.texts[key]

    def counter(self, image, seconds, phi):
        """The inner cylinder's turns so far, as a number with a small label."""
        if WIND[0] < seconds < WIND[1]:
            label = "turns forward"
        elif UNWIND[0] < seconds < UNWIND[1]:
            label = "turning back"
        else:
            label = "turns"
        alpha = self.counter_alpha(f"{abs(phi) / TAU:.1f}", label)
        top = COUNTER_Y - int(0.42 * alpha.shape[0]) + COUNTER_SIZES[0] // 3
        region = image[top : top + alpha.shape[0]]
        region += (self.counter_colour - region) * alpha
        return image

    @staticmethod
    def wrapped(d, sigma, sigma_eff):
        """A wrapped Gaussian of width `sigma`, smoothed to `sigma_eff`, peak 1."""
        d = torch.remainder(d + np.pi, TAU) - np.pi
        near = sum(
            torch.exp(-0.5 * ((d + TAU * k) / sigma_eff) ** 2) for k in (-1, 0, 1)
        )
        return near * (sigma / sigma_eff)

    def frame(self, seconds):
        phi = float(crank(seconds))
        rate = abs(float(crank_rate(seconds)))

        # Where each pixel's fluid started, and its footprint there (in theta0).
        theta0 = torch.remainder(self.theta - self.omega * phi + np.pi, TAU)
        foot2 = (
            self.s_pixel2
            + (self.omega * rate * SHUTTER / FPS) ** 2 / 12
            + (0.4 * self.domega * phi / R_PX) ** 2
        )
        texel = TAU / TEX_THETA
        level = (0.5 * torch.log2(foot2 / texel**2)).clamp(0, LEVELS - 1)
        column = theta0 / texel + 0.5  # in the padded texture
        grid = torch.stack(
            (
                2 * column / (TEX_THETA + 1) - 1,
                self.grid_r,
                2 * level / (LEVELS - 1) - 1,
            ),
            -1,
        )[None, None]
        optical = F.grid_sample(self.tex, grid, align_corners=True)[0, :, 0]
        syrup = self.light * torch.exp(-optical.permute(1, 2, 0))

        # The inner cylinder turns rigidly by phi: notches on its rim.
        s_n = 0.035
        s_eff = np.sqrt(s_n**2 + (rate * SHUTTER / FPS) ** 2 / 12)
        notch = self.wrapped(
            (self.theta - phi) * NOTCHES, s_n * NOTCHES, s_eff * NOTCHES
        )
        steel = self.steel_colour + self.hi * 0.9 * (notch * self.notch_band)[..., None]

        image = self.background * (1 - self.syrup - self.steel - self.glass)[..., None]
        image = image + syrup * self.syrup[..., None]
        image = image + steel * self.steel[..., None]
        image = image + self.glass_colour * self.glass[..., None]
        image = self.counter(image, seconds, phi)
        return (image.clamp(0, 1) * 255.0).flip(-1)  # BGR


# ------------------------------------------------------------------- the sound
def drop_crossings(drop_list):
    """
    (time, drop index, whole turn?) whenever a drop's centre comes round to
    its starting angle or to the opposite side: omega(r0) * crank(t) crosses
    a multiple of half a turn.
    """
    t = np.linspace(0, DURATION, 60001)
    phi = crank(t)
    events = []
    for i, (r0, *_) in enumerate(drop_list):
        turns = 2 * omega(r0) * phi / TAU  # in half turns
        k = np.floor(turns + 1e-9)
        for j in np.nonzero(np.diff(k))[0]:
            level = max(k[j], k[j + 1])
            a, b = turns[j], turns[j + 1]
            when = t[j] + (level - a) / (b - a) * (t[j + 1] - t[j])
            events.append((float(when), i, bool(level % 2 == 0)))
    return sorted(events)


def coherence(drop_list, seconds):
    """How whole the drops still look: mean of 1 / sqrt(1 + stretch^2), 1 .. ~0."""
    phi = np.abs(crank(seconds))
    c = 0
    for r0, *_ in drop_list:
        stretch = 2 * SHEAR / r0**3 * DROP_SIZE * phi / (DROP_SIZE / r0)
        c = c + 1 / np.sqrt(1 + stretch**2)
    return c / len(drop_list)


def soundtrack(drop_list, style_name, rng):
    """
    Drone & pad under phase-like struck notes. Every time a drop's centre
    comes round to its starting angle (drop_crossings) its own note sounds,
    and softer on the far side half a turn later. The inner drops turn faster,
    so they are higher and more frequent; panned by where the drop is,
    louder when the crank turns faster. Wound back,
    the same notes play in reverse order. Two pads, the home chord and a
    clouded seventh chord a step up, crossfade with the drops' coherence
    (1 / sqrt(1 + stretch^2)): the harmony smears as the ink smears and
    clears as it comes home. Low notes mark the start of the winding, the
    turn back, and the moment the drops are whole again, where the five drop
    notes roll (and roll out again as the winding starts). A loop: everything
    circular.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    hz = lambda note: float(audio.midi_to_hz(note))

    t = np.linspace(0, DURATION, 20001)
    peak = np.abs(crank_rate(t)).max()
    for when, i, whole in drop_crossings(drop_list):
        r0, a0 = drop_list[i][:2]
        speed = abs(float(crank_rate(when))) / peak
        velocity = 0.2 + 0.3 * speed + 0.03 * (len(drop_list) - i)
        velocity *= 1.0 if whole else 0.6
        side = np.cos(a0 + (0 if whole else np.pi))
        note = style.voice(scale.hz(DROP_DEGREES[i]), velocity)
        audio.add(track, audio.pan(note, 0.6 * side), when)

    # Pads crossfading with the coherence of the ink.
    t = np.arange(len(track)) / audio.SAMPLE_RATE
    c = coherence(drop_list, t)[:, None]
    home, cloud = audio.silence(DURATION), audio.silence(DURATION)
    for note in scale.chord(0, 3, -1):
        audio.add(home, audio.pad(hz(note), DURATION, 0.2, 0.01, 0.01, 0.45), 0)
    for note in scale.chord(1, 4, -1):
        audio.add(cloud, audio.pad(hz(note), DURATION, 0.18, 0.01, 0.01, 0.25), 0)
    track += home * c + audio.lowpass(cloud, 900.0) * (1 - c)

    for when, degree in ((WIND[0], 0), (UNWIND[0], 3), (UNWIND[1], 0)):
        audio.add(track, style.bass(scale.hz(degree, -2), 0.5), when)
    # All five drops' notes, rolled outward as the winding starts and rolled
    # back inward as they come home: a palindrome, like the ink.
    for k, (degree, (r0, a0, *_)) in enumerate(zip(DROP_DEGREES, drop_list)):
        note = audio.pan(style.voice(scale.hz(degree), 0.32), 0.6 * np.cos(a0))
        audio.add(track, note, WIND[0] + 0.09 * k)
        audio.add(track, note, UNWIND[1] - 0.09 * k)
    return audio.master(style.room(track, rng))


def generate(settings: ImageProcessingSettings = None):
    """
    Taylor-Couette reversibility, in closed form.

    Stokes (zero Reynolds number) flow between a rotating inner cylinder of
    radius R_INNER and a fixed outer one of radius 1 is pure circular Couette
    flow, u_theta = A r + B / r, i.e. each ring turns at angular velocity
    omega(r) = Omega * SHEAR * (1/r^2 - 1). Being linear and inertia-free, the
    displacement depends only on the inner wall's total angle phi(t), not on
    its speed: fluid starting at (r, theta0) sits at theta0 + omega(r) phi.
    `crank` winds phi up by TURNS turns over WIND (a beta-function speed
    profile, slow at first), holds
    it, and winds it back over UNWIND as the exact time mirror, so
    phi(0) = phi(DURATION) = 0 and the video is an exact loop.

    The ink at the start (five slightly irregular round drops and a straight
    spoke across the gap) is rasterised once as Beer-Lambert optical density
    on a polar grid (r, theta0), and blurred along theta0 into LEVELS
    mip levels by FFT (Gaussian, exact on the circle). Per frame, each pixel
    is mapped back to theta0 = theta - omega(r) phi and the texture is
    sampled trilinearly in (theta0, r, level), the level chosen by the
    pixel's footprint in theta0: its own angular size, the shear
    omega'(r) phi times its radial size, and the motion-blur sweep
    omega(r) phi' SHUTTER / FPS, in quadrature. Threads thinner than a pixel
    therefore fade into the right average tint instead of aliasing. Colour
    is backlight * exp(-density). Steel notches on the inner cylinder show
    phi; a counter shows the turns. One torch pass per frame, native
    resolution. The seed sets the drops' angles, jitter, shapes and ink
    order, and (via audio_rng) the music's key.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    renderer = Renderer(device, settings.rng)

    def render(frame: int):
        with torch.no_grad():
            return renderer.frame(frame / FPS)

    render_frames(settings, render, FRAME_COUNT, parallel=False)
    drop_list = renderer.drops
    del renderer
    gc.collect()
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(drop_list, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)


if __name__ == "__main__":
    generate()
