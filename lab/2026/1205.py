"""
Why Hydrogen Glows Red

A hydrogen atom giving off one particle of light. The violet cloud is its
electron, caught halfway between two energy levels, 3d and 2p. A quantum state
with a single energy never moves: the cloud of a pure 3d or 2p state stays
perfectly still. Mix the two and the cloud starts to slosh back and forth,
and a sloshing charge sends out light, the red ripples.

The cloud swings at a rate set by the energy gap between the two levels,
E₃ − E₂ = hν, so the colour of the light is fixed by the gap. For this jump it
is 656 nm, the deep red of hydrogen and the glow of every pink nebula in
space photos. As the light carries the energy away, the electron drains into
the lower level, the cloud shrinks and goes still, and the hydrogen atom stops
glowing. (Slowed down about 10¹⁵ times.)

Listen: each swing of the cloud strikes a handpan, and the notes step down
as the electron falls, over a soft choir that moves from the upper chord to
the home chord as the atom comes to rest.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#hydrogenatom #quantummechanics #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

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
HOOK = 'Why does hydrogen glow red?'
HOOK_Y = 280
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The transition ----------------------------------------------------------------
P0 = 0.97  # |c_3|^2 at s = 0: the atom has just started to leave 3d
TAU = 2.2  # s: lifetime of the upper level (|c_3|^2 decays as e^{-s/τ})
SETTLE = (8.5, 12.0)  # s: the tail of the decay is eased to exactly zero here
SWING = 1.25  # Hz: the beat (E_3 - E_2)/h, slowed down for the eye
FADE = 0.4  # s: fade to the background at the end

# --- Layout (px) -------------------------------------------------------------------
CX, CY = WIDTH // 2, 900  # the nucleus
SCALE = 21.0  # px per Bohr radius
AXIS = np.radians(28.0)  # the dipole axis, in the image plane, above horizontal
TILT = np.radians(22.0)  # and tilted towards the viewer
DEPTH = 34.0  # Bohr radii integrated either side along the line of sight
DEPTH_SAMPLES = 112
CLOUD_HALF = 700  # px: rows CY ± CLOUD_HALF hold the cloud
C_LIGHT = 360.0  # px/s: speed of the ripples
CREST_POW = 9.0  # sharpness of the crests
NEAR = (110.0, 260.0)  # px: the ripples fade in over this range of radius
TITLE_Y, FORMULA_Y = 96, 196
BARS_Y = (1640, 1712)  # the |c_3|^2 and |c_2|^2 bars
BAR_X = (250, 930)
BAR_H = 14
LINE_Y = 1792

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#05040a"
CLOUD = ["#05040a", "#17123a", "#3b2f86", "#7d6fd6", "#c9c0ff", "#f4f1ff"]
CLOUD_GAIN, CLOUD_GAMMA = 2.0, 0.6  # tone map 1 - exp(-gain (ρ / max ρ_2p)^gamma)
LIGHT = "#ff2a3a"  # H-alpha
LIGHT_CORE = "#ffb0a8"
GLOW_SIGMA, GLOW_GAIN = 8.0, 0.65
TEXT, TEXT_DIM = "#efeaff", "#9a90bf"

# --- Sound -------------------------------------------------------------------------
STYLE = "handpan"  # common.music.STYLES; an --audio-only tag naming one overrides it
UPPER, HOME = 4, 0  # choir chords (scale degrees) for the 3d and 2p states
STEPS = 7  # the struck melody falls this many scale degrees as the atom decays


def upper_weight(s):
    """|c_3(s)|^2: exponential decay from P0, eased to exactly 0 by SETTLE[1]."""
    s = np.asarray(s, float)
    p = (
        P0
        * np.exp(-s / TAU)
        * (1 - smoothstep((s - SETTLE[0]) / (SETTLE[1] - SETTLE[0])))
    )
    return np.clip(p, 0.0, 1.0)


def dipole(s):
    """2|c_3 c_2|, the swing amplitude of the cloud and of the light it gives off."""
    p = upper_weight(s)
    return 2 * np.sqrt(p * (1 - p))


def orbitals(r, cos_theta):
    """ψ_320 and ψ_210 of hydrogen in atomic units (real)."""
    r32 = 4 / (81 * np.sqrt(30)) * r**2 * np.exp(-r / 3)
    r21 = 1 / (2 * np.sqrt(6)) * r * np.exp(-r / 2)
    y20 = np.sqrt(5 / (16 * np.pi)) * (3 * cos_theta**2 - 1)
    y10 = np.sqrt(3 / (4 * np.pi)) * cos_theta
    return r32 * y20, r21 * y10


def projections():
    """
    Column densities ∫|ψ_3|², ∫|ψ_2|² and ∫ψ_3ψ_2 along the line of sight, at half
    resolution over the cloud rows, then resized to full: (3, rows, WIDTH) float32.
    """
    axis = np.array(
        [np.cos(TILT) * np.cos(AXIS), np.cos(TILT) * np.sin(AXIS), np.sin(TILT)]
    )
    rows = 2 * CLOUD_HALF
    u = ((np.arange(WIDTH // 2) + 0.5) * 2 - CX) / SCALE
    v = -((np.arange(rows // 2) + 0.5) * 2 - CLOUD_HALF) / SCALE
    w = np.linspace(-DEPTH, DEPTH, DEPTH_SAMPLES)
    dw = w[1] - w[0]
    out = np.zeros((3, rows // 2, WIDTH // 2), np.float64)
    for i, vi in enumerate(v):
        uu, ww = np.meshgrid(u, w, indexing="ij")
        r = np.sqrt(uu**2 + vi**2 + ww**2) + 1e-9
        cos_theta = (uu * axis[0] + vi * axis[1] + ww * axis[2]) / r
        p3, p2 = orbitals(r, cos_theta)
        out[0, i] = (p3 * p3).sum(1) * dw
        out[1, i] = (p2 * p2).sum(1) * dw
        out[2, i] = (p3 * p2).sum(1) * dw
    return np.stack(
        [
            cv2.resize(
                o.astype(np.float32), (WIDTH, rows), interpolation=cv2.INTER_CUBIC
            )
            for o in out
        ]
    )


def soundtrack(style_name, rng):
    """
    The decay made audible, from the same |c_3(s)|² as the picture.

    - Strikes: every swing of the cloud (cos(2π SWING s) = ±1) strikes the style's voice
      while the dipole 2|c_3 c_2| is above 0.08; louder for a bigger swing, the far swing an
      octave lower and softer. The pitch falls STEPS scale degrees as |c_3|² drains to 0.
    - Choir: an "u" chord on degree UPPER whose level follows |c_3|², and the home chord
      whose level follows |c_2|² = 1 - |c_3|², so the harmony moves with the electron.
    - Landing: the home root and fifth struck as the atom settles (SETTLE[1]), ringing out.
    One-shot: nothing wraps (circular=False) and `master(loop=False)`.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    sr = audio.SAMPLE_RATE
    track = audio.silence(DURATION)
    n = len(track)
    s = np.arange(n) / sr
    upper = upper_weight(s)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.clip((DURATION - s) / 1.2, 0, 1))
    for degree, level in ((UPPER, upper), (HOME, 1 - upper)):
        for note in scale.chord(degree, 3, octave=-1):
            sung = audio.choir(
                float(audio.midi_to_hz(note)), DURATION, 0.32, attack=0.6, vowel="u"
            )[:n]
            track += sung * (level * fade)[:, None]

    strikes = 0
    for k in range(int(2 * SWING * DURATION)):
        when = k / (2 * SWING)
        amp = float(dipole(when))
        if amp < 0.08:
            continue
        degree = int(round(STEPS * float(upper_weight(when))))
        near = k % 2 == 0  # cos = +1: the swing towards the upper right
        freq = scale.hz(degree, octave=0 if near else -1)
        velocity = (0.3 + 0.5 * amp) * (1.0 if near else 0.6)
        audio.add(
            track,
            audio.pan(style.voice(freq, velocity), 0.35 if near else -0.35),
            when,
            circular=False,
        )
        strikes += 1
    for degree, velocity in ((HOME, 0.7), (HOME + 4, 0.45)):
        audio.add(
            track, style.voice(scale.hz(degree), velocity), SETTLE[1], circular=False
        )
    logger.info(f"Soundtrack: {style_name}, root {scale.root}, {strikes} strikes")
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A hydrogen atom radiating H-alpha (3d -> 2p, m = 0), as a one-shot.

    The state is ψ = c_3 ψ_320 e^{-iE_3 t} + c_2 ψ_210 e^{-iE_2 t} with real hydrogen
    eigenfunctions (atomic units). Its density is
    |c_3|² ψ_3² + |c_2|² ψ_2² + 2|c_3 c_2| ψ_3 ψ_2 cos(ω s), ω = 2π SWING standing in for
    (E_3 - E_2)/ħ. |c_3(s)|² = P0 e^{-s/τ} (the Weisskopf-Wigner decay), eased to 0
    over SETTLE so the 2p cloud ends perfectly still; |c_2|² = 1 - |c_3|².

    Cloud: the three column densities ∫ψ_3², ∫ψ_2², ∫ψ_3ψ_2 along the line of sight (the
    m = 0 axis at AXIS in the image plane, tilted TILT towards the viewer) are integrated
    once at half resolution; each frame is their weighted sum, tone-mapped through the
    CLOUD palette. Light: the far field of the oscillating dipole d(s) ∝ 2|c_3 c_2| cos(ω s)
    along the axis, E ∝ sin θ d(s - r/c) / r, drawn as crests (½ + ½cos)^CREST_POW of the
    retarded phase with brightness 2|c_3 c_2|(s - r/c) sin²θ / √r, in H-alpha red with
    a glow. Wavelength and atom size are not to scale (λ is ~10⁴ atoms really).

    Text (Computer Modern via common.video.latex_mask, typeset once) and two bars for
    |c_3|² and |c_2|². Every frame is a pure function of s on the render_frames thread
    pool. Nothing is random: --seed does not change the picture. The soundtrack
    (`soundtrack`, STYLE handpan over a choir) uses settings.audio_rng for the root and the
    reverb; the posted one used --seed 0. An --audio-only tag naming a common.music style
    swaps the struck voice.
    """
    settings = settings or ImageProcessingSettings(1)
    cloud_maps = projections()
    ref = float(cloud_maps[1].max())
    cloud_maps /= ref
    lut = (palette_lut(CLOUD, 1024)[:, ::-1] * 255).astype(np.float32)  # BGR
    top = CY - CLOUD_HALF

    # Light: radius and angle of every pixel from the nucleus, once.
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    dx, dy = xx - CX, -(yy - CY)
    radius = np.sqrt(dx**2 + dy**2) + 1e-3
    axis = np.array(
        [np.cos(TILT) * np.cos(AXIS), np.cos(TILT) * np.sin(AXIS)], np.float32
    )
    sin2 = 1 - ((dx * axis[0] + dy * axis[1]) / radius) ** 2  # w = 0 plane
    # Dim the light behind the text bands so the type stays readable.
    row = np.arange(HEIGHT, dtype=np.float32)
    calm = 1 - 0.65 * smoothstep((FORMULA_Y + 90 - row) / 60)
    calm *= 1 - 0.65 * smoothstep((row - (BARS_Y[0] - 70)) / 60)
    sin2 *= calm[:, None]
    bins = 4  # per px
    rbin = np.round(radius * bins).astype(np.int32)
    r1d = np.arange(rbin.max() + 1) / bins
    reach = smoothstep((r1d - NEAR[0]) / (NEAR[1] - NEAR[0])) * np.sqrt(
        NEAR[1] / np.maximum(r1d, NEAR[1])
    )
    del xx, yy, dx, dy, radius
    light_colour = np.array(hex_to_bgr(LIGHT), np.float32)
    core_colour = np.array(hex_to_bgr(LIGHT_CORE), np.float32)
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    omega = 2 * np.pi * SWING

    title = latex_mask(r"$3d\ \rightarrow\ 2p$", 66)
    formula = latex_mask(
        r"$\psi = c_3\,\psi_{3d}\,e^{-iE_3t/\hbar} + c_2\,\psi_{2p}\,e^{-iE_2t/\hbar}$",
        44,
    )
    line = latex_mask(r"$E_3 - E_2 = h\nu{,}\quad \lambda = 656\ \mathrm{nm}$", 42)
    labels = [latex_mask(r"$|c_3|^2$", 40), latex_mask(r"$|c_2|^2$", 40)]
    text, text_dim = hex_to_bgr(TEXT), hex_to_bgr(TEXT_DIM)
    bar_colours = [np.array(hex_to_bgr(c), np.float32) for c in (CLOUD[4], CLOUD[3])]
    cols = np.arange(WIDTH, dtype=np.float32) + 0.5

    def render(frame: int) -> np.ndarray:
        s = frame / (FRAME_COUNT - 1) * DURATION
        upper = float(upper_weight(s))
        swing = float(dipole(s)) * np.cos(omega * s)
        image = np.empty((HEIGHT, WIDTH, 3), np.float32)
        image[:] = bg

        # The cloud.
        rho = (
            upper * cloud_maps[0] + (1 - upper) * cloud_maps[1] + swing * cloud_maps[2]
        )
        tone = 1 - np.exp(-CLOUD_GAIN * np.clip(rho, 0, None) ** CLOUD_GAMMA)
        image[top : top + 2 * CLOUD_HALF] = lut[(tone * 1023).astype(np.int32)]

        # The light, by retarded time.
        ret = s - r1d / C_LIGHT
        crest = (0.5 + 0.5 * np.cos(omega * ret)) ** CREST_POW
        profile = (dipole(ret) * (ret > -0.1) * crest * reach).astype(np.float32)
        light = profile[rbin] * sin2
        image += light[..., None] * light_colour
        image += (light**2)[..., None] * core_colour * 0.5
        glow = cv2.GaussianBlur(
            cv2.resize(np.clip(image - bg, 0, None), (WIDTH // 2, HEIGHT // 2)),
            (0, 0),
            GLOW_SIGMA / 2,
        )
        image += GLOW_GAIN * cv2.resize(glow, (WIDTH, HEIGHT))

        # Text and the two weights.
        blit_mask(image, title, WIDTH // 2, TITLE_Y, text, anchor="centre")
        blit_mask(image, formula, WIDTH // 2, FORMULA_Y, text_dim, anchor="centre")
        for i, weight in enumerate((upper, 1 - upper)):
            y = BARS_Y[i]
            mask = labels[i]
            blit_mask(
                image, mask, BAR_X[0] - 30 - mask.shape[1], y - mask.shape[0] // 2, text
            )
            end = BAR_X[0] + weight * (BAR_X[1] - BAR_X[0])
            cover = np.clip(end - cols, 0, 1) * (cols >= BAR_X[0])
            cover *= smoothstep((end - BAR_X[0]) / 10)  # a bar under 10 px fades out
            band = image[y - BAR_H // 2 : y + BAR_H // 2]
            track = (cols >= BAR_X[0]) & (cols <= BAR_X[1])
            band[:, track] = 0.75 * band[:, track] + 0.25 * np.array(
                hex_to_bgr(CLOUD[1]), np.float32
            )
            band += cover[None, :, None] * (bar_colours[i] - band)
        blit_mask(image, line, WIDTH // 2, LINE_Y, text_dim, anchor="centre")

        out = float(smoothstep((DURATION - s) / FADE))
        if out < 1:
            image = bg + out * (image - bg)
        return image

    logger.info(
        f"Cloud maps {cloud_maps.shape}, peak 3d/2p column density "
        f"{cloud_maps[0].max():.2f}; swing {SWING} Hz, λ = {C_LIGHT / SWING:.0f} px"
    )
    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    tag = settings.audio_tag
    track = soundtrack(tag if tag in STYLES else STYLE, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
