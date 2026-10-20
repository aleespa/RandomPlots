"""
Why Every Equation Has a Root

Every polynomial equation has a solution, if you allow complex numbers. This is the
Fundamental Theorem of Algebra, and here is a proof you can watch. Top: a loop in the
complex plane. Bottom: where a degree-5 polynomial p sends every point of that loop.

On a big loop, p(z) behaves like z⁵, so its image coils five times around zero (the bright
point). On a tiny loop, p(z) barely moves from p(0), so its image winds zero times. As the
loop shrinks, the coil must unwind, and the only way to slip a turn is to snag on zero. Each
snag means p(z) = 0 somewhere on the loop: a root, found without solving anything.

Colours follow the angle of p(z), so you can count the turns on both loops.

Listen: a music box plays one note for every turn of the coil. Each time a root is found,
one note drops out and the tune winds down, until the last root lands on the home note.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#fundamentaltheoremofalgebra #complexanalysis #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

BACKGROUND = "#05060a"
SATURATION = 0.62  # of the phase colours: below 1 for a pastel hue wheel
HUE_BUCKETS = 48  # curve segments are drawn in this many colour groups

# Layout: the z-plane on top, the w = p(z) plane below.
Z_CENTRE = (540, 500)  # px
Z_SCALE = 212.0  # px per unit of z
W_CENTRE = (540, 1430)  # px
W_RADIUS = 440.0  # px: where the largest |p| on the first loop lands

KNOBS = dict(
    moduli=(0.36, 0.72, 1.06, 1.42, 1.84),  # |root|, smallest first
    radius_start=2.25,  # the loop at 0 s
    crossings=(12.0, 9.6, 7.2, 4.8, 2.4),  # s: the loop meets each root (same order)
    collapse_at=13.6,  # s: the loop has shrunk to a point
    ease=0.55,  # 0 steady shrink .. 1 stops dead at every snag
    w_soft=0.04,  # |p| below this is drawn on a linear scale, above it logarithmic
    trail=14,  # earlier images drawn behind the current one
    trail_step=0.07,  # s between them
    reveal=0.2,  # brightness of the swept part of the z-plane
    glow=0.9,
)

# --- Sound -------------------------------------------------------------------------
STYLE = "music_box"  # common.music.STYLES; an --audio-only tag naming one overrides it
CHORD = (0, 2, 4, 6, 8)  # scale degrees, one per turn of the coil (lowest first)
BAR = 0.8  # s: the music box plays the remaining notes once per bar


def hue_lut(size: int = 256) -> np.ndarray:
    """(size, 3) float32 BGR in 0..1: the phase wheel, angle 0 .. 2 pi, pastel."""
    hsv = np.zeros((1, size, 3), np.uint8)
    hsv[0, :, 0] = np.arange(size) * 180 // size
    hsv[0, :, 1] = int(255 * SATURATION)
    hsv[0, :, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32) / 255.0


def radius_at(seconds, knobs) -> float:
    """
    The loop's radius: through radius_start, then each |root| at its crossing time,
    then 0 at collapse_at; each stretch eases, slowing (not stopping) at every snag.
    """
    times = (0.0,) + tuple(sorted(knobs["crossings"])) + (knobs["collapse_at"],)
    radii = (knobs["radius_start"],) + tuple(sorted(knobs["moduli"], reverse=True))
    radii = radii + (0.0,)
    if seconds >= times[-1]:
        return 0.0
    i = int(np.searchsorted(times, seconds, side="right")) - 1
    u = (seconds - times[i]) / (times[i + 1] - times[i])
    last = i == len(times) - 2
    eased = (
        smoothstep(u)
        if last
        else (1 - knobs["ease"]) * u + knobs["ease"] * smoothstep(u)
    )
    return float(radii[i] + (radii[i + 1] - radii[i]) * eased)


def soundtrack(roots, root_times, style_name, rng):
    """
    A one-shot music box that winds down with the winding number n (5 at first, one
    less at each crossing). Every BAR it plays the n lowest notes of CHORD as a rising
    arpeggio, n evenly spaced notes, so the tune thins and slows as roots are found;
    a soft pad holds the same n notes. At each crossing the note that leaves (the
    top one) is struck hard, panned to the root's side of the plane, over a low
    marimba note; the last crossing strikes the home note and an open fifth, which
    ring out.
    """
    style = STYLES[style_name]
    root = int(rng.choice(style.roots))
    scale = Scale(root, style.mode)
    kw = dict(circular=False)
    track = audio.silence(DURATION)
    times = np.sort(root_times)  # crossings, earliest first: n drops 5 -> 0
    order = np.argsort(root_times)
    hz = lambda degree, octave=-1: scale.hz(degree, octave)

    def winding(seconds):
        return len(CHORD) - int(np.searchsorted(times, seconds, side="right"))

    for i, degree in enumerate(CHORD):  # the top note leaves first
        until = times[len(CHORD) - 1 - i]
        voice = audio.pad(hz(degree), until, 0.12, 1.2, 1.0, 0.3)
        audio.add(track, voice, 0.0, **kw)

    notes = 0
    for bar in np.arange(0.0, times[-1], BAR):
        n = winding(bar)
        for j in range(n):
            when = bar + j * BAR / n
            if when >= times[-1] or winding(when) < n or n == 0:
                break
            velocity = 0.32 + 0.06 * (j == 0)
            note = style.voice(hz(CHORD[j]), velocity)
            audio.add(
                track, audio.pan(note, 0.5 * (j / max(n - 1, 1) - 0.5)), when, **kw
            )
            notes += 1

    for k, when in enumerate(times):
        x = float(np.clip(roots[order[k]].real / 2.0, -1, 1))
        if k < len(times) - 1:
            leaving = CHORD[len(CHORD) - 1 - k]
            audio.add(track, audio.pan(style.voice(hz(leaving), 0.75), x), when, **kw)
            audio.add(track, style.bass(hz(leaving, -3), 0.3, 6.0), when, **kw)
        else:  # home: the root, its fifth and octave, ringing out
            for degree, gain in ((0, 0.8), (4, 0.5), (7, 0.45)):
                audio.add(
                    track, audio.pan(style.voice(hz(degree), gain), 0.3 * x), when, **kw
                )
            audio.add(track, style.bass(hz(0, -3), 0.45, 6.0), when, **kw)
            audio.add(
                track,
                audio.pad(hz(0, -2), DURATION - when - 1.2, 0.25, 0.3, 1.0, 0.3),
                when,
                **kw,
            )
    logger.info(f"Soundtrack: {style_name}, root MIDI {root}, {notes} arpeggio notes")
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.4)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The winding-number proof of the Fundamental Theorem of Algebra, for a monic quintic
    p(z) = prod (z - r_k).

    The roots r_k have the moduli in KNOBS and angles from settings.rng (golden-angle
    spaced with jitter, so they spread round the plane). A circle |z| = R(s) shrinks
    from radius_start to 0, passing each |r_k| at its crossing time (`radius_at`:
    piecewise, eased). The winding number of p round 0 along the circle counts the
    roots inside it (argument principle): 5 at first, 0 at the end, dropping by one at
    each crossing, where the image curve passes through 0.

    Top panel, the z-plane: the circle drawn with each point coloured by arg p(z); the
    region already swept (|z| > R) shows a dim domain colouring of p (hue = arg p,
    brightness rings at every doubling of |p|); each root, once crossed, is a star.
    Bottom panel, the w-plane: the image p(circle), coloured the same way, under a
    radial compression w -> w / |w| * A log(1 + |w| / w_soft) that keeps every winding
    and fits both the first coil (|p| up to ~10^3) and the final point p(0). Faint
    images from the last trail * trail_step seconds sit behind it. At each crossing the
    origin flashes and the root's star is born. Segments are drawn anti-aliased with
    cv2 (sub-pixel, shift 4) in HUE_BUCKETS colour groups, plus a bloom from a
    quarter-size blur. A one-shot, frames in parallel on the thread pool; it rests from
    collapse_at and the last 0.4 s fade out.

    Sound (see `soundtrack`): driven by the winding number and the crossing times. The
    key comes from settings.audio_rng; the posted soundtrack uses --seed 0 (the
    default).
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS
    rng = settings.rng

    golden = np.pi * (3 - np.sqrt(5))
    angles = rng.uniform(0, 2 * np.pi) + golden * np.arange(5) * 2
    angles = angles + rng.uniform(-0.35, 0.35, 5)
    roots = np.array(knobs["moduli"]) * np.exp(1j * angles)
    logger.info(f"Roots: {np.round(roots, 3)}, p(0) = {np.prod(-roots):.3f}")

    def p(z):
        out = np.ones_like(z)
        for r in roots:
            out = out * (z - r)
        return out

    samples = 6000
    theta = np.linspace(0, 2 * np.pi, samples + 1)
    circle = np.exp(1j * theta)
    w_max = np.abs(p(knobs["radius_start"] * circle)).max()
    w_gain = W_RADIUS / np.log1p(w_max / knobs["w_soft"])

    def w_to_px(w):
        rho = w_gain * np.log1p(np.abs(w) / knobs["w_soft"])
        unit = w / np.maximum(np.abs(w), 1e-300)
        return np.stack(
            [W_CENTRE[0] + rho * unit.real, W_CENTRE[1] - rho * unit.imag], 1
        )

    def z_to_px(z):
        return np.stack(
            [Z_CENTRE[0] + Z_SCALE * z.real, Z_CENTRE[1] - Z_SCALE * z.imag], 1
        )

    lut = hue_lut()
    background = np.array(
        [int(BACKGROUND[i : i + 2], 16) for i in (5, 3, 1)], np.float32
    )

    # The z-plane's domain colouring, revealed where the loop has swept.
    top = 2 * Z_CENTRE[1]
    ys, xs = np.mgrid[0:top, 0:WIDTH].astype(np.float32)
    zz = (xs - Z_CENTRE[0]) / Z_SCALE - 1j * (ys - Z_CENTRE[1]) / Z_SCALE
    pz = p(zz)
    hue = ((np.angle(pz) / (2 * np.pi)) % 1.0 * 255).astype(int)
    rings = np.log2(np.abs(pz) + 1e-12) % 1.0
    edge = np.clip((top - ys) / 160.0, 0, 1)  # the panel fades out at its foot
    shade = (0.45 + 0.55 * rings) * smoothstep(edge) * 255 * knobs["reveal"]
    domain = lut[hue] * shade[..., None]
    z_abs = np.abs(zz).astype(np.float32)
    root_px = z_to_px(roots)
    crossing = dict(zip(knobs["moduli"], knobs["crossings"]))
    root_times = np.array([crossing[m] for m in knobs["moduli"]])

    # A soft round sprite for stars and the origin's flash.
    sprite_r = 60
    yy, xx = np.mgrid[-sprite_r : sprite_r + 1, -sprite_r : sprite_r + 1]
    d2 = (xx * xx + yy * yy).astype(np.float32)
    halo = np.exp(-d2 / (2 * 14.0**2)) + 0.6 * np.exp(-d2 / (2 * 4.0**2))
    spikes = np.exp(-(np.minimum(np.abs(xx), np.abs(yy)) ** 2) / 2.0) * np.exp(
        -np.sqrt(d2) / 22.0
    )

    def stamp(image, centre, sprite, colour):
        x, y = int(round(centre[0])), int(round(centre[1]))
        h, w = image.shape[:2]
        x0, x1 = max(0, x - sprite_r), min(w, x + sprite_r + 1)
        y0, y1 = max(0, y - sprite_r), min(h, y + sprite_r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        part = sprite[
            y0 - y + sprite_r : y1 - y + sprite_r, x0 - x + sprite_r : x1 - x + sprite_r
        ]
        image[y0:y1, x0:x1] += part[..., None] * colour

    def draw_curve(layer, pts, phase, thickness, gain):
        """Anti-aliased polyline, each segment coloured by its phase (0..1)."""
        segments = np.stack([pts[:-1], pts[1:]], 1)
        fixed = np.round(segments * 16).astype(np.int32)
        bucket = (phase[:-1] % 1.0 * HUE_BUCKETS).astype(int) % HUE_BUCKETS
        for b in np.unique(bucket):
            colour = lut[int((b + 0.5) * 256 / HUE_BUCKETS)] * 255 * gain
            cv2.polylines(
                layer,
                fixed[bucket == b],
                False,
                colour.tolist(),
                thickness,
                cv2.LINE_AA,
                shift=4,
            )

    def phase_of(w):
        return np.angle(w) / (2 * np.pi)

    p0 = complex(p(np.array([0j]))[0])
    fade_frames = 0.4 * FPS
    white = np.array([255, 255, 255], np.float32)

    def render(frame):
        seconds = frame / (FRAME_COUNT - 1) * DURATION
        radius = radius_at(seconds, knobs)
        lines = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # Trail of earlier images in the w-plane, oldest and faintest first.
        for k in range(knobs["trail"], 0, -1):
            past = seconds - k * knobs["trail_step"]
            if past < 0:
                continue
            w = p(radius_at(past, knobs) * circle[::4])
            fade = 0.32 * (1 - k / (knobs["trail"] + 1)) ** 1.5
            draw_curve(lines, w_to_px(w), phase_of(w), 1, fade)

        z = radius * circle
        w = p(z)
        phase = phase_of(w)
        draw_curve(lines, w_to_px(w), phase, 3, 1.0)
        if radius > 0.01:
            draw_curve(lines, z_to_px(z), phase, 3, 1.0)

        image = lines.astype(np.float32)

        # The origin of the w-plane, and its flash when the image snags on it.
        since = seconds - root_times
        flash = float(np.exp(-np.clip(since, 0, None) / 0.35)[since >= 0].sum())
        approach = float(np.exp(-(since[since < 0] ** 2) / 0.08).sum())
        stamp(image, W_CENTRE, halo, white * (0.35 + 0.4 * approach + 2.2 * flash))
        stamp(image, W_CENTRE, spikes, white * 1.6 * flash)
        # Found roots: a star born with a burst, then steady.
        for k in np.flatnonzero(since >= 0):
            burst = np.exp(-since[k] / 0.5)
            stamp(image, root_px[k], halo, white * (0.55 + 1.8 * burst))
            stamp(image, root_px[k], spikes, white * (0.7 + 1.5 * burst))
        # As the loop becomes a point, both ends become a glowing dot: 0 and p(0).
        point = float(smoothstep(1 - radius / 0.12))
        if point > 0:
            colour = lut[int(phase_of(p0) % 1.0 * 255)] * 255 * 1.2 * point
            stamp(image, z_to_px(np.array([0j]))[0], halo, colour)
            stamp(image, w_to_px(np.array([p0]))[0], halo, colour)

        small = cv2.resize(
            image, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA
        )
        bloom = cv2.GaussianBlur(small, (0, 0), 5.0)
        image += knobs["glow"] * cv2.resize(
            bloom, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        # Under the light: the background, and the swept part of the z-plane.
        image += background
        swept = np.clip((z_abs - radius) * Z_SCALE / 3.0, 0, 1)
        image[:top] += domain * swept[..., None]
        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return np.clip(image * fade, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT)
    style_name = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(roots, root_times, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
