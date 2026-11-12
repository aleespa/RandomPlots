"""
Inside a Raindrop

Sunlight falls on a single raindrop, and this is where a rainbow is made.
Each ray bends as it enters the drop, bounces off the back wall and leaves
again, and each colour bends by a slightly different amount.

Rays leave at every angle, but they pile up at one angle they can never
pass: 42° for light that bounced once inside. That bright edge, a caustic,
is the rainbow. Light that bounces twice piles up at 51° with its colours
reversed: the faint second rainbow, and the darker sky between the two.
Descartes worked this out in 1637 with a glass ball full of water.

Each colour plays its own note, red lowest and violet highest. When a
rainbow is born its colours play in order, rising for the first bow and
falling for the reversed second.

Made with Python & PyTorch. Sound synthesized in NumPy.
#rainbow #optics #raytracing #mathart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
from loguru import logger

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The drop (drop radii; the sun is straight above) ---------------------------------
CENTRE = (WIDTH / 2, 1010.0)  # px
SCALE = 270.0  # px per drop radius
MAX_CHORDS = 5  # paths with up to 4 internal reflections (the 4th-order bow)
LAMBDA = (405.0, 695.0)  # nm, the visible band sampled
LIFT = 0.55  # each internal reflection's loss is shown as loss^(1 - LIFT)
ORDER_PROB = (0.10, 0.18, 0.30, 0.22, 0.10, 0.10)  # photons sampled per exit order

# --- Time: s is how far the light has travelled, in drop radii -----------------------
START = -CENTRE[1] / SCALE - 0.1  # the beam's starting line, above the frame
S0, S1 = 1.75, 13.4  # path length shown at t = 0 and when the light stops
STOP = (10.0, 13.6)  # s: the light slows to a stop over this span
PHOTONS_PER_UNIT = 90_000_000  # photons per drop radius of travel
CHUNK = 4_000_000

# --- Look ---------------------------------------------------------------------------
EXPOSURE, GAMMA = 0.8, 0.5  # brightness = 1 - exp(-EXPOSURE * (Y^GAMMA + glow))
BEAM = 0.2  # the incoming sunlight's weight, against the light that left the drop
SATURATION = 2.5  # colour grading: chroma (rgb / Y - 1) scaled by this
CHROMA_GATE = (0.15, 0.4)  # only tints this strong get the full boost
SOFT, CHROMA_SOFT = 0.8, 3.0  # px blur of the light, and of its colour (less noise)
FRONT_GAIN, FRONT_HALF_LIFE = 8.0, 0.07  # the travelling wavefront, s
GLOWS = ((10.0, 2, 0.08), (40.0, 8, 0.05))  # (sigma px, shrink, gain) linear halos
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 14.0, 4, 0.2
RIM_COLOUR, RIM_ALPHA, RIM_WIDTH = (0.62, 0.78, 0.92), 0.35, 2.0

# --- Sound -------------------------------------------------------------------------
STYLE = "music_box"  # common.music.STYLES; an --audio-only tag naming one overrides it
COLOURS = 7  # notes per rainbow arpeggio, red to violet
ARPEGGIO_GAP = 0.09  # s between the colours of one bow
ECHO = 0.36  # s, ping-pong echoes on the struck notes


def refractive_index(wavelength):
    """Water at 20 C, a Cauchy fit: 1.331 at 700 nm, 1.344 at 400 nm."""
    return 1.3247 + 3046.0 / wavelength**2


def spectrum_rgb(wavelength: np.ndarray) -> np.ndarray:
    """
    Linear sRGB of monochromatic light (Wyman, Sloan & Shirley's fit of the CIE
    1931 observer), negatives clipped, each channel scaled so that the band
    LAMBDA averages to white.
    """

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    w = np.asarray(wavelength, float)
    x = 1.056 * g(w, 599.8, 37.9, 31.0) + 0.362 * g(w, 442.0, 16.0, 26.7)
    x -= 0.065 * g(w, 501.1, 20.4, 26.2)
    y = 0.821 * g(w, 568.8, 46.9, 40.5) + 0.286 * g(w, 530.9, 16.3, 31.1)
    z = 1.217 * g(w, 437.0, 11.8, 36.0) + 0.681 * g(w, 459.0, 26.0, 13.8)
    m = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb = np.clip(np.stack([x, y, z], -1) @ m.T, 0, None)
    band = np.linspace(*LAMBDA, 1024)
    xb = 1.056 * g(band, 599.8, 37.9, 31.0) + 0.362 * g(band, 442.0, 16.0, 26.7)
    xb -= 0.065 * g(band, 501.1, 20.4, 26.2)
    yb = 0.821 * g(band, 568.8, 46.9, 40.5) + 0.286 * g(band, 530.9, 16.3, 31.1)
    zb = 1.217 * g(band, 437.0, 11.8, 36.0) + 0.681 * g(band, 459.0, 26.0, 13.8)
    mean = np.clip(np.stack([xb, yb, zb], -1) @ m.T, 0, None).mean(0)
    return rgb / mean


def path_length(seconds):
    """How far the light has travelled (drop radii) at video time `seconds`."""
    t0, t1 = STOP
    s = np.clip(np.asarray(seconds, float), 0, None)
    speed = (S1 - S0) / (t0 + 0.5 * (t1 - t0))  # constant, then eased to zero
    u = np.clip((s - t0) / (t1 - t0), 0, 1)
    eased = (t1 - t0) * (u - u**3 + 0.5 * u**4)  # integral of 1 - smoothstep
    return S0 + speed * np.where(s < t0, s, t0 + eased)


def video_time(length):
    """Inverse of path_length: when the light has travelled `length`."""
    grid = np.linspace(0, DURATION, 30001)
    return np.interp(length, path_length(grid), grid)


def fresnel(cos_i, cos_r, n):
    """Unpolarised reflectance of a water surface (the same inside and out)."""
    rs = ((cos_i - n * cos_r) / (cos_i + n * cos_r)) ** 2
    rp = ((n * cos_i - cos_r) / (n * cos_i + cos_r)) ** 2
    return 0.5 * (rs + rp)


def rainbow_ray(chords: int, n: float):
    """
    The ray of minimum deviation for light leaving after `chords` chords
    (chords - 1 internal reflections): its impact parameter b and its angle
    from the antisolar point (straight up), degrees. cos^2 i = (n^2 - 1) /
    (k^2 - 1) with k = chords.
    """
    k = chords
    cos_i = np.sqrt((n * n - 1) / (k * k - 1))
    i = np.arccos(cos_i)
    r = np.arcsin(np.sin(i) / n)
    deviation = 2 * (i - r) + (k - 1) * (np.pi - 2 * r)
    return float(np.sin(i)), float(np.degrees(np.pi - deviation) % 360)


def exit_length(b: float, chords: int, n: float) -> float:
    """Path length (drop radii from START) at which a ray of impact b leaves the drop."""
    i = np.arcsin(b)
    r = np.arcsin(b / n)
    return float(-np.cos(i) - START + chords * 2 * np.cos(r))


def photons(s_lo, s_hi, count, gen, device, sample_rgb):
    """
    `count` photons at path lengths uniform in (s_lo, s_hi]: their positions
    (px), RGB weights and path lengths. Each has a random impact parameter b
    across the drop, a random wavelength and a random exit order m (the
    number of chords inside; 0 is the outside reflection). Weights are the
    exact Fresnel fractions of the stretch the photon is on, divided by the
    chance of a photon being there, so every region is unbiased.
    """
    f = dict(device=device, dtype=torch.float32)
    u = torch.rand(4, count, generator=gen, **f)
    s = s_lo + (s_hi - s_lo) * u[0]
    b = 2 * u[1] - 1
    wl_index = (u[2] * (len(sample_rgb) - 1)).round().long()
    wavelength = LAMBDA[0] + (LAMBDA[1] - LAMBDA[0]) * wl_index / (len(sample_rgb) - 1)
    q = torch.tensor(ORDER_PROB, **f)
    cdf = torch.cumsum(q, 0)
    m = torch.searchsorted(cdf / cdf[-1], u[3].contiguous()).clamp(max=MAX_CHORDS)
    n = refractive_index(wavelength)
    i = torch.asin(b)
    r = torch.asin(b / n)
    cos_i, cos_r = torch.cos(i), torch.cos(r)
    reflect = fresnel(cos_i, cos_r, n)
    transmit = 1 - reflect
    shown = reflect ** (1 - LIFT)  # an internal reflection, as displayed

    entry = -cos_i - START
    chord = 2 * cos_r
    turn = np.pi - 2 * r
    theta0 = np.pi - i
    stretch = torch.floor((s - entry) / chord)  # -1 before entry, then chord index
    inside = (s >= entry) & (stretch < m)
    before = s < entry
    after = ~before & ~inside

    k = stretch.clamp(min=0)
    th_a = theta0 - k * turn
    th_b = th_a - turn
    frac = (s - entry - k * chord) / chord
    in_u = torch.cos(th_a) + frac * (torch.cos(th_b) - torch.cos(th_a))
    in_v = torch.sin(th_a) + frac * (torch.sin(th_b) - torch.sin(th_a))

    mf = m.float()
    th_m = theta0 - mf * turn
    phi = -(2 * (i - r) + (mf - 1) * turn)
    out_len = s - entry - mf * chord
    out_u = torch.cos(th_m) + out_len * torch.cos(phi)
    out_v = torch.sin(th_m) + out_len * torch.sin(phi)

    pos_u = torch.where(before, START + s, torch.where(inside, in_u, out_u))
    pos_v = torch.where(before, b, torch.where(inside, in_v, out_v))

    # Probability of being on each stretch, and its displayed light.
    tail = torch.flip(torch.cumsum(torch.flip(q, [0]), 0), [0]) / q.sum()  # P(m >= j)
    p_inside = tail[(k + 1).long().clamp(max=MAX_CHORDS)]
    w_inside = transmit * shown**k
    w_out = torch.where(
        m == 0, reflect, transmit * transmit * shown ** (mf - 1).clamp(min=0)
    )
    p_out = q[m] / q.sum()
    weight = torch.where(
        before,
        torch.full_like(s, BEAM),
        torch.where(inside, w_inside / p_inside, w_out / p_out),
    )
    px = CENTRE[0] + SCALE * pos_v
    py = CENTRE[1] + SCALE * pos_u
    colour = sample_rgb[wl_index] * weight[:, None]
    return px, py, colour, s


def rim_mask() -> np.ndarray:
    """The drop's outline, an alpha mask drawn at 4x and shrunk."""
    big = np.zeros((HEIGHT * 4, WIDTH * 4), np.uint8)
    centre = (int(CENTRE[0] * 4 * 16), int(CENTRE[1] * 4 * 16))
    cv2.circle(
        big, centre, int(SCALE * 4 * 16), 255, int(RIM_WIDTH * 4), cv2.LINE_AA, 4
    )
    small = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def edge_distance(chords: int, n: float) -> float:
    """How far (drop radii) the rainbow ray of `chords` travels from the drop to the frame edge."""
    b, _ = rainbow_ray(chords, n)
    i, r = np.arcsin(b), np.arcsin(b / n)
    theta = np.pi - i - chords * (np.pi - 2 * r)
    phi = -(2 * (i - r) + (chords - 1) * (np.pi - 2 * r))
    u, v = np.cos(theta), np.sin(theta)
    du, dv = np.cos(phi), np.sin(phi)
    limits = []
    for pos, step, lo, hi in (
        (u, du, -CENTRE[1] / SCALE, (HEIGHT - CENTRE[1]) / SCALE),
        (v, dv, -CENTRE[0] / SCALE, (WIDTH - CENTRE[0]) / SCALE),
    ):
        if abs(step) > 1e-9:
            limits.append(((hi if step > 0 else lo) - pos) / step)
    return float(min(limits))


def bow_events():
    """
    The moments the soundtrack marks, computed from the same geometry (path
    lengths turned into video seconds): the beam reaching the drop; the light
    that passed straight through meeting at its focus below the drop; each
    bow's birth (its rainbow ray leaving the drop), with the path length at
    which each colour's rainbow ray leaves; and each bow's rainbow ray
    reaching the edge of the frame. Returns (seconds, kind, data), sorted.
    """
    n_mid = refractive_index(550.0)
    events = [(float(video_time(-1 - START)), "hit", None)]
    focus = n_mid / (2 * (n_mid - 1))  # paraxial focus of a ball lens, radii
    events.append((float(video_time(-1 - START + 2 + (focus - 1))), "focus", None))
    wavelengths = np.linspace(LAMBDA[1], LAMBDA[0], COLOURS)  # red .. violet
    for chords in range(2, MAX_CHORDS + 1):
        exits = []
        for c, w in enumerate(wavelengths):
            n = refractive_index(w)
            b, angle = rainbow_ray(chords, n)
            exits.append((exit_length(b, chords, n), c, angle))
        born = min(e[0] for e in exits)
        events.append((float(video_time(born)), "bow", (chords, exits)))
        b, _ = rainbow_ray(chords, n_mid)
        edge = exit_length(b, chords, n_mid) + edge_distance(chords, n_mid)
        edges = [e[0] for e in events if e[1] == "edge"]
        when = float(video_time(edge))
        if edge < S1 and all(abs(when - e) > 0.3 for e in edges):
            events.append((when, "edge", chords))
    return sorted(events, key=lambda e: e[0])


def soundtrack(style_name: str, rng: np.random.Generator):
    """
    A one-shot keyed to the light's travel (video_time of each path length).

    - Each colour has a note: COLOURS scale degrees, red lowest, violet highest.
    - The falling sunlight: one high note at once, and a soft choir "u" on
      the home chord held to the end; a brighter "o" chord joins while the
      bows are born. The beam reaching the drop: the home chord's bass.
    - Struck notes get ping-pong echoes (ECHO), which carry the bows on
      while the light spreads out to the edges.
    - The glint as the beam meets the drop (high), and the focus of the
      light passing straight through (a fifth): single notes.
    - Each bow's rainbow ray reaching the frame edges: a low note on both
      sides, a different chord tone per bow.
    - Each bow's birth (its rainbow ray leaving the drop, `bow_events`): its
      colours played in angular order from the outside in, ARPEGGIO_GAP apart:
      red to violet (rising) for the 1st and 3rd bows, violet to red (falling)
      for the reversed 2nd and 4th; the higher orders an octave up and softer.
      Panned left/right alternately, as the bow's two arms leave both sides.
    - When the light stops: the home chord rings out over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    struck = audio.silence(DURATION)  # the events, which get echoes
    kw = dict(circular=False)
    events = bow_events()
    hit = events[0][0]
    first_bow = next(w for w, k, _ in events if k == "bow")

    bass = scale.hz(0, -2)
    bass = 2 * bass if bass < 65 else bass
    audio.add(track, style.bass(bass, 0.45, length=4.0), hit, **kw)
    sun = audio.pan(style.voice(scale.hz(0, 1), 0.25), 0.0)
    audio.add(struck, sun, 0.1, **kw)  # the sunlight, already falling
    for n in scale.chord(0, 3, octave=-1):
        hz = float(audio.midi_to_hz(n))
        held = audio.choir(hz, DURATION - 1.8, 0.10, 1.2, 1.5, vowel="u")
        audio.add(track, held, 0.05, **kw)
    for n in scale.chord(4, 3, octave=-1):  # a brighter chord while the bows are born
        hz = float(audio.midi_to_hz(n))
        lift = audio.choir(hz, STOP[1] - 1.0 - first_bow, 0.07, 1.5, 1.2, vowel="o")
        audio.add(track, lift, first_bow, **kw)

    notes = 0
    for when, kind, data in events:
        if kind in ("hit", "focus"):
            degree = 7 if kind == "hit" else 4  # a glint off the top; the focus
            note = style.voice(scale.hz(degree), 0.4)
            audio.add(struck, audio.pan(note, 0.0), when, **kw)
            notes += 1
        elif kind == "edge":
            order = data - 1
            low = scale.hz((0, 4, 2, 5)[order - 1], -1)
            for side in (-0.7, 0.7):  # both arms of the bow reach the edges
                note = style.voice(low, (0.36, 0.3, 0.24, 0.2)[order - 1])
                audio.add(struck, audio.pan(note, side), when, **kw)
            notes += 2
        elif kind == "bow":
            chords, exits = data
            order = chords - 1
            reversed_colours = order % 2 == 0
            colours = sorted(range(COLOURS), reverse=reversed_colours)
            octave = 0 if order <= 2 else 1
            velocity = (0.55, 0.42, 0.28, 0.24)[order - 1]
            for j, c in enumerate(colours):
                side = -0.55 if (j % 2 == 0) else 0.55
                note = style.voice(
                    scale.hz(c, octave), velocity * rng.uniform(0.9, 1.05)
                )
                audio.add(struck, audio.pan(note, side), when + j * ARPEGGIO_GAP, **kw)
                notes += 1

    track += audio.delay(struck, ECHO, 0.4, 0.3, circular=False)
    rest = float(STOP[1])
    for k, n in enumerate(scale.chord(0, 4)):
        hz = float(audio.midi_to_hz(n))
        note = audio.pan(style.voice(hz, 0.32), 0.3 * (k - 1.5))
        audio.add(track, note, rest - 0.6 + 0.1 * k, **kw)
    audio.add(track, style.bass(bass, 0.4, length=4.0), rest - 0.6, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} notes; "
        + ", ".join(f"{k} {w:.2f} s" for w, k, _ in events)
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A light sheet through a circular cross-section of a raindrop, Monte Carlo.

    Parallel sunlight (impact parameter b in [-1, 1], drop radius 1) falls on
    a water circle. A ray at incidence i = asin b refracts to r = asin(b/n),
    with n(lambda) from a Cauchy fit for water. Inside, every chord has length
    2 cos r and turns the hit point by pi - 2r round the circle, so the k-th
    hit is at angle (pi - i) - k (pi - 2r) and a ray leaving after m chords
    has turned by D = 2(i - r) + (m - 1)(pi - 2r) (m = 0: the outside
    reflection, pi - 2i). D(b) is stationary at cos^2 i = (n^2 - 1)/(m^2 - 1):
    the caustic, the rainbow ray: 42 deg from the antisolar point for m = 2,
    51 deg for m = 3, colour-separated by the dispersion of n.

    Everything is closed form, so a photon is a point (b, lambda, m, s) and
    its position follows directly from s, the distance travelled. Photons are
    drawn on the GPU with Fresnel weights (unpolarised): the fraction
    surviving to each stretch divided by the chance of sampling it, so the
    beam and each chord carry exact weights and only the exit order is
    importance-sampled (ORDER_PROB). Each internal reflection's loss R is
    shown as R^(1 - LIFT) so the higher-order bows read at all.

    Time is the light's travel: path_length(t) runs from S0 (the beam already
    in frame) at constant speed, then eases to a stop over STOP, and the image
    rests. Each frame adds the photons with s in (s(prev), s(now)] to an
    accumulating canvas (frames render in order; any frame is still a pure
    function of t, since the canvas holds exactly the photons with s <=
    s(t)) and into a short-lived front layer (FRONT_HALF_LIFE) that shows the
    wavefront. The luminance Y is tone-mapped as 1 - exp(-EXPOSURE (Y^GAMMA +
    glow)), with linear halos (GLOWS) round the caustics; the hue comes from
    the (slightly blurred) linear colour with its chroma scaled by SATURATION,
    so the rainbow's colour fringes read on a phone; the hottest cores go
    white; then a bloom and the drop's rim on top. Photon draws use a torch generator seeded from
    settings.rng per frame, so a seed reproduces the video exactly. The
    posted render uses --seed 0 (the default), with the music_box soundtrack.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    base_seed = int(settings.rng.integers(1, 2**31))
    f = dict(device=device, dtype=torch.float32)
    wavelengths = np.linspace(*LAMBDA, 512)
    sample_rgb = torch.tensor(spectrum_rgb(wavelengths), **f)
    normal = 2 * SCALE * SCALE / PHOTONS_PER_UNIT  # the beam's density -> BEAM
    canvas = torch.zeros(HEIGHT, WIDTH, 3, **f)
    front = torch.zeros(HEIGHT, WIDTH, 3, **f)
    soft = gaussian_kernel(SOFT, device)
    chroma_soft = gaussian_kernel(CHROMA_SOFT, device)
    rim = torch.tensor(rim_mask(), **f)[..., None]
    rim_rgb = torch.tensor(RIM_COLOUR, **f)
    decay = 0.5 ** (1.0 / (FPS * FRONT_HALF_LIFE))
    speed = (S1 - S0) / (STOP[0] + 0.5 * (STOP[1] - STOP[0]))  # radii per second
    front_length = speed * FRONT_HALF_LIFE / np.log(2)
    state = {"frame": -1, "s": 0.0}
    for chords in (2, 3):
        b, angle = rainbow_ray(chords, refractive_index(550.0))
        logger.info(f"Bow of {chords - 1} reflection(s): b = {b:.3f}, {angle:.1f} deg")

    def add_light(s_lo, s_hi, frame):
        count = int(round((s_hi - s_lo) * PHOTONS_PER_UNIT))
        chunks = max(1, -(-count // CHUNK))
        for c in range(chunks):
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed + 1000 * frame + c)
            lo = s_lo + (s_hi - s_lo) * c / chunks
            hi = s_lo + (s_hi - s_lo) * (c + 1) / chunks
            px, py, colour, s = photons(
                lo, hi, count // chunks, gen, device, sample_rgb
            )
            splat(canvas, px, py, colour * normal)
            recent = torch.exp((s - s_hi) / front_length)  # the newest light glows
            splat(front, px, py, colour * (normal * recent)[:, None])

    def render(frame: int) -> torch.Tensor:
        s_now = float(path_length(frame / FPS))
        if frame != state["frame"] + 1:  # out of order: rebuild from nothing
            canvas.zero_()
            state["s"] = 0.0
        front.mul_(decay)
        if s_now > state["s"] + 1e-9:
            add_light(state["s"], s_now, frame)
        state["frame"], state["s"] = frame, s_now
        light = blur(canvas + FRONT_GAIN * front, soft)
        lum = light.mean(-1, keepdim=True).clamp_min(1e-6)
        glow = sum(g * wide_blur(lum, sigma, k) for sigma, k, g in GLOWS)
        brightness = 1 - torch.exp(-EXPOSURE * (lum**GAMMA + glow))
        smooth = blur(light, chroma_soft)
        chroma = smooth / smooth.mean(-1, keepdim=True).clamp_min(1e-6)
        tint = chroma - 1
        strength = tint.abs().amax(
            -1, keepdim=True
        )  # grain tints faintly, fringes strongly
        gate = ((strength - CHROMA_GATE[0]) / (CHROMA_GATE[1] - CHROMA_GATE[0])).clamp(
            0, 1
        )
        image = (brightness * (1 + (1 + (SATURATION - 1) * gate) * tint)).clamp_min(0)
        image = image + (brightness - image) * brightness**4  # hot cores go white
        image = image + BLOOM_GAIN * wide_blur(image, BLOOM_SIGMA, BLOOM_FACTOR)
        image = image * (1 - RIM_ALPHA * rim) + RIM_ALPHA * rim * rim_rgb
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
