"""
How a Fourier Transform Hears a Chord

The Fourier transform, drawn as a machine. Top: the sound wave of a chord,
three notes added together. Below, the same wave is wound around a circle,
its height becoming the distance from the centre, and the winding speed
slowly turns up.

At most speeds the loops spread evenly and their centre of mass, the bright
dot (drawn four times further out so you can see it), stays near the middle.
But when the winding speed matches one of the notes, every crest lands on the
same side, the drawing tips over and the dot swings out. Plot how far it
swings against the speed and you get the graph at the bottom: the Fourier
transform, ĝ(f) = ∫ g(t) e^(−2πift) dt. Each peak is a note hidden in the
wave, pulled out one at a time.

Listen: the chord hums muffled underneath while a soft tone glides up with
the winding speed. Near each note the two beat, slower and slower, until they
lock, and that note rings out clear and stays, so the chord builds itself.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#fouriertransform #signalprocessing #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger
from numba import njit

from common import audio
from common.image_processing import ImageProcessingSettings
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    hex_to_bgr,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

# The signal: g(tau) = w(tau) (1 + sum a_j cos(2 pi f_j tau + phi_j)), tau in [0, TW).
CHORDS = {  # just-intonation ratios, scaled so the lowest note is BASE_F
    "major": (4, 5, 6),
    "minor": (10, 12, 15),
    "major_sixth": (5, 6, 8),
}
BASE_F = 2.0  # winding cycles per unit of tau for the lowest note
TW = 8.0  # length of the wound signal: peak width ~ 2 / TW
TAPER = 0.5  # Tukey window: fraction of TW spent fading in and out
SAMPLES = 2400  # for the wave and the transform
WIND_SAMPLES = 90000  # points splatted along the winding
OMEGA_START, OMEGA_END = 1.3, 3.5  # the winding-speed sweep
SWEEP_START, SWEEP_END = 0.0, 12.8  # s
EASE_OUT = 0.3  # last fraction of the sweep spent slowing to a stop
LINGER = 1.6  # the sweep slows this many times over at each note
LINGER_WIDTH = 0.1  # in winding cycles per unit tau

# Layout (px).
X0, X1 = 90, 990
WAVE_Y, WAVE_AMP = 235, 75
LANE_Y, LANE_AMP = (390, 445, 500), 17
CX, CY = WIDTH / 2, 915
RADIUS = 370  # px for the largest value of g
COM_ZOOM = 4.0  # the centre of mass is drawn this many times further out
SPEC_Y, SPEC_H = 1570, 220
BOX = 2 * RADIUS + 40  # the winding's square
BX, BY = int(CX - BOX / 2), int(CY - BOX / 2)

SS = 2
SHIFT = 4
BG = np.array(hex_to_bgr("#07060c"), np.float32)
INK = hex_to_bgr("#ebe6ff")  # the curves
GLOW = hex_to_bgr("#5b4bd6")  # their halo
GUIDE = hex_to_bgr("#221f2e")
NOTE_COLOURS = ("#ff5c8a", "#ffa45c", "#4fd8e8")  # low, middle, high note

# Sound: a sustained voice holds each note once found, a struck one marks it.
SOUNDS = {  # tag: (sustain, strike, reverb rt60, wet)
    "strings": ("strings", audio.piano, 2.4, 0.3),
    "choir": ("choir", audio.bell, 3.0, 0.32),
    "bells": ("pad", audio.bell, 2.8, 0.3),
}
SOUND = "strings"  # an --audio-only tag naming one of SOUNDS overrides it
ROOTS = range(55, 61)  # MIDI of the lowest chord note (G3 .. C4)
PROBE_FADE = (11.4, 12.8)  # s: the winding tone fades out once all notes are found
WIND_LO = np.array(hex_to_bgr("#6a58e8"), np.float32)  # sparse winding
WIND_HI = np.array(hex_to_bgr("#f4f0ff"), np.float32)  # where loops pile up


@njit(nogil=True, cache=True)
def wind_density(out, g, tau, omega, scale, cx, cy):
    """
    Splat the winding z = g e^(-2 pi i omega tau) bilinearly into `out`,
    centred at (cx, cy) px, each point weighted by its arc length so the
    density is line coverage whatever the speed along the curve.
    """
    n = len(g)
    h, w = out.shape
    xs = np.empty(n)
    ys = np.empty(n)
    for i in range(n):
        a = -2.0 * np.pi * omega * tau[i]
        xs[i] = cx + scale * g[i] * np.cos(a)
        ys[i] = cy - scale * g[i] * np.sin(a)
    for i in range(n):
        j = (i + 1) % n
        weight = np.hypot(xs[j] - xs[i], ys[j] - ys[i])
        x0 = int(np.floor(xs[i]))
        y0 = int(np.floor(ys[i]))
        fx = xs[i] - x0
        fy = ys[i] - y0
        if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
            continue
        out[y0, x0] += weight * (1 - fx) * (1 - fy)
        out[y0, x0 + 1] += weight * fx * (1 - fy)
        out[y0 + 1, x0] += weight * (1 - fx) * fy
        out[y0 + 1, x0 + 1] += weight * fx * fy


def tukey(x):
    """The window: 0 at the ends of [0, 1], 1 in the middle, cosine tapers."""
    edge = np.minimum(x, 1.0 - x) / (TAPER / 2)
    return np.where(edge >= 1, 1.0, 0.5 - 0.5 * np.cos(np.pi * np.clip(edge, 0, 1)))


def signal(tau, amps, freqs, phases):
    """The windowed chord and its three windowed components."""
    w = tukey(tau / TW)
    parts = amps[:, None] * np.cos(2 * np.pi * freqs[:, None] * tau + phases[:, None])
    return w * (1.0 + parts.sum(0)), parts


def sweep_progress(t):
    """Normalised sweep progress in t: steady from the start, easing to a stop."""
    x = np.linspace(0.0, 1.0, 2001)
    speed = np.where(
        x < 1 - EASE_OUT,
        1.0,
        0.5 + 0.5 * np.cos(np.pi * (x - 1 + EASE_OUT) / EASE_OUT),
    )
    u = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return np.interp((t - SWEEP_START) / (SWEEP_END - SWEEP_START), x, u / u[-1])


def sweep_table(freqs):
    """omega as a function of sweep progress in [0, 1]: slower near the notes."""
    omega = np.linspace(OMEGA_START, OMEGA_END, 4001)
    density = 1.0 + LINGER * sum(
        np.exp(-(((omega - f) / LINGER_WIDTH) ** 2)) for f in freqs
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(omega))]
    )
    return omega, cumulative / cumulative[-1]


def chord_maths(rng):
    """Every random choice and the precomputed transform, shared by picture and sound."""
    chord = list(CHORDS)[int(rng.integers(len(CHORDS)))]
    ratios = np.array(CHORDS[chord], float)
    freqs = BASE_F * ratios / ratios[0]
    amps = rng.uniform(0.75, 1.0, 3)
    amps *= 0.97 / amps.sum()
    phases = rng.uniform(0, 2 * np.pi, 3)
    omega_grid, progress_grid = sweep_table(freqs)

    def omega_at(t):
        return np.interp(sweep_progress(t), progress_grid, omega_grid)

    tau = (np.arange(SAMPLES) + 0.5) * TW / SAMPLES
    g, _ = signal(tau, amps, freqs, phases)
    spec_w = np.linspace(OMEGA_START, OMEGA_END, 1500)
    spec_mag = np.abs(np.exp(-2j * np.pi * spec_w[:, None] * tau) @ g) / SAMPLES
    times = np.linspace(0, DURATION, 30001)
    found = np.array([times[np.searchsorted(omega_at(times), f)] for f in freqs])
    logger.info(
        f"Chord {chord}: f = {np.round(freqs, 3)}, a = {np.round(amps, 3)}, "
        f"found at t = {np.round(found, 2)} s"
    )
    return dict(
        chord=chord,
        ratios=ratios,
        freqs=freqs,
        amps=amps,
        phases=phases,
        omega_at=omega_at,
        spec_w=spec_w,
        spec_mag=spec_mag,
        found=found,
    )


def sustain(kind, freq, duration, velocity):
    """A held note of the chosen sustained voice, stereo."""
    if kind == "strings":
        return audio.strings(freq, duration, velocity, 0.5, 2.0, 0.3)
    if kind == "choir":
        return audio.choir(freq, duration, velocity, 0.6, 2.0, vowel="o")
    return audio.pad(freq, duration, velocity, 0.5, 2.0, 0.4)


def soundtrack(m, tag, rng):
    """
    The chord the picture analyses, made audible. The lowest note is a seeded
    MIDI root in ROOTS; the others follow the chord's just-intonation ratios.

    - The wave: the whole chord held quietly for the full 15 s and low-passed
      at 600 Hz, muffled, as heard before analysis.
    - The winding: a soft sine whose pitch is the winding speed, f(t) =
      f_low omega(t) / BASE_F, gliding up with the sweep. As it nears a note
      of the chord it beats against the bed, slower and slower, and locks in
      at the peak; it fades out over PROBE_FADE.
    - Each note found (omega = f_j, the same times the peaks colour and the
      lanes appear) is struck (`strike` voice, an octave up, loudness by its
      amplitude a_j, panned low-left to high-right) and then held clear by the
      sustained voice to the end, so the chord assembles note by note.
    - When the sweep stops (SWEEP_END) the full chord is struck once more,
      rolled low to high, and rings out over the rest.
    """
    name = tag if tag in SOUNDS else SOUND
    kind, strike, rt60, wet = SOUNDS[name]
    kw = dict(circular=False)
    sr = audio.SAMPLE_RATE
    f_low = float(audio.midi_to_hz(int(rng.choice(ROOTS))))
    notes = f_low * m["ratios"] / m["ratios"][0]
    pans = (-0.45, 0.0, 0.45)
    track = audio.silence(DURATION)

    bed = audio.silence(DURATION)
    for f, a in zip(notes, m["amps"]):
        audio.add(bed, sustain(kind, f, DURATION, 0.85 * a), 0.0, **kw)
    track += audio.lowpass(bed, 600.0, circular=False)

    t = np.arange(len(track)) / sr
    freq = f_low * m["omega_at"](t) / BASE_F
    phase = 2 * np.pi * np.cumsum(freq) / sr
    level = 0.14 * np.clip(t / 0.05, 0, 1)
    level *= 1 - smoothstep((t - PROBE_FADE[0]) / (PROBE_FADE[1] - PROBE_FADE[0]))
    probe = level * (np.sin(phase) + 0.15 * np.sin(2 * phase))
    track += probe[:, None] * np.sqrt(0.5)

    for f, a, when, p in zip(notes, m["amps"], m["found"], pans):
        audio.add(track, audio.pan(strike(2 * f, 0.5 + 0.8 * a), p), when, **kw)
        held = sustain(kind, f, DURATION - when, 0.55 + 0.8 * a)
        audio.add(track, held, when - 0.15, **kw)
    for i, (f, p) in enumerate(zip([*notes, 2 * notes[0]], [*pans, 0.2])):
        audio.add(track, audio.pan(strike(2 * f, 0.55), p), SWEEP_END + 0.07 * i, **kw)
    track = audio.lowpass(track, 7000.0, circular=False)
    track = audio.reverb(track, rng, rt60, wet, tone=4500.0, circular=False)
    logger.info(
        f"Soundtrack ({name}): notes {np.round(notes, 1)} Hz, "
        f"found at {np.round(m['found'], 2)} s"
    )
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The Fourier transform as a winding machine (after 3Blue1Brown). A chord
    g(tau) = w(tau) (1 + sum_j a_j cos(2 pi f_j tau + phi_j)): three
    just-intonation notes (a seeded choice of major 4:5:6, minor 10:12:15 or
    major sixth 5:6:8, the lowest at BASE_F), random amplitudes summing to
    0.97 (so g >= 0) and random phases, under a Tukey window w (TAPER), so the
    wave is a tone burst and the transform has almost no sidelobes. Each frame
    winds it at speed omega: z(tau) = g(tau) e^(-2 pi i omega tau). The
    winding is WIND_SAMPLES points splatted bilinearly into a density buffer
    (weighted by arc length, so density is line coverage) and tone-mapped
    1 - e^(-d): where loops pile up, at a matching note, the drawing burns
    white on one side. The dot is the centre of mass c(omega) = mean z, the
    finite Fourier transform of g, drawn COM_ZOOM times further out (as the
    caption says); |c| over the sweep is precomputed and
    traced at the bottom up to the current omega.

    omega(t) sweeps OMEGA_START -> OMEGA_END over SWEEP_START..SWEEP_END at a
    steady rate easing to a stop over the last EASE_OUT, and lingers near each
    note (local speed / (1 + LINGER e^(-(d/w)^2))); lookup tables invert both,
    so every frame is a pure function of t. When omega passes a note, its
    peak takes the note's colour and that note's pure cosine fades in as a lane
    under the wave. A bead runs along tau on both pictures every 2.5 s. Lines
    are drawn with cv2 at 2x and downsampled, glow from a blurred 1x layer, on
    the thread pool.
    """
    settings = settings or ImageProcessingSettings(1)
    m = chord_maths(settings.rng)
    freqs, amps, phases = m["freqs"], m["amps"], m["phases"]
    omega_at, found = m["omega_at"], m["found"]
    spec_w, spec_mag = m["spec_w"], m["spec_mag"]

    tau = (np.arange(SAMPLES) + 0.5) * TW / SAMPLES
    g, parts = signal(tau, amps, freqs, phases)
    scale = RADIUS / g.max()
    tau_fine = (np.arange(WIND_SAMPLES) + 0.5) * TW / WIND_SAMPLES
    g_fine, _ = signal(tau_fine, amps, freqs, phases)

    spec_scale = SPEC_H / spec_mag.max()
    spec_x = X0 + (spec_w - OMEGA_START) / (OMEGA_END - OMEGA_START) * (X1 - X0)
    spec_y = SPEC_Y - spec_mag * spec_scale

    note_bgr = [np.array(hex_to_bgr(c), np.float32) for c in NOTE_COLOURS]
    ink = np.array(INK, np.float32)
    near = np.stack([np.exp(-(((spec_w - f) / 0.16) ** 2)) for f in freqs])
    spec_col = ink[None] * (1 - near.sum(0).clip(0, 1))[:, None]
    for j in range(3):
        spec_col += near[j][:, None] * note_bgr[j][None]

    window = tukey(tau / TW)
    wave_x = X0 + tau / TW * (X1 - X0)
    wave_y = WAVE_Y - (g - window) * WAVE_AMP
    lane_y = [LANE_Y[j] - window * parts[j] / amps[j] * LANE_AMP for j in range(3)]

    def px(v):
        return np.round(np.asarray(v, float) * SS * (1 << SHIFT)).astype(np.int32)

    def pts(x, y):
        return px(np.stack([x, y], -1))

    def p1(x, y):  # a point on the 1x glow layer
        return int(round(x * (1 << SHIFT))), int(round(y * (1 << SHIFT)))

    wave_pts = pts(wave_x, wave_y)
    lane_pts = [pts(wave_x, ly) for ly in lane_y]
    spec_pts = pts(spec_x, spec_y)

    # The static guides: the circle g = 1, the axes, the baselines.
    base = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
    base[:] = BG.astype(np.uint8)
    cv2.circle(base, tuple(px((CX, CY))), int(px(scale)), GUIDE, SS, cv2.LINE_AA, SHIFT)
    for a, b in (
        ((CX - RADIUS - 20, CY), (CX + RADIUS + 20, CY)),
        ((CX, CY - RADIUS - 20), (CX, CY + RADIUS + 20)),
        ((X0, SPEC_Y), (X1, SPEC_Y)),
        ((X0, WAVE_Y), (X1, WAVE_Y)),
    ):
        cv2.line(base, tuple(px(a)), tuple(px(b)), GUIDE, SS, cv2.LINE_AA, SHIFT)

    def render(frame):
        t = frame / FPS
        omega = float(omega_at(t))
        big = base.copy()
        glow = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # The wave and the notes found so far.
        cv2.polylines(big, [wave_pts], False, INK, 2 * SS, cv2.LINE_AA, SHIFT)
        cv2.polylines(glow, [wave_pts // SS], False, GLOW, 5, cv2.LINE_AA, SHIFT)
        for j in range(3):
            a = smoothstep((t - found[j] + 0.1) / 0.6)
            if a > 0:
                col = tuple(float(c) for c in BG + a * (note_bgr[j] - BG))
                cv2.polylines(
                    big, [lane_pts[j]], False, col, 2 * SS, cv2.LINE_AA, SHIFT
                )
                cv2.polylines(
                    glow,
                    [lane_pts[j] // SS],
                    False,
                    tuple(float(c) * a for c in note_bgr[j]),
                    6,
                    cv2.LINE_AA,
                    SHIFT,
                )

        # Nearness to a note colours the centre of mass.
        weights = np.exp(-(((omega - freqs) / 0.1) ** 2))
        dot = ink * (1 - min(1.0, weights.sum()))
        for j in range(3):
            dot += weights[j] * note_bgr[j]
        dot_c = tuple(float(c) for c in dot)

        z = np.exp(-2j * np.pi * omega * tau)
        c = (g * z).mean()
        cx = CX + COM_ZOOM * scale * c.real
        cy = CY - COM_ZOOM * scale * c.imag
        cv2.line(
            big,
            tuple(px((CX, CY))),
            tuple(px((cx, cy))),
            dot_c,
            3 * SS,
            cv2.LINE_AA,
            SHIFT,
        )
        cv2.line(glow, p1(CX, CY), p1(cx, cy), dot_c, 6, cv2.LINE_AA, SHIFT)

        # The bead running along tau on both pictures.
        phase = (t / 2.5) % 1.0
        k = int(phase * (SAMPLES - 1))
        bead_a = np.sin(np.pi * phase) ** 0.5
        bead_c = tuple(255.0 * bead_a for _ in range(3))
        bz = g[k] * z[k]
        for bx, by in (
            (wave_x[k], wave_y[k]),
            (CX + scale * bz.real, CY - scale * bz.imag),
        ):
            cv2.circle(
                big, tuple(px((bx, by))), int(px(5)), bead_c, -1, cv2.LINE_AA, SHIFT
            )
            cv2.circle(glow, p1(bx, by), 14 << SHIFT, bead_c, -1, cv2.LINE_AA, SHIFT)

        # The transform, traced up to the current omega.
        n = int(np.searchsorted(spec_w, omega))
        if n >= 2:
            fill = np.concatenate(
                [spec_pts[:n], px([[spec_x[n - 1], SPEC_Y], [spec_x[0], SPEC_Y]])]
            )
            top = (SPEC_Y - SPEC_H - 40) * SS
            strip = big[top : (SPEC_Y + 20) * SS]
            under = np.zeros_like(strip)
            fill[:, 1] -= top << SHIFT
            cv2.fillPoly(under, [fill], (58, 34, 44), cv2.LINE_AA, SHIFT)
            cv2.add(strip, under, dst=strip)
            step = 16
            for i in range(0, n - 1, step):
                seg = spec_pts[i : min(n, i + step + 1)]
                col = tuple(float(v) for v in spec_col[min(i + step // 2, n - 1)])
                cv2.polylines(big, [seg], False, col, 3 * SS, cv2.LINE_AA, SHIFT)
                cv2.polylines(glow, [seg // SS], False, col, 8, cv2.LINE_AA, SHIFT)
        sx = X0 + (omega - OMEGA_START) / (OMEGA_END - OMEGA_START) * (X1 - X0)
        sy = SPEC_Y - abs(c) * spec_scale
        cv2.line(
            big,
            tuple(px((sx, SPEC_Y + 14))),
            tuple(px((sx, SPEC_Y - SPEC_H - 20))),
            GUIDE,
            SS,
            cv2.LINE_AA,
            SHIFT,
        )
        for x, y in ((sx, sy), (cx, cy)):
            cv2.circle(
                big, tuple(px((x, y))), int(px(9)), dot_c, -1, cv2.LINE_AA, SHIFT
            )
            cv2.circle(glow, p1(x, y), 26 << SHIFT, dot_c, -1, cv2.LINE_AA, SHIFT)

        image = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)

        # The winding, as light: overlapping loops add up.
        density = np.zeros((BOX, BOX), np.float32)
        wind_density(density, g_fine, tau_fine, omega, scale, CX - BX, CY - BY)
        halo = cv2.GaussianBlur(density, (0, 0), 6)
        core = 1 - np.exp(-2.5 * density)
        hot = 1 - np.exp(-0.15 * density * density)
        lit = (
            core[..., None] * WIND_LO
            + hot[..., None] * (WIND_HI - WIND_LO)
            + (1 - np.exp(-0.5 * halo))[..., None] * (0.7 * WIND_LO)
        )
        region = image[BY : BY + BOX, BX : BX + BOX]
        np.maximum(region, BG + lit, out=region)  # under the dot and its line
        image += 0.8 * cv2.GaussianBlur(glow, (0, 0), 7).astype(np.float32)
        # Redraw the dot crisp on top of the winding.
        cv2.circle(
            image,
            tuple(px((cx, cy)) // SS),
            int(px(9)) // SS,
            dot_c,
            -1,
            cv2.LINE_AA,
            SHIFT,
        )
        image = BG + (image - BG) * smoothstep((DURATION - t) / FADE)
        return image

    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(m, settings.audio_tag, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
