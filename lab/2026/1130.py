"""
Random Numbers That Refuse to Touch

Each thread is one eigenvalue of a random matrix, a 72×72 grid of random numbers
that slowly changes. At the top only its diagonal is switched on, so the eigenvalues
are just independent random numbers: they wander, collide and cross freely. Going
down, the other entries fade in and the eigenvalues start to feel each other.

By the bottom, no two threads touch. They swerve apart at the last moment and keep
an almost even spacing. This is level repulsion, the heart of random matrix theory:
Eugene Wigner found it in the energy levels of heavy nuclei, and the same spacings
turn up in the zeros of the Riemann zeta function and in bus arrivals in Cuernavaca.

Every collision strikes a marimba bar, pitched by where it happens. The clatter
thins to the odd near miss as the threads learn to dodge, a soft chord settles in,
and the last row of eigenvalues plays itself, each note waiting out the gap to
its neighbour.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#randommatrixtheory #eigenvalues #probability #mathart #randomplots
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
    frame_count,
    hex_to_bgr,
    palette_lut,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Watch random numbers learn to avoid each other'
HOOK_Y = 780
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds, one-shot
FRAME_COUNT = frame_count(DURATION)

# --- The matrices -------------------------------------------------------------------
N = 72  # levels
SIGMA_D = 0.8  # spread of the independent (diagonal) levels
EPS = 0.45  # how far the free levels wander: M0 + EPS (M1 cos θ + M2 sin θ)
EPS_G = 1.2  # the same for the coupled matrix, whose levels move less
TURNS = 1.1  # turns of θ from the top of the screen to the bottom
U_ON, U_FULL = 0.22, 0.78  # coupling starts / is full at these fractions of the height
G_MIN = 0.01  # the first notch of the mixing angle, as a fraction of 90°

# --- Layout & time --------------------------------------------------------------------
Y_TOP, Y_BOTTOM = 150, 1790  # px: the drawing area
X_LEFT, X_RIGHT = 40, WIDTH - 40
LAMBDA_MAX = 2.25  # |λ| mapped to the margins
T_PEN = 13.3  # s: the pen reaches the bottom
FADE = 0.4  # s fade at the end
SS = 2  # supersampling of the static drawing

# --- Collisions ----------------------------------------------------------------------
TOUCH = 0.12  # a closest approach under this many local spacings is a touch
CLOSE = 0.45  # ... and under this, once coupled, a close call (sound only)
U_CLOSE = 0.6  # close calls are heard below this height
FLASH = 0.55  # s of afterglow for a touch

# --- Look -------------------------------------------------------------------------
BACKGROUND = "#05070d"
PALETTE = ["#2f6bff", "#6fc3e8", "#f3eedb", "#ffb94a", "#ff6a3d"]  # by position
BEAD = hex_to_bgr("#fff4dc")
SPARK = np.array(hex_to_bgr("#fffaf0"), np.float32)
LINE_W = 1.6  # px

# --- Sound ----------------------------------------------------------------------------
STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
NOTE_GAP = 0.13  # s: at most one touch note per gap
CLOSE_GAP = 0.4  # s: ... and one close-call note
ROLL = 9  # central levels of the last row played as the closing roll


def gue(rng):
    """A complex Hermitian Gaussian matrix with E|H_ij|^2 = 1/N (spectrum ~ [-2, 2])."""
    x = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    return (x + x.conj().T) / np.sqrt(2 * N)


def mixing(u):
    """
    The mixing angle φ(u): 0 above U_ON, then (π/2) G_MIN^(1 - w), w eased from
    0 to 1 by U_FULL, so the coupling sin φ grows geometrically and lands
    smoothly on a pure random matrix.
    """
    w = smoothstep((np.asarray(u, float) - U_ON) / (U_FULL - U_ON))
    phi = 0.5 * np.pi * G_MIN ** (1.0 - w)
    return np.where(np.asarray(u) < U_ON, 0.0, phi)


def pen_u(t):
    """Fraction of the height drawn at time t: steady, easing into the bottom."""
    x = np.clip(np.asarray(t, float) / T_PEN, 0.0, 1.0)
    return 0.06 + 0.94 * (1.0 - (1.0 - x) ** 1.6)


def pen_time(u):
    """Inverse of pen_u (u below its start maps to 0)."""
    x = np.clip((np.asarray(u, float) - 0.06) / 0.94, 0.0, 1.0)
    return T_PEN * (1.0 - (1.0 - x) ** (1 / 1.6))


def spectra(rng, rows):
    """Sorted eigenvalues of H(u) for the given heights u: shape (len(rows), N)."""
    diag = [SIGMA_D * rng.standard_normal(N) for _ in range(3)]
    full = [gue(rng) for _ in range(3)]
    norm, norm_g = np.sqrt(1 + EPS**2), np.sqrt(1 + EPS_G**2)
    out = np.empty((len(rows), N))
    for k, u in enumerate(rows):
        theta = 2 * np.pi * TURNS * u
        c, s = np.cos(theta), np.sin(theta)
        d = (diag[0] + EPS * (c * diag[1] + s * diag[2])) / norm
        f = (full[0] + EPS_G * (c * full[1] + s * full[2])) / norm_g
        phi = float(mixing(u))
        out[k] = np.linalg.eigvalsh(np.cos(phi) * np.diag(d) + np.sin(phi) * f)
    return out


def touches(lam, limit=TOUCH):
    """
    Closest approaches of neighbouring levels: local minima in u of the gap,
    in units of the local mean spacing (the mean of the 7 surrounding gaps).
    Returns arrays (row index, pair index, normalised gap) with gap < limit.
    """
    gaps = np.diff(lam, axis=1)  # (rows, N - 1)
    pad = np.pad(gaps, ((0, 0), (3, 3)), mode="edge")
    local = np.mean([pad[:, k : k + N - 1] for k in range(7)], axis=0)
    s = gaps / local
    minimum = (s[1:-1] <= s[:-2]) & (s[1:-1] < s[2:]) & (s[1:-1] < limit)
    r, i = np.nonzero(minimum)
    return r + 1, i, s[r + 1, i]


def lam_to_x(lam):
    return X_LEFT + (X_RIGHT - X_LEFT) * (np.asarray(lam) + LAMBDA_MAX) / (
        2 * LAMBDA_MAX
    )


def soundtrack(lam, rows, events, style_name, rng):
    """
    One-shot. Touches (closest approaches under TOUCH spacings) in the order the
    pen reaches them strike the style's voice, at most one per NOTE_GAP s: pitch
    from the position (left low, right high, scale degrees -4 .. 9), louder the
    closer the touch, panned by x. They are dense while the levels are free and
    thin out as the coupling grows. Below U_CLOSE, close calls (closest
    approaches between TOUCH and CLOSE spacings, the swerves of repelled
    levels) strike softer notes, at most one per CLOSE_GAP s. A soft pad on the
    home chord swells in with the coupling. When the pen lands, the ROLL central levels of the last row
    are struck left to right with onsets proportional to their positions, so
    the rhythm is their (repelled, near-even) spacing, over a bass note.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    when, xs, s = events
    last = -1.0
    notes = 0
    for k in np.argsort(when):
        if when[k] - last < NOTE_GAP or when[k] > T_PEN:
            continue
        last = when[k]
        pos = float(np.clip(xs[k] / LAMBDA_MAX, -1, 1))
        degree = int(round(-4 + 13 * (pos + 1) / 2))
        velocity = 0.2 + 0.4 * (1 - s[k] / TOUCH)
        note = audio.pan(style.voice(scale.hz(degree), velocity), 0.7 * pos)
        audio.add(track, note, float(when[k]), **kw)
        notes += 1

    r, i, s = touches(lam, CLOSE)
    keep = (rows[r] >= U_CLOSE) & (s >= TOUCH)
    r, i, s = r[keep], i[keep], s[keep]
    close_when = pen_time(rows[r])
    close_lam = 0.5 * (lam[r, i] + lam[r, i + 1])
    last = -1.0
    calls = 0
    for k in np.argsort(close_when):
        if close_when[k] - last < CLOSE_GAP or close_when[k] > T_PEN:
            continue
        last = close_when[k]
        pos = float(np.clip(close_lam[k] / LAMBDA_MAX, -1, 1))
        degree = int(round(-4 + 13 * (pos + 1) / 2))
        velocity = 0.12 + 0.2 * (1 - s[k] / CLOSE)
        note = audio.pan(style.voice(scale.hz(degree), velocity), 0.7 * pos)
        audio.add(track, note, float(close_when[k]), **kw)
        calls += 1

    on = float(pen_time(U_ON + 0.25 * (U_FULL - U_ON)))
    length = DURATION - on
    for degree in (0, 2, 4):
        voice = audio.pad(scale.hz(degree, -1), length, 0.16, 4.0, 1.5, 0.3)
        audio.add(track, voice, on, **kw)

    last_row = lam[-1]
    centre = N // 2 - ROLL // 2
    roll = last_row[centre : centre + ROLL]
    land = T_PEN + 0.1
    span = 1.4  # s for the whole roll
    onsets = land + span * (roll - roll[0]) / (roll[-1] - roll[0])
    for j, (at, value) in enumerate(zip(onsets, roll)):
        pos = float(value / LAMBDA_MAX)
        note = audio.pan(style.voice(scale.hz(j), 0.45), 0.7 * pos)
        audio.add(track, note, float(at), **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.55, length=3.0), land, **kw)
    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} touch notes, "
        f"{calls} close calls, "
        f"roll gaps {np.round(np.diff(onsets), 3)}"
    )
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Level dynamics of a random Hermitian matrix with a coupling switched on.

    H(u) = cos φ(u) D(u) + sin φ(u) G(u), where u in [0, 1] is the height on screen,
    D(u) is diagonal (N independent N(0, SIGMA_D²) levels) and G(u) a GUE matrix
    (E|G_ij|² = 1/N, semicircle on [-2, 2]); both wander as M0 + EPS (M1 cos θ +
    M2 sin θ), normalised, with θ = 2π TURNS u, so each stays in its ensemble.
    φ(u) = 0 above U_ON (the levels are independent and cross), then grows from
    (π/2) G_MIN geometrically to π/2 at U_FULL (`mixing`; Wigner-Dyson
    repulsion, P(s) ~ s²). The
    sorted eigenvalues (eigvalsh) are computed once for every output pixel row
    (seconds) and drawn as polylines, each segment coloured by position, into a
    static 2x image.

    A frame is a pure function of t: the pen height pen_u(t) reveals the static
    drawing above it (a soft edge and a fading "wet ink" glow just behind it),
    beads sit at the interpolated eigenvalues on the pen row with a blurred glow,
    and every touch (closest approach of neighbours, under TOUCH local mean
    spacings) the pen has passed in the last FLASH s sparks. Frames render on the
    thread pool. The posted video and soundtrack use --seed 0 (the default).
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    rows = (np.arange(Y_TOP, Y_BOTTOM + 1) - Y_TOP) / (Y_BOTTOM - Y_TOP)
    lam = spectra(rng, rows)
    xs = lam_to_x(lam)
    ys = Y_TOP + rows * (Y_BOTTOM - Y_TOP)

    r, i, s = touches(lam)
    ev_time = pen_time(rows[r])
    ev_lam = 0.5 * (lam[r, i] + lam[r, i + 1])
    ev_x, ev_y = lam_to_x(ev_lam), ys[r]
    ev_strength = 1.0 - s / TOUCH
    logger.info(
        f"{len(r)} touches; {np.sum(rows[r] < U_ON)} free, "
        f"{np.sum(rows[r] >= U_FULL)} fully coupled"
    )

    # The static drawing.
    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    lut = (palette_lut(PALETTE) * 255)[:, ::-1]
    big = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
    shift = 4
    px = np.round(xs * SS * (1 << shift)).astype(np.int64)
    py = np.round(ys * SS * (1 << shift)).astype(np.int64)
    colour_index = np.clip(
        np.round(255 * (lam + LAMBDA_MAX) / (2 * LAMBDA_MAX)), 0, 255
    ).astype(int)
    width = max(1, int(round(LINE_W * SS)))
    for n in range(N):
        for k in range(len(rows) - 1):
            c = tuple(float(v) for v in lut[colour_index[k, n]])
            cv2.line(
                big,
                (int(px[k, n]), int(py[k])),
                (int(px[k + 1, n]), int(py[k + 1])),
                c,
                width,
                cv2.LINE_AA,
                shift,
            )
    drawing = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    drawing = drawing.astype(np.float32)
    top_fade = smoothstep((np.arange(HEIGHT) - Y_TOP + 4) / 70.0).astype(np.float32)
    drawing *= top_fade[:, None, None]  # the threads emerge from the dark
    del big
    y_axis = np.arange(HEIGHT, dtype=np.float32)

    def render(frame):
        t = frame / (FRAME_COUNT - 1) * DURATION
        u = float(pen_u(t))
        pen_y = Y_TOP + u * (Y_BOTTOM - Y_TOP)
        behind = pen_y - y_axis
        reveal = np.clip(behind / 4.0 + 0.5, 0.0, 1.0)
        wet = 1.0 + 0.35 * np.exp(-np.maximum(behind, 0.0) / 90.0) * reveal
        if t > T_PEN:  # the ink dries over the rest
            wet = 1.0 + (wet - 1.0) * np.exp(-(t - T_PEN) / 0.4)
        image = bg + drawing * (reveal * wet)[:, None, None]

        glow = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        # Sparks at touches the pen has just passed.
        age = t - ev_time
        live = np.nonzero((age >= 0) & (age < FLASH))[0]
        for k in live:
            a = 1.0 - age[k] / FLASH
            amp = ev_strength[k] * a * a
            centre = (int(round(ev_x[k] * 16)), int(round(ev_y[k] * 16)))
            cv2.circle(
                glow,
                centre,
                int((6 + 22 * (1 - a)) * 16),
                tuple(float(v) for v in SPARK * amp * 1.2),
                -1,
                cv2.LINE_AA,
                4,
            )
            cv2.circle(
                glow,
                centre,
                int(3 * 16),
                tuple(float(v) for v in SPARK * amp * 3.0),
                -1,
                cv2.LINE_AA,
                4,
            )

        # Beads on the pen row, while the pen moves (they fade into the rest).
        if t <= T_PEN + 1.0:
            fade_beads = 1.0 - smoothstep((t - T_PEN) / 1.0)
            row = u * (len(rows) - 1)
            k0 = min(int(row), len(rows) - 2)
            f = row - k0
            bx = xs[k0] * (1 - f) + xs[k0 + 1] * f
            colour = tuple(float(v) * fade_beads for v in BEAD)
            for x in bx:
                c = (int(round(x * 16)), int(round(pen_y * 16)))
                cv2.circle(image, c, 4 * 16, colour, -1, cv2.LINE_AA, 4)
                cv2.circle(
                    glow, c, 9 * 16, tuple(0.5 * v for v in colour), -1, cv2.LINE_AA, 4
                )
            cv2.line(
                glow,
                (0, int(pen_y)),
                (WIDTH - 1, int(pen_y)),
                tuple(25.0 * fade_beads for _ in range(3)),
                1,
            )

        image += cv2.GaussianBlur(glow, (0, 0), 7) * 1.3 + glow * 0.3
        image *= smoothstep((DURATION - t) / FADE)
        return image

    render_frames(settings, render, FRAME_COUNT, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    tag = settings.audio_tag
    track = soundtrack(
        lam,
        rows,
        (ev_time, ev_lam, s),
        tag if tag in STYLES else STYLE,
        settings.audio_rng,
    )
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
