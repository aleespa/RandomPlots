"""
The Primes Are Made of Waves

How the zeros of the zeta function build the prime numbers. Part 3 of 3.

The ink curve counts the primes up to x. It starts as Riemann's smooth guess, R(x), which
knows roughly how many primes there are but not where they are. Then the waves come in:
each zero of the zeta function, ½ + iγ, adds one wave, and its height γ is the wave's
frequency. Ten waves, and steps begin to appear at 2, 3, 5 and 7. A thousand waves, and
the curve is a perfect staircase: it climbs by exactly one at every prime, 2, 3, 5, 7,
11, … 97, and stays flat in between.

This is Riemann's explicit formula from 1859: π(x) = R(x) − Σ R(xᵖ), one term for every
zero ρ. The Riemann hypothesis says every zero sits on the line ½, so every wave grows
only like √x: the primes are as orderly as they can possibly be. In November 2026 that
was proved, with a computer-checked proof in Lean.

Send this to someone who thinks the primes are random.

Made with Python, NumPy & OpenCV.
#primenumbers #riemannhypothesis #numbertheory #mathart #randomplots
"""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mpmath
import numpy as np
from loguru import logger
from scipy.special import exp1
from scipy.special import zeta as real_zeta

from common.image_processing import ImageProcessingSettings
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    hook_masks,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 34.0  # seconds
FRAME_COUNT = frame_count(DURATION)

HOOK = "Add waves. Get the primes."
HOOK_Y = 270
HOOK_COLOUR = "#f4f0e7"
COVER_SECONDS = 19.5  # grid cover (tools.post_reel): the finished staircase

# dark: "paper" is the background, "ink" the curve and text
PAPER = "#060912"
INK = "#f4f0e7"
RED = "#ff6f61"
WAVE = "#5fc9e0"
MUTED = "#7c8396"

X_MIN, X_MAX = 1.5, 100.0  # the count π(x) is drawn on this range
SAMPLES = 2400  # points of x
MOBIUS_TERMS = 8  # R(x^ρ) = Σ_m μ(m)/m Ei(ρ ln x / m), m ≤ this
ZEROS = 1000  # waves used at most
CACHE = Path("outputs") / "_cache"
# the plot: x from 0 to 100 across, π from 0 to 26 up
PLOT = (110, 990, 1300, 500)  # x0, x1, y(π=0), y(π=26)
PI_TOP = 26.0
WAVE_PANEL = (1400, 1570)  # y range of the newest-wave panel
SUPERSAMPLE = 2  # curves are drawn at this multiple of the frame size
# phases (seconds)
# the explanation: (start, end) of each caption, in seconds
STAIRS = (0.0, 1.8)  # the true count, lit: it steps up by 1 at every prime
GUESS = (1.8, 3.8)  # Riemann's smooth R(x) fades in
ADD = (3.8, 4.6)  # now add one wave for each zero
FIRST = (4.4, 10.0)  # waves 1-10 come in one at a time
MANY = (10.0, 19.0)  # 10 -> 1000 waves
PRIMES = (19.0, 23.5)  # each prime lights up on its step
FORMULA = (24.0, 30.0)
HOOK_END = 23.8
WAVE_IN = 0.45  # seconds for a wave to fade fully into the sum (first phase)

MU = {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0, 9: 0, 10: 1}


def ei(z: np.ndarray) -> np.ndarray:
    """The exponential integral Ei of a complex argument off the real axis."""
    z = np.asarray(z, complex)
    return -exp1(-z) + 1j * np.pi * np.sign(z.imag)


def riemann_r(x: np.ndarray, terms: int = 100) -> np.ndarray:
    """Riemann's R(x) = 1 + Σ_k (ln x)^k / (k k! ζ(k+1)) (Gram series)."""
    lx = np.log(x)
    total = np.ones_like(x)
    term = np.ones_like(x)
    for k in range(1, terms + 1):
        term = term * lx / k
        total = total + term / (k * real_zeta(k + 1))
    return total


def zero_heights(count: int = ZEROS) -> np.ndarray:
    """γ_1 < γ_2 < ... of the first `count` zeros ½ + iγ, cached (mpmath is slow)."""
    path = CACHE / f"zeta_zeros_{count}.npy"
    if path.exists():
        return np.load(path)
    with ThreadPoolExecutor(8) as pool:
        heights = list(
            pool.map(lambda k: float(mpmath.zetazero(k).imag), range(1, count + 1))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(heights))
    return np.array(heights)


def waves(x: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    W_k(x) = -2 Re R(x^ρ_k) for ρ_k = ½ + iγ_k: the k-th zero's wave (with its
    conjugate zero), so that π(x) ≈ R(x) + Σ_k W_k(x) + small terms. Cached.
    """
    path = CACHE / f"prime_waves_{len(gammas)}_{len(x)}_{MOBIUS_TERMS}.npy"
    if path.exists():
        return np.load(path)
    lx = np.log(x)

    def one(g):
        rho = 0.5 + 1j * g
        total = np.zeros_like(x, dtype=complex)
        for m in range(1, MOBIUS_TERMS + 1):
            if MU[m]:
                total += MU[m] / m * ei(rho * lx / m)
        return (-2 * total.real).astype(np.float32)

    with ThreadPoolExecutor(16) as pool:
        table = np.stack(list(pool.map(one, gammas)))
    np.save(path, table)
    return table


def primes_below(n: int) -> list[int]:
    sieve = np.ones(n + 1, bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return [int(p) for p in np.flatnonzero(sieve)]


def halo(mask, sigma=6, gain=2.0):
    pad = int(2 * sigma)
    return np.clip(cv2.GaussianBlur(np.pad(mask, pad), (0, 0), sigma) * gain, 0, 1)


def window(seconds, start, end, fade=0.4):
    rise = smoothstep(np.clip((seconds - start) / fade, 0, 1))
    fall = 1 - smoothstep(np.clip((seconds - end + fade) / fade, 0, 1))
    return float(rise * fall)


def shown_count(c: float) -> int:
    """The counter: waves in the sum, counting one that is still fading in."""
    return int(np.ceil(c - 1e-9)) if c < ZEROS else ZEROS


def wave_count(seconds: float) -> float:
    """How many waves are in the sum at a moment (fractional while one fades in)."""
    if seconds < FIRST[0]:
        return 0.0
    if seconds < FIRST[1]:
        step = (FIRST[1] - FIRST[0]) / 10
        k, into = divmod(seconds - FIRST[0], step)
        return k + smoothstep(min(into / WAVE_IN, 1.0))
    if seconds < MANY[1]:
        u = smoothstep((seconds - MANY[0]) / (MANY[1] - MANY[0]))
        return 10.0 * (ZEROS / 10.0) ** u
    return float(ZEROS)


def generate(settings: ImageProcessingSettings = None):
    """
    Riemann's explicit formula for the prime-counting function,
    π(x) ≈ R(x) - Σ_ρ R(x^ρ) - 1/ln x + arctan(π/ln x)/π, with R(x) by the Gram series
    and R(x^ρ) = Σ_{m≤8} μ(m)/m Ei(ρ ln x / m) (Ei of complex argument through
    scipy's exp1); each zero ρ = ½ + iγ (with its conjugate) contributes the wave
    W_k(x) = -2 Re R(x^ρ_k). The first 1000 zeros (mpmath.zetazero) and the 1000 x 2400
    wave table are cached in outputs/_cache; partial sums are cumulative, so a frame
    with N(t) waves is one row plus a fraction of the next.

    0-4.4 s: the explanation (the true staircase lit, then R(x) fading in over it,
    then "add one wave for each zero"). 4.4-10 s: waves 1-10 fade in one at a time (each over 0.45 s), the newest drawn
    alone in a panel under the plot (normalised) with its zero and frequency. 10-19 s: N grows as
    10 * 100^smoothstep(u) to 1000. 19-23.5 s: each prime's step corner lights in red
    with its label. 24-30 s: the formula; then the closing lines. The true staircase
    is a faint red ghost throughout.

    Rendering: cv2 on a thread pool, curves drawn at 2x and averaged down, text as
    LaTeX masks built once. Frames are pure functions of t.
    """
    settings = settings or ImageProcessingSettings(1)

    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    gammas = zero_heights()
    table = waves(x, gammas)  # (ZEROS, SAMPLES)
    partial = np.vstack([np.zeros(SAMPLES, np.float32), np.cumsum(table, axis=0)])
    lx = np.log(x)
    base = riemann_r(x) - 1 / lx + np.arctan(np.pi / lx) / np.pi
    primes = primes_below(int(X_MAX))
    true_count = np.searchsorted(primes, x, side="right").astype(float)
    full = base + partial[-1]
    off = np.abs(full - true_count)[np.abs(x - np.round(x)) > 0.3]  # away from jumps
    logger.info(
        f"{len(gammas)} zeros up to γ = {gammas[-1]:.1f}; with all of them the formula "
        f"is within {off.max():.3f} of π(x) away from the jumps; {len(primes)} primes"
    )

    # --- geometry and colours ---
    px0, px1, py0, py1 = PLOT
    sx = px0 + (px1 - px0) * x / X_MAX
    to_y = lambda v: py0 - (py0 - py1) * v / PI_TOP
    paper = np.array(hex_to_bgr(PAPER), np.float32)
    ink = np.array(hex_to_bgr(INK), np.float32)
    red = np.array(hex_to_bgr(RED), np.float32)
    blue = np.array(hex_to_bgr(WAVE), np.float32)
    muted = np.array(hex_to_bgr(MUTED), np.float32)
    background = np.empty((HEIGHT, WIDTH, 3), np.float32)
    background[:] = paper
    # axes and the true staircase (faint), drawn once
    fixed = 4 * SUPERSAMPLE

    def pts(xs, ys):
        return np.round(np.stack([xs, ys], -1) * fixed).astype(np.int32)

    big = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE, 3), np.float32)
    big[:] = paper
    grey = tuple(float(v) for v in muted)
    cv2.line(
        big,
        *[tuple(p) for p in pts(np.array([px0, px1]), np.array([py0, py0]))],
        grey,
        SUPERSAMPLE,
        cv2.LINE_AA,
        2,
    )
    cv2.line(
        big,
        *[tuple(p) for p in pts(np.array([px0, px0]), np.array([py0, py1]))],
        grey,
        SUPERSAMPLE,
        cv2.LINE_AA,
        2,
    )
    stair_x = [0.0]
    stair_y = [0.0]
    for k, p in enumerate(primes, start=1):
        stair_x += [p, p]
        stair_y += [k - 1, k]
    stair_x.append(X_MAX)
    stair_y.append(len(primes))
    ghost_colour = tuple(float(0.35 * r + 0.65 * q) for r, q in zip(red, paper))
    cv2.polylines(
        big,
        [pts(px0 + (px1 - px0) * np.array(stair_x) / X_MAX, to_y(np.array(stair_y)))],
        False,
        ghost_colour,
        2 * SUPERSAMPLE,
        cv2.LINE_AA,
        2,
    )
    for k, p in enumerate(primes):  # red ticks under the axis
        tx = px0 + (px1 - px0) * p / X_MAX
        cv2.line(
            big,
            *[tuple(q) for q in pts(np.array([tx, tx]), np.array([py0 + 4, py0 + 14]))],
            ghost_colour,
            SUPERSAMPLE,
            cv2.LINE_AA,
            2,
        )
    background = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    stair_big = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE), np.float32)
    cv2.polylines(
        stair_big,
        [pts(px0 + (px1 - px0) * np.array(stair_x) / X_MAX, to_y(np.array(stair_y)))],
        False,
        1.0,
        3 * SUPERSAMPLE,
        cv2.LINE_AA,
        2,
    )
    stair_mask = cv2.resize(stair_big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    del stair_big
    del big

    # --- text, built once ---
    hook_lines = hook_masks(HOOK)
    halo_light = lambda m: halo(m, 7, 2.2)
    hook_halos = [halo_light(m) for m in hook_lines]
    axis_labels = [
        (latex_mask(r"$x$", 40), px1 + 14, py0 - 22),
        (latex_mask(r"$\pi(x)$", 40), px0 - 20, py1 - 56),
    ]
    tick_labels = [
        (latex_mask(rf"$\mathrm{{{v}}}$", 30), v) for v in (0, 25, 50, 75, 100)
    ]
    count_masks = {}  # built for every count shown

    seconds_all = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION
    counts = [wave_count(s) for s in seconds_all]
    for c in counts:
        n = shown_count(c)
        if n not in count_masks:
            word = "wave" if n == 1 else "waves"  # one line: one baseline
            count_masks[n] = latex_mask(rf"$\mathrm{{{n}\ {word}}}$", 46)
    newest_masks = {
        k: latex_mask(
            rf"$\mathrm{{zero\ {k + 1}{{:}}\ \ }}\frac{{1}}{{2}}+{gammas[k]:.2f}\,i"
            rf"\ \ \to\ \ \mathrm{{a\ wave\ of\ frequency\ {gammas[k]:.2f}}}$",
            36,
        )
        for k in range(10)
    }
    intro = [  # (window, [(mask, colour)]) under the plot, before the waves
        (
            STAIRS,
            [
                (latex_mask(r"$\mathrm{Count\ the\ primes\ up\ to\ }x{:}$", 44), red),
                (
                    latex_mask(
                        r"$\mathrm{the\ line\ steps\ up\ by\ 1\ at\ every\ prime}$", 40
                    ),
                    ink,
                ),
            ],
        ),
        (
            GUESS,
            [
                (
                    latex_mask(
                        r"$\mathrm{Riemann\text{'}s\ smooth\ guess\ }R(x){:}$", 44
                    ),
                    ink,
                ),
                (
                    latex_mask(
                        r"$\mathrm{close{,}\ but\ it\ can\text{'}t\ see\ the\ primes}$",
                        40,
                    ),
                    ink,
                ),
            ],
        ),
        (
            ADD,
            [
                (latex_mask(r"$\mathrm{Now\ add\ one\ wave}$", 44), blue),
                (latex_mask(r"$\mathrm{for\ each\ zero\ of\ }\zeta$", 44), blue),
            ],
        ),
    ]
    prime_masks = [latex_mask(rf"$\mathrm{{{p}}}$", 30) for p in primes]
    primes_count = latex_mask(rf"$\mathrm{{{len(primes)}\ primes\ below\ 100}}$", 46)
    formula = [
        latex_mask(r"$\pi(x)\ =\ R(x)\ -\ \sum_{\rho}\ R(x^{\rho})$", 54),
        latex_mask(
            r"$\mathrm{one\ wave\ for\ every\ zero\ }\rho=\frac{1}{2}+i\gamma$", 40
        ),
    ]
    rh_lines = [
        latex_mask(
            r"$\mathrm{On\ the\ line\ }\frac{1}{2}\mathrm{{,}\ every\ wave\ grows\ only\ like\ }\sqrt{x}$",
            40,
        ),
        latex_mask(r"$\mathrm{the\ primes\ are\ as\ orderly\ as\ they\ can\ be}$", 40),
    ]
    close_lines = [
        latex_mask(r"$\mathrm{The\ primes\ are\ made\ of\ waves}$", 52),
        latex_mask(r"$\mathrm{Part\ 3\ of\ 3\ \cdot\ @random\_plot}$", 34),
    ]
    halos = {
        id(m): halo_light(m) for m in formula + rh_lines + close_lines + [primes_count]
    }

    def text(image, mask, x0, y0, colour, alpha, anchor="centre"):
        if alpha <= 0:
            return
        g = halos.get(id(mask))
        if g is not None:
            pad = (g.shape[0] - mask.shape[0]) // 2
            gx = x0 - pad if anchor == "left" else x0
            blit_mask(image, g, gx, y0 - pad, paper, 0.85 * alpha, anchor)
        blit_mask(image, mask, x0, y0, colour, alpha, anchor)

    def render(frame):
        seconds = seconds_all[frame]
        c = counts[frame]
        n = int(np.floor(c + 1e-9)) if c < ZEROS else ZEROS
        frac = c - n if n < ZEROS else 0.0
        y_count = base + partial[n] + (frac * table[n] if n < ZEROS else 0)
        image = background.copy()

        # the count, drawn at 2x
        big = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE, 3), np.float32)
        y_screen = np.clip(to_y(y_count), py1 - 80, py0 + 60)
        cv2.polylines(
            big,
            [pts(sx, y_screen)],
            False,
            (1.0, 1.0, 1.0),
            round(3 * SUPERSAMPLE),
            cv2.LINE_AA,
            2,
        )
        # the newest wave, alone, in the panel below (first phase), or the latest
        if seconds < MANY[1] + 0.5 and n >= 1 or (FIRST[0] < seconds and frac > 0):
            k = min(max(n - 1 + (1 if frac > 0 else 0), 0), ZEROS - 1)
            w = table[k]
            w = w / (np.abs(w).max() + 1e-9)
            mid = 0.5 * (WAVE_PANEL[0] + WAVE_PANEL[1])
            amp = 0.45 * (WAVE_PANEL[1] - WAVE_PANEL[0])
            a = window(seconds, FIRST[0], MANY[1] + 0.3, 0.4)
            panel = np.zeros_like(big)
            cv2.polylines(
                panel,
                [pts(sx, mid - amp * w)],
                False,
                (1.0, 1.0, 1.0),
                round(2 * SUPERSAMPLE),
                cv2.LINE_AA,
                2,
            )
            mask_panel = cv2.resize(
                panel[..., 0], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
            )
            image += (a * mask_panel)[..., None] * (blue - image)
        mask_curve = cv2.resize(
            big[..., 0], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
        )
        a_stairs = window(seconds, -1.0, GUESS[1], 0.6)
        if a_stairs > 0:  # the true count, lit while it is explained
            image += (0.85 * a_stairs * stair_mask)[..., None] * (red - image)
        a_curve = smoothstep(np.clip((seconds - GUESS[0]) / 0.7, 0, 1))
        mask_curve = a_curve * mask_curve
        image += mask_curve[..., None] * (ink - image)
        small = cv2.resize(
            mask_curve, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA
        )
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (WIDTH, HEIGHT))
        image += (0.9 * glow)[..., None] * (ink - paper) * 0.5  # a soft glow

        # axes labels and ticks
        for m, ax, ay in axis_labels:
            blit_mask(image, m, int(ax), int(ay), muted, 1.0)
        for m, v in tick_labels:
            blit_mask(
                image,
                m,
                int(px0 + (px1 - px0) * v / X_MAX),
                py0 + 20,
                muted,
                1.0,
                "centre",
            )

        # the explanation under the plot
        for (start, end), lines in intro:
            a = window(seconds, start - (0.6 if start == 0 else 0.0), end, 0.3)
            yy = WAVE_PANEL[0] + 20
            for mask, colour in lines:
                blit_mask(image, mask, WIDTH // 2, yy, colour, a, "centre")
                yy += mask.shape[0] + 16

        # the newest wave's zero (first phase)
        if FIRST[0] <= seconds < FIRST[1] + 0.4:
            k = min(int(np.ceil(c)) - 1, 9)
            if k >= 0:
                a = window(seconds, FIRST[0], FIRST[1] + 0.4, 0.3)
                blit_mask(image, newest_masks[k], px0, WAVE_PANEL[1] + 20, blue, a)

        # the wave count
        a_count = window(seconds, FIRST[0] - 0.3, FORMULA[0], 0.5)
        if a_count > 0:
            m = count_masks[shown_count(c)]
            y0 = 420
            blit_mask(image, m, px1 - m.shape[1], y0, ink, a_count)

        # primes light up on their steps
        if seconds >= PRIMES[0]:
            step = (PRIMES[1] - PRIMES[0]) / len(primes)
            for k, p in enumerate(primes):
                t_on = PRIMES[0] + k * step
                a = smoothstep(np.clip((seconds - t_on) / 0.25, 0, 1)) * window(
                    seconds, PRIMES[0], 99, 0.3
                )
                if a <= 0:
                    continue
                cx = int(px0 + (px1 - px0) * p / X_MAX)
                cy = int(to_y(k + 1))
                cv2.circle(
                    image,
                    (cx, cy),
                    6,
                    tuple(
                        float(v) for v in (image[cy, cx] + a * (red - image[cy, cx]))
                    ),
                    -1,
                    cv2.LINE_AA,
                )
                m = prime_masks[k]
                blit_mask(image, m, cx - 10 - m.shape[1], cy - m.shape[0] - 4, red, a)
            text(
                image,
                primes_count,
                WIDTH // 2,
                WAVE_PANEL[0] + 30,
                red,
                window(seconds, PRIMES[1] - 0.3, FORMULA[0], 0.4),
            )

        # top text: the hook, then the formula, then the close
        a_hook = window(seconds, -1.0, HOOK_END, 0.6)
        if a_hook > 0:
            yy = HOOK_Y
            for mask, g in zip(hook_lines, hook_halos):
                blit_mask(image, g, WIDTH // 2, yy - 14, paper, 0.8 * a_hook, "centre")
                blit_mask(image, mask, WIDTH // 2, yy, ink, a_hook, "centre")
                yy += int(mask.shape[0] * 1.25)
        a_formula = window(seconds, FORMULA[0], 99, 0.6)
        if a_formula > 0:
            yy = 250
            for mask in formula:
                text(image, mask, WIDTH // 2, yy, ink, a_formula)
                yy += mask.shape[0] + 18
            a_rh = window(seconds, FORMULA[0] + 1.6, FORMULA[1], 0.5)
            yy = WAVE_PANEL[0]
            for mask in rh_lines:
                text(image, mask, WIDTH // 2, yy, ink, a_rh)
                yy += mask.shape[0] + 14
        a_close = window(seconds, FORMULA[1], 99, 0.6)
        if a_close > 0:
            yy = WAVE_PANEL[0]
            for mask, colour in zip(close_lines, (red, ink)):
                text(image, mask, WIDTH // 2, yy, colour, a_close)
                yy += mask.shape[0] + 18
        fade = 1 - smoothstep(np.clip((seconds - (DURATION - 0.5)) / 0.5, 0, 1))
        return np.clip(image * fade + paper * (1 - fade), 0, 255)

    render_frames(settings, render, FRAME_COUNT)
    settings.save_video(fps=FPS, crf=18)
    gc.collect()


if __name__ == "__main__":
    generate()
