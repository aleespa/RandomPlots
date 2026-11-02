"""
The Map of All Julia Sets: the maths, as an Instagram Story.

Five 1080x1920 cards explaining the Reel of 2026-11-02 (lab/2026/1102.py):
the iteration, Julia sets, the Mandelbrot set as their map, the walk round
the main cardioid.
"""

import gc
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings
from common.video import palette_lut

W, H, DPI = 1080, 1920, 100
MARGIN = 80  # px, left and right
TEXT_WIDTH = (W - 2 * MARGIN) * 72 / DPI  # pt, for LaTeX's \parbox
TOP, BOTTOM = 270, 1640  # px: content stays between these (Story UI above and below)

BG = "#021a1a"
INK = "#f3fff8"
ACCENT = "#5fe3c0"
MUTED = "#8fb8ae"
WARM = "#ffb070"

BODY, DISPLAY, TITLE = 31, 38, 52  # pt
RABBIT = complex(-0.1226, 0.7449)  # Douady's rabbit
CARDS = 5

LATEX = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def reel():
    """The Reel's module (its name starts with a digit, so importlib)."""
    return importlib.import_module("lab.2026.1102")


def axes_px(fig, x, y, w, h):
    """Axes at pixel box (x, y, w, h), y from the top, nothing drawn round it."""
    ax = fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H))
    ax.set_axis_off()
    return ax


def tex(fig, x, y, s, size=BODY, color=INK, ha="left", wrap=False):
    """
    LaTeX text with its top at pixel y; returns the y just below it, measured
    from the rendered box. wrap: a ragged-right \\parbox the width of the
    text column, so LaTeX breaks the lines (inline maths included).
    """
    if wrap:
        s = (
            rf"\parbox[t]{{{TEXT_WIDTH:.0f}pt}}{{\raggedright"
            rf"\linespread{{1.2}}\selectfont {s}}}"
        )
    t = fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, ha=ha, va="top")
    box = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return y + box.height


def para(fig, y, s, size=BODY, color=INK, gap=1.0):
    """A wrapped paragraph from the left margin; returns the y below it, plus a gap."""
    return tex(fig, MARGIN, y, s, size, color, wrap=True) + gap * size


def display(fig, y, formula, size=DISPLAY, color=ACCENT, gap=1.0):
    """A centred display formula; returns the y below it, plus a gap."""
    s = rf"$\displaystyle {formula}$"
    return tex(fig, W / 2, y, s, size, color, "center") + gap * size


def frame(fig, n, title, kicker):
    """Background, kicker, title and footer common to every card."""
    fig.patch.set_facecolor(BG)
    tex(fig, MARGIN, TOP, rf"\textbf{{{kicker}}}", 21, ACCENT)
    y = tex(fig, MARGIN, TOP + 42, rf"\bfseries {title}", TITLE, INK, wrap=True)
    tex(fig, MARGIN, BOTTOM - 30, r"@random\_plot", 19, MUTED)
    tex(fig, W - MARGIN, BOTTOM - 30, f"{n} / {CARDS}", 19, MUTED, "right")
    return y + 55


def fractal(c, centre, scale, w, h, julia=True):
    """RGB image (h, w, 3) in [0, 1]: the Reel's escape + glow + palette."""
    m = reel()
    xs = centre.real + (np.arange(w) - w / 2 + 0.5) / scale
    ys = centre.imag - (np.arange(h) - h / 2 + 0.5) / scale
    count, distance = m.escape(c.real, c.imag, xs, ys, julia)
    light = m.glow(count, distance, scale)
    if not julia:
        light = np.maximum(light, 0.2 * (count < 0))
    lut = palette_lut(m.PALETTE, 1024)
    return lut[(light * 1023).astype(np.int32)], count


def show(fig, image, x, y):
    h, w = image.shape[:2]
    ax = axes_px(fig, x, y, w, h)
    ax.imshow(image, interpolation="lanczos")
    return ax


def card_cover(fig):
    y = frame(fig, 1, r"The Map of\\All Julia Sets", "THE MATHS BEHIND TODAY'S REEL")
    image, _ = fractal(RABBIT, 0j, 255, W - 2 * MARGIN, 600)
    show(fig, image, MARGIN, y)
    y += 640
    y = display(fig, y, r"z \;\longmapsto\; z^2 + c", 50, gap=1.0)
    y = para(
        fig,
        y,
        "Every shape in the video comes from this one rule. "
        "Next: orbits, Julia sets, the Mandelbrot set and the path of the dot.",
    )
    para(fig, y, r"\textit{Hold the screen to pause.}", 23, MUTED)


def card_orbits(fig):
    y = frame(fig, 2, "One rule, repeated", r"1 $\cdot$ ORBITS")
    y = para(
        fig, y, r"Fix a complex number $c$, start at $z_0 = 0$ and iterate:", gap=0.8
    )
    y = display(fig, y, r"z_{n+1} = z_n^2 + c", gap=0.2)
    ax = axes_px(fig, MARGIN, y, W - 2 * MARGIN, 520)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.1, 2.1)
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", lw=1.2, color=MUTED))
    ax.text(1.45, -1.85, r"$|z| = 2$", color=MUTED, fontsize=21)
    for c, colour in ((RABBIT, ACCENT), (complex(0.27, 0.0), WARM)):
        z, orbit = 0j, [0j]
        for _ in range(60):
            z = z * z + c
            orbit.append(z)
            if abs(z) > 2.5:
                break
        orbit = np.array(orbit)
        ax.plot(orbit.real, orbit.imag, "-", color=colour, lw=1.0, alpha=0.6)
        ax.plot(orbit.real, orbit.imag, "o", color=colour, ms=5)
    ax.plot(0, 0, "o", color=INK, ms=7)
    ax.text(-0.1, -0.35, r"$z_0$", color=INK, fontsize=21)
    ax.text(
        -0.35, 1.0, r"$c = -0.123 + 0.745i$", color=ACCENT, fontsize=22, ha="center"
    )
    ax.text(1.45, 0.2, r"$c = 0.27$", color=WARM, fontsize=22, ha="center")
    y += 560
    y = para(
        fig,
        y,
        r"Green: the orbit falls into a 3-cycle and stays bounded. "
        r"Orange: it crawls right, then escapes.",
    )
    para(
        fig,
        y,
        r"Once $|z_n| > \max(2, |c|)$ it never comes back, so the computer "
        r"can stop there.",
        color=MUTED,
    )


def card_julia(fig):
    y = frame(fig, 3, "Julia sets", r"2 $\cdot$ WHICH STARTING POINTS STAY?")
    y = para(
        fig,
        y,
        r"Fix $c$ and iterate from every $z_0$. Those that stay bounded form "
        r"the filled Julia set $K_c$; its boundary is the Julia set $J_c$.",
    )
    half = (W - 2 * MARGIN - 20) // 2
    m = reel()
    xs = (np.arange(half) - half / 2 + 0.5) / 260
    ys = -(np.arange(half) - half / 2 + 0.5) / 260
    count, distance = m.escape(RABBIT.real, RABBIT.imag, xs, ys, True)
    bands = np.where(count < 0, 0, (np.sin(count * 0.9) + 1) / 2 * 0.8 + 0.1)
    lut = palette_lut(m.PALETTE, 1024)
    show(fig, lut[(bands * 1023).astype(np.int32)], MARGIN, y)
    glow, _ = fractal(RABBIT, 0j, 260, half, half)
    show(fig, glow, MARGIN + half + 20, y)
    y += half + 12
    tex(fig, MARGIN + half / 2, y, "escape time", 22, MUTED, "center")
    tex(fig, MARGIN + 1.5 * half + 20, y, "distance estimate", 22, MUTED, "center")
    y += 80
    y = para(fig, y, r"The video draws the distance to $J_c$ (right):", gap=0.8)
    y = display(
        fig,
        y,
        r"d \approx \frac{|z_n|\,\ln|z_n|}{|z_n'|}, \qquad z_{n+1}' = 2\,z_n z_n'",
    )
    para(
        fig,
        y,
        r"A bright line where $d < 1$ px keeps hair-thin filaments visible.",
        color=MUTED,
    )


def card_map(fig):
    y = frame(fig, 4, "The Mandelbrot set", r"3 $\cdot$ THE MAP OF ALL JULIA SETS")
    y = para(
        fig,
        y,
        r"$z = 0$ is the critical point, $f_c'(0) = 0$, and its orbit decides "
        r"everything:",
        gap=0.8,
    )
    y = display(
        fig, y, r"M = \{\, c : \text{the orbit of } 0 \text{ stays bounded} \,\}", 32
    )
    y = para(
        fig,
        y,
        r"Fatou \& Julia, 1919: $c \in M$ gives a connected $J_c$, "
        r"$c \notin M$ gives dust.",
        color=MUTED,
    )
    half = (W - 2 * MARGIN - 20) // 2
    for k, (c, label) in enumerate(
        ((0.25, r"$c = 0.25 \in M$"), (0.36, r"$c = 0.36 \notin M$"))
    ):
        image, _ = fractal(complex(c, 0), 0j, 205, half, half)
        x = MARGIN + k * (half + 20)
        show(fig, image, x, y)
        tex(fig, x + half / 2, y + half + 12, label, 23, MUTED, "center")
    y += half + 90
    para(fig, y, "Step past the edge and the inside vanishes: the burst in the video.")


def card_walk(fig):
    y = frame(fig, 5, "The path of the dot", r"4 $\cdot$ THE MAIN CARDIOID")
    y = para(
        fig,
        y,
        r"Here $f_c$ has a fixed point $z^*$ with multiplier $\lambda = 2z^*$, "
        r"so $c = \lambda/2 - \lambda^2/4$. Its edge is $|\lambda| = 1$:",
        gap=0.8,
    )
    y = display(fig, y, r"c(\theta) = \frac{e^{i\theta}}{2} - \frac{e^{2i\theta}}{4}")
    m = reel()
    scale, w, h = 280, W - 2 * MARGIN, 480
    centre = complex(-0.33, 0.0)
    image, _ = fractal(0j, centre, scale, w, h, julia=False)
    ax = show(fig, image, MARGIN, y)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    def px(c):
        x = (c.real - centre.real) * scale + w / 2
        return x, (centre.imag - c.imag) * scale + h / 2

    ax.plot(*px(m.walk(np.linspace(0, 1, 2000))), color=INK, lw=1.3)
    for p, q in ((1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (2, 5), (3, 5)):
        theta = 2 * np.pi * p / q
        c = np.exp(1j * theta) / 2 - np.exp(2j * theta) / 4
        out = c + 0.12 * (-1j * np.exp(1.5j * theta))
        ax.plot(*px(c), "o", color=WARM, ms=5)
        ax.text(
            *px(out),
            rf"$\mathbf{{{p}/{q}}}$",
            color=WARM,
            fontsize=21,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.75),
        )
    y += h + 40
    y = para(
        fig,
        y,
        r"At $\theta = 2\pi p/q$ a bulb is attached; its Julia sets have $q$ "
        r"arms (the rabbit: $q = 3$). The dot is pushed out five times a loop:",
        gap=0.8,
    )
    display(
        fig,
        y,
        r"c = c(\theta) + 0.03\,\max(0, \sin 5\theta)^2\,\bigl(-i\,e^{3i\theta/2}\bigr)",
        30,
    )


def generate(settings: ImageProcessingSettings = None):
    """
    Five 1080x1920 Story cards (PNG) for the Reel of 2026-11-02. All text is
    typeset by LaTeX (matplotlib text.usetex; paragraphs are \\parbox-es so
    LaTeX breaks the lines), over images computed with the Reel's own
    `escape`, `glow`, `walk` and palette. Content stays between TOP and
    BOTTOM px, clear of Instagram's Story UI. Saved as story_<n>.png.
    Needs a LaTeX install with dvipng (MiKTeX here). Nothing is random:
    --seed does not change anything.
    """
    settings = settings or ImageProcessingSettings(1)
    folder = settings.output_path / settings.filename
    cards = (card_cover, card_orbits, card_julia, card_map, card_walk)
    with plt.rc_context(LATEX):
        for n, card in enumerate(cards, 1):
            fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
            card(fig)
            path = folder / f"story_{n}.png"
            fig.savefig(path, facecolor=BG, dpi=DPI)
            plt.close(fig)
            gc.collect()
            logger.info(f"Saved {path}")


if __name__ == "__main__":
    generate()
