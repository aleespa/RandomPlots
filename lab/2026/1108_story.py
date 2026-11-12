"""
Why Every Equation Has a Root: the maths, as an Instagram Story.

Four 1080x1920 cards explaining the Reel of 2026-11-08 (lab/2026/1108.py):
the theorem, the winding number, why it counts roots, and the proof.
"""

import gc
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from common.image_processing import ImageProcessingSettings

W, H, DPI = 1080, 1920, 100
MARGIN = 80  # px, left and right
TEXT_WIDTH = (W - 2 * MARGIN) * 72 / DPI  # pt, for LaTeX's \parbox
TOP, BOTTOM = 270, 1640  # px: content stays between these (Story UI above and below)

BG = "#05060a"
INK = "#f4f2ff"
ACCENT = "#9fd8ff"
MUTED = "#9a98b0"
WARM = "#ffbf80"

BODY, DISPLAY, TITLE = 31, 38, 52  # pt
CARDS = 4
SEED = 0  # the posted Reel's --seed: it sets the roots' angles

LATEX = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def reel():
    """The Reel's module (its name starts with a digit, so importlib)."""
    return importlib.import_module("lab.2026.1108")


def roots():
    """The Reel's five roots, drawn exactly as its generate() draws them."""
    rng = np.random.default_rng(SEED)
    moduli = np.array(reel().KNOBS["moduli"])
    golden = np.pi * (3 - np.sqrt(5))
    angles = rng.uniform(0, 2 * np.pi) + golden * np.arange(5) * 2
    angles = angles + rng.uniform(-0.35, 0.35, 5)
    return moduli * np.exp(1j * angles)


def p(z, rs):
    out = np.ones_like(z, dtype=complex)
    for r in rs:
        out = out * (z - r)
    return out


def hue_rgb(phase):
    """RGB in 0..1 for phase (turns), the Reel's pastel wheel."""
    lut = reel().hue_lut()[:, ::-1]
    return lut[(np.asarray(phase) % 1.0 * 255).astype(int)]


def axes_px(fig, x, y, w, h):
    """Axes at pixel box (x, y, w, h), y from the top, nothing drawn round it."""
    ax = fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H))
    ax.set_axis_off()
    return ax


def tex(fig, x, y, s, size=BODY, color=INK, ha="left", wrap=False):
    """LaTeX text with its top at pixel y; returns the y just below it."""
    if wrap:
        s = (
            rf"\parbox[t]{{{TEXT_WIDTH:.0f}pt}}{{\raggedright"
            rf"\linespread{{1.2}}\selectfont {s}}}"
        )
    t = fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, ha=ha, va="top")
    box = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return y + box.height


def para(fig, y, s, size=BODY, color=INK, gap=1.0):
    return tex(fig, MARGIN, y, s, size, color, wrap=True) + gap * size


def display(fig, y, formula, size=DISPLAY, color=ACCENT, gap=1.0):
    s = rf"$\displaystyle {formula}$"
    return tex(fig, W / 2, y, s, size, color, "center") + gap * size


def frame(fig, n, title, kicker):
    fig.patch.set_facecolor(BG)
    tex(fig, MARGIN, TOP, rf"\textbf{{{kicker}}}", 21, ACCENT)
    y = tex(fig, MARGIN, TOP + 42, rf"\bfseries {title}", TITLE, INK, wrap=True)
    tex(fig, MARGIN, BOTTOM - 30, r"@random\_plot", 19, MUTED)
    tex(fig, W - MARGIN, BOTTOM - 30, f"{n} / {CARDS}", 19, MUTED, "right")
    return y + 55


def coloured_curve(ax, pts, phase, lw=2.4):
    """A polyline in the plane, each segment coloured by its phase (turns)."""
    from matplotlib.collections import LineCollection

    seg = np.stack([pts[:-1], pts[1:]], 1)
    ax.add_collection(LineCollection(seg, colors=hue_rgb(phase[:-1]), linewidths=lw))


def compress(w, soft=0.04):
    """The Reel's w-plane squash: same angle, radius log(1 + |w| / soft)."""
    rho = np.log1p(np.abs(w) / soft)
    unit = w / np.maximum(np.abs(w), 1e-300)
    return np.stack([rho * unit.real, rho * unit.imag], 1)


def card_cover(fig, settings):
    y = frame(
        fig, 1, r"Why Every Equation\\Has a Root", "THE MATHS BEHIND TODAY'S REEL"
    )
    still = settings.output_path / "20261108" / "frames" / "frame0060.png"
    image = np.asarray(Image.open(still).convert("RGB"))[960:1900]
    side = W - 2 * MARGIN
    h = int(image.shape[0] * side / image.shape[1] * 0.8)
    ax = axes_px(fig, MARGIN + side * 0.1, y - 20, side * 0.8, h)
    ax.imshow(image, interpolation="lanczos")
    y += h - 10
    y = para(
        fig,
        y,
        r"\textbf{The Fundamental Theorem of Algebra.} Every polynomial "
        r"of degree $n \ge 1$,",
        gap=0.6,
    )
    y = display(fig, y, r"p(z) = z^n + a_{n-1}z^{n-1} + \dots + a_1 z + a_0,", 34)
    y = para(fig, y, r"has a root: some $z$ in $\mathbb{C}$ with $p(z) = 0$.")
    para(
        fig,
        y,
        r"Not in $\mathbb{R}$: $x^2 + 1 = 0$ has no real solution. "
        r"The proof in the video needs the whole plane.",
        color=MUTED,
    )


def card_winding(fig, settings):
    rs = roots()
    y = frame(fig, 2, "Counting turns", r"1 $\cdot$ THE WINDING NUMBER")
    y = para(
        fig,
        y,
        r"Walk once round a closed curve and count how many full turns it makes "
        r"around $0$: its \textit{winding number}. The video's degree-5 $p$:",
        gap=0.6,
    )
    half = (W - 2 * MARGIN - 20) // 2
    theta = np.linspace(0, 2 * np.pi, 4001)
    for k, (radius, label) in enumerate(
        (
            (2.25, r"$|z| = 2.25$: \textbf{5 turns}"),
            (0.2, r"$|z| = 0.2$: \textbf{0 turns}"),
        )
    ):
        w = p(radius * np.exp(1j * theta), rs)
        x = MARGIN + k * (half + 20)
        ax = axes_px(fig, x, y, half, half)
        ax.set_aspect("equal")
        lim = 11.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        coloured_curve(ax, compress(w), np.angle(w) / (2 * np.pi), 2.0)
        ax.plot(0, 0, "o", color=INK, ms=8)
        ax.text(0.6, -1.4, r"$0$", color=INK, fontsize=22)
        tex(fig, x + half / 2, y + half + 10, label, 23, MUTED, "center")
    y += half + 80
    y = para(
        fig,
        y,
        r"\textbf{Big loop.} There $p(z) = z^5(1 + \text{small})$, so "
        r"$\arg p \approx 5 \arg z$: one trip round the loop, five turns of the image.",
        gap=0.6,
    )
    para(
        fig,
        y,
        r"\textbf{Tiny loop.} $p(z)$ barely moves from $p(0) \neq 0$, "
        r"so it never goes round $0$.",
        color=INK,
    )
    tex(
        fig,
        MARGIN,
        BOTTOM - 90,
        r"Images squashed radially, $|w| \mapsto \log(1 + |w|/0.04)$, so both fit.",
        20,
        MUTED,
    )


def card_count(fig, settings):
    rs = roots()
    y = frame(fig, 3, "Turns count roots", r"2 $\cdot$ THE ARGUMENT PRINCIPLE")
    y = para(fig, y, r"Factor $p$ through its roots. Angles of a product add:", gap=0.6)
    y = display(
        fig,
        y,
        r"\arg p(z) = \sum_{k=1}^{5} \arg\,(z - r_k)",
        gap=0.4,
    )
    size = W - 2 * MARGIN
    h = 500
    ax = axes_px(fig, MARGIN, y, size, h)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.95, 1.75)
    radius = 1.25
    theta = np.linspace(0, 2 * np.pi, 2001)
    loop = radius * np.exp(1j * theta)
    w = p(loop, rs)
    coloured_curve(
        ax, np.stack([loop.real, loop.imag], 1), np.angle(w) / (2 * np.pi), 3.0
    )
    z = radius * np.exp(1j * 2.6)
    for r in rs:
        inside = abs(r) < radius
        colour = ACCENT if inside else WARM
        ax.annotate(
            "",
            xy=(z.real, z.imag),
            xytext=(r.real, r.imag),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6, alpha=0.85),
        )
        ax.plot(r.real, r.imag, "*", color=colour, ms=17)
    ax.plot(z.real, z.imag, "o", color=INK, ms=9)
    ax.text(z.real - 0.3, z.imag + 0.12, r"$z$", color=INK, fontsize=24)
    y += h + 30
    y = para(
        fig,
        y,
        r"As $z$ goes once round the loop, an arrow from a root "
        r"inside (blue) makes one full turn; "
        r"from a root outside (orange) it swings and comes back.",
        gap=0.6,
    )
    y = display(fig, y, r"\text{turns of } p(\text{loop}) = \#\,\text{roots inside}")
    para(
        fig,
        y,
        r"Here: 3. Colours on the loop are the angle of $p(z)$, as in the video.",
        color=MUTED,
    )


def card_proof(fig, settings):
    m = reel()
    knobs = m.KNOBS
    y = frame(fig, 4, "Shrink the loop", r"3 $\cdot$ THE PROOF YOU WATCHED")
    y = para(
        fig,
        y,
        r"Shrink the circle from huge to a point. Its image moves continuously, "
        r"and a moving curve can only change its turns around $0$ by "
        r"\textit{passing through} $0$.",
        gap=0.6,
    )
    y = para(
        fig,
        y,
        r"It starts at $n$ turns and ends at $0$. So somewhere on the way the "
        r"image hits $0$: there, $p(z) = 0$. \textbf{A root.}",
        gap=0.8,
    )
    h = 430
    ax = axes_px(fig, MARGIN + 70, y, W - 2 * MARGIN - 90, h)
    ax.set_axis_on()
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=20)
    t = np.linspace(0, 15, 1500)
    crossings = np.sort(knobs["crossings"])
    turns = 5 - np.searchsorted(crossings, t, side="right")
    ax.step(t, turns, where="post", color=ACCENT, lw=3)
    radius = np.array([m.radius_at(s, knobs) for s in t])
    ax.plot(t, radius * 5 / 2.25, color=MUTED, lw=1.5, ls="--")
    for c in crossings:
        ax.plot(
            c,
            5 - np.searchsorted(crossings, c, side="right") + 0.5,
            "*",
            color=WARM,
            ms=16,
        )
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.3, 5.6)
    ax.set_yticks(range(6))
    ax.set_xlabel(r"seconds into the video", color=MUTED, fontsize=21)
    ax.text(10.4, 4.6, r"turns", color=ACCENT, fontsize=23)
    ax.text(10.4, 3.9, r"loop radius", color=MUTED, fontsize=21)
    ax.text(10.4, 3.2, r"$\bigstar$ a root found", color=WARM, fontsize=21)
    y += h + 95
    y = para(
        fig,
        y,
        r"The video goes further: the count drops by one at each snag, "
        r"5, 4, 3, 2, 1, 0, so it finds \textit{all five} roots. "
        r"The music box drops a note each time.",
        gap=0.6,
    )
    para(
        fig,
        y,
        r"No formula needed, and no general formula in radicals exists for degree 5 (Abel, 1824).",
        color=MUTED,
    )


def generate(settings: ImageProcessingSettings = None):
    """
    Four 1080x1920 Story cards (PNG) for the Reel of 2026-11-08. All text is
    typeset by LaTeX (matplotlib text.usetex, \\parbox paragraphs), over plots
    computed with the Reel's own roots (seed SEED), hue wheel, radius_at and
    w-plane squash; the cover shows frame 60 of the Reel's last full render
    (outputs/20261108/frames). Content stays between TOP and BOTTOM px, clear
    of Instagram's Story UI. Saved as story_<n>.png. Needs a LaTeX install
    with dvipng. --seed does not change anything.
    """
    settings = settings or ImageProcessingSettings(1)
    folder = settings.output_path / settings.filename
    cards = (card_cover, card_winding, card_count, card_proof)
    with plt.rc_context(LATEX):
        for n, card in enumerate(cards, 1):
            fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
            card(fig, settings)
            path = folder / f"story_{n}.png"
            fig.savefig(path, facecolor=BG, dpi=DPI)
            plt.close(fig)
            gc.collect()
            logger.info(f"Saved {path}")


if __name__ == "__main__":
    generate()
