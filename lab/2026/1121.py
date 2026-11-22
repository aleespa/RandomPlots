"""
The Proof Is 17 Million Lines Long

The proof of the Riemann hypothesis, every file of it. Part 1 of 3.

On 20 November 2026, Levent Alpöge released a computer-checked proof of the most famous
open problem in mathematics, written in the proof language Lean: 175,187 files, 818,755
theorems, 17,374,215 lines. At MIT, Andrew Sutherland independently rechecked it with the
Lean kernel.

Each point is one file, and each thread joins it to a file it builds on. Light spreads
from the final theorem through everything it depends on, 43 layers deep. Then a flight
through abelian varieties (more than half of the proof), the Weil bound and étale
cohomology, with real theorem names from the files, and the longest chain: a file 43
steps from the theorem, traced all the way back. It ends on the statement itself, in Lean
and in words.

The hypothesis, from 1859, says every nontrivial zero of the zeta function lies on the
line Re(s) = ½. It governs how the prime numbers are spread out.

Listen: a string section changes chord with each part of the journey, holds its breath
as "hypothesis" is struck out, and comes home as it becomes "theorem".

Send this to the friend who said it would never be solved.

Made with Python & PyTorch, from the proof's own source files. Sound synthesized in NumPy.
#riemannhypothesis #leanprover #formalproof #mathart #randomplots
"""

import gc
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.interpolate import PchipInterpolator

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    palette_lut,
    hook_masks,
    render_frames,
    smoothstep,
)

DURATION = 36.0  # seconds
FRAME_COUNT = frame_count(DURATION)

HOOK = "This is the proof of the Riemann hypothesis"
HOOK_Y = 270
HOOK_COLOUR = "#f4f0e7"

# The archive released on 2026-11-20, unpacked next to this repository.
RIEMANN_SRC = Path(__file__).resolve().parents[3] / "RiemannH" / "lean" / "src"
CACHE = Path("outputs") / "_cache" / "rh_lean_graph_v2.npz"

BACKGROUND = "#05060a"
INK = "#f4f0e7"
CORE = "#ffd37a"  # the theorem, the files that state it, the longest chain
# Around the disc, folder by folder (files sorted by path, so a folder is a sector).
PALETTE = [
    "#4f8fe8",
    "#7a6cf0",
    "#b46ee0",
    "#e0699f",
    "#f09a5c",
    "#e8c86a",
    "#5fc9c9",
    "#4f8fe8",
]

CENTRE = (WIDTH / 2, 1060.0)  # screen position of the camera's target
RADIUS = 440.0  # screen radius of the whole proof at zoom 1
ZOOM_MAX = 30.0  # closest the camera gets (frame 0)
FRONT_START = 0.38  # screen radius of the light front, as a share of RADIUS, at first
# The light front's depth over time (s, layers): the sparse first layers go by fast,
# then the bulk sweeps out steadily and eases into the rim.
WAVE_T = (0.25, 0.8, 1.4, 3.0, 6.0, 9.5)
WAVE_D = (0.0, 2.0, 6.0, 14.0, 29.0, 44.0)
REVEAL_END = 10.0
OVERVIEW_END = 12.6
# The tour: folder, title, note, zoom, and (leave, arrive, depart) times in seconds.
STOPS = [
    (
        "Development/AbelianVarietyTheory",
        r"$\mathrm{Abelian\ varieties}$",
        r"$\mathrm{more\ than\ half\ of\ the\ proof}$",
        2.1,
        (12.6, 13.8, 16.4),
    ),
    (
        "Development/WeilBound",
        r"$\mathrm{The\ Weil\ bound}$",
        None,
        3.6,
        (16.4, 17.6, 20.0),
    ),
    (
        "Development/EtaleCohomology",
        r"$\mathrm{\acute{E}tale\ cohomology}$",
        r"$\mathrm{the\ machinery\ behind\ the\ Weil\ conjectures}$",
        6.0,
        (20.0, 21.2, 23.6),
    ),
]
FINAL_ARRIVE = 25.0
FINAL_ZOOM = 0.86
FINAL_DISC_Y = 1150.0  # screen y of the theorem in the closing view
DIVE = (28.6, 30.4)  # seconds: back into the theorem at the centre
DIVE_ZOOM = 16.0
DIVE_Y = 1180.0  # screen y of the theorem after the dive
STATEMENT_IN = 29.6  # the statement, in Lean and in words, from here to the end
HEADER_Y = 250  # "Riemann hypothesis" across the top, once the hook has faded
HOOK_END = 28.0  # the hook stays up this long (drawn here, not by render_frames)
HEADER_IN = 28.0
STRIKE = (33.0, 33.6)  # seconds: "hypothesis" is struck through...
SWAP = (33.7, 34.4)  # ...and becomes "theorem"
CREDIT_IN = 30.4  # the account's handle under the statement, so reposts carry it
CREDIT_Y = 1000
COVER_SECONDS = (
    12.0  # grid cover (tools.post_reel thumb_offset): the overview + counters
)
# The statement as the archive states it (Statements.lean), typeset line by line:
# (indent in px, LaTeX). Lean's own symbols in maths font, the rest in typewriter.
LEAN_LINES = [
    (0, r"$\mathtt{theorem\ riemannHypothesis\ :}$"),
    (44, r"$\forall\mathtt{\ s\ :\ }\mathbb{C}\mathtt{,\ riemannZeta\ s\ =\ 0\ }\to$"),
    (
        44,
        r"$\mathtt{(}\neg\exists\mathtt{\ n\ :\ }\mathbb{N}\mathtt{,\ s\ =\ {-}2\ *\ (n\ +\ 1))\ }\to$",
    ),
    (44, r"$\mathtt{s\ }\neq\mathtt{\ 1\ }\to\mathtt{\ s.re\ =\ 1\ /\ 2}$"),
]
WORD_LINES = [
    r"$\mathrm{Every\ zero\ of\ }\zeta(s)\mathrm{\ except\ }{-}2{,}\ {-}4{,}\ {-}6{,}\ \ldots$",
    r"$\mathrm{lies\ on\ the\ line\ \ Re}(s)=\frac{1}{2}$",
]
CHAIN_FILE = "box_of_fibres"  # the theorem in one of the files 43 steps deep
CHAIN_TRACE = (25.0, 27.2)  # seconds: traced from that file back to the theorem
# Regions named in the overview: (folder, label, radius), radii staggered to fit
REGIONS = [
    ("Development/AbelianVarietyTheory", r"$\mathrm{abelian\ varieties}$", 0.6),
    ("Development/AutomorphicArithmetic", r"$\mathrm{automorphic\ arithmetic}$", 0.62),
    ("Development/WeilBound", r"$\mathrm{Weil\ bound}$", 0.86),
    ("Formalism", r"$\mathrm{formalism}$", 0.52),
    ("GroundTheory", r"$\mathrm{ground\ theory}$", 0.8),
]

FLASH = 1.6  # layers over which a newly lit point cools from white to its colour
SUPERSAMPLE_PX = 2.6  # target spacing of samples along a thread, screen px
LONG_THREAD_PX = 60.0  # threads longer than this (at zoom 1) fade as 1/length
FILE_GAIN = 0.35
THREAD_GAIN = 0.05
GHOST = 0.08  # brightness of a file or thread the light hasn't reached
BLOOM = (0.35, 0.2)
EXPOSURE = 1.0
STAR_DEPTH = 4
STAR_GAIN = 2.5
FOCUS_DIM = 0.6  # how much the rest of the proof dims while the camera visits a folder

KINDS = (
    b"theorem",
    b"lemma",
    b"def",
    b"abbrev",
    b"structure",
    b"class",
    b"instance",
    b"inductive",
    b"axiom",
    b"opaque",
)
DECLARATION = re.compile(
    rb"^(?:@\[[^\]\n]*\]\s*)?"
    rb"(?:(?:private|protected|noncomputable|nonrec|partial|unsafe|public)\s+)*"
    rb"(" + b"|".join(KINDS) + rb")\b[ \t]*([^\s:({\[]*)",
    re.M,
)
PLAIN_NAME = re.compile(rb"^[A-Za-z][A-Za-z0-9_]*$")


def _scan(path: Path):
    """A Lean file's imports (its header), line count, declarations by kind, and the
    longest plain theorem name in it (10-28 characters, for labels)."""
    data = path.read_bytes()
    lines = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
    imports = []
    for raw in data.split(b"\n"):
        s = raw.strip()
        if s.startswith(b"public import "):
            s = s[7:]
        if s.startswith(b"import "):
            imports.extend(x.decode() for x in s[7:].split())
        elif not (s == b"" or s.startswith((b"--", b"/-", b"module"))):
            break
    counts = [0] * len(KINDS)
    best = b""
    for match in DECLARATION.finditer(data):
        kind = KINDS.index(match.group(1))
        counts[kind] += 1
        name = match.group(2).split(b".")[-1]
        if kind < 2 and 10 <= len(name) <= 28 and len(name) > len(best):
            if PLAIN_NAME.match(name):
                best = name
    return imports, lines, counts, best.decode()


def build_graph(src: Path = RIEMANN_SRC) -> dict:
    """
    Parse every .lean file of the archive into an import graph and run a breadth-first
    search from Main: depth (layers from the theorem), parent (the file that first
    reached it), lines, declaration counts, a theorem name and the two-level folder of
    every file. Cached in outputs/_cache.
    """
    if CACHE.exists():
        return dict(np.load(CACHE))
    if not src.exists():
        raise FileNotFoundError(f"Lean sources not found at {src}")
    start = time.time()
    files = sorted(
        Path(root) / n
        for root, _, names in os.walk(src)
        for n in names
        if n.endswith(".lean")
    )
    parts = [p.relative_to(src).with_suffix("").parts for p in files]
    modules = [".".join(p) for p in parts]
    with ThreadPoolExecutor(32) as pool:
        scanned = list(pool.map(_scan, files, chunksize=256))
    index = {m: i for i, m in enumerate(modules)}
    adjacency = [[index[m] for m in s[0] if m in index] for s in scanned]
    edges = sum(len(a) for a in adjacency)

    root = index["Main"]
    depth = np.full(len(files), -1, np.int32)
    parent = np.full(len(files), -1, np.int32)
    depth[root] = 0
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if depth[b] < 0:
                depth[b], parent[b] = depth[a] + 1, a
                queue.append(b)
    graph = {
        "depth": depth,
        "parent": parent,
        "lines": np.array([s[1] for s in scanned], np.int64),
        "counts": np.array([s[2] for s in scanned], np.int64),
        "name": np.array([s[3] for s in scanned]),
        "folder": np.array(["/".join(p[: min(2, len(p) - 1)]) or p[0] for p in parts]),
        "edges": np.array(edges),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **graph)
    logger.info(
        f"Lean graph: {len(files)} files, {edges} imports, "
        f"{graph['lines'].sum()} lines, depth {depth.max()} ({time.time() - start:.1f}s)"
    )
    return graph


def number_masks(size: float) -> tuple[list[np.ndarray], int]:
    """One LaTeX mask per digit, plus the fixed advance width that aligns them."""
    digits = [latex_mask(rf"$\mathrm{{{d}}}$", size) for d in range(10)]
    return digits, max(m.shape[1] for m in digits) + 1


def blit_number(image, value, digits, advance, x_right, y, colour, alpha):
    """Draw `value` right-aligned at x_right with thin gaps between groups of three."""
    x = x_right
    for ch in reversed(f"{value:,}"):
        if ch == ",":
            x -= advance // 2
            continue
        mask = digits[int(ch)]
        x -= advance
        blit_mask(image, mask, x + (advance - mask.shape[1]) // 2, y, colour, alpha)


def halo(mask, sigma=6, gain=2.0):
    """A soft dark backing for a text mask, padded by 2 sigma on every side."""
    pad = int(2 * sigma)
    return np.clip(cv2.GaussianBlur(np.pad(mask, pad), (0, 0), sigma) * gain, 0, 1)


def window(seconds, start, end, fade=0.35):
    """1 inside [start, end], easing in and out over `fade` seconds at each end."""
    rise = smoothstep(np.clip((seconds - start) / fade, 0, 1))
    fall = 1 - smoothstep(np.clip((seconds - end + fade) / fade, 0, 1))
    return float(rise * fall)


def tex_name(name: str) -> str:
    return r"$\mathtt{" + name.replace("_", r"\_") + "}$"


# --- soundtrack: a string section, one simple chord per section of the video ---
MUSIC = (
    "strings"  # default variant; `--audio-only <tag>` picks another from MUSIC_PLANS
)
KEY = 50  # D3: the home note
QUALITY = {"maj": (0, 4, 7), "min": (0, 3, 7), "sus4": (0, 5, 7)}
RESOLVE = 33.9  # the home chord lands as "hypothesis" turns into "theorem" (SWAP)
# (start in seconds, root in semitones above D, quality, loudness 0..1): the chord
# changes where the picture changes -- reveal, overview, the three stops, the chain,
# the dive, the statement, the strike-through, and home on "theorem".
MUSIC_PLANS = {
    "strings": [  # D major: I vi IV | I vi IV | V vi IV V Vsus4 | I
        (0.0, 0, "maj", 0.30),
        (6.0, 9, "min", 0.38),
        (REVEAL_END, 5, "maj", 0.45),
        (STOPS[0][4][1] - 0.6, 0, "maj", 0.42),
        (STOPS[1][4][1] - 0.6, 9, "min", 0.42),
        (STOPS[2][4][1] - 0.6, 5, "maj", 0.44),
        (STOPS[2][4][2], 7, "maj", 0.45),
        (CHAIN_TRACE[0], 9, "min", 0.46),
        (DIVE[0], 5, "maj", 0.48),
        (31.5, 7, "maj", 0.52),
        (STRIKE[0] - 0.2, 7, "sus4", 0.56),
        (RESOLVE, 0, "maj", 0.68),
    ],
    "picardy": [  # D minor: i VI III VII | i VI iv | VII i VI iv Vsus4 | I (major!)
        (0.0, 0, "min", 0.30),
        (6.0, 8, "maj", 0.38),
        (REVEAL_END, 3, "maj", 0.45),
        (STOPS[0][4][1] - 0.6, 10, "maj", 0.42),
        (STOPS[1][4][1] - 0.6, 0, "min", 0.42),
        (STOPS[2][4][1] - 0.6, 8, "maj", 0.44),
        (STOPS[2][4][2], 5, "min", 0.45),
        (CHAIN_TRACE[0], 10, "maj", 0.46),
        (DIVE[0], 0, "min", 0.48),
        (31.5, 8, "maj", 0.52),
        (STRIKE[0] - 0.2, 7, "sus4", 0.56),
        (RESOLVE, 0, "maj", 0.70),
    ],
    "drift": [  # D major, I and IV rocking, home on "theorem"
        (0.0, 0, "maj", 0.30),
        (REVEAL_END, 5, "maj", 0.42),
        (STOPS[1][4][1] - 0.6, 0, "maj", 0.42),
        (STOPS[2][4][2], 5, "maj", 0.45),
        (DIVE[0], 0, "maj", 0.48),
        (31.5, 5, "maj", 0.52),
        (RESOLVE, 0, "maj", 0.68),
    ],
}
# The reveal opens on the bass alone and adds a voice as each part of the proof lights:
# (seconds, voices sounding above the bass).
ENTRIES = ((0.0, 0), (1.4, 1), (3.0, 2), (6.0, 3))


def voice_lead(previous, pitch_classes, low=57, high=76):
    """Upper voices for a chord: each voice moves to the nearest chord tone (no doubling
    when avoidable), keeping the line smooth from chord to chord."""
    options = [n for n in range(low, high + 1) if n % 12 in pitch_classes]
    voices, used = [], set()
    for prev in previous:
        ranked = sorted(options, key=lambda n: (n % 12 in used, abs(n - prev)))
        voices.append(ranked[0])
        used.add(ranked[0] % 12)
    return voices


def soundtrack(variant: str, rng: np.random.Generator) -> np.ndarray:
    """
    Strings only, one-shot. Each section of the video gets one chord (MUSIC_PLANS):
    a bass on the root (D2-D3 range) and three upper voices voice-led from chord to
    chord, each chord held until the next with overlapping bows. Over the reveal the
    texture grows from the bass alone to four voices, brightening as the light
    spreads; the last change is a suspended fourth under the strike-through,
    released into the home chord as "theorem" appears, which rings over the end.
    """
    plan = MUSIC_PLANS.get(variant, MUSIC_PLANS[MUSIC])
    track = audio.silence(DURATION)
    upper = [KEY + 7, KEY + 12, KEY + 16]  # A3 D4 F#4, re-voiced per chord below
    for i, (start, root, quality, loud) in enumerate(plan):
        end = plan[i + 1][0] if i + 1 < len(plan) else DURATION
        last = i + 1 == len(plan)
        pcs = {(KEY + root + q) % 12 for q in QUALITY[quality]}
        upper = voice_lead(upper, pcs)
        bass = KEY - 12 + root if root <= 7 else KEY - 24 + root
        bright = 0.25 + 0.5 * min(start / REVEAL_END, 1.0) + (0.15 if last else 0.0)
        notes = [(bass, 1.0)] + [(n, 0.8) for n in upper]
        if last:  # the home chord gets a high octave on top
            notes.append((upper[-1] + 12, 0.45))
        for k, (midi, weight) in enumerate(notes):
            entry = start
            if start < REVEAL_END and k > 0:  # the reveal adds voices one by one
                entry = max(start, next(t for t, n in ENTRIES if n >= min(k, 3)))
                if entry >= end:
                    continue
            sound = audio.strings(
                float(audio.midi_to_hz(midi)),
                (end - entry + 0.5) if not last else DURATION - entry,
                velocity=loud * weight,
                attack=1.6 if entry < 1.0 else (0.5 if last else 0.9),
                release=1.4 if not last else 2.5,
                brightness=bright,
            )
            audio.add(track, sound, entry, circular=False)
    track = audio.reverb(track, rng, rt60=2.6, wet=0.32, circular=False)
    audio.fade(track, 0.01, 1.0)  # out with the picture's fade
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The import graph of the Lean archive claiming the Riemann hypothesis (2026-11-20):
    175,187 files, 753,569 internal imports. A breadth-first search from Main gives every
    file a depth d (0..43) and a parent, the file that first reached it. File i sits at
    light depth L_i = d_i + u_i (u_i ~ U[0, 1) from settings.rng), at radius
    0.45·L_i/44 + 0.55·sqrt(share of files with smaller L) -- half depth, half
    area-preserving -- and at angle 2π·rank/N by sorted path, so each folder is a
    sector. Its parent thread is the polar interpolation between the two positions.

    0-10 s: a front at depth D(t) (monotone cubic through WAVE_T/WAVE_D) sweeps out; a
    point lights once D ≥ L, flashing white and cooling to its sector colour over 1.6
    layers, and a thread lights from parent to child. The zoom puts the front at a
    planned screen radius (never zooming back in), from 30x down to 1x. Counters sum
    files, theorems and lines over lit files. 10-12.6 s: overview with region names.
    12.6-23.6 s: the camera (centre and log-zoom, smoothstep between keyframes) visits
    three folders; the rest of the proof dims, and real theorem names are pinned to
    files chosen to spread over the view. 23.6-30 s: back to the whole proof; the
    BFS-parent chain from a file 43 steps deep is traced back to Main in gold, over a
    closing panel of totals.

    Rendering: each frame splats ~10M points (files plus thread samples, sampled by each
    thread's on-screen length at the closest zoom it is seen at) on the GPU with
    common.gpu.splat, adds a star glow for the first layers and a wide bloom,
    tone-maps 1 - exp(-x), then composites LaTeX text, labels and the chain on the CPU.
    Frames are pure functions of t.
    """
    settings = settings or ImageProcessingSettings(1)
    if settings.audio_only:  # the soundtrack needs none of the picture's arrays
        track = soundtrack(settings.audio_tag or MUSIC, settings.audio_rng)
        settings.save_video(fps=FPS, crf=18, audio=track)
        return
    rng = settings.rng
    device = select_device()

    graph = build_graph()
    depth = graph["depth"].astype(np.float64)
    parent = graph["parent"]
    lines = graph["lines"]
    counts = graph["counts"]
    names = graph["name"]
    folder = graph["folder"]
    theorems = counts[:, 0] + counts[:, 1]
    n = depth.size
    d_max = depth.max() + 1.0
    totals = {
        "files": n,
        "theorems": int(theorems.sum()),
        "definitions": int(counts[:, 2].sum() + counts[:, 3].sum()),
        "lines": int(lines.sum()),
    }
    logger.info(f"totals: {totals}, axioms declared: {counts[:, 8].sum()}")

    # --- file positions; light depth L, radius, angle, world coordinates ---
    light = depth + rng.random(n)
    light[depth == 0] = 0.0
    sorted_light = np.sort(light)
    share = np.searchsorted(sorted_light, light) / n
    radius = 0.45 * light / d_max + 0.55 * np.sqrt(share)
    angle = 2 * np.pi * (np.arange(n) + 0.5) / n
    wx, wy = radius * np.cos(angle), radius * np.sin(angle)
    lut = palette_lut(PALETTE, 512)
    colour = lut[np.minimum((angle / (2 * np.pi) * 511).astype(int), 511)]
    top = np.array([f.split("/")[0] for f in folder])
    core = np.isin(top, ["Main", "Statements", "Supplements"])
    colour[core] = np.array(palette_lut([CORE, CORE], 2)[0])
    weight = np.clip((lines / np.median(lines)) ** 0.35, 0.4, 3.0)
    per_layer = np.bincount(graph["depth"])
    weight *= 1 + 6.0 / np.sqrt(per_layer[graph["depth"]])  # sparse layers: stars
    star = depth <= STAR_DEPTH

    # --- camera ---
    grid = np.linspace(0, d_max, 2001)
    r_front = 0.45 * grid / d_max + 0.55 * np.sqrt(
        np.searchsorted(sorted_light, grid) / n
    )
    target = FRONT_START + (1 - FRONT_START) * (grid / d_max) ** 0.8
    zoom_grid = np.minimum.accumulate(
        np.clip(target / np.maximum(r_front, 1e-6), 1, ZOOM_MAX)
    )
    zoom_grid[-1] = 1.0

    def zoom_at(d):
        return np.interp(d, grid, zoom_grid)

    in_stop = [folder == f for f, *_ in STOPS]
    final_cy = (CENTRE[1] - FINAL_DISC_Y) / (FINAL_ZOOM * RADIUS)
    keys = [
        (REVEAL_END, 0.0, 0.0, 1.0),  # settle lower, clear of the counters
        (REVEAL_END + 1.0, 0.0, final_cy, FINAL_ZOOM),
        (OVERVIEW_END, 0.0, final_cy, FINAL_ZOOM),
    ]
    for mask, (_, _, _, zoom, (_, arrive, depart)) in zip(in_stop, STOPS):
        cx_w, cy_w = wx[mask].mean(), wy[mask].mean()
        out = np.array([cx_w, cy_w]) / np.hypot(cx_w, cy_w)
        drift = 0.025 / zoom * out  # a slow push outward while the camera holds
        keys.append((arrive, cx_w - drift[0], cy_w - drift[1], zoom))
        keys.append((depart, cx_w + drift[0], cy_w + drift[1], zoom * 1.12))
    keys.append((FINAL_ARRIVE, 0.0, final_cy, FINAL_ZOOM))
    keys.append((DIVE[0], 0.0, final_cy, FINAL_ZOOM * 1.03))
    dive_cy = (CENTRE[1] - DIVE_Y) / (DIVE_ZOOM * RADIUS)
    keys.append((DIVE[1], 0.0, dive_cy, DIVE_ZOOM))
    dive_end_cy = (CENTRE[1] - DIVE_Y) / (DIVE_ZOOM * 1.15 * RADIUS)
    keys.append((DURATION, 0.0, dive_end_cy, DIVE_ZOOM * 1.15))
    key_t = np.array([k[0] for k in keys])

    def camera(seconds):
        """(world x, world y, zoom) at the screen centre."""
        if seconds < REVEAL_END:
            return 0.0, 0.0, float(zoom_at(min(front(seconds), d_max)))
        i = min(np.searchsorted(key_t, seconds, side="right") - 1, len(keys) - 2)
        (t0, x0, y0, z0), (t1, x1, y1, z1) = keys[i], keys[i + 1]
        s = smoothstep(np.clip((seconds - t0) / (t1 - t0), 0, 1))
        zoom = float(np.exp(np.log(z0) + s * (np.log(z1) - np.log(z0))))
        return x0 + s * (x1 - x0), y0 + s * (y1 - y0), zoom

    def to_screen(x, y, cam):
        cam_x, cam_y, zoom = cam
        scale = zoom * RADIUS
        return CENTRE[0] + scale * (x - cam_x), CENTRE[1] + scale * (y - cam_y)

    # --- parent threads, sampled by their on-screen length at their closest view ---
    child = np.flatnonzero(parent >= 0)
    p = parent[child]
    r0, r1 = radius[p], radius[child]
    a0 = angle[p]
    da = (angle[child] - a0 + np.pi) % (2 * np.pi) - np.pi
    length = np.sqrt((r1 - r0) ** 2 + ((r0 + r1) / 2 * da) ** 2)  # unit-radius lengths
    zoom_need = zoom_at(light[p])
    mid_x, mid_y = 0.5 * (wx[p] + wx[child]), 0.5 * (wy[p] + wy[child])
    for key in keys[3:]:
        _, kx, ky, kz = key
        seen = (np.abs(mid_x - kx) < 0.7 * WIDTH / (kz * RADIUS)) & (
            np.abs(mid_y - ky) < 0.7 * HEIGHT / (kz * RADIUS)
        )
        zoom_need = np.where(seen, np.maximum(zoom_need, kz), zoom_need)
    samples = np.clip(
        np.ceil(length * RADIUS * zoom_need / SUPERSAMPLE_PX), 2, 500
    ).astype(np.int64)
    edge_of = np.repeat(np.arange(child.size), samples)
    offsets = np.arange(edge_of.size) - np.repeat(np.cumsum(samples) - samples, samples)
    u = (offsets + 0.5) / samples[edge_of]
    e_radius = r0[edge_of] + u * (r1 - r0)[edge_of]
    e_angle = a0[edge_of] + u * da[edge_of]
    e_light = light[p][edge_of] + u * (light[child] - light[p])[edge_of]
    e_colour = 0.5 * (colour[p][edge_of] + colour[child][edge_of])
    inner = 1 + 10.0 / np.sqrt(per_layer[graph["depth"][child]])
    # px of thread per sample at zoom 1; long cross-folder threads are dimmer
    e_weight = (length / samples / (1 + length * RADIUS / LONG_THREAD_PX) * inner)[
        edge_of
    ] * RADIUS
    group = np.zeros(n, np.int8)
    for k, mask in enumerate(in_stop):
        group[mask] = k + 1
    e_group = group[child][edge_of]
    logger.info(f"{n} files, {child.size} threads, {edge_of.size} thread samples")

    def t_(x, dtype=torch.float32):
        return torch.tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

    pt_x = t_(np.concatenate([wx, e_radius * np.cos(e_angle)]))
    pt_y = t_(np.concatenate([wy, e_radius * np.sin(e_angle)]))
    pt_light = t_(np.concatenate([light, e_light]))
    pt_colour = t_(np.concatenate([colour, e_colour]))
    is_file = t_(np.concatenate([np.ones(n), np.zeros(e_radius.size)])) > 0
    pt_weight = t_(np.concatenate([FILE_GAIN * weight, THREAD_GAIN * e_weight]))
    pt_group = t_(np.concatenate([group, e_group]), torch.int8)
    st_x, st_y = t_(wx[star]), t_(wy[star])
    st_light, st_colour = t_(light[star]), t_(colour[star])
    st_weight = t_(
        STAR_GAIN * (1 + 3 * (depth[star] == 0)) * weight[star] / weight[star].max()
    )
    del e_radius, e_angle, e_light, e_colour, e_weight, e_group, edge_of, offsets, u
    star_canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    star_kernel = gaussian_kernel(5.0, device)
    canvas = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    ghost_tint = torch.tensor([0.45, 0.5, 0.65], device=device)
    background = torch.tensor(
        np.array(hex_to_bgr(BACKGROUND)[::-1], np.float32), device=device
    )

    # counters: files, theorems and lines reached, by light depth
    order = np.argsort(light)
    theorems_cum = np.concatenate([[0], np.cumsum(theorems[order])])
    lines_cum = np.concatenate([[0], np.cumsum(lines[order])])

    # --- text, built once ---
    ink = np.array(hex_to_bgr(INK), np.float32)[::-1]  # RGB canvas
    gold = np.array(hex_to_bgr(CORE), np.float32)[::-1]
    theorem = [
        latex_mask(r"$\mathtt{theorem\ riemannHypothesis\ :}$", 46),
        latex_mask(r"$\mathtt{RiemannHypothesis\ :\!\!=\ rhOfAll}$", 46),
    ]
    theorem_halo = [halo(m) for m in theorem]
    digits, advance = number_masks(50)
    word = {
        k: latex_mask(rf"$\mathrm{{{v}}}$", 38)
        for k, v in {
            "files": "files",
            "theorems": "theorems",
            "definitions": "definitions",
            "lines": r"lines\ of\ proof",
            "unproven": r"steps\ left\ unproven",
        }.items()
    }
    stop_title = [latex_mask(s[1], 60) for s in STOPS]
    stop_note = [latex_mask(s[2], 36) if s[2] else None for s in STOPS]
    region_masks = []
    for f, tex, r in REGIONS:
        members = (folder == f) | (top == f)
        mid = np.median(angle[members])
        mask = latex_mask(tex, 34)
        region_masks.append((r * np.cos(mid), r * np.sin(mid), mask, halo(mask)))
    # a dark band behind the panel at the top while the camera is close in
    rows_y = np.arange(HEIGHT)
    band = smoothstep(np.clip((rows_y - 300) / 180, 0, 1)) * (
        1 - smoothstep(np.clip((rows_y - 700) / 220, 0, 1))
    )
    band = band.astype(np.float32)[:900, None, None]
    band_statement = smoothstep(np.clip((rows_y - 320) / 120, 0, 1)) * (
        1 - smoothstep(np.clip((rows_y - 850) / 150, 0, 1))
    )
    band_statement = band_statement.astype(np.float32)[:1000, None, None]

    # labels: real theorem names pinned to files spread over each stop's view
    labels = []
    for mask, (_, _, _, zoom, (_, arrive, depart)) in zip(in_stop, STOPS):
        cam = camera(0.5 * (arrive + depart))
        sx, sy = to_screen(wx, wy, cam)
        ok = (
            mask
            & (np.char.str_len(names) > 0)
            & (sx > 140)
            & (sx < WIDTH - 140)
            & (sy > 760)
            & (sy < 1560)
        )
        candidates = np.flatnonzero(ok)
        candidates = candidates[np.argsort(-lines[candidates])][:4000]
        chosen = []
        for i in candidates:
            if all(np.hypot(sx[i] - sx[j], sy[i] - sy[j]) > 190 for j in chosen):
                chosen.append(i)
            if len(chosen) == 5:
                break
        stop_labels = []
        for i in chosen:
            m = latex_mask(tex_name(str(names[i])), 30)
            stop_labels.append((i, m, halo(m, 5)))
        labels.append(stop_labels)

    # the longest chain: a file 43 steps from Main, back along BFS parents
    deep = np.flatnonzero((names == CHAIN_FILE) & (depth == depth.max()))
    deep = int(deep[0]) if deep.size else int(np.argmax(depth))
    chain = [deep]
    while parent[chain[-1]] >= 0:
        chain.append(int(parent[chain[-1]]))
    path_x, path_y = [], []
    for a, b in zip(chain[:-1], chain[1:]):
        step = (angle[b] - angle[a] + np.pi) % (2 * np.pi) - np.pi
        v = np.linspace(0, 1, 40, endpoint=False)
        rr, aa = radius[a] + v * (radius[b] - radius[a]), angle[a] + v * step
        path_x.append(rr * np.cos(aa))
        path_y.append(rr * np.sin(aa))
    path_x = np.concatenate(path_x + [[wx[chain[-1]]]])
    path_y = np.concatenate(path_y + [[wy[chain[-1]]]])
    chain_name = latex_mask(tex_name(str(names[deep])), 38)
    chain_note = latex_mask(
        rf"$\mathrm{{is\ {len(chain) - 1}\ steps\ from\ the\ theorem}}$", 36
    )
    chain_masks = [(m, halo(m, 6)) for m in (chain_name, chain_note)]
    lean_masks = [(dx, latex_mask(tex, 38)) for dx, tex in LEAN_LINES]
    lean_masks = [(dx, m, halo(m, 6)) for dx, m in lean_masks]
    lean_x = (WIDTH - max(dx + m.shape[1] for dx, m, _ in lean_masks)) // 2
    word_masks = [latex_mask(tex, 44) for tex in WORD_LINES]
    word_masks = [(m, halo(m, 6)) for m in word_masks]
    caption_lean = latex_mask(r"$\mathrm{in\ Lean}$", 30)
    caption_words = latex_mask(r"$\mathrm{in\ words}$", 30)
    credit = latex_mask(r"$\mathrm{@random\_plot}$", 40)
    credit_halo = halo(credit, 6)
    # header: "Riemann" stays put while "hypothesis" turns into "theorem"
    head_r = latex_mask(r"$\mathrm{Riemann}$", 62)
    head_a = latex_mask(r"$\mathrm{Riemann\ hypothesis}$", 62)
    head_b = latex_mask(r"$\mathrm{Riemann\ theorem}$", 62)

    def word_start(mask):  # first ink column after "Riemann" and the space
        ink_cols = mask.max(0) > 0.05
        col = head_r.shape[1] - 2
        while not ink_cols[col]:
            col += 1
        return col

    cut_a, cut_b = word_start(head_a), word_start(head_b)
    head_x = (WIDTH - head_a.shape[1]) // 2
    head_parts = {
        "riemann": head_a[:, : cut_a - 4],
        "hypothesis": head_a[:, cut_a - 4 :],
        "theorem": head_b[:, cut_b - 4 :],
    }
    head_halos = {k: halo(m, 7) for k, m in head_parts.items()}
    # the hook, held far longer than render_frames' 2.5 s, in the same style
    hook_lines = hook_masks(HOOK)
    hook_halos = [
        np.clip(cv2.GaussianBlur(np.pad(m, 14), (0, 0), 7) * 2.2, 0, 1)
        for m in hook_lines
    ]
    hook_ink = np.array(hex_to_bgr(HOOK_COLOUR), np.float32)[::-1]
    logger.info(f"longest chain: {names[deep]} ({len(chain) - 1} steps)")

    wave = PchipInterpolator(WAVE_T, np.array(WAVE_D) * d_max / WAVE_D[-1])

    def front(seconds):
        if seconds >= WAVE_T[-1]:  # keeps going, so the rim cools like the rest
            return d_max + 4 * (seconds - WAVE_T[-1])
        return max(float(wave(max(seconds, WAVE_T[0]))), 0.02)  # the theorem is lit

    def draw_rows(image, rows, y, alpha):
        """Rows of (number, word mask), numbers right-aligned on one column."""
        for value, label in rows:
            if alpha > 0:
                blit_number(image, value, digits, advance, 520, y, ink, alpha)
                base = y + digits[0].shape[0] - 4
                blit_mask(
                    image, label, 546, base - int(0.694 * 38) - 2, ink, 0.8 * alpha
                )
            y += 66
        return y

    def render(frame):
        t = frame / (FRAME_COUNT - 1)
        seconds = t * DURATION
        D = front(seconds)
        cam = camera(seconds)
        cam_x, cam_y, zoom = cam
        scale = zoom * RADIUS
        cx, cy = CENTRE
        focus = [window(seconds, s[4][0] + 0.6, s[4][2] - 0.1, 0.6) for s in STOPS]

        # points in view: position, lit fraction, flash, focus
        px = cx + scale * (pt_x - cam_x)
        py = cy + scale * (pt_y - cam_y)
        view = torch.nonzero(
            (px > -2) & (px < WIDTH + 1) & (py > -2) & (py < HEIGHT + 1)
        ).squeeze(1)
        px, py = px[view], py[view]
        file = is_file[view]
        w = pt_weight[view] * torch.where(file, 1.0, zoom)
        c = pt_colour[view]
        if D < d_max + 8 * FLASH:  # still lighting up or cooling down
            behind = D - pt_light[view]
            lit = GHOST + (1 - GHOST) * (behind / 0.12).clamp(0, 1)
            flash = torch.exp(-behind.clamp(min=0) / FLASH) * (behind > 0)
            w = w * lit * (1.0 + torch.where(file, 2.5, 1.5) * flash)
            c = c * (1 - 0.7 * flash[:, None]) + 0.7 * flash[:, None]
            c = torch.where((behind <= 0)[:, None], ghost_tint, c)
        dim = 1 - FOCUS_DIM * sum(focus)
        if dim < 1:
            k = int(np.argmax(focus))
            w = w * torch.where(pt_group[view] == k + 1, 1.0 + 0.4 * (1 - dim), dim)
        chain_dim = window(seconds, CHAIN_TRACE[0] - 0.3, DIVE[0] + 0.6, 0.6)
        if chain_dim > 0:
            w = w * (1 - 0.5 * chain_dim)
        canvas.zero_()
        splat(canvas, px, py, c * w[:, None])

        star_canvas.zero_()
        s_behind = D - st_light
        s_w = st_weight * (0.25 + 0.75 * (s_behind / 0.12).clamp(0, 1))
        s_w = s_w * (1 + 2 * torch.exp(-s_behind.clamp(min=0) / FLASH) * (s_behind > 0))
        splat(
            star_canvas,
            cx + scale * (st_x - cam_x),
            cy + scale * (st_y - cam_y),
            (0.5 + 0.5 * st_colour) * s_w[:, None],
        )
        canvas.add_(
            blur(star_canvas, star_kernel) * 30 + wide_blur(star_canvas, 28.0, 4) * 60
        )
        glow = (
            canvas
            + BLOOM[0] * wide_blur(canvas, 8.0, 4)
            + BLOOM[1] * wide_blur(canvas, 40.0, 8)
        )
        image = 1 - torch.exp(-EXPOSURE * glow)
        image = background / 255 * (1 - image) + image
        fade = 1 - smoothstep(np.clip((seconds - (DURATION - 0.5)) / 0.5, 0, 1))
        image = (image * fade * 255).cpu().numpy()

        # the theorem line, over the first second and a half
        a_text = 1 - smoothstep(np.clip((seconds - 0.9) / 0.8, 0, 1))
        if a_text > 0:
            y = int(cy - 150 - 40 * smoothstep(np.clip(seconds / 2.4, 0, 1)))
            for mask, glow_mask in zip(theorem, theorem_halo):
                blit_mask(
                    image, glow_mask, int(cx), y - 12, (0, 0, 0), 0.7 * a_text, "centre"
                )
                blit_mask(image, mask, int(cx), y, gold, a_text, "centre")
                y += int(mask.shape[0] * 1.45)

        # counters while the light spreads, then the overview's region names
        a_count = window(seconds, 0.6, OVERVIEW_END + 0.2, 0.5) * fade
        if a_count > 0:
            reached = int(np.searchsorted(sorted_light, D, side="right"))
            rows = [
                (reached, word["files"]),
                (int(theorems_cum[reached]), word["theorems"]),
                (int(lines_cum[reached]), word["lines"]),
            ]
            y = draw_rows(image, rows, 470, a_count)
            draw_rows(
                image,
                [(0, word["unproven"])],
                y,
                a_count * window(seconds, 10.0, 99, 0.6),
            )
        a_regions = window(seconds, REVEAL_END + 0.3, OVERVIEW_END + 0.1, 0.5)
        if a_regions > 0:
            for rx, ry, mask, glow_mask in region_masks:
                x, y = to_screen(rx, ry, cam)
                pad = (glow_mask.shape[0] - mask.shape[0]) // 2
                top_y = int(y - mask.shape[0] / 2)
                blit_mask(
                    image,
                    glow_mask,
                    int(x),
                    top_y - pad,
                    (0, 0, 0),
                    0.8 * a_regions,
                    "centre",
                )
                blit_mask(image, mask, int(x), top_y, ink, a_regions, "centre")

        # the tour: title, stats and pinned theorem names
        for k, (f, _, _, _, (_, arrive, depart)) in enumerate(STOPS):
            a_stop = window(seconds, arrive - 0.5, depart - 0.15, 0.4)
            if a_stop <= 0:
                continue
            image[:900] *= 1 - 0.7 * a_stop * band
            blit_mask(image, stop_title[k], int(cx), 470, ink, a_stop, "centre")
            members = in_stop[k]
            rows = [
                (int(members.sum()), word["files"]),
                (int(theorems[members].sum()), word["theorems"]),
                (int(lines[members].sum()), word["lines"]),
            ]
            y = draw_rows(image, rows, 470 + stop_title[k].shape[0] + 26, a_stop)
            if stop_note[k] is not None:
                blit_mask(
                    image, stop_note[k], int(cx), y + 4, ink, 0.85 * a_stop, "centre"
                )
            a_label = window(seconds, arrive - 0.1, depart - 0.3, 0.35)
            for i, mask, glow_mask in labels[k]:
                x, y = to_screen(wx[i], wy[i], cam)
                x, y = int(round(x)), int(round(y))
                cv2.circle(
                    image,
                    (x, y),
                    7,
                    tuple(float(v) * a_label for v in ink),
                    2,
                    cv2.LINE_AA,
                )
                right = x < cx
                x_text = x + 24 if right else x - 24 - mask.shape[1]
                end = x + 20 if right else x - 20
                cv2.line(
                    image,
                    (x + (8 if right else -8), y),
                    (end, y),
                    tuple(0.8 * float(v) * a_label for v in ink),
                    1,
                    cv2.LINE_AA,
                )
                pad = (glow_mask.shape[0] - mask.shape[0]) // 2
                top_y = y - mask.shape[0] // 2
                blit_mask(
                    image,
                    glow_mask,
                    x_text - pad,
                    top_y - pad,
                    (0, 0, 0),
                    0.75 * a_label,
                )
                blit_mask(image, mask, x_text, top_y, ink, a_label)

        # the longest chain, traced from the deepest file back to the theorem
        a_chain = window(seconds, CHAIN_TRACE[0], DIVE[0] + 0.3, 0.3) * fade
        if a_chain > 0:
            progress = smoothstep(
                np.clip(
                    (seconds - CHAIN_TRACE[0]) / (CHAIN_TRACE[1] - CHAIN_TRACE[0]), 0, 1
                )
            )
            count = max(2, int(progress * len(path_x)))
            sx, sy = to_screen(path_x[:count], path_y[:count], cam)
            pts = np.stack([sx, sy], 1)
            x_lo = int(max(min(sx.min(), sx[-1]) - 40, 0))
            y_lo = int(max(min(sy.min(), sy[-1]) - 40, 0))
            x_hi = int(min(sx.max() + 40, WIDTH))
            y_hi = int(min(sy.max() + 40, HEIGHT))
            pts = pts - [x_lo, y_lo]
            sx, sy = sx - x_lo, sy - y_lo
            line = np.zeros((y_hi - y_lo, x_hi - x_lo), np.float32)
            cv2.polylines(
                line,
                [np.round(pts * 4).astype(np.int32)],
                False,
                1.0,
                3,
                cv2.LINE_AA,
                shift=2,
            )
            for j in range(0, count, 40):  # the 44 files along the chain
                cv2.circle(line, (int(sx[j]), int(sy[j])), 4, 1.0, -1, cv2.LINE_AA)
            cv2.circle(line, (int(sx[-1]), int(sy[-1])), 7, 1.0, -1, cv2.LINE_AA)
            glow_line = cv2.GaussianBlur(line, (0, 0), 5)
            image[y_lo:y_hi, x_lo:x_hi] += (
                a_chain
                * gold[None, None, :]
                * np.clip(line + 2.5 * glow_line, 0, 1.5)[..., None]
            )
            x0, y0 = to_screen(wx[deep], wy[deep], cam)
            a_note = a_chain * window(seconds, CHAIN_TRACE[0] + 0.2, 99, 0.4)
            xr = min(int(x0) + 60, WIDTH - 40)
            ty = int(y0) - 30 - sum(m.shape[0] + 10 for m, _ in chain_masks)
            for mask, g in chain_masks:
                pad = (g.shape[0] - mask.shape[0]) // 2
                blit_mask(
                    image,
                    g,
                    xr - mask.shape[1] - pad,
                    ty - pad,
                    (0, 0, 0),
                    0.75 * a_note,
                )
                blit_mask(
                    image,
                    mask,
                    xr - mask.shape[1],
                    ty,
                    gold if mask is chain_name else ink,
                    a_note,
                )
                ty += mask.shape[0] + 10

        # closing totals
        a_final = window(seconds, FINAL_ARRIVE - 0.6, DIVE[0] + 0.4, 0.6) * fade
        if a_final > 0:
            rows = [
                (totals["files"], word["files"]),
                (totals["theorems"], word["theorems"]),
                (totals["definitions"], word["definitions"]),
                (totals["lines"], word["lines"]),
                (0, word["unproven"]),
            ]
            draw_rows(image, rows, 460, a_final)
        # the handle, for reposts that don't credit the account
        a_credit = window(seconds, CREDIT_IN, 99, 0.6) * fade
        if a_credit > 0:
            pad = (credit_halo.shape[0] - credit.shape[0]) // 2
            blit_mask(
                image,
                credit_halo,
                int(cx),
                CREDIT_Y - pad,
                (0, 0, 0),
                0.7 * a_credit,
                "centre",
            )
            blit_mask(image, credit, int(cx), CREDIT_Y, ink, 0.85 * a_credit, "centre")

        # the statement, in Lean and in words
        a_state = window(seconds, STATEMENT_IN, 99, 0.6) * fade
        if a_state > 0:
            image[:1000] *= 1 - 0.6 * a_state * band_statement
            blit_mask(image, caption_lean, int(cx), 400, ink, 0.6 * a_state, "centre")
            y = 452
            for dx, mask, g in lean_masks:
                pad = (g.shape[0] - mask.shape[0]) // 2
                blit_mask(
                    image, g, lean_x + dx - pad, y - pad, (0, 0, 0), 0.7 * a_state
                )
                blit_mask(image, mask, lean_x + dx, y, gold, a_state)
                y += 60
            a_words = window(seconds, STATEMENT_IN + 0.8, 99, 0.6) * fade
            y += 40
            blit_mask(image, caption_words, int(cx), y, ink, 0.6 * a_words, "centre")
            y += 52
            for mask, g in word_masks:
                pad = (g.shape[0] - mask.shape[0]) // 2
                blit_mask(
                    image, g, int(cx), y - pad, (0, 0, 0), 0.7 * a_words, "centre"
                )
                blit_mask(image, mask, int(cx), y, ink, a_words, "centre")
                y += mask.shape[0] + 22
        # the hook over most of the video, then the short header
        a_hook = window(seconds, -1.0, HOOK_END, 0.6) * fade
        if a_hook > 0:
            y = HOOK_Y
            for mask, g in zip(hook_lines, hook_halos):
                blit_mask(image, g, int(cx), y - 14, (0, 0, 0), 0.6 * a_hook, "centre")
                blit_mask(image, mask, int(cx), y, hook_ink, a_hook, "centre")
                y += int(mask.shape[0] * 1.25)

        # the header, and at the very end its last word changes
        a_head = window(seconds, HEADER_IN, 99, 0.6) * fade
        if a_head > 0:
            strike = smoothstep(
                np.clip((seconds - STRIKE[0]) / (STRIKE[1] - STRIKE[0]), 0, 1)
            )
            swap = smoothstep(np.clip((seconds - SWAP[0]) / (SWAP[1] - SWAP[0]), 0, 1))
            # glide so that "Riemann theorem" ends up centred
            shift = int(
                round(
                    swap * ((head_a.shape[1] - cut_a) - (head_b.shape[1] - cut_b)) / 2
                )
            )
            x_word = head_x + cut_a - 4 + shift
            parts = [
                ("riemann", head_x + shift, ink, 1.0),
                ("hypothesis", x_word, ink, 1.0 - swap),
                ("theorem", x_word, gold, swap),
            ]
            for key, x, colour_, alpha in parts:
                if alpha <= 0:
                    continue
                mask, g = head_parts[key], head_halos[key]
                pad = (g.shape[0] - mask.shape[0]) // 2
                blit_mask(
                    image, g, x - pad, HEADER_Y - pad, (0, 0, 0), 0.7 * a_head * alpha
                )
                if key == "theorem":  # a warm glow as the new word arrives
                    pulse = 4 * swap * (1 - swap) + 0.35 * swap
                    blit_mask(
                        image, g, x - pad, HEADER_Y - pad, gold, 0.5 * pulse * a_head
                    )
                blit_mask(image, mask, x, HEADER_Y, colour_, a_head * alpha)
            if strike > 0 and swap < 1:
                width = head_parts["hypothesis"].shape[1]
                y_line = HEADER_Y + int(0.52 * head_r.shape[0])
                x1 = x_word + int(strike * width)
                strike_layer = np.zeros((14, width + 8), np.float32)
                cv2.line(
                    strike_layer, (4, 7), (x1 - x_word + 4, 7), 1.0, 4, cv2.LINE_AA
                )
                blit_mask(
                    image,
                    strike_layer,
                    x_word - 4,
                    y_line - 7,
                    gold,
                    a_head * (1 - swap),
                )
        return np.clip(image, 0, 255)

    render_frames(
        settings,
        render,
        FRAME_COUNT,
        parallel=False,
        rgb=True,
    )
    track = soundtrack(MUSIC, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
