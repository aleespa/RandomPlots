"""
Made of Code

Post number 400 of this account, and this time the picture is made from the
code itself. It opens on the very first script behind @random_plot, from
7 November 2019, the one that drew fifty spinning lemniscates, in Python. Then
the camera pulls back through seven years of scripts, the code behind the posts
you have seen here, ending with the script of this video.

Then the code turns to dust. Every grain is a speck of a letter, still in the
colour it had as code, and the dust is pulled into a spirograph: a small wheel
rolling inside a ring, with eight pens on one spoke, one for each colour of the
code. Names, keywords, strings, numbers and comments each trace their own curve,
loop after loop, until all eight close on themselves: thirteen points, eight
trips around the ring, drawn with the code that made this account.

x(θ) = (R − r) cos θ + d cos((R − r)θ / r),  y(θ) = (R − r) sin θ − d sin((R − r)θ / r)

Listen: the first script types itself with text-box blips, arpeggios race as
the years of code come into view, and the dust powers up. Then a melody note
sounds each time the outer pen reaches a point of the star, the arpeggio speeding
up and slowing down with the pen, until the curve closes on the home chord.

Thank you for 400 posts.

Made with Python & PyTorch. Sound synthesized in NumPy.
#spirograph #hypotrochoid #mathart #creativecoding #randomplots
"""

import builtins
import datetime as dt
import gc
import io
import keyword
import tokenize
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from common import audio
from common.gpu import select_device, splat, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import Scale, beat
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)
ROOT = Path(__file__).resolve().parents[2]
TODAY = dt.date(2026, 10, 20)

# --- The wall of code ------------------------------------------------------------
# Distances in the wall ("doc" units) are character widths.
COLUMNS = 8  # the scripts flow down these columns, oldest first
COLUMN_CHARS = 96  # longer lines are cut at the column edge
GUTTER = 8
LINE = 2.1  # line pitch
RAGGED = 24  # the last column's lines run this far past the screen's right edge
EDGE = 1.0  # space left of the first column and above the first line
START_MARGIN = 2.0  # characters either side of the first script's longest line
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
FINE = (24, 48)  # ink samples per character (across, down): the first script...
COARSE = (5, 10)  # ...and every other, only ever seen small
INK_MIN = 0.04  # samples with less coverage are dropped

# --- The spirograph ----------------------------------------------------------------
NAME, KEYWORD, BUILTIN, DEFINED, STRING, NUMBER, COMMENT, OPERATOR = range(8)
RING, WHEEL = 13, 8  # R and r: R/r lobes, r trips round the ring
# One pen per syntax class, all on one spoke of the wheel, from its centre out:
# their distances d run evenly over HOLE_RANGE.
HOLES = (COMMENT, OPERATOR, BUILTIN, KEYWORD, NAME, STRING, NUMBER, DEFINED)
HOLE_RANGE = (1.2, 7.6)
FIGURE_RADIUS = 470.0  # px, the outer curve's reach
TURN = np.radians(25.0)  # the figure turns this much while drawn, easing to rest
EASE = 0.85  # how much the pens slow at the start and end of the drawing
JITTER, SCATTER, SCATTER_SHARE = 1.6, 6.0, 0.1  # px across the curve

# --- Timing (s) --------------------------------------------------------------------
TYPE = (-0.15, 1.35)  # the first script types itself out
ZOOM = (2.2, 5.6)  # the camera pulls back to the whole wall
APPEAR_FADE = 0.35  # the other scripts fade in as the view reaches them
CRUMBLE = (5.8, 7.8)  # characters turn to dust, oldest first
CRUMBLE_JITTER = 0.12
DRAW = (7.6, 13.0)  # the pen draws
FLIGHT = (0.45, 0.9)  # a grain's trip from the cloud to the pen
GEAR_FADE = 0.6
REST = 14.3  # the figure stops turning
FADE_OUT = 0.35

# --- The dust ----------------------------------------------------------------------
SETTLE = 0.9  # s, how fast the dust spirals in to its orbit
START_LAG = 0.4  # s, the dust starts from rest
OMEGA = 0.9  # rad/s at FIGURE_RADIUS, faster inside
WOBBLE = 14.0  # px
PUFF, PUFF_TIME = (20.0, 90.0), 0.5  # px/s kick as a grain comes loose
BREAK = 0.3  # s for a loose character to break into grains
# A few grains are bright (DOTS_PER_PX per px of each curve, the rarest class
# sets the limit) and carry the light of the dust and of the curves; the rest
# make a faint haze in the cloud and fade out on their way to the pen.
DOTS_PER_PX = 2.5
BRIGHT_SPREAD = 0.5  # lognormal sigma of a bright grain's light
DUST_DOT, DUST_HAZE = 0.5, 0.15  # bright grain; the others, times their ink
TWINKLE, TWINKLE_RATE = 0.5, (1.5, 4.0)  # depth, Hz
CURVE_DOT = 0.6

# --- Look --------------------------------------------------------------------------
BACKGROUND = "#07080d"
SYNTAX = (
    "#c9d1e0",  # names
    "#c792ea",  # keywords
    "#82aaff",  # builtins
    "#ffcb6b",  # names after def / class
    "#a5d36e",  # strings
    "#f7a35c",  # numbers
    "#6b7a90",  # comments and labels
    "#7fdbca",  # operators
)
GEAR_COLOUR, GEAR_WEIGHT, NIB_WEIGHT = "#8fa3c0", 0.35, 1.2
FLASH_GAIN, FLASH_TIME = 3.0, 0.15  # a grain flares as it lands
CURSOR = 0.8
EXPOSURE = 2.0
BLOOM_SIGMA, BLOOM_FACTOR, BLOOM_GAIN = 10.0, 4, 0.45

# --- Sound -------------------------------------------------------------------------
# Retro, in a major key. An --audio-only tag naming a RETRO flavour overrides SOUND.
SOUND = "synthwave"
RETRO = {
    # lead: pulse duty of the melody (None: an 80s FM electric piano); arp: duty
    # of the arpeggios; pads: detuned-saw chords under them, this bright;
    # cutoff: lowpass (Hz) on the pulse layer, None for the raw 8-bit console;
    # echo: lead delay wet; rt60, wet: the room.
    "chiptune": dict(
        lead=0.25,
        arp=0.125,
        pads=False,
        brightness=0.75,
        cutoff=None,
        echo=0.25,
        rt60=0.6,
        wet=0.1,
    ),
    "gameboy": dict(
        lead=0.125,
        arp=0.5,
        pads=False,
        brightness=0.75,
        cutoff=None,
        echo=0.15,
        rt60=0.4,
        wet=0.06,
    ),
    "synthwave": dict(
        lead=None,
        arp=0.5,
        pads=True,
        brightness=0.55,
        cutoff=2800.0,
        echo=0.35,
        rt60=2.2,
        wet=0.3,
    ),
}
TEMPO = 128  # BPM of the arpeggios before the drawing: 8 bars in DURATION
ROOTS = range(60, 67)
TYPING = (0, 2, 4, 7, 9, 7, 4, 2, 4, 7, 11)  # text-box blips as the code types
TYPED_EVERY = 2  # a blip every this many typed lines
REVEAL = ((2.0, 0), (4.0, 5))  # (start s, chord degree) under the zoom out: I, vi
HATS_FROM = 3.0  # s, hi-hats join the arpeggio
DUST = 3  # IV while the code turns to dust, its arpeggio climbing CLIMB octaves
CLIMB = 2
SPARKLE = (0.4, 0.25)  # share of steps with a glint: as it crumbles, as it lands
ROLL = (
    0.6,
    8,
)  # the snare roll starts this long into the dust, in 16ths for the last 8
# One melody note per lobe tip of the outer curve, over a chord per tip:
# I I I I V V V vi vi vi IV IV IV, then home (I) as the curve closes.
DRAWING = (0, 0, 0, 0, 4, 4, 4, 5, 5, 5, 3, 3, 3)
MELODY = (4, 2, 4, 7, 8, 6, 8, 9, 7, 9, 10, 9, 11)
ARP_STEPS = 4  # arpeggio notes per tip: it speeds up and slows down with the pen
LEVEL_CLEAR = (0, 2, 4, 7, 9, 11, 14)  # the run into the last chord

STRINGS = {tokenize.STRING} | {
    getattr(tokenize, n)
    for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
    if hasattr(tokenize, n)
}
BUILTINS = set(dir(builtins))


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def archive() -> list[tuple[dt.date, str, str]]:
    """Every dated lab script posted before TODAY, oldest first: (date, path, source)."""
    scripts = []
    for path in sorted((ROOT / "lab").glob("[0-9]" * 4 + "/" + "[0-9]" * 4 + ".py")):
        year, mmdd = int(path.parent.name), path.stem
        try:
            date = dt.date(year, int(mmdd[:2]), int(mmdd[2:]))
        except ValueError:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if date < TODAY and source.strip():
            scripts.append((date, f"lab/{year}/{mmdd}.py", source))
    return scripts


def lines_of(source: str) -> list[str]:
    return source.expandtabs(4).splitlines()


def select(scripts: list, budget: int) -> list:
    """
    Scripts spread over the archive, oldest first, whose lines (each with its
    label and a blank) fill `budget` exactly: the most evenly spaced ones that
    fit whole, then others from the gaps while they fit, and one more cut short
    to the lines left.
    """
    size = [len(lines_of(s)) + 2 for _, _, s in scripts]
    for count in range(len(scripts), 0, -1):
        pick = set(np.round(np.linspace(0, len(scripts) - 1, count)).astype(int))
        if sum(size[i] for i in pick) <= budget:
            break
    left = budget - sum(size[i] for i in pick)
    spare = sorted(set(range(len(scripts))) - pick, key=lambda i: (i * 0.618034) % 1)
    for i in spare:
        if size[i] <= left:
            pick.add(i)
            left -= size[i]
    chosen = {i: scripts[i] for i in pick}
    longer = [i for i in spare if i not in pick and size[i] > left]
    if left >= 3 and longer:
        date, name, source = scripts[longer[0]]
        chosen[longer[0]] = (date, name, "\n".join(lines_of(source)[: left - 2]))
    return [chosen[i] for i in sorted(chosen)]


def syntax(lines: list[str]) -> list[np.ndarray]:
    """A syntax class (NAME, KEYWORD, ...) for every character of every line."""
    classes = [np.full(len(line), NAME, np.int64) for line in lines]

    def paint(start, end, cls):
        (r0, c0), (r1, c1) = start, end
        for r in range(r0, min(r1, len(lines)) + 1):
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(lines[r - 1])
            classes[r - 1][a:b] = cls

    previous = None
    readline = io.StringIO("\n".join(lines) + "\n").readline
    try:
        for tok in tokenize.generate_tokens(readline):
            if tok.type == tokenize.COMMENT:
                cls = COMMENT
            elif tok.type in STRINGS:
                cls = STRING
            elif tok.type == tokenize.NUMBER:
                cls = NUMBER
            elif tok.type == tokenize.OP:
                cls = OPERATOR
            elif tok.type == tokenize.NAME:
                if keyword.iskeyword(tok.string):
                    cls = KEYWORD
                elif previous in ("def", "class"):
                    cls = DEFINED
                elif tok.string in BUILTINS:
                    cls = BUILTIN
                else:
                    cls = NAME
            else:
                continue
            paint(tok.start, tok.end, cls)
            previous = tok.string
    except (tokenize.TokenError, SyntaxError):
        pass  # keep what was coloured so far
    return classes


def wall(scripts: list, column_lines: int) -> dict:
    """
    The scripts poured line by line down COLUMNS columns, each headed by a
    comment-coloured label. Returns one entry per visible character: its doc
    position (x, y), glyph, syntax class and script, plus the typing order of
    the first script (doc positions of every character and line end in turn).
    """
    flow = []
    for k, (date, name, source) in enumerate(scripts):
        lines = lines_of(source)
        flow.append((k, f"# {date:%Y-%m-%d} · {name}", None))
        flow += [(k, line, cls) for line, cls in zip(lines, syntax(lines))]
        flow.append((k, "", None))
    if len(flow) > COLUMNS * column_lines:
        logger.warning(f"{len(flow) - COLUMNS * column_lines} lines don't fit the wall")

    x, y, glyph, cls, script, typed = [], [], [], [], [], []
    for index, (k, text, classes) in enumerate(flow[: COLUMNS * column_lines]):
        column, row = divmod(index, column_lines)
        left = column * (COLUMN_CHARS + GUTTER)
        text = text[:COLUMN_CHARS]
        if k == 0 and classes is not None:
            typed += [(left + j, row * LINE) for j in range(len(text) + 1)]
        for j, ch in enumerate(text):
            if ch.isspace():
                continue
            x.append(left + j)
            y.append(row * LINE)
            glyph.append(ch)
            cls.append(COMMENT if classes is None else classes[j])
            script.append(k)
    return dict(
        x=np.array(x, float),
        y=np.array(y, float),
        glyph=np.array(glyph),
        cls=np.array(cls),
        script=np.array(script),
        typed=np.array(typed, float),
        lines=len(flow),
    )


def glyph_ink(chars, across: int, down: int) -> dict:
    """
    Each glyph as ink samples on an across x down grid over its character cell:
    (x, y) offsets in doc units and the coverage of each sample, from the
    glyph drawn large and area-averaged down.
    """
    width = 120
    height = int(round(width * LINE))
    probe = ImageFont.truetype(str(FONT), 200)
    font = ImageFont.truetype(str(FONT), int(round(200 * width / probe.getlength("M"))))
    ascent, descent = font.getmetrics()
    top = (height - ascent - descent) / 2
    ink = {}
    for ch in chars:
        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).text((0, top), ch, fill=255, font=font)
        small = cv2.resize(
            np.asarray(image, np.float32) / 255.0,
            (across, down),
            interpolation=cv2.INTER_AREA,
        )
        v, u = np.nonzero(small > INK_MIN)
        ink[ch] = ((u + 0.5) / across, (v + 0.5) / down * LINE, small[v, u])
    return ink


def grains(chars: dict) -> dict:
    """
    Every character broken into its ink samples: doc position, ink area (doc
    units squared) and the index of the character it came from. The first
    script (script 0) is sampled FINE, the rest COARSE.
    """
    parts = []
    for fine, (across, down) in ((True, FINE), (False, COARSE)):
        mask = (chars["script"] == 0) == fine
        area = (1 / across) * (LINE / down)
        ink = glyph_ink(np.unique(chars["glyph"][mask]), across, down)
        for ch, (u, v, coverage) in ink.items():
            index = np.flatnonzero(mask & (chars["glyph"] == ch))
            parts.append(
                (
                    (chars["x"][index, None] + u).ravel(),
                    (chars["y"][index, None] + v).ravel(),
                    np.broadcast_to(coverage * area, (len(index), len(u))).ravel(),
                    np.repeat(index, len(u)),
                )
            )
    x, y, ink, char = (np.concatenate(p) for p in zip(*parts))
    return dict(x=x, y=y, ink=ink, char=char)


def hypotrochoid(theta, pen):
    """
    The point of a pen `pen` from the wheel's centre at wheel angle `theta`, and
    the unit normal of its curve, in spirograph units.
    """
    k = (RING - WHEEL) / WHEEL
    a = RING - WHEEL
    x = a * np.cos(theta) + pen * np.cos(k * theta)
    y = a * np.sin(theta) - pen * np.sin(k * theta)
    tx = -a * np.sin(theta) - pen * k * np.sin(k * theta)
    ty = a * np.cos(theta) - pen * k * np.cos(k * theta)
    norm = np.hypot(tx, ty) + 1e-12
    return x, y, -ty / norm, tx / norm


def soundtrack(events: dict, kind: str, rng: np.random.Generator):
    """
    A retro one-shot in a major key (RETRO[kind]) that follows the story.

    - Typing: a text-box blip every TYPED_EVERY typed lines (TYPING), then a
      rolled start-up chord while the cursor blinks.
    Before the drawing everything sits on one 16th-note grid at TEMPO that
    lands exactly on DRAW[0], so the build-up resolves on the first lead note.

    - The zoom out: a 16th-note arpeggio on REVEAL's chords (I, vi),
      a bass note on every beat, hi-hats from HATS_FROM, louder as the wall
      comes into view (with pads: saw chords and a kick under it).
    - The dust: a power-up arpeggio on IV climbing CLIMB octaves over a snare
      roll in 8ths, then 16ths (ROLL), and glints on a share of its steps.
    - The drawing: a lead note (MELODY over DRAWING's chords) each time the
      outer pen reaches a lobe tip, panned to the tip, held to the next and
      echoed; under it ARP_STEPS arpeggio notes per tip, so the arpeggio
      speeds up and slows down with the pen, a bass note and a drum on each
      tip, quieter glints on some of the arpeggio's notes as the grains land.
    - Home as the curve closes: a level-clear run (LEVEL_CLEAR) into the home
      chord, dry, a low root and a crash, ringing over the rest.

    The 8-bit flavours keep the raw console: pulse waves, the stepped triangle
    bass, LFSR noise drums and coin blips. Synthwave smooths every one: the
    pulse layer through a lowpass (`cutoff`), a plucked bass, soft noise drums
    and bell glints.
    """
    spec = RETRO[kind]
    eightbit = spec["cutoff"] is None
    scale = Scale(int(rng.choice(ROOTS)), "major")
    track = audio.silence(DURATION)
    pulses = audio.silence(DURATION)  # arpeggios and blips, lowpassed for synthwave
    leads = audio.silence(DURATION)  # the melody, echoed
    kw = dict(circular=False)
    sixteenth = beat(TEMPO) / 4

    def hz(degree: int, octave: int = 0, high: int = 86) -> float:
        midi = scale.note(degree, octave)
        while midi > high:
            midi -= 12
        return float(audio.midi_to_hz(midi))

    def lead(degree, velocity, length, x=0.0):
        f = hz(degree)
        if spec["lead"] is None:
            note = audio.rhodes(f, velocity, max(length + 0.6, 1.2))
        else:
            note = audio.chip(f, velocity, length, spec["lead"])
        return audio.pan(note, x)

    def pulse(f, when, velocity, length, duty, x=0.0, into=None):
        note = audio.pan(audio.chip(f, velocity, length, duty), x)
        audio.add(pulses if into is None else into, note, when, **kw)

    def bass(degree, when, length, velocity=0.55):
        f = hz(degree % 7, -2)
        if eightbit:
            note = audio.chip_bass(f, velocity, length)
        else:
            note = audio.pluck(f, velocity, max(length, 0.5))
        audio.add(track, note, when, **kw)

    def pads(degree, begin, end, velocity, brightness=None):
        if spec["pads"]:
            brightness = spec["brightness"] if brightness is None else brightness
            for midi in scale.chord(degree, 3):
                f = float(audio.midi_to_hz(48 + (midi - 48) % 12))
                tone = audio.pad(f, end - begin, velocity, 0.4, 0.8, brightness)
                audio.add(track, tone, begin, **kw)

    def noise(when, velocity, which):
        """A hat, snare or crash: LFSR noise on the 8-bit flavours, else soft noise."""
        decay, pitch, tone = dict(
            hat=(0.02, 9000.0, 8000.0),
            snare=(0.09, 2200.0, 2500.0),
            crash=(0.5, 5000.0, 6000.0),
        )[which]
        if eightbit:
            sound = audio.chip_noise(velocity, decay, pitch)
        else:
            sound = audio.tick(rng, 0.7 * velocity, 0.7 * decay, tone)
        audio.add(track, sound, when, **kw)

    def kick(when, velocity):
        if spec["pads"]:
            audio.add(track, audio.kick(velocity, 52.0), when, **kw)
        else:
            audio.add(track, audio.kick(0.8 * velocity, 72.0), when, **kw)

    def grid(begin: float, end: float):
        """
        The 16th-note steps in [begin, end) of the one grid that lands on DRAW[0],
        and each step's count back from it (a multiple of 4 is a beat).
        """
        first = int(np.floor((DRAW[0] - begin) / sixteenth + 1e-9))
        last = int(np.floor((DRAW[0] - end) / sixteenth + 1e-9)) + 1
        counts = np.arange(first, last - 1, -1)
        return counts, DRAW[0] - counts * sixteenth

    def glints(times, share, velocity):
        """
        Coin blips (bells when smooth) on the major pentatonic, two octaves up,
        on a random `share` of `times`, so always in time with the rest.
        """
        for when in np.asarray(times)[rng.uniform(size=len(times)) < share]:
            degree = int(rng.choice((0, 1, 2, 4, 5)))  # pentatonic in a major scale
            f, x = hz(degree + 14, high=96), rng.uniform(-0.8, 0.8)
            if eightbit:
                pulse(f, when, velocity, 0.06, 0.5, x, into=track)
            else:
                audio.add(track, audio.pan(audio.bell(f, velocity), x), when, **kw)

    def midi_hz(midi: int) -> float:
        return float(audio.midi_to_hz(midi))

    # Typing, then the start-up chord while the cursor blinks.
    for i, when in enumerate(events["typed"][::TYPED_EVERY]):
        f = hz(TYPING[i % len(TYPING)], 1)
        pulse(f, when, 0.24 if eightbit else 0.45, 0.035, 0.5, rng.uniform(-0.3, 0.3))
    for i, degree in enumerate((0, 2, 4, 7)):
        note = lead(degree, 0.3, 0.7 - 0.06 * i, 0.25 * (i - 1.5))
        audio.add(track, note, TYPE[1] + 0.05 * i, **kw)
    bass(0, TYPE[1], 0.7, 0.45)

    # The zoom out: a steady arpeggio that grows.
    counts, steps = grid(REVEAL[0][0] + 0.2, CRUMBLE[0])
    for k, (count, when) in enumerate(zip(counts, steps)):
        degree = [d for t, d in REVEAL if t <= when][-1]
        grow = (when - steps[0]) / (CRUMBLE[0] - steps[0])
        chord = scale.chord(degree, 3) + [scale.chord(degree, 3)[0] + 12]
        tone = (0, 1, 2, 3, 2, 1)[k % 6]
        pulse(
            midi_hz(chord[tone]), when, 0.1 + 0.14 * grow, 0.8 * sixteenth, spec["arp"]
        )
        if count % 4 == 0:
            bass(degree, when, 0.45 * beat(TEMPO), 0.35 + 0.2 * grow)
            if when >= REVEAL[1][0]:
                kick(when, 0.3 + 0.2 * grow)
        if count % 2 == 1 and when >= HATS_FROM:
            noise(when, 0.06 + 0.1 * grow, "hat")
    for i, (when, degree) in enumerate(REVEAL):
        end = REVEAL[i + 1][0] if i + 1 < len(REVEAL) else CRUMBLE[0]
        pads(degree, when + 0.2, end + 0.2, 0.25 + 0.1 * i)

    # The dust: a power-up arpeggio and a snare roll into the drawing.
    counts, steps = grid(CRUMBLE[0], DRAW[0])
    chord = scale.chord(DUST, 3)
    for k, (count, when) in enumerate(zip(counts, steps)):
        midi = chord[k % 3] + 12 * int(CLIMB * k / len(steps))
        grow = k / len(steps)
        pulse(midi_hz(midi), when, 0.14 + 0.12 * grow, 0.7 * sixteenth, 0.5)
        if count % 4 == 0:
            bass(DUST, when, 0.45 * beat(TEMPO), 0.45)
        rolling = when >= CRUMBLE[0] + ROLL[0]
        if rolling and (count % 2 == 0 or count <= ROLL[1]):  # 8ths, then 16ths
            noise(when, 0.1 + 0.35 * grow, "snare")
    pads(DUST, CRUMBLE[0], DRAW[0], 0.35)
    glints(steps, SPARKLE[0], 0.12)

    # The drawing: a lead note per lobe tip, the arpeggio following the pen.
    tips, xs = events["tips"], events["tips_x"]
    arp_times = []
    for m, degree in enumerate(MELODY):
        begin, end = tips[m], tips[m + 1]
        span = end - begin
        chord = DRAWING[m]
        grow = m / len(MELODY)
        audio.add(
            leads,
            lead(degree, 0.3 + 0.12 * grow, 0.85 * span, 0.7 * xs[m]),
            begin,
            **kw,
        )
        bass(chord, begin, 0.8 * span, 0.55)
        if m % 2:
            noise(begin, 0.4 + 0.15 * grow, "snare")
        else:
            kick(begin, 0.4 + 0.15 * grow)
        if m == 0 or DRAWING[m - 1] != chord:
            change = [n for n in range(m + 1, len(DRAWING)) if DRAWING[n] != chord]
            pads(chord, begin, tips[change[0]] if change else tips[-1], 0.3)
        tones = scale.chord(chord, 3) + [scale.chord(chord, 3)[0] + 12]
        for k in range(ARP_STEPS):
            when = begin + span * k / ARP_STEPS
            arp_times.append(when)
            midi = tones[(0, 2, 1, 3)[k % 4]]
            velocity = 0.1 + 0.06 * grow
            pulse(midi_hz(midi), when, velocity, 0.8 * span / ARP_STEPS, spec["arp"])
            noise(when + 0.5 * span / ARP_STEPS, 0.07, "hat")
    glints(arp_times, SPARKLE[1], 0.07)

    # Home: the level-clear run, the chord and a crash.
    end = tips[-1]
    for i, degree in enumerate(LEVEL_CLEAR):
        pulse(hz(degree), end - 0.055 * (len(LEVEL_CLEAR) - i), 0.3, 0.05, 0.5)
    for i, degree in enumerate((0, 4, 7, 14)):
        note = lead(degree, 0.34, DURATION - end - 0.4, 0.3 * (i - 1.5))
        audio.add(track, note, end + 0.03 * i, **kw)  # dry: echoes would smear it
    bass(0, end, DURATION - end - 0.5, 0.6)
    kick(end, 0.6)
    noise(end, 0.45, "crash")
    pads(0, end, DURATION - 0.6, 0.4, 0.5)

    logger.info(
        f"Soundtrack: {kind}, root MIDI {scale.root}, {len(MELODY)} lead notes "
        f"from {tips[0]:.2f} s, home at {end:.2f} s"
    )
    if not eightbit:
        pulses = audio.lowpass(pulses, spec["cutoff"], circular=False)
    leads = audio.delay(leads, 0.18, 0.3, spec["echo"], circular=False)
    mix = track + pulses + leads
    track = audio.reverb(mix, rng, spec["rt60"], spec["wet"], circular=False)
    audio.fade(track, 0.0, FADE_OUT)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Code to dust to spirograph: every character of a wall of scripts becomes
    ink grains, and the grains draw a hypotrochoid.

    The wall: the first script of the account (lab/2019/1107.py), an even
    selection of the later lab scripts posted before TODAY, and this script,
    poured line by line down COLUMNS columns (oldest first, each under a
    "# date · path" label), syntax-coloured with `tokenize`. Each character's
    glyph (DejaVu Sans Mono) is drawn large and area-averaged onto a grid of
    ink samples, FINE for the first script (read up close) and COARSE for the
    rest; every sample with coverage over INK_MIN is a grain carrying its
    coverage times its cell area as ink, in its token's colour. All the
    pictures are these grains, splatted on the GPU at 2x and averaged down,
    tone-mapped (1 - exp(-EXPOSURE x)) and bloomed.

    Each frame is a closed form in t (seconds):
    - Text: the camera maps doc units to the screen at zoom z(t) px per
      character about centre c(t). The first script types itself out over TYPE
      at the zoom that fits its longest line (with START_MARGIN either side),
      then z falls log-smoothly to the zoom that fits the wall
      over ZOOM, with c moving in proportion to 1/z (the view widens around
      the first script). A grain's light is its ink times z^2, so text keeps
      its brightness at every zoom. The other scripts fade in as a front
      spreading out from the first script, in step with the view.
    - Dust: characters come loose in reading order over CRUMBLE (per-grain
      jitter). A loose grain starts from rest (lagged time s = tau^2/(tau +
      START_LAG)), spirals from its place in the wall to a random orbit radius
      (relaxing over SETTLE) while turning at OMEGA (sqrt(R/r) faster inside,
      so the cloud winds into arms), with a decaying kick and a slow wobble.
      Over BREAK its light crossfades from its ink to its dust light: a few
      grains per class are bright (lognormal, BRIGHT_SPREAD) and twinkle, the
      rest are a faint haze (DUST_HAZE times their ink), so the cloud reads as
      grains, not smoke.
    - Spirograph: a wheel of radius r rolls inside a ring of radius R; a pen
      at distance d from its centre traces x = (R-r) cos th + d cos((R-r)th/r),
      y = (R-r) sin th - d sin((R-r)th/r), closing after th = 2 pi r (R/r and
      r coprime: R lobes, r trips round the ring). There is one pen per syntax
      class, all on one spoke (HOLES, d evenly over HOLE_RANGE), so the eight
      curves interlace in the eight colours. The wheel angle runs from 0 to
      2 pi r over DRAW with an ease (EASE) that starts and ends slow. Grains
      are given wheel angles in reading order within their class, so each
      curve is drawn from the oldest code to this script; each grain leaves
      the cloud FLIGHT seconds before its pen gets there, flies to the moving
      pen, lands on the curve offset along its normal (JITTER, a few SCATTER)
      and flares as it lands. Each class has DOTS_PER_PX bright grains per px
      of its curve (all of the rarest class), which carry CURVE_DOT; the rest
      fade out in flight, so every curve is the same density of dots. The
      ring, the wheel, the spoke and a coloured nib on each pen are drawn
      while the pens move, and the whole figure turns by TURN, easing to a
      stop at REST.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    # The wall fills the screen edge to edge, every column to the bottom.
    wall_chars = COLUMNS * COLUMN_CHARS + (COLUMNS - 1) * GUTTER
    end_zoom = WIDTH / (EDGE + wall_chars - RAGGED)
    column_lines = int(np.ceil((HEIGHT / end_zoom - EDGE) / LINE))
    first, *rest = archive()
    this = (TODAY, "lab/2026/1020.py", Path(__file__).read_text(encoding="utf-8"))
    budget = COLUMNS * column_lines - sum(
        len(lines_of(s)) + 2 for *_, s in (first, this)
    )
    middle = select(rest, budget)
    scripts = [first, *middle, this]
    chars = wall(scripts, column_lines)
    dust = grains(chars)
    n = len(dust["x"])
    logger.info(
        f"{len(scripts)} scripts, {chars['lines']} lines in {COLUMNS} x {column_lines}, "
        f"{len(chars['x'])} characters, {n} grains"
    )

    # The camera: the whole first script across the screen, then the whole wall.
    first_lines = lines_of(first[2])
    widest = max(map(len, first_lines))
    start_zoom = WIDTH / (widest + 2 * START_MARGIN)
    start_centre = np.array([widest / 2, (len(first_lines) + 1) * LINE / 2])
    end_centre = np.array([WIDTH, HEIGHT]) / 2 / end_zoom - EDGE

    def zoom(seconds: float) -> float:
        u = smoothstep((seconds - ZOOM[0]) / (ZOOM[1] - ZOOM[0]))
        return float(start_zoom * (end_zoom / start_zoom) ** u)

    def centre(seconds: float) -> np.ndarray:
        u = (1 / zoom(seconds) - 1 / start_zoom) / (1 / end_zoom - 1 / start_zoom)
        return start_centre + u * (end_centre - start_centre)

    # When each character appears: typed, or reached by the spreading front.
    typed = chars["typed"]
    type_time = TYPE[0] + (TYPE[1] - TYPE[0]) * np.arange(len(typed)) / len(typed)
    lookup = {(px, py): t for (px, py), t in zip(map(tuple, typed), type_time)}
    first_char = chars["script"] == 0
    distance = np.hypot(chars["x"] - start_centre[0], chars["y"] - start_centre[1])
    reach0 = distance[first_char].max()
    t_table = np.linspace(ZOOM[0], ZOOM[1] - APPEAR_FADE - 0.2, 400)
    reach = 1 / np.array([zoom(t) for t in t_table])
    reach = reach0 + (distance.max() - reach0) * (reach - reach[0]) / (
        reach[-1] - reach[0]
    )
    appear = np.interp(distance, reach, t_table)
    fade = np.full(len(distance), APPEAR_FADE)
    for i in np.flatnonzero(first_char):
        appear[i] = lookup.get((chars["x"][i], chars["y"][i]), -1.0)  # labels: -1
        fade[i] = 0.04

    # Dust: release in reading order, orbits, wobble and kick, all drawn now.
    char = dust["char"]
    cls = chars["cls"][char]
    release = (
        CRUMBLE[0]
        + (CRUMBLE[1] - CRUMBLE[0]) * char / len(chars["x"])
        + rng.uniform(0, CRUMBLE_JITTER, n)
    )
    wall_x = (dust["x"] - end_centre[0]) * end_zoom  # maths axes about the centre
    wall_y = -(dust["y"] - end_centre[1]) * end_zoom
    r0, phi0 = np.hypot(wall_x, wall_y), np.arctan2(wall_y, wall_x)
    orbit = FIGURE_RADIUS * (0.12 + 0.95 * rng.uniform(size=n) ** 0.8)
    omega = OMEGA * np.sqrt(FIGURE_RADIUS / orbit)
    kick = rng.uniform(*PUFF, n) * PUFF_TIME
    kick_angle = rng.uniform(0, 2 * np.pi, n)
    wobble_f = rng.uniform(0.2, 0.6, (2, n)) * 2 * np.pi
    wobble_p = rng.uniform(0, 2 * np.pi, (2, n))

    # The pens: one per class, each grain given a wheel angle in reading order
    # within its class, and when the wheel reaches it.
    holes = np.empty(len(SYNTAX))
    holes[list(HOLES)] = np.linspace(*HOLE_RANGE, len(HOLES))
    counts = np.bincount(cls, minlength=len(SYNTAX))
    order = np.lexsort((char + rng.uniform(size=n), cls))
    rank = np.empty(n)
    rank[order] = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
    theta_end = 2 * np.pi * WHEEL / np.gcd(RING, WHEEL)
    x_table = np.linspace(0, 1, 4001)
    progress = x_table - EASE * np.sin(2 * np.pi * x_table) / (2 * np.pi)
    theta = theta_end * (rank + 0.5) / counts[cls]
    land = DRAW[0] + (DRAW[1] - DRAW[0]) * np.interp(
        theta / theta_end, progress, x_table
    )
    depart = np.maximum(land - rng.uniform(*FLIGHT, n), release + 0.25)
    flight = np.maximum(land - depart, 0.05)
    depart = land - flight

    def pen_angle(seconds: float) -> float:
        x = np.clip((seconds - DRAW[0]) / (DRAW[1] - DRAW[0]), 0, 1)
        return float(theta_end * (x - EASE * np.sin(2 * np.pi * x) / (2 * np.pi)))

    def turn(seconds: float) -> float:
        return float(TURN * smoothstep((seconds - DRAW[0]) / (REST - DRAW[0])))

    unit = FIGURE_RADIUS / (RING - WHEEL + HOLE_RANGE[1])
    cx, cy, nx, ny = hypotrochoid(theta, holes[cls])
    offset = np.where(
        rng.uniform(size=n) < SCATTER_SHARE, SCATTER, JITTER
    ) * rng.standard_normal(n)
    curve_x, curve_y = cx * unit + offset * nx, cy * unit + offset * ny

    # The bright grains: the same number per px of every curve.
    fine_t = np.linspace(0, theta_end, 20001)
    length = np.array(
        [
            np.hypot(*np.diff(hypotrochoid(fine_t, d)[:2], axis=1)).sum() * unit
            for d in holes
        ]
    )
    share = np.minimum(1.0, DOTS_PER_PX * length / np.maximum(counts, 1))
    bright = rng.uniform(size=n) < share[cls]
    light = rng.lognormal(-(BRIGHT_SPREAD**2) / 2, BRIGHT_SPREAD, n)
    dust_weight = np.where(
        bright, DUST_DOT * light, DUST_HAZE * dust["ink"] * end_zoom**2
    )
    curve_weight = np.where(bright, CURVE_DOT * light, 0.0)
    twinkle_f = rng.uniform(*TWINKLE_RATE, n) * 2 * np.pi
    twinkle_p = rng.uniform(0, 2 * np.pi, n)
    for k in HOLES:
        logger.info(
            f"pen d = {holes[k]:.2f}: {SYNTAX[k]}, {counts[k]} grains, "
            f"{share[k] * counts[k]:.0f} bright, curve {length[k]:.0f} px"
        )

    # To the GPU.
    to = dict(device=device, dtype=torch.float32)

    def tensor(a):
        return torch.tensor(np.ascontiguousarray(a), **to)

    g = {
        name: tensor(value)
        for name, value in dict(
            doc_x=dust["x"],
            doc_y=dust["y"],
            ink=dust["ink"],
            appear=appear[char],
            fade=fade[char],
            release=release,
            r0=r0,
            phi0=phi0,
            orbit=orbit,
            omega=omega,
            kick_x=kick * np.cos(kick_angle),
            kick_y=kick * np.sin(kick_angle),
            wf0=wobble_f[0],
            wf1=wobble_f[1],
            wp0=wobble_p[0],
            wp1=wobble_p[1],
            depart=depart,
            flight=flight,
            land=land,
            curve_x=curve_x,
            curve_y=curve_y,
            dust_weight=dust_weight,
            curve_weight=curve_weight,
            twinkle_f=twinkle_f,
            twinkle_p=twinkle_p,
        ).items()
    }
    grain_class = torch.tensor(cls, device=device)
    palette = tensor(np.array([_rgb(c) for c in SYNTAX]))
    colours = palette[grain_class]
    del dust, order, rank, release, wall_x, wall_y, r0, phi0, orbit, omega
    gc.collect()

    gear_rgb = tensor(_rgb(GEAR_COLOUR))
    circle = torch.linspace(0, 2 * np.pi, 4001, **to)[:-1]
    arm = torch.linspace(0, 1, 400, **to)
    nib = 1.2 * torch.stack(  # a small hexagon of points for each pen's nib
        [torch.cos(circle[::667]), torch.sin(circle[::667])], 1
    )
    nib = torch.cat([torch.zeros(1, 2, **to), nib])
    background = tensor(_rgb(BACKGROUND))
    canvas = torch.zeros(2 * HEIGHT, 2 * WIDTH, 3, **to)

    def ease01(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        return x * x * (3 - 2 * x)

    def pens(seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Every pen's position in px (maths axes, before the figure's turn)."""
        px, py, _, _ = hypotrochoid(pen_angle(seconds), holes)
        return px * unit, py * unit

    def gears(seconds: float, rotation: tuple[float, float]):
        """
        The faint ring, wheel and spoke, and a coloured nib on every pen, as
        points (maths axes) with their light (N, 3); None when not shown.
        """
        alpha = smoothstep((seconds - DRAW[0] + GEAR_FADE) / GEAR_FADE) * (
            1 - smoothstep((seconds - DRAW[1]) / GEAR_FADE)
        )
        if alpha <= 0:
            return None
        th = pen_angle(seconds)
        a = (RING - WHEEL) * unit
        wx, wy = a * np.cos(th), a * np.sin(th)
        px, py = pens(seconds)
        tip = HOLES[-1]  # the outermost pen ends the spoke
        xs = torch.cat(
            [
                RING * unit * torch.cos(circle),
                wx + WHEEL * unit * torch.cos(circle),
                wx + (float(px[tip]) - wx) * arm,
            ]
        )
        ys = torch.cat(
            [
                RING * unit * torch.sin(circle),
                wy + WHEEL * unit * torch.sin(circle),
                wy + (float(py[tip]) - wy) * arm,
            ]
        )
        rgb = gear_rgb.expand(len(xs), 3) * GEAR_WEIGHT
        nibs = tensor(np.stack([px, py], 1))[:, None, :] + nib[None]  # (8, 7, 2)
        xs = torch.cat([xs, nibs[..., 0].reshape(-1)])
        ys = torch.cat([ys, nibs[..., 1].reshape(-1)])
        rgb = torch.cat([rgb, palette.repeat_interleave(len(nib), 0) * NIB_WEIGHT])
        c, s = rotation
        return xs * c - ys * s, xs * s + ys * c, rgb * alpha

    def render(frame: int) -> torch.Tensor:
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        z = zoom(seconds)
        ox, oy = centre(seconds)
        psi = turn(seconds)
        c, s = np.cos(psi), np.sin(psi)
        canvas.zero_()

        # In the wall, or loose in the cloud.
        x = (g["doc_x"] - ox) * z
        y = -(g["doc_y"] - oy) * z
        weight = g["ink"] * z * z * ease01((seconds - g["appear"]) / g["fade"])
        tau = (seconds - g["release"]).clamp_min(0)
        loose = tau > 0
        if bool(loose.any()):
            lag = tau * tau / (tau + START_LAG)
            r = g["r0"] + (g["orbit"] - g["r0"]) * (1 - torch.exp(-lag / SETTLE))
            phi = g["phi0"] + g["omega"] * lag
            spread = 1 - torch.exp(-lag / PUFF_TIME)
            wob = WOBBLE * (1 - torch.exp(-lag / 0.8))
            dust_x = (
                r * torch.cos(phi)
                + g["kick_x"] * spread
                + wob * torch.sin(g["wf0"] * tau + g["wp0"])
            )
            dust_y = (
                r * torch.sin(phi)
                + g["kick_y"] * spread
                + wob * torch.sin(g["wf1"] * tau + g["wp1"])
            )
            x = torch.where(loose, dust_x, x)
            y = torch.where(loose, dust_y, y)
            in_wall = g["ink"] * end_zoom**2
            dust = g["dust_weight"] * (
                1 + TWINKLE * torch.sin(g["twinkle_f"] * seconds + g["twinkle_p"])
            )
            broken = ease01(tau / BREAK)
            weight = torch.where(loose, in_wall + (dust - in_wall) * broken, weight)

        # To the pens, and onto the curves.
        if seconds > DRAW[0] - FLIGHT[1] - 0.3:
            px, py = (tensor(v)[grain_class] for v in pens(seconds))
            landed = seconds >= g["land"]
            to_x = torch.where(landed, g["curve_x"], px)
            to_y = torch.where(landed, g["curve_y"], py)
            to_x, to_y = to_x * c - to_y * s, to_x * s + to_y * c
            pull = ease01((seconds - g["depart"]) / g["flight"])
            x = x + (to_x - x) * pull
            y = y + (to_y - y) * pull
            weight = weight + (g["curve_weight"] - weight) * pull
            since = (seconds - g["land"]).clamp_min(0)
            weight = weight * (1 + FLASH_GAIN * landed * torch.exp(-since / FLASH_TIME))

        splat(
            canvas,
            2 * (WIDTH / 2 + x),
            2 * (HEIGHT / 2 - y),
            colours * (4 * weight)[:, None],
        )
        parts = gears(seconds, (c, s))
        if parts is not None:
            xs, ys, rgb = parts
            splat(canvas, 2 * (WIDTH / 2 + xs), 2 * (HEIGHT / 2 - ys), 4 * rgb)

        light = F.avg_pool2d(canvas.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)

        # The cursor while typing, blinking until the camera pulls back.
        blink = seconds < TYPE[1] or (seconds < ZOOM[0] and (seconds * 2) % 1 < 0.5)
        if blink:
            k = min(len(typed) - 1, max(0, int(np.searchsorted(type_time, seconds))))
            left = int(WIDTH / 2 + (typed[k, 0] - ox) * z)
            top = int(HEIGHT / 2 + (typed[k, 1] - oy + 0.2) * z)
            light[top : top + int(LINE * z * 0.75), left : left + int(z)] += CURSOR

        light = 1 - torch.exp(-EXPOSURE * light)
        image = (
            background
            + light
            + BLOOM_GAIN * wide_blur(light, BLOOM_SIGMA, BLOOM_FACTOR)
        )
        image = image * (1 - smoothstep((seconds - (DURATION - FADE_OUT)) / FADE_OUT))
        return image.clamp(0, 1) * 255.0

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)

    # For the soundtrack, from the same closed forms: when lines are typed, and
    # when and where the outer pen reaches each lobe tip (the last closes it).
    lobes = RING // np.gcd(RING, WHEEL)
    tip_theta = 2 * np.pi * WHEEL / RING * np.arange(lobes + 1)
    tips = DRAW[0] + (DRAW[1] - DRAW[0]) * np.interp(
        tip_theta / theta_end, progress, x_table
    )
    tx, ty, _, _ = hypotrochoid(tip_theta, holes[HOLES[-1]])
    angle = np.array([turn(t) for t in tips])
    events = dict(
        typed=type_time[typed[:, 0] == 0],
        tips=tips,
        tips_x=(tx * np.cos(angle) - ty * np.sin(angle)) * unit / FIGURE_RADIUS,
    )
    tag = settings.audio_tag
    track = soundtrack(events, tag if tag in RETRO else SOUND, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
