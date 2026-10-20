"""
A Quarter of a Million Grains

Drop grains of sand, one at a time, onto a single square of a grid. One rule: any square
holding four grains topples, passing one grain to each of its four neighbours, which may
topple in turn. That's the whole game. This sandpile ends with 262,144 grains, and the
camera pulls back as it grows.

Each colour is a height: empty, one grain (gold), two (teal) or three (deep blue, one grain
from toppling).
The rule is square, yet the pile grows round, and inside it a fractal of nested triangles
and stripes appears that nobody designed. Stranger still, the order of the topplings
doesn't matter: every order ends in exactly this pattern. That is why it's called the
abelian sandpile, a model of how avalanches, earthquakes and forest fires can organise
themselves at the edge of collapse.

Listen: an 8-bit tune that follows the sand. The melody leaps with every big avalanche,
the hiss is the grains pouring in, a coin chimes each time the pile doubles, and when the
last grain settles: level clear.

Made with Python, NumPy, Numba & OpenCV. Sound synthesized in NumPy.
#abeliansandpile #cellularautomata #mathart #generativeart #randomplots
"""

import gc
import time

import cv2
import numpy as np
from loguru import logger
from numba import njit

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import PROGRESSIONS, STYLES, Scale, beat, beat_times
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# Heights 0..3 inside the pile, then the untouched ground outside it.
COLOURS = ["#0b0b18", "#f2c76e", "#2fa6a0", "#1c2559", "#05050a"]
DENSITY = 2.1  # mean grains per square of a large pile: sets its radius

KNOBS = dict(
    grains_start=64,
    grains_end=2**18,
    grow_until=13.4,  # s: the pile reaches grains_end and rests
    ease=1.7,  # > 1: growth (in log grains) slows towards the end
    fill=0.46,  # the pile's radius as a fraction of the frame width
    gap=0.12,  # dark seam between cells, as a fraction of a cell (when cells are big)
    glow=0.35,
)

# --- Sound: chiptune, by --audio-only tag ------------------------------------------
SOUNDS = {
    "nes": dict(mode="major", chords="pop", bpm=128.0, duty=0.25, arp=2, echo=0.0),
    "gameboy": dict(
        mode="dorian", chords="drift", bpm=112.0, duty=0.125, arp=2, echo=0.3
    ),
    "arcade": dict(mode="minor", chords="epic", bpm=128.0, duty=0.5, arp=4, echo=0.15),
}
SOUND = "arcade"  # the default; an --audio-only tag naming another overrides it
JINGLE = (0, 2, 4, 7, 9, 11, 14)  # scale degrees of the "level clear" run
SETTLE = 0.7  # s before growth stops: the pile is still enough for "level clear"


@njit(nogil=True, cache=True)
def relax(height, reached, stack, top):
    """Topple every square holding 4 or more grains until none does; count topplings."""
    n = height.shape[1]
    topplings = 0
    while top > 0:
        top -= 1
        i = stack[top] // n
        j = stack[top] % n
        q = height[i, j] >> 2
        if q == 0:
            continue
        height[i, j] &= 3
        topplings += q
        for k in range(4):
            a = i + (1 if k == 0 else -1 if k == 1 else 0)
            b = j + (1 if k == 2 else -1 if k == 3 else 0)
            before = height[a, b]
            height[a, b] = before + q
            reached[a, b] = True
            if before < 4 and before + q >= 4:
                stack[top] = a * n + b
                top += 1
    return topplings


def grains_at(seconds, knobs) -> float:
    """Grains dropped by `seconds`: exponential growth, easing out, then held."""
    u = min(seconds / knobs["grow_until"], 1.0)
    eased = 1 - (1 - u) ** knobs["ease"]
    start, end = np.log(knobs["grains_start"]), np.log(knobs["grains_end"])
    return float(np.exp(start + (end - start) * eased))


def pile_radius(grains) -> float:
    """Radius of a pile of `grains`, in cells (it is nearly a disc of mean height DENSITY)."""
    return np.sqrt(grains / (DENSITY * np.pi)) + 0.5


def simulate(grains):
    """
    The stable pile after each total in `grains` (non-decreasing), all dropped on the
    centre square. Returns heights (frames, n, n) uint8, with 4 marking squares never
    reached, and the topplings each frame's new grains caused.
    """
    n = 2 * int(1.15 * pile_radius(grains[-1])) + 21
    centre = n // 2
    height = np.zeros((n, n), np.int64)
    reached = np.zeros((n, n), np.bool_)
    reached[centre, centre] = True
    stack = np.zeros(n * n, np.int64)
    states = np.empty((len(grains), n, n), np.uint8)
    topplings = np.zeros(len(grains), np.int64)
    dropped = 0
    for f, total in enumerate(grains):
        height[centre, centre] += total - dropped
        dropped = total
        stack[0] = centre * n + centre
        topplings[f] = relax(height, reached, stack, 1)
        states[f] = np.where(reached, height, 4)
    assert not reached[[0, -1], :].any() and not reached[:, [0, -1]].any()
    return states, topplings


def soundtrack(grains_cum, topplings_cum, frame_seconds, sound, rng):
    """
    A one-shot chiptune on a steady tempo (a whole number of bars in DURATION).

    Per arpeggio step (an 8th or 16th note) the sand gives two numbers: the grains
    dropped in the step and the topplings they caused (interpolated from the per-frame
    cumulative counts). Topplings per grain, detrended against the pile's size, is the
    avalanche surprise z: the pulse-wave lead climbs the bar's chord (PROGRESSIONS) as
    an arpeggio, jumps up a chord tone and plays louder when z is high, and rises an
    octave as the pile grows. The noise hi-hat's level follows the pour rate (log grains
    per second). Bar 2 adds the triangle bass, bar 3 the drums. A coin (two quick
    square notes) marks every doubling of the pile; when growth stops, a rising run
    and a held home chord: level clear, SETTLE s before growth stops (it has eased
    almost to a halt). A 9 kHz low-pass tames the pulse and noise harmonics.
    """
    spec = SOUNDS[sound]
    style = STYLES["chiptune"]
    root = int(rng.choice(style.roots))
    scale = Scale(root, spec["mode"])
    chords = PROGRESSIONS[spec["chords"]]
    bpm, kw = spec["bpm"], dict(circular=False)
    bar = 4 * beat(bpm)
    step = beat(bpm) / spec["arp"]
    track = audio.silence(DURATION)

    def hz(degree, octave=0):
        return scale.hz(degree, octave)

    def at(times, values):
        return np.interp(times, frame_seconds, values)

    end = round((KNOBS["grow_until"] - SETTLE) / step) * step  # level clear
    starts = beat_times(end, bpm, spec["arp"])
    stops = starts + step
    g = at(stops, grains_cum) - at(starts, grains_cum)
    t = at(stops, topplings_cum) - at(starts, topplings_cum)
    size = np.log(at(stops, grains_cum))
    surprise = np.log((t + 1) / (g + 1))
    trend = np.polyval(np.polyfit(size, surprise, 2), size)
    z = (surprise - trend) / max(np.std(surprise - trend), 1e-9)
    pour = np.log(g / step + 1)
    pour = (pour - pour.min()) / max(np.ptp(pour), 1e-9)
    growth = (size - size[0]) / max(size[-1] - size[0], 1e-9)

    lead = audio.silence(DURATION)
    for i, when in enumerate(starts):
        b = int(when // bar)
        degree = chords[b % len(chords)]
        tones = scale.chord(degree, 4)
        index = i % 4 + (1 if z[i] > 1.0 else 0)
        octave = 1 if growth[i] > 0.55 else 0
        midi = tones[index % 4] + 12 * (index // 4 + octave)
        velocity = 0.32 + 0.12 * float(np.clip(z[i], 0, 2))
        note = audio.chip(
            float(audio.midi_to_hz(midi)), velocity, 0.8 * step, spec["duty"]
        )
        audio.add(lead, audio.pan(note, 0.15 * (-1) ** i), when, **kw)
        hat = audio.chip_noise(0.08 + 0.3 * float(pour[i]), 0.025, 9000.0)
        audio.add(track, audio.pan(hat, 0.3), when, **kw)
        if b >= 1 and i % spec["arp"] == 0:  # 8th-note bass: root, octave, root...
            low = -2 if (i // spec["arp"]) % 2 == 0 else -1
            bass = audio.chip_bass(hz(degree, low), 0.6, 0.4 * beat(bpm))
            audio.add(track, bass, when, **kw)
        on_beat = i % spec["arp"] == 0
        beat_index = i // spec["arp"]
        if b >= 2 and on_beat:
            if beat_index % 2 == 0:
                audio.add(track, audio.kick(0.45, 72.0), when, **kw)
            else:
                audio.add(track, audio.chip_noise(0.5, 0.09, 2200.0), when, **kw)
    if spec["echo"]:
        lead = audio.delay(lead, 0.75 * beat(bpm), 0.3, spec["echo"], circular=False)
    track += lead

    # A coin at every doubling of the pile.
    first = int(np.log2(grains_cum[0])) + 1
    for k in range(first, int(np.log2(grains_cum[-1])) + 1):
        when = float(np.interp(2.0**k, grains_cum, frame_seconds))
        if when >= end - 0.3:
            continue
        for offset, midi, length in ((0.0, 83, 0.06), (0.07, 88, 0.3)):
            coin = audio.chip(float(audio.midi_to_hz(midi)), 0.22, length, 0.5)
            audio.add(track, audio.pan(coin, -0.4), when + offset, **kw)

    # Level clear: a rising run, then the home chord held.
    run = step / 2
    for j, degree in enumerate(JINGLE):
        note = audio.chip(hz(degree, 1), 0.45, 0.9 * run, spec["duty"])
        audio.add(track, note, end + j * run, **kw)
    land = end + len(JINGLE) * run
    ring = DURATION - land - 0.2
    for degree, side in ((0, 0.0), (2, -0.3), (4, 0.3)):
        note = audio.chip(hz(degree, 1), 0.4, ring, 0.5)
        audio.add(track, audio.pan(note, side), land, **kw)
    audio.add(track, audio.chip_bass(hz(0, -2), 0.7, ring), land, **kw)

    logger.info(
        f"Soundtrack: {sound}, root MIDI {root}, {len(starts)} steps, level clear at "
        f"{end:.2f} s"
    )
    track = audio.lowpass(
        style.room(track, rng, circular=False), 9000.0, circular=False
    )
    audio.fade(track, 0.0, 0.4)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    The abelian sandpile (Bak, Tang & Wiesenfeld 1987) grown from a single square.

    Grains are dropped on the centre of a square grid; a square with h >= 4 grains
    topples, h -> h - 4 and +1 to each of its four neighbours (several topplings at
    once when h >= 8). The stable result after N grains doesn't depend on the order of
    topplings (the abelian property), so each frame adds its new grains to the last
    frame's stable pile and relaxes it (`relax`, a numba stack), which is exact.
    N(s) grows exponentially from grains_start to grains_end, easing out over
    grow_until seconds (`grains_at`), then holds. No randomness in the picture: the
    seed only sets the soundtrack's key (settings.audio_rng; the posted one uses
    --seed 0, the default).

    The simulation runs once, in frame order (about 20 s for 2^18 grains), storing every
    frame's heights. Frames then render in parallel: heights -> COLOURS, scaled about
    the centre so the pile's radius (from N and the mean height DENSITY) stays at
    fill * WIDTH, nearest-neighbour at 2x then area-downsampled, with dark seams
    between cells while they are big and a soft bloom. The last 0.4 s fade out.

    Sound (see `soundtrack`): from the per-frame grain and toppling counts, saved as
    sand.npy so --audio-only runs skip the simulation.
    """
    settings = settings or ImageProcessingSettings(1)
    knobs = KNOBS

    seconds = np.arange(FRAME_COUNT) / (FRAME_COUNT - 1) * DURATION
    grains = np.array([round(grains_at(s, knobs)) for s in seconds], np.int64)
    saved = settings.output_path / settings.filename / "sand.npy"
    if settings.audio_only and saved.exists():
        states, n = None, 0
        topplings = np.load(saved)
    else:
        start = time.perf_counter()
        states, topplings = simulate(grains)
        n = states.shape[1]
        logger.info(
            f"Sandpile: {grains[-1]} grains on a {n}^2 grid, "
            f"{topplings.sum():.3g} topplings in {time.perf_counter() - start:.1f} s"
        )
        np.save(saved, topplings)

    lut = np.array(
        [[int(c[i : i + 2], 16) for i in (5, 3, 1)] for c in COLOURS], np.uint8
    )
    fade_frames = 0.4 * FPS
    big_w, big_h = 2 * WIDTH, 2 * HEIGHT

    def render(frame):
        cells = lut[states[frame]]  # (n, n, 3) BGR
        scale = 2 * knobs["fill"] * WIDTH / pile_radius(grains[frame])  # px per cell
        c = n // 2 + 0.5  # the centre square's middle, in cell units
        matrix = np.array(
            [[scale, 0, big_w / 2 - c * scale], [0, scale, big_h / 2 - c * scale]]
        )
        big = cv2.warpAffine(
            cells,
            matrix,
            (big_w, big_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=lut[4].tolist(),
        )
        # Seams between cells, fading out as the cells shrink to a few pixels.
        seam = knobs["gap"] * scale
        strength = np.clip((seam - 1.0) / 2.0, 0, 1)
        if strength > 0:
            x = (np.arange(big_w) - matrix[0, 2]) / scale % 1.0
            y = (np.arange(big_h) - matrix[1, 2]) / scale % 1.0
            half = knobs["gap"] / 2
            inside_x = (np.minimum(x, 1 - x) > half).astype(np.float32)
            inside_y = (np.minimum(y, 1 - y) > half).astype(np.float32)
            keep = 1 - strength * 0.6 * (1 - inside_y[:, None] * inside_x[None, :])
            big = (big * keep[..., None]).astype(np.uint8)
        image = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        small = cv2.resize(
            image, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA
        )
        bloom = cv2.GaussianBlur(small, (0, 0), 6.0)
        image += knobs["glow"] * cv2.resize(bloom, (WIDTH, HEIGHT))
        fade = min(1.0, (FRAME_COUNT - 1 - frame) / fade_frames)
        return np.clip(image * fade, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT)
    sound = settings.audio_tag if settings.audio_tag in SOUNDS else SOUND
    track = soundtrack(
        grains.astype(float),
        np.cumsum(topplings).astype(float),
        seconds,
        sound,
        settings.audio_rng,
    )
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
