"""
Three Answers to One Probability Question

Pick a random chord of a circle. What is the probability it is longer than a
side of the triangle inside? Three circles answer with 20,000 random chords
each, and disagree: 1/3, 1/2 and 1/4.

This is Bertrand's paradox (1889), and no answer is wrong. Two random points
on the circle, a random point on a random radius, a random midpoint inside:
each is a fair way to pick a chord "at random", but they are different ways,
and probability must know which. Red chords are the long ones: their midpoints
lie in the triangle's inner circle.

Listen: a note for each early chord and as each circle passes 10, 100 and 1,000
chords; at the end, three notes at 3, 2 and 4 times one pitch.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#bertrandparadox #probability #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
from loguru import logger

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    hex_to_bgr,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds, one-shot
FRAME_COUNT = frame_count(DURATION)

# --- The chords ----------------------------------------------------------------------
CHORDS = 20000  # per circle
SLOW = 5  # the first chords of each circle land one at a time, shown being built
SLOW_START, SLOW_STEP = 0.15, 0.8  # s
METHOD_LAG = 0.27  # s between the three circles' slow chords
FAST_END = 11.0  # s, the last chord lands
REVEAL = 11.4  # s, the exact answers appear
ANSWERS = (3, 2, 4)  # 1 / k for the three methods
MILESTONES = (10, 100, 1000)

# --- Layout (px) -----------------------------------------------------------------------
R = 212  # circle radius
CX = 292
CYS = (612, 1102, 1592)
MARGIN = 10
SIDE = 2 * (R + MARGIN)  # crop around a circle, 1x
SS = 2  # supersampling of the ink
STEP = 0.7  # sample spacing along a chord, supersampled px
TX = 572  # left edge of the panel text
TITLE_YS = (112, 186)
LEGEND_Y = 282

# --- Look ------------------------------------------------------------------------------
PAPER = "#f4f0e7"
LONG = "#b8432f"  # brick red: longer than the triangle's side
SHORT = "#27466b"  # ink blue: shorter
LINE = "#4a4440"  # circle and triangle
TEXT, TEXT_DIM = "#2a2622", "#7a7068"
INK = 0.04  # absorbance per unit of chord coverage, with every chord laid
INK_FIRST = 0.7  # ... and the most it reaches while only a few chords are down
INK_POWER = 0.5  # ink per chord ~ n^-0.5, so the density grows like sqrt(n)
FLASH = 0.6  # s for a new chord's highlight to fade by e

# --- Sound -----------------------------------------------------------------------------
# "box" (bells over pad and choir, posted) or "keys" (Rhodes over pad and strings)
STYLE = "box"
VARIANTS = {
    # lead voice, mode, MIDI roots, second bed voice
    "keys": (audio.rhodes, "dorian", range(55, 61), "strings"),
    "box": (audio.bell, "lydian", range(67, 73), "choir"),
}
PROGRESSION = (0, 5, 3, 4)  # one chord per ~2.9 s, then home at the reveal


def chord_times():
    """Landing time (s) of chord i of a circle, before the circle's lag."""
    i = np.arange(CHORDS, dtype=float)
    slow = SLOW_START + SLOW_STEP * i
    last_slow = SLOW_START + SLOW_STEP * (SLOW - 1)
    # then exponentially faster: chord i lands at a time linear in log i
    u = np.log(np.maximum(i, SLOW - 1) / (SLOW - 1)) / np.log((CHORDS - 1) / (SLOW - 1))
    fast = last_slow + (FAST_END - 2 * METHOD_LAG - last_slow) * u
    return np.where(i < SLOW, slow, fast)


def ink_strength(n):
    """
    Absorbance per unit coverage with n chords down: strong for the first chords
    so each one shows, thinning as they pile up so the finished weave stays
    readable instead of turning black.
    """
    return min(INK_FIRST, INK * (CHORDS / n) ** INK_POWER)


def sample_chords(rng):
    """
    CHORDS chords of the unit circle for each of Bertrand's three methods, as
    (endpoints (n, 2, 2), midpoint distance d (n,), construction (n, 2)).
    Construction: method 0, the angle of the second endpoint; 1 and 2, the
    midpoint's angle phi.
    """
    out = []
    # 0: two independent uniform points on the circle
    a, b = rng.uniform(0, 2 * np.pi, (2, CHORDS))
    ends = np.stack(
        [np.stack([np.cos(a), np.sin(a)], 1), np.stack([np.cos(b), np.sin(b)], 1)], 1
    )
    out.append((ends, np.abs(np.cos((a - b) / 2)), np.stack([a, b], 1)))
    # 1: a uniform radius, a uniform point on it; 2: a uniform point in the disc
    for method in (1, 2):
        phi = rng.uniform(0, 2 * np.pi, CHORDS)
        u = rng.uniform(0, 1, CHORDS)
        d = u if method == 1 else np.sqrt(u)
        mid = d[:, None] * np.stack([np.cos(phi), np.sin(phi)], 1)
        along = np.stack([-np.sin(phi), np.cos(phi)], 1) * np.sqrt(1 - d * d)[:, None]
        ends = np.stack([mid - along, mid + along], 1)
        out.append((ends, d, np.stack([phi, phi], 1)))
    return out


def ink_samples(ends, long):
    """
    Pixel indices (into a (2, S, S) supersampled crop, class 0 = long, 1 = short)
    of points every STEP px along each chord, concatenated in chord order, and the
    offset where each chord's samples start.
    """
    s = SS * SIDE
    p = (SIDE / 2 + R * ends * [1, -1]) * SS  # (n, 2, 2) supersampled px
    length = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    counts = np.ceil(length / STEP).astype(int) + 1
    offsets = np.concatenate([[0], np.cumsum(counts)])
    chord = np.repeat(np.arange(len(ends)), counts)
    u = (np.arange(offsets[-1]) - offsets[chord]) / (counts[chord] - 1)
    xy = p[chord, 0] + u[:, None] * (p[chord, 1] - p[chord, 0])
    xy = np.clip(np.rint(xy).astype(np.int64), 0, s - 1)
    index = xy[:, 1] * s + xy[:, 0] + np.where(long[chord], 0, s * s)
    return index.astype(np.int32), offsets


def glyphs(size):
    """Digit and point masks for a running estimate, built once."""
    return {c: latex_mask(rf"$\mathrm{{{c}}}$", size) for c in "0123456789."}


def blit_number(image, masks, text, x, bottom, colour, advance):
    """Draw `text` digit by digit, bottoms on a common baseline, digits monospaced."""
    for c in text:
        m = masks[c]
        if c == ".":
            blit_mask(image, m, x + 2, bottom - m.shape[0], colour)
            x += m.shape[1] + 6
        else:
            blit_mask(
                image, m, x + (advance - m.shape[1]) // 2, bottom - m.shape[0], colour
            )
            x += advance


def generate(settings: ImageProcessingSettings = None):
    """
    Bertrand's paradox (1889) as a 15 s one-shot. "A random chord is longer than
    the side of the inscribed equilateral triangle" (length > sqrt(3) R, i.e. its
    midpoint lies within R/2 of the centre, the triangle's incircle) has
    probability 1/3, 1/2 or 1/4 depending on how the chord is chosen: two uniform
    endpoints on the circle (P(d < 1/2) for d = |cos(dtheta/2)| is 1/3), a uniform
    point on a uniform radius (d uniform: 1/2), or a uniform midpoint in the disc
    (d = sqrt(u): 1/4).

    Data: 20000 chords per method from settings.rng, drawn up front, so the picture
    depends on the seed (the posted video uses --seed 0). Chord i of a circle
    lands at a fixed time: the first five 0.8 s apart (each circle 0.27 s behind
    the one above), then at times linear in log i, so the count grows
    exponentially to 20000 at 11 s. The exact answers fade in from 11.4 s and the
    finished image rests to the end.

    Frame at time s (t = frame / (FRAME_COUNT - 1), s = 15 t): for each circle the
    first n(s) chords are laid as ink. Every chord is precomputed as samples 0.7 px
    apart on a 2x grid; a frame bincounts the first n chords' samples (long and
    short separately), area-downsamples to coverage, and multiplies the paper by
    exp(-k(n) coverage A), k(n) = min(0.7, 0.04 sqrt(20000 / n)), with A the absorbance of brick red (long) or ink blue
    (short) -- subtractive, like ink. New chords flash as thick lines with their
    construction (the two endpoints; the radius and the point on it; the
    midpoint), fading over 0.6 s, only during the first ~5 s. Running estimates
    (long / n, three decimals) are composed from LaTeX digit masks built once; all
    text is LaTeX (latex_mask / blit_mask). Pure function of s on the render_frames
    thread pool.

    Sound (posted: --seed 0, no --audio-only tag, STYLE "box" = music_box bells
    in lydian over pad and "u" choir, audio seed = seed, root 70):
    see soundtrack().
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    methods = sample_chords(rng)
    base = chord_times()
    times = [base + k * METHOD_LAG for k in range(3)]
    longs = [d < 0.5 for _, d, _ in methods]
    ink = [ink_samples(ends, long) for (ends, _, _), long in zip(methods, longs)]
    running = [np.cumsum(long) / np.arange(1, CHORDS + 1) for long in longs]
    for k, long in enumerate(longs):
        logger.info(f"Method {k}: {long.mean():.4f} long (exact 1/{ANSWERS[k]})")

    paper = np.array(hex_to_bgr(PAPER), np.float32)
    colours = {
        name: np.array(hex_to_bgr(c), np.float32)
        for name, c in (("long", LONG), ("short", SHORT), ("line", LINE))
    }
    text_c, dim_c = hex_to_bgr(TEXT), hex_to_bgr(TEXT_DIM)
    absorb = np.stack(
        [-np.log(colours["long"] / paper), -np.log(colours["short"] / paper)]
    )  # (2, 3)

    # --- Static picture: paper, labels, legend --------------------------------------
    background = np.empty((HEIGHT, WIDTH, 3), np.float32)
    background[:] = paper
    labels = (
        (r"$\mathrm{two\ random\ points}$", r"$\mathrm{on\ the\ circle}$"),
        (r"$\mathrm{a\ random\ point\ on}$", r"$\mathrm{a\ random\ radius}$"),
        (r"$\mathrm{a\ random\ midpoint}$", r"$\mathrm{inside\ the\ disc}$"),
    )
    for (first, second), cy in zip(labels, CYS):
        blit_mask(background, latex_mask(first, 40), TX, cy - 178, dim_c)
        blit_mask(background, latex_mask(second, 40), TX, cy - 128, dim_c)
    approx = latex_mask(r"$P\ \approx$", 70)  # drawn once a circle has a chord
    # Legend: a red and a blue stroke with their meaning.
    long_word = latex_mask(r"$\mathrm{longer\ than\ its\ side}$", 38)
    short_word = latex_mask(r"$\mathrm{shorter}$", 38)
    stroke, gap = 56, 14
    total = 2 * (stroke + gap) + long_word.shape[1] + 60 + short_word.shape[1]
    x = (WIDTH - total) // 2
    for word, colour in ((long_word, LONG), (short_word, SHORT)):
        y_mid = LEGEND_Y + word.shape[0] // 2
        cv2.line(
            background,
            (x, y_mid),
            (x + stroke, y_mid),
            hex_to_bgr(colour),
            5,
            cv2.LINE_AA,
        )
        x += stroke + gap
        blit_mask(background, word, x, LEGEND_Y, dim_c)
        x += word.shape[1] + 60

    titles = [
        (
            latex_mask(r"$\mathrm{Pick\ a\ chord\ of\ a\ circle\ at\ random.}$", 52),
            latex_mask(
                r"$\mathrm{Is\ it\ longer\ than\ the\ triangle\text{'}s\ side?}$", 52
            ),
        ),
        (
            latex_mask(r"$\mathrm{Three\ fair\ ways\ to\ pick\ it{,}}$", 52),
            latex_mask(r"$\mathrm{three\ different\ answers.}$", 52),
        ),
    ]
    for pair in titles:
        for m in pair:
            assert m.shape[1] < WIDTH - 60, m.shape
    exact = [
        latex_mask(rf"$\mathrm{{exactly}}\ \ \dfrac{{1}}{{{k}}}$", 66) for k in ANSWERS
    ]
    digits = glyphs(70)
    advance = max(digits[c].shape[1] for c in "0123456789") + 3
    number_x = TX + approx.shape[1] + 16

    # Circle and triangle outlines, one alpha mask shared by the three crops.
    outline = np.zeros((SIDE, SIDE), np.uint8)
    centre = (SIDE * 8, SIDE * 8)  # 4-bit subpixel
    cv2.circle(outline, centre, R * 16, 150, 2, cv2.LINE_AA, 4)
    vertices = np.array(
        [
            [SIDE / 2 + R * np.cos(a), SIDE / 2 - R * np.sin(a)]
            for a in np.pi / 2 + 2 * np.pi / 3 * np.arange(3)
        ]
    )
    cv2.polylines(
        outline, [np.rint(vertices * 16).astype(np.int32)], True, 95, 1, cv2.LINE_AA, 4
    )
    outline_alpha = outline.astype(np.float32)[..., None] / 255

    def to_px(point):
        """Unit-circle point -> 4-bit subpixel crop coordinates."""
        return (
            int(round((SIDE / 2 + R * point[0]) * 16)),
            int(round((SIDE / 2 - R * point[1]) * 16)),
        )

    def flash(k, s):
        """Highlight masks (long, short, construction) of circle k's recent chords."""
        gate = 1.0 - float(smoothstep((s - 3.6) / 1.4))
        if gate <= 0:
            return None
        ends, d, angle = methods[k]
        age = s - times[k]
        recent = np.flatnonzero((age >= 0) & (age < 5 * FLASH))
        if len(recent) == 0:
            return None
        masks = np.zeros((3, SIDE, SIDE), np.uint8)
        for i in recent[np.argsort(-age[recent])]:  # oldest first
            h = gate * np.exp(-age[i] / FLASH) * float(smoothstep(age[i] / 0.08))
            value = int(255 * h)
            if value < 2:
                continue
            target = masks[0] if longs[k][i] else masks[1]
            a, b = to_px(ends[i, 0]), to_px(ends[i, 1])
            cv2.line(target, a, b, value, 3, cv2.LINE_AA, 4)
            mark = masks[2]
            dot = int(6.5 * 16)
            if k == 0:
                cv2.circle(mark, a, dot, value, -1, cv2.LINE_AA, 4)
                cv2.circle(mark, b, dot, value, -1, cv2.LINE_AA, 4)
            else:
                phi = angle[i, 0]
                mid = d[i] * np.array([np.cos(phi), np.sin(phi)])
                if k == 1:
                    rim = to_px((np.cos(phi), np.sin(phi)))
                    cv2.line(mark, to_px((0, 0)), rim, value // 2, 2, cv2.LINE_AA, 4)
                cv2.circle(mark, to_px(mid), dot, value, -1, cv2.LINE_AA, 4)
        return masks.astype(np.float32) / 255

    s2 = SS * SIDE

    def render(frame):
        s = DURATION * frame / (FRAME_COUNT - 1)
        image = background.copy()

        for k, cy in enumerate(CYS):
            n = int(np.searchsorted(times[k], s, side="right"))
            index, offsets = ink[k]
            y0, x0 = cy - SIDE // 2, CX - SIDE // 2
            crop = image[y0 : y0 + SIDE, x0 : x0 + SIDE]
            if n:
                hits = np.bincount(index[: offsets[n]], minlength=2 * s2 * s2)
                hits = hits.astype(np.float32).reshape(2, s2, s2)
                cover = [
                    cv2.resize(h, (SIDE, SIDE), interpolation=cv2.INTER_AREA) * STEP
                    for h in hits
                ]
                density = (
                    cover[0][..., None] * absorb[0] + cover[1][..., None] * absorb[1]
                )
                crop *= np.exp(-ink_strength(n) * density)
            crop += outline_alpha * (colours["line"] - crop)
            masks = flash(k, s)
            if masks is not None:
                for mask, name in zip(masks, ("long", "short", "line")):
                    a = mask[..., None]
                    crop += a * (colours[name] - crop)

            # Running estimate, then the exact answer.
            if n:
                shown = float(smoothstep((s - times[k][0]) / 0.15))
                top = cy + 18 - approx.shape[0]
                blit_mask(image, approx, TX, top, colours["long"], alpha=shown)
                value = f"{running[k][n - 1]:.3f}"
                blit_number(
                    image, digits, value, number_x, cy + 18, colours["long"], advance
                )
            fade = float(smoothstep((s - REVEAL - 0.25 * k) / 0.5))
            if fade > 0:
                blit_mask(image, exact[k], TX, cy + 52, text_c, alpha=fade)

        # The question, then the conclusion.
        swap = float(smoothstep((s - REVEAL) / 0.5))
        for pair, alpha in ((titles[0], 1 - swap), (titles[1], swap)):
            if alpha <= 0:
                continue
            for m, y in zip(pair, TITLE_YS):
                blit_mask(image, m, WIDTH // 2, y, text_c, alpha=alpha, anchor="centre")
        return image

    render_frames(settings, render, FRAME_COUNT)
    track = soundtrack(
        times, longs, methods, settings.audio_tag or STYLE, settings.audio_rng
    )
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def soundtrack(times, longs, methods, variant, rng):
    """
    Ambient one-shot: a sustained bed with sparse struck accents from the chords.

    Bed: pad chords I vi IV V in the variant's mode (dorian for "keys", lydian for
    "box"), ~2.9 s each, an octave and a half under the lead, with strings
    ("keys") or an "u" choir ("box") an octave above; it swells as the ink
    builds, then lands on the home chord at the reveal and rings out.
    Accents (Rhodes for "keys", bells for "box"): each of the first five chords of
    every circle, as it lands -- longer chords lower, panned by the chord's
    midpoint x; one note per circle as its count passes 10, 100 and 1000 (tones
    of the current bed chord, rising); at the reveal, three notes at 3, 2 and 4
    times one frequency, one per circle as its exact answer 1/3, 1/2, 1/4
    appears (the answers as harmonics). Echo, long room, circular=False,
    master(loop=False).
    """
    voice, mode, roots, second = VARIANTS[variant]
    root = int(rng.choice(roots))
    scale = Scale(root, mode)
    bed_octave = -2 if variant == "box" else -1
    track = audio.silence(DURATION)

    bed = audio.silence(DURATION)
    bar = REVEAL / len(PROGRESSION)
    sections = [(i * bar, bar, deg) for i, deg in enumerate(PROGRESSION)]
    sections.append((REVEAL, DURATION - REVEAL, 0))
    for i, (start, length, degree) in enumerate(sections):
        level = 0.2 + 0.1 * i / len(PROGRESSION)
        notes = scale.chord(degree, 3, octave=bed_octave)
        for midi in notes:
            hz = float(audio.midi_to_hz(midi))
            attack = 0.5 if i == 0 else 1.2
            held = audio.pad(hz, length + 0.6, level, attack, 2.5, brightness=0.3)
            audio.add(bed, held, start - (0 if i == 0 else 0.6), circular=False)
            upper = float(audio.midi_to_hz(midi + 12))
            if second == "strings":
                sound = audio.strings(upper, length + 0.6, 0.55 * level, 1.4, 2.5, 0.3)
            else:
                sound = audio.choir(upper, length + 0.6, 0.6 * level, 1.4, 2.5, "u")
            audio.add(bed, sound, start - (0 if i == 0 else 0.6), circular=False)
    track += audio.lowpass(bed, 2400.0, circular=False)

    accents = audio.silence(DURATION)
    count = 0
    for k in range(3):
        ends, d, _ = methods[k]
        for i in range(SLOW):
            length = 2 * np.sqrt(1 - d[i] ** 2)  # 0 .. 2 radii
            degree = int(round((1 - length / 2) * 7))
            x = float(ends[i, :, 0].mean())
            note = voice(scale.hz(degree), 0.42)
            audio.add(accents, audio.pan(note, 0.6 * x), times[k][i], circular=False)
            count += 1
        for j, n in enumerate(MILESTONES):
            when = times[k][n - 1]
            chord = scale.chord(PROGRESSION[min(int(when // bar), 3)], 3)
            midi = chord[k] + 12 * (j // 2)
            note = voice(float(audio.midi_to_hz(midi)), 0.34)
            audio.add(accents, audio.pan(note, 0.4 * (k - 1)), when, circular=False)
            count += 1
    base = scale.hz(0, octave=-1)
    for k, harmonic in enumerate(ANSWERS):
        when = REVEAL + 0.25 * k + 0.15
        note = voice(harmonic * base, 0.5)
        audio.add(accents, audio.pan(note, 0.4 * (k - 1)), when, circular=False)
        count += 1
    accents = audio.lowpass(accents, 6000.0, circular=False)
    accents = audio.delay(accents, 0.45, feedback=0.3, wet=0.22, circular=False)
    track += accents
    track = audio.highpass(track, 80.0, circular=False)
    track = audio.reverb(track, rng, rt60=4.5, wet=0.4, tone=4200.0, circular=False)
    logger.info(f"Soundtrack ({variant}): root {root} {mode}, {count} accents")
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
