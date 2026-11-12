"""
A 4D Cube Passing Through Our World

What would you see if a four-dimensional cube (a tesseract) drifted through
our three-dimensional world, corner first? Not the cube itself, only the slice
of it that is here right now: a point, then a growing tetrahedron, its corners
cut off, a perfect octahedron at the halfway mark, and back down to a point.

The top panel is the same trick one dimension down. A cube pushed corner first
through a flat world shows its flat inhabitants a triangle, a hexagon, a
triangle. Leave the shutter open and the slice's corners trace the cube's
edges: the familiar drawing of a cube. Below, the corners of the 3D slices
trace the shadow of the 4D cube. It meets our world in layers of 1, 4, 6, 4
and 1 corners, a row of Pascal's triangle.

Strings and a choir swell as the slice grows and settle as it shrinks, and a
bell rings every corner as it crosses into our world.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#tesseract #fourthdimension #topology #mathart #randomplots
"""

import gc
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import audio
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
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

# Timeline (s): the pass runs from 0 to PASS_END, slowing round the octahedron
# and easing into the last corner; then the long exposure rests, then fades.
PASS_END = 12.2
MID_SLOW, MID_WIDTH = 0.72, 1.15  # how much, and over how many s, the middle slows
END_EASE = 1.6  # s of easing into the last corner
START_BOOST = 0.6  # extra speed over the first second, so the slices grow at once
REVEAL = 0.9  # s for the finished drawings to brighten
FADE = 0.4  # s fade to black at the end

# Layout (output px).
SS = 2  # supersampling
SHIFT = 4  # cv2 sub-pixel bits
TOP_CENTRE, TOP_SCALE = (540.0, 452.0), 238.0  # the flat world (cube slices)
BOT_CENTRE, BOT_SCALE = (540.0, 1205.0), 345.0  # our world (tesseract slices)
CAMERA = 6.0  # perspective distance, in units of the tesseract's circumradius
TILT = 0.62  # rad, the camera looks slightly down
SPIN = 0.55  # turns of the 3D view over the clip
HEADER_Y = (122, 726)  # header text centres
NAME_Y = (186, 790)  # shape-name text centres

BACKGROUND = "#03050c"
# One colour per axis of the cube / tesseract: a face is cut from the facet
# x_i = 0 (deeper) or x_i = 1 (lighter) of that axis.
AXES = ["#2fd8c0", "#3d8bff", "#a46bff", "#ff6fa8"]
TRAIL = "#ffc874"  # the long exposure: the path of every corner
EDGE = "#f4f8ff"
HEADER_COLOUR = "#8a93b0"
NAME_COLOUR = "#f4f8ff"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
HEADERS = ("a cube passing through a flat world", "a 4D cube passing through ours")
ENDINGS = ("its corners drew a cube", "its corners drew a tesseract")

# The soundtrack: "swell" (strings, choir, bells); a tag naming one of
# common.music.STYLES swaps the struck voice for that style's.
STYLE = "swell"
ROOTS = range(50, 56)  # home key, MIDI
PROGRESSION = (0, 5, 3, 4, 0)  # a chord per corner layer of the tesseract: I vi IV V I
ROOM = (3.2, 0.34, 4200.0)  # reverb rt60, wet, tone


# --- Geometry --------------------------------------------------------------------


def hypercube(dim):
    """Vertices of [0, 1]^dim, and its edges as (vertex, axis) with vertex[axis] = 0."""
    verts = np.array(
        [[(v >> k) & 1 for k in range(dim)] for v in range(2**dim)], np.float64
    )
    edges = [(v, k) for v in range(2**dim) for k in range(dim) if not (v >> k) & 1]
    return verts, edges


def plane_basis(dim):
    """Orthonormal rows spanning the hyperplane perpendicular to (1, ..., 1)."""
    rows = []
    for k in range(1, dim):
        row = np.zeros(dim)
        row[:k] = 1.0
        row[k] = -k
        rows.append(row / np.linalg.norm(row))
    return np.array(rows)


def section(verts, edges, s):
    """
    Where the hyperplane x_1 + ... + x_n = s cuts each edge of [0, 1]^n: the
    slice's corners, each tagged with its edge. Only edges that it crosses.
    """
    points, tags = [], []
    for v, k in edges:
        level = verts[v].sum()
        if level - 1e-9 <= s <= level + 1 + 1e-9:
            p = verts[v].copy()
            p[k] = np.clip(s - level, 0.0, 1.0)
            points.append(p)
            tags.append((v, k))
    return np.array(points).reshape(-1, verts.shape[1]), tags


def unique_rows(points, tol=1e-7):
    keep = []
    for p in points:
        if all(np.abs(p - q).max() > tol for q in keep):
            keep.append(p)
    return np.array(keep).reshape(-1, points.shape[1])


def slice_faces(s, verts, edges, basis):
    """
    The 3D slice of the tesseract at level s, as polygons in hyperplane
    coordinates: one per facet x_i = b it crosses (outward normal, axis, b).
    """
    points, _ = section(verts, edges, s)
    faces = []
    if len(points) < 4:
        return faces
    centre = np.full(4, 0.5)
    for i in range(4):
        for b in (0.0, 1.0):
            on = unique_rows(points[np.abs(points[:, i] - b) < 1e-9])
            if len(on) < 3:
                continue
            p3 = (on - centre) @ basis.T
            normal = basis[:, i] * (1.0 if b else -1.0)
            normal /= np.linalg.norm(normal)
            c = p3.mean(0)
            a1 = p3[0] - c
            a1 -= normal * (a1 @ normal)
            if np.linalg.norm(a1) < 1e-9:
                continue
            a1 /= np.linalg.norm(a1)
            a2 = np.cross(normal, a1)
            order = np.argsort(np.arctan2((p3 - c) @ a2, (p3 - c) @ a1))
            faces.append((p3[order], normal, i, int(b)))
    return faces


def slice_polygon(s, verts, edges, basis):
    """The 2D slice of the cube at level s: polygon corners in plane coords, and the
    axis of the cube face each polygon side lies on."""
    points, _ = section(verts, edges, s)
    points = unique_rows(points)
    if len(points) < 3:
        return None, None
    p2 = (points - 0.5) @ basis.T
    c = p2.mean(0)
    order = np.argsort(np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]))
    points, p2 = points[order], p2[order]
    axes = []
    for j in range(len(points)):
        a, b = points[j], points[(j + 1) % len(points)]
        shared = [i for i in range(3) if abs(a[i] - b[i]) < 1e-9 and a[i] in (0, 1)]
        axes.append(shared[0] if shared else 0)
    return p2, axes


# --- Time --------------------------------------------------------------------------


def pass_curve():
    """
    u(seconds): how far through the pass (0 = first corner, 1 = last). Full speed
    from the first frame, slower round the middle (the octahedron), easing into
    the end. Returned as a fine table, for interpolation both ways.
    """
    ts = np.linspace(0.0, PASS_END, 6001)
    mid = 0.5 * PASS_END
    for _ in range(6):  # centre the slow stretch on the octahedron (u = 1/2)
        speed = 1.0 - MID_SLOW * np.exp(-(((ts - mid) / MID_WIDTH) ** 2))
        speed *= 1.0 + START_BOOST * np.exp(-((ts / 1.4) ** 2))
        speed *= 0.04 + 0.96 * smoothstep((PASS_END - ts) / END_EASE)
        steps = 0.5 * (speed[1:] + speed[:-1]) * np.diff(ts)
        u = np.concatenate([[0.0], np.cumsum(steps)])
        u /= u[-1]
        mid = float(np.interp(0.5, u, ts))
    return ts, u


def u_at(seconds, ts, us):
    return float(np.interp(seconds, ts, us))


def when_u(u, ts, us):
    return float(np.interp(u, us, ts))


# --- Text --------------------------------------------------------------------------


def text_sprite(text, size):
    """A float alpha mask of `text`, tight-cropped."""
    font = ImageFont.truetype(str(FONT), size)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("L", (right - left + 8, bottom - top + 8), 0)
    ImageDraw.Draw(image).text((4 - left, 4 - top), text, font=font, fill=255)
    return np.asarray(image, np.float32) / 255.0


def paste(out, sprite, cy, colour, alpha):
    """Composite a centred text sprite onto the float BGR frame."""
    if alpha <= 0.002:
        return
    h, w = sprite.shape
    y0, x0 = int(cy - h / 2), int((WIDTH - w) / 2)
    region = out[y0 : y0 + h, x0 : x0 + w]
    a = (alpha * sprite)[..., None]
    region *= 1.0 - a
    region += a * colour


def label_segments(levels, names, dim, ts, us):
    """(start s, end s, text) on the slice level -> (start, end) seconds."""
    return [
        (when_u(a / dim, ts, us), when_u(b / dim, ts, us), text)
        for (a, b), text in zip(levels, names)
    ]


def segment_alpha(seconds, start, end, ramp=0.14):
    return float(
        np.clip((seconds - start) / ramp, 0, 1) * np.clip((end - seconds) / ramp, 0, 1)
    )


# --- Soundtrack ----------------------------------------------------------------------


def soundtrack(layers3, layers4, style_name, rng):
    """
    Cinematic swell, a one-shot. `layers4` is (time, corners, pan) for each of
    the five corner layers of the tesseract crossing our world (1, 4, 6, 4, 1
    corners at u = 0, 1/4, 1/2, 3/4, 1); `layers3` the same for the cube's
    four layers (1, 3, 3, 1 at u = 0, 1/3, 2/3, 1).

    Each tesseract layer starts a chord of the progression I vi IV V I, held to
    the next layer, on strings, louder and brighter as the slice grows towards
    the octahedron (the middle layer) and softer as it shrinks; a sung "a" joins
    in the middle half. Each layer's corners ring as a quick roll of bells, one
    bell per corner, the roll as long as the layer is wide (1, 4, 6, 4, 1),
    panned by where the corners are. The cube's layers sound an octave higher,
    softer, three bells at most. A riser leads into the octahedron, where a
    deep bell and a low kick land. The final corner lands the home chord on
    strings and choir, which rings through the rest.
    """
    struck = audio.bell if style_name not in STYLES else STYLES[style_name].voice
    root = int(rng.choice(ROOTS))
    scale = Scale(root, "major")
    hz = lambda midi: float(audio.midi_to_hz(midi))
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    times = [when for when, _, _ in layers4]

    # Strings: one chord per layer, swelling with the slice.
    for k, degree in enumerate(PROGRESSION[:-1]):
        start, end = times[k], times[k + 1]
        size = 1.0 - abs(k + 0.5 - 2.0) / 2.0  # 0.25, 0.75, 0.75, 0.25
        velocity = 0.14 + 0.16 * size
        brightness = 0.2 + 0.5 * size
        attack = 0.5 if k == 0 else 0.35
        for note in scale.chord(degree, 3, -1):
            sound = audio.strings(
                hz(note), end - start, velocity, attack, 1.1, brightness
            )
            audio.add(track, sound, start, **kw)
        bass = audio.strings(
            hz(scale.note(degree, -1)), end - start, 0.2, attack, 1.1, 0.2
        )
        audio.add(track, bass, start, **kw)
        if k in (1, 2):
            for note in scale.chord(degree, 3, 0):
                voice = audio.choir(
                    hz(note), end - start, 0.11 + 0.05 * size, 0.6, 1.2, "a"
                )
                audio.add(track, voice, start, **kw)

    # A riser into the octahedron.
    rise_start = times[1] + 0.35 * (times[2] - times[1])
    rise = audio.riser(times[2] - rise_start, rng, 160.0, 900.0, 0.16)
    audio.add(track, audio.pan(rise, 0.0), rise_start, **kw)
    audio.add(track, audio.kick(0.45, 46.0), times[2], **kw)
    audio.add(track, audio.bell(hz(scale.note(0, -1)), 0.7, 5.0), times[2], **kw)

    # Bells: a roll per corner layer, one bell per corner.
    def roll(when, count, pans, degree0, octave, velocity, gap):
        for j in range(count):
            note = scale.hz(degree0 + 2 * j, octave)
            sound = audio.pan(struck(note, velocity), 0.7 * pans[j % len(pans)])
            audio.add(track, sound, when + j * gap, **kw)

    for k, (when, count, pans) in enumerate(layers4):
        degree = PROGRESSION[k]
        roll(when, count, pans, degree, 0, 0.42 + 0.06 * min(k, 4 - k), 0.075)
    for when, count, pans in layers3:
        roll(when + 0.02, count, pans, 4, 1, 0.22, 0.06)

    # Home: the last corner leaves, the home chord rings out.
    home = times[-1]
    hold = DURATION - home - 0.3
    for note in scale.chord(0, 3, -1) + [scale.note(0, -1)]:
        audio.add(
            track, audio.strings(hz(note), hold, 0.26, 0.4, 1.4, 0.45), home, **kw
        )
    for note in scale.chord(0, 3, 0):
        audio.add(track, audio.choir(hz(note), hold, 0.16, 0.6, 1.4, "o"), home, **kw)
    audio.add(track, audio.bell(hz(scale.note(0, 1)), 0.5, 4.0), home + 0.3, **kw)
    track = audio.reverb(track, rng, ROOM[0], ROOM[1], tone=ROOM[2], circular=False)
    return audio.master(track, loop=False)


# --- The video ---------------------------------------------------------------------------


def rotation(seconds, phase):
    """The 3D view: a slow spin about the vertical, seen from slightly above."""
    a = phase + 2 * np.pi * SPIN * seconds / DURATION
    spin = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c, s = np.cos(TILT), np.sin(TILT)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return tilt @ spin


def generate(settings: ImageProcessingSettings = None):
    """
    A tesseract [0, 1]^4 crossing the hyperplane x1 + x2 + x3 + x4 = s, with s
    sweeping 0 -> 4 (corner first), against a cube [0, 1]^3 crossing the plane
    x1 + x2 + x3 = s' (s' = 3s/4) in the panel above. A one-shot.

    Slices: the hyperplane meets each of the 32 edges (a vertex at level k plus
    an axis) at most once, at vertex + (s - k) e_axis; those points are the
    slice's corners. Its faces are the cuts of the 8 cubic facets x_i = b: the
    corners on that facet, sorted by angle round their centroid. Everything is
    drawn in an orthonormal basis of the hyperplane (so the slice keeps its true
    shape): a tetrahedron (0 < s < 1), a truncated tetrahedron (1 < s < 3), the
    octahedron at s = 2. The cube gives triangle, hexagon, triangle.

    The long exposure: each edge is drawn from its lower vertex to where the
    hyperplane is now (or whole, once passed), so the slice's corners trace the
    orthogonal projection of the whole figure along (1, ..., 1): the cube
    drawing (hexagon and spokes) on top, the tesseract's rhombic dodecahedron
    and its eight spokes below. Corner layers (1, 4, 6, 4, 1 and 1, 3, 3, 1
    vertices) flash as they are crossed.

    Time: s(t) from a speed profile integrated once (full speed at the start,
    0.72 slower round the octahedron, faster over the first second, eased into the last corner at 12.2 s);
    then the drawings brighten, rest and fade. Every frame is a pure function of
    t: cv2 at 2x on the thread pool (faces added as light, edges and trails as
    antialiased lines), a blurred half-resolution glow layer, PIL text sprites.
    The seed sets only the 3D view's starting angle. The soundtrack is computed
    from the same layer times; the posted one used --seed 0.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    phase = 0.55 + rng.uniform(-0.25, 0.25)

    v3, e3 = hypercube(3)
    v4, e4 = hypercube(4)
    b3, b4 = plane_basis(3), plane_basis(4)
    lv3, lv4 = v3.sum(1), v4.sum(1)
    p3_all = (v3 - 0.5) @ b3.T  # cube corners in the flat world
    p4_all = (v4 - 0.5) @ b4.T  # tesseract corners in our world
    ts, us = pass_curve()

    bg = np.array(hex_to_bgr(BACKGROUND), np.float32)
    axis_bgr = [np.array(hex_to_bgr(c), np.float32) for c in AXES]
    trail_bgr = np.array(hex_to_bgr(TRAIL), np.float32)
    edge_bgr = np.array(hex_to_bgr(EDGE), np.float32)
    header_bgr = np.array(hex_to_bgr(HEADER_COLOUR), np.float32)
    name_bgr = np.array(hex_to_bgr(NAME_COLOUR), np.float32)
    light = np.array([-0.45, 0.6, 0.66])
    light /= np.linalg.norm(light)

    headers = [text_sprite(h, 34) for h in HEADERS]
    endings = [text_sprite(h, 46) for h in ENDINGS]
    names3 = [
        ((0, 1.12), "triangle"),
        ((1.12, 1.88), "hexagon"),
        ((1.88, 3), "triangle"),
    ]
    names4 = [
        ((0, 1.1), "tetrahedron"),
        ((1.1, 1.86), "truncated tetrahedron"),
        ((1.86, 2.14), "octahedron"),
        ((2.14, 2.9), "truncated tetrahedron"),
        ((2.9, 4), "tetrahedron"),
    ]
    sprites = {n: text_sprite(n, 50) for _, n in names3 + names4}
    seg3 = label_segments(*zip(*names3), 3, ts, us)
    seg4 = label_segments(*zip(*names4), 4, ts, us)
    layer_t3 = [when_u(k / 3, ts, us) for k in range(4)]
    layer_t4 = [when_u(k / 4, ts, us) for k in range(5)]
    end_seg = (PASS_END + 0.9, 99.0)

    def pts(xy, scale=SS):
        return np.round(np.asarray(xy) * scale * (1 << SHIFT)).astype(np.int32)

    def project(p, rot):
        q = p @ rot.T
        f = CAMERA / (CAMERA - q[:, 2])
        cx, cy = BOT_CENTRE
        return (
            np.c_[cx + BOT_SCALE * q[:, 0] * f, cy - BOT_SCALE * q[:, 1] * f],
            q[:, 2],
        )

    def flat(p):
        cx, cy = TOP_CENTRE
        return np.c_[cx + TOP_SCALE * p[:, 0], cy - TOP_SCALE * p[:, 1]]

    def add_poly(canvas, xy, colour):
        """Additive antialiased fill of one polygon (output px) into the 2x canvas."""
        poly = pts(xy)
        x0, y0 = np.maximum(poly.min(0) >> SHIFT, 0) - 2
        x1, y1 = (poly.max(0) >> SHIFT) + 3
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        tmp = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        offset = np.array([x0, y0]) << SHIFT
        cv2.fillPoly(tmp, [poly - offset], colour.tolist(), cv2.LINE_AA, SHIFT)
        cv2.add(canvas[y0:y1, x0:x1], tmp, dst=canvas[y0:y1, x0:x1])

    def line(img, a, b, colour, width, scale):
        cv2.line(
            img,
            tuple(pts(a, scale)),
            tuple(pts(b, scale)),
            colour.tolist(),
            max(1, int(round(width * scale))),
            cv2.LINE_AA,
            SHIFT,
        )

    def dot(img, a, radius, colour, scale):
        cv2.circle(
            img,
            tuple(pts(a, scale)),
            int(round(radius * scale * (1 << SHIFT))),
            colour.tolist(),
            -1,
            cv2.LINE_AA,
            SHIFT,
        )

    def render(frame):
        sec = frame / (FRAME_COUNT - 1) * DURATION
        u = u_at(sec, ts, us)
        s3, s4 = 3 * u, 4 * u
        rest = smoothstep((sec - PASS_END) / REVEAL)
        fade = smoothstep((DURATION - sec) / FADE)
        sharp = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
        glow = np.zeros((HEIGHT // 2, WIDTH // 2, 3), np.uint8)
        g = 0.5  # glow layer scale

        # --- Top: the cube through a flat world --------------------------------
        trail_c = trail_bgr * (0.62 + 0.38 * rest)
        corners = []
        for v, k in e3:
            a = p3_all[v]
            lit = np.clip(s3 - lv3[v], 0.0, 1.0)
            if lit <= 0:
                continue
            w = v | (1 << k)
            b = a + lit * (p3_all[w] - a)
            xa, xb = flat(np.array([a, b]))
            line(sharp, xa, xb, trail_c, 2.4, SS)
            line(glow, xa, xb, trail_c * 0.55, 5.0, g)
            if 0 < lit < 1:
                corners.append(xb)
        poly, axes = slice_polygon(s3, v3, e3, b3) if 0 < s3 < 3 else (None, None)
        if poly is not None:
            xy = flat(poly)
            add_poly(sharp, xy, np.full(3, 26.0))
            for j in range(len(xy)):
                c = axis_bgr[axes[j]]
                line(sharp, xy[j], xy[(j + 1) % len(xy)], c, 4.0, SS)
                line(glow, xy[j], xy[(j + 1) % len(xy)], c * 0.8, 9.0, g)
        for xb in corners:
            dot(sharp, xb, 5.0, edge_bgr, SS)
            dot(glow, xb, 9.0, edge_bgr * 0.6, g)
        for k, when in enumerate(layer_t3):
            age = sec - when
            if age < 0:
                continue
            flash = np.exp(-age / 0.35)
            for v in np.nonzero(lv3 == k)[0]:
                x = flat(p3_all[v][None])[0]
                dot(sharp, x, 4.0 + 3.0 * rest, trail_c, SS)
                if flash > 0.02:
                    dot(glow, x, 10 + 26 * flash, edge_bgr * flash, g)

        # --- Bottom: the tesseract through our world -----------------------------
        rot = rotation(sec, phase)
        xy_all, z_all = project(p4_all, rot)
        corners = []
        for v, k in e4:
            lit = np.clip(s4 - lv4[v], 0.0, 1.0)
            if lit <= 0:
                continue
            w = v | (1 << k)
            a, b = p4_all[v], p4_all[v] + lit * (p4_all[w] - p4_all[v])
            (xa, xb), (za, zb) = project(np.array([a, b]), rot)
            depth = 0.55 + 0.45 * np.clip(0.5 + 0.5 * (za + zb), 0, 1)
            c = trail_c * depth
            line(sharp, xa, xb, c, 2.6, SS)
            line(glow, xa, xb, c * 0.5, 5.5, g)
            if 0 < lit < 1:
                corners.append((xb, zb))
        faces = slice_faces(s4, v4, e4, b4) if 0 < s4 < 4 else []
        drawn = []
        for poly, normal, i, side in faces:
            xy, _ = project(poly, rot)
            n = rot @ normal
            front = n[2] > 0
            shade = 0.3 + 0.7 * max(0.0, float(n @ light))
            tone = axis_bgr[i] * (0.55 + 0.45 * side)
            fill = tone * (0.36 * shade + 0.05) if front else tone * 0.07
            add_poly(sharp, xy, fill)
            drawn.append((front, xy, tone))
        for front, xy, tone in sorted(drawn, key=lambda d: d[0]):
            for j in range(len(xy)):
                a, b = xy[j], xy[(j + 1) % len(xy)]
                if front:
                    line(sharp, a, b, 0.55 * edge_bgr + 0.45 * tone, 3.6, SS)
                    line(glow, a, b, tone * 0.7, 9.0, g)
                else:
                    line(sharp, a, b, tone * 0.42, 2.0, SS)
        for xb, zb in corners:
            dot(sharp, xb, 5.5, edge_bgr, SS)
            dot(glow, xb, 10.0, edge_bgr * 0.6, g)
        for k, when in enumerate(layer_t4):
            age = sec - when
            if age < 0:
                continue
            flash = np.exp(-age / 0.35)
            for v in np.nonzero(lv4 == k)[0]:
                dot(sharp, xy_all[v], 4.5 + 3.0 * rest, trail_c, SS)
                if flash > 0.02:
                    dot(glow, xy_all[v], 12 + 30 * flash, edge_bgr * flash, g)

        # --- Compose ---------------------------------------------------------------
        image = cv2.resize(sharp, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        halo = cv2.GaussianBlur(glow, (0, 0), 7.0)
        halo = cv2.resize(halo, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        out = bg + image.astype(np.float32) + (1.15 + 0.5 * rest) * halo
        for panel, (segs, dim_s) in enumerate(((seg3, s3), (seg4, s4))):
            paste(out, headers[panel], HEADER_Y[panel], header_bgr, 1.0 - rest)
            for start, end, text in segs:
                alpha = segment_alpha(sec, start, min(end, PASS_END + 0.15))
                if panel == 0 and start == 0:
                    alpha = segment_alpha(sec, -1, min(end, PASS_END + 0.15))
                if panel == 1 and start == 0:
                    alpha = segment_alpha(sec, -1, end)
                paste(out, sprites[text], NAME_Y[panel], name_bgr, alpha)
            ending = segment_alpha(sec, *end_seg, ramp=0.5)
            paste(out, endings[panel], NAME_Y[panel] - 30, name_bgr, ending)
        return np.clip(bg + (out - bg) * fade, 0, 255).astype(np.uint8)

    render_frames(settings, render, FRAME_COUNT)

    def layer_pans(dim, k, when, rot=None):
        verts = v3 if dim == 3 else v4
        chosen = np.nonzero(verts.sum(1) == k)[0]
        if dim == 3:
            x = p3_all[chosen, 0] / 0.82
        else:
            x = (p4_all[chosen] @ rotation(when, phase).T)[:, 0]
        return [float(np.clip(v, -1, 1)) for v in x]

    layers3 = [
        (layer_t3[k], (1, 3, 3, 1)[k], layer_pans(3, k, layer_t3[k])) for k in range(4)
    ]
    layers4 = [
        (layer_t4[k], (1, 4, 6, 4, 1)[k], layer_pans(4, k, layer_t4[k]))
        for k in range(5)
    ]
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(layers3, layers4, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
