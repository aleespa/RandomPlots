"""
A Cube Through a Cube

Can a cube pass through a hole drilled in a cube of exactly the same size? It
sounds impossible, but yes: tilt the hole along the right slanted direction
and there is room for a square tunnel slightly wider than the cube itself, with
the drilled cube still in one piece.

Prince Rupert of the Rhine bet on this in the 1690s and won. Seen along the
tunnel, a cube's outline is a hexagon, and a square a little bigger than its own
faces fits inside it. A century later Pieter Nieuwland found the best tunnel:
a cube about 6% bigger, 3√2/4 ≈ 1.06 times the size, can still slip through.
Many solids share this property. In 2025 mathematicians found the first convex
shape that does not, the Noperthedron.

Listen: a slow, breathing chord opens as the tunnel is drilled, holds its
breath while the second cube slides through, and settles on the home chord the
moment it is clear, with a soft bell at each turn.

Made with Python & PyTorch. Sound synthesized in NumPy.
#princerupertscube #geometry #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    blit_mask,
    frame_count,
    latex_mask,
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds, one-shot
FRAME_COUNT = frame_count(DURATION)

# --- The geometry ------------------------------------------------------------------
# Unit cube A = [-1/2, 1/2]^3. Nieuwland's tunnel runs along d = (2, 2, -1)/3: the
# square with corners (1/4, -1/2, -1/2), (-1/2, 1/4, -1/2), (-1/4, 1/2, 1/2),
# (1/2, -1/4, 1/2) lies inside A, is perpendicular to d and has side 3√2/4.
TUNNEL = np.array(
    [
        [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],  # e1
        [1 / np.sqrt(18), 1 / np.sqrt(18), 4 / np.sqrt(18)],  # e2
        [2 / 3, 2 / 3, -1 / 3],  # d = e1 x e2
    ]
).T  # columns e1, e2, d: a rotation
NIEUWLAND = 3 * np.sqrt(2) / 4  # the widest square tunnel, 1.0607
HOLE = 1.03  # side of the drilled tunnel; cube B (side 1) passes with 1.5% clearance
REACH = 0.5 * np.abs(TUNNEL[:, 2]).sum()  # half the extent of A along d: 5/6
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) / 2

# --- Timeline (seconds) ------------------------------------------------------------
DRILL = (0.5, 3.4)  # the tunnel opens from nothing to HOLE
THROUGH = (3.7, 4.5)  # the camera looks straight down the tunnel
FLIGHT = (4.5, 6.3)  # B flies round behind A and turns to the tunnel's angle
SLIDE = (6.3, 10.4)  # B slides along the tunnel, towards the camera
B_START, B_END = -2.1, 1.75  # B's centre along d during the slide
ABOVE = 1.8  # B's height above A (camera up) at the start and the end
OBLIQUE = 50.0  # degrees the camera tilts away from d over TILT
TILT = (5.8, 11.0)
ORBIT = 75.0  # degrees the camera turns about d from then until the end
QUESTION_OUT, ANSWER_IN = (5.4, 6.1), (10.8, 11.6)
FADE = 0.4  # s: fade to the background at the end

# --- Camera ------------------------------------------------------------------------
SS = 2  # supersampling
DISTANCE = 8.0  # camera to target
FOCAL = 3000.0  # px at 1x
LIFT = 0.12  # the target sits this far below the pair's middle (room for text)
MARGIN = 8  # px around the cubes' projected bounding box that gets traced

# --- Look --------------------------------------------------------------------------
BG_TOP, BG_BOTTOM = "#121a2e", "#05070d"
CUBE_A = "#efe7d6"  # porcelain
WALL = "#d9a43a"  # the tunnel's inner walls, gold
CUBE_B = "#2bb0a0"  # teal
INK = "#1b1e28"
GHOST = 0.32  # opacity of the drilled-away edges of A
TEXT, TEXT_DIM = "#ece6da", "#9aa2b6"
LINE_PX = 1.6  # edge line half-width, px at 1x
LIGHT = np.array([-0.45, 0.75, 0.6])  # camera frame (right, up, back)
AMBIENT, DIFFUSE = 0.42, 0.62

# --- Sound -------------------------------------------------------------------------
STYLE = (
    "mallet_garden"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
AMBIENT_ROOTS = (50, 52, 53)  # MIDI roots for the posted ambient track (D3, E3, F3)
DRILL_NOTES = 8  # struck as the tunnel widens through k/8 of its size
SLIDE_STEP = 0.25  # a note each time B advances this far while it overlaps A


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def _normalise(v):
    return v / np.linalg.norm(v)


def camera_frame(back) -> np.ndarray:
    """Columns (right, up, back) for a camera looking along -back, world z up."""
    back = _normalise(np.asarray(back, float))
    up = _normalise(np.array([0.0, 0.0, 1.0]) - back[2] * back)
    return np.stack([np.cross(up, back), up, back], 1)


def eased(s, start, end):
    return float(smoothstep((s - start) / (end - start)))


def _inverse_ease(fraction, start, end):
    """When smoothstep over (start, end) reaches `fraction` (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if smoothstep(mid) < fraction else (lo, mid)
    return start + 0.5 * (lo + hi) * (end - start)


def hole_time(fraction):
    """When the tunnel reaches `fraction` of its final size."""
    return _inverse_ease(fraction, *DRILL)


def slide_time(x):
    """When B's centre reaches x along d during the slide."""
    return _inverse_ease((x - B_START) / (B_END - B_START), *SLIDE)


def _bezier(p0, p1, p2, v):
    return (1 - v) ** 2 * p0 + 2 * v * (1 - v) * p1 + v**2 * p2


def timeline():
    """
    Per-frame scene parameters, all closed forms in time: the camera frame and
    origin, the tunnel size, B's centre and orientation.
    """
    d = TUNNEL[:, 2]
    side = _normalise([-0.6, 0.9, 0.9] - np.dot([-0.6, 0.9, 0.9], d) * d)
    side2 = np.cross(d, side)
    approach = Slerp(
        [0, 1],
        Rotation.from_matrix(
            [camera_frame([1.0, -1.25, 0.95]), camera_frame(d)]  # three faces -> d
        ),
    )

    def camera_at(s):
        if s < THROUGH[0]:
            return approach([eased(s, 0.0, THROUGH[0])]).as_matrix()[0]
        tilt = np.radians(OBLIQUE) * eased(s, TILT[0], TILT[1])
        turn = np.radians(ORBIT) * eased(s, THROUGH[1] + 0.5, DURATION)
        around = np.cos(turn) * side + np.sin(turn) * side2
        return camera_frame(np.cos(tilt) * d + np.sin(tilt) * around)

    turn = Slerp(
        [0, 1],
        Rotation.concatenate([Rotation.identity(), Rotation.from_matrix(TUNNEL)]),
    )
    up_through = camera_frame(d)[:, 1]  # camera up while it looks down the tunnel
    fly = (ABOVE * up_through, 1.9 * up_through + 1.3 * B_START * d, B_START * d)

    frames = []
    for frame in range(FRAME_COUNT):
        s = frame / (FRAME_COUNT - 1) * DURATION
        camera = camera_at(s)
        if s < FLIGHT[0]:
            centre, orient = ABOVE * camera[:, 1], np.eye(3)
        elif s < SLIDE[0]:
            v = eased(s, *FLIGHT)
            centre, orient = _bezier(*fly, v), turn([v]).as_matrix()[0]
        else:
            x = B_START + (B_END - B_START) * eased(s, *SLIDE)
            centre, orient = x * d, TUNNEL
        target = 0.5 * centre - LIFT * camera[:, 1]
        frames.append(
            dict(
                s=s,
                camera=camera,
                origin=target + DISTANCE * camera[:, 2],
                hole=HOLE * eased(s, *DRILL),
                centre=centre,
                orient=orient,
            )
        )
    return frames


def _crop(p):
    """The 2x pixel box (rows, cols) that both cubes project into, plus a margin."""
    R, O = p["camera"], p["origin"]
    points = np.concatenate([CORNERS, p["centre"] + CORNERS @ p["orient"].T])
    cam = (points - O) @ R  # (right, up, back) coordinates
    depth = -cam[:, 2]
    x = WIDTH / 2 + FOCAL * cam[:, 0] / depth
    y = HEIGHT / 2 - FOCAL * cam[:, 1] / depth
    x0 = int(np.clip(np.floor(x.min()) - MARGIN, 0, WIDTH))
    x1 = int(np.clip(np.ceil(x.max()) + MARGIN, 0, WIDTH))
    y0 = int(np.clip(np.floor(y.min()) - MARGIN, 0, HEIGHT))
    y1 = int(np.clip(np.ceil(y.max()) + MARGIN, 0, HEIGHT))
    return y0, max(y1, y0 + 1), x0, max(x1, x0 + 1)


def _slabs(q, v, half):
    """Ray-slab intersection: entry/exit distances and the axes they happen on."""
    v = torch.where(v.abs() < 1e-9, torch.full_like(v, 1e-9), v)
    inv = 1.0 / v
    t1, t2 = (-half - q) * inv, (half - q) * inv
    near, far = torch.minimum(t1, t2), torch.maximum(t1, t2)
    t_near, axis_near = near.max(-1)
    t_far, axis_far = far.min(-1)
    return t_near, t_far, axis_near, axis_far, v


def _face_normal(frame, v, axis):
    """World normal facing the ray on slab face `axis` of a box with axes `frame`."""
    direction = torch.gather(v, 1, axis[:, None])[:, 0]
    return frame.T[axis] * -torch.sign(direction)[:, None]


def _edge_distance(q, half, skip):
    """Distance from a point in box coordinates to the nearest edge of its face."""
    m = (half - q.abs()).scatter(1, skip[:, None], 1e9)
    return m.min(-1).values


def generate(settings: ImageProcessingSettings = None):
    """
    Prince Rupert's cube, as a one-shot ray-traced on the GPU. Cube A is the unit
    cube at the origin; the tunnel is the infinite square prism of side h(s)
    (0 -> HOLE = 1.03 over DRILL, smoothstep) about the axis d = (2, 2, -1)/3, with
    cross-section axes e1 = (-1, 1, 0)/√2, e2 = (1, 1, 4)/√18. Nieuwland's square
    (side 3√2/4) is the cross-section of A through its centre perpendicular to d, so
    any tunnel up to that size leaves A in one piece (here 12% of its volume).

    Cube B (side 1) starts above A with A's orientation (camera up × ABOVE). The
    camera slerps from a three-face view to straight down the tunnel while it is
    drilled, holds, then B flies on a quadratic Bézier round behind A while slerping
    to the tunnel frame (e1, e2, d), so it sits framed in the hole, then slides
    along d from B_START to B_END, towards the camera and out of the front of A.
    Meanwhile the camera tilts OBLIQUE = 50° away from d over TILT and turns ORBIT
    = 75° about d until the end, so the passage is seen from the side. The look-at
    target is half of B's centre, so both stay framed. All of it is a closed form
    in the time s; nothing is random, so --seed does not change the picture.

    Rendering: only the projected bounding box of the two cubes is traced, one
    perspective ray per pixel at 2x supersampling. A minus the tunnel is solved per
    ray with slab intersections (convex CSG: A's entry face unless that point lies
    in the prism, else the prism's exit wall if it is still inside A); B is a second
    oriented slab box; the nearest hit wins. Faces are Lambert-shaded from a light
    fixed to the camera, the tunnel walls are gold and darken with depth, ink lines
    mark edges, the tunnel's rim and corners at a constant pixel width, and the
    drilled-away edges of A stay as faint ghost lines. Premultiplied colour and
    coverage are average-pooled to 1x and composited over a gradient on CUDA, with
    LaTeX text layers (mathtext masks, `blit_mask`-ed once into full-frame layers).
    render_frames(parallel=False).

    Soundtrack (`ambient`, Drone & pad), from the same timeline: pad chords vi, IV,
    a suspended V from the moment B touches A (centre at −(5/6 + 1/2) along d) until
    it is clear (+(5/6 + 1/2)), then I, each section's lowpass opening as it goes;
    a soft "u" choir throughout; bell tones at 0 s, the straight-through view,
    first contact and the release; delay and a long reverb. A STYLES tag to
    --audio-only (e.g. mallet_garden) plays the older struck Rise & resolve
    version (`soundtrack`) instead. The posted
    soundtrack uses --seed 0 (settings.audio_rng picks the root and the room).
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    focal = FOCAL * SS
    plan = timeline()
    logger.info(f"tunnel {HOLE} wide, Nieuwland limit {NIEUWLAND:.4f}")

    def tensor(a):
        return torch.tensor(np.asarray(a), device=device, dtype=torch.float32)

    tunnel = tensor(TUNNEL)
    eye = torch.eye(3, device=device)
    cube_a, wall, cube_b, ink = (tensor(_rgb(c)) for c in (CUBE_A, WALL, CUBE_B, INK))
    light_cam = _normalise(LIGHT)

    # Background: a vertical gradient with a soft vignette, RGB 0..255.
    v = np.linspace(0, 1, HEIGHT)[:, None, None]
    bg = (1 - v) * _rgb(BG_TOP) + v * _rgb(BG_BOTTOM)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    r2 = ((xx - WIDTH / 2) / WIDTH) ** 2 + ((yy - HEIGHT * 0.55) / HEIGHT) ** 2
    bg = bg * (1.0 - 0.35 * np.clip(r2 / 0.35, 0, 1))[..., None]
    background = tensor(255 * bg)

    # Text, typeset once into full-frame layers: alpha and premultiplied colour.
    def layer(lines):
        alpha = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        colour = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        for tex, size, y, rgb in lines:
            mask = latex_mask(tex, size)
            blit_mask(alpha, mask, WIDTH // 2, y, (1, 1, 1), anchor="centre")
            blit_mask(colour, mask, WIDTH // 2, y, 255 * rgb, anchor="centre")
        return tensor(alpha[..., :1]), tensor(colour)

    text, dim = _rgb(TEXT), _rgb(TEXT_DIM)
    question = layer(
        [
            (r"$\mathrm{Can\ a\ cube\ pass\ through}$", 66, 120, text),
            (r"$\mathrm{a\ hole\ in\ an\ identical\ cube?}$", 66, 210, text),
        ]
    )
    answer = layer(
        [
            (r"$\mathrm{Yes{,}\ with\ room\ to\ spare.}$", 66, 110, text),
            (r"$\mathrm{Prince\ Rupert\ was\ right{,}\ in\ 1693}$", 44, 204, dim),
            (
                r"$\mathrm{Even\ a\ cube}\ \frac{3\sqrt{2}}{4}\approx 1.06"
                r"\ \mathrm{times\ as\ big\ fits.}$",
                52,
                272,
                text,
            ),
        ]
    )

    def trace(p, rows, cols):
        """Premultiplied colour (N, 3) in [0, 1] and coverage (N,) for a 2x box."""
        R = tensor(p["camera"])
        yy, xx = torch.meshgrid(
            (HEIGHT * SS / 2 - (rows + 0.5)) / focal,
            (cols + 0.5 - WIDTH * SS / 2) / focal,
            indexing="ij",
        )
        dirs = torch.stack([xx, yy, -torch.ones_like(xx)], -1).reshape(-1, 3)
        D = (dirs / dirs.norm(dim=1, keepdim=True)) @ R.T  # world directions
        O = tensor(p["origin"])
        n = D.shape[0]
        inf = torch.full((n,), float("inf"), device=device)
        h = p["hole"]
        drilled = h > 1e-4

        # Cube A minus the tunnel.
        tnA, tfA, iaA, _, vA = _slabs(O[None], D, 0.5)
        hitA = (tnA < tfA) & (tfA > 0)
        if drilled:
            qH, vH0 = O @ tunnel[:, :2], D @ tunnel[:, :2]
            tnH, tfH, _, ibH, vH = _slabs(qH[None], vH0, h / 2)
            inside = (tnH <= tnA) & (tnA < tfH)
            outer = hitA & ~inside
            inner = hitA & inside & (tfH < tfA)
            tA = torch.where(outer, tnA, torch.where(inner, tfH, inf))
        else:
            outer, inner = hitA, torch.zeros_like(hitA)
            tA = torch.where(hitA, tnA, inf)

        # Cube B.
        RB, pB = tensor(p["orient"]), tensor(p["centre"])
        tnB, tfB, iaB, _, vB = _slabs(((O - pB) @ RB)[None], D @ RB, 0.5)
        tB = torch.where((tnB < tfB) & (tfB > 0), tnB, inf)

        t_hit = torch.minimum(tA, tB)
        hit = torch.isfinite(t_hit)
        isB = hit & (tB <= tA)
        isOuter, isInner = hit & ~isB & outer, hit & ~isB & inner
        t = torch.where(hit, t_hit, torch.zeros_like(t_hit))
        P = O[None] + t[:, None] * D
        width = LINE_PX * SS / focal  # world half-width of a line per unit depth

        normal = _face_normal(eye, vA, iaA)
        dist = _edge_distance(P, 0.5, iaA)
        base = cube_a[None].expand(n, 3)
        shade = torch.ones(n, device=device)
        if drilled:
            normal = torch.where(
                isInner[:, None], _face_normal(tunnel[:, :2], vH, ibH), normal
            )
            r = (P @ tunnel[:, :2]).abs()
            rim = (r.max(-1).values - h / 2).abs()
            dist = torch.where(isOuter, torch.minimum(dist, rim), dist)
            gap = (0.5 - P.abs()).min(-1).values  # how deep inside A
            corner = h / 2 - r.min(-1).values
            dist = torch.where(isInner, torch.minimum(gap, corner), dist)
            depth = torch.clamp(gap / 0.3, 0, 1)
            shade = torch.where(isInner, 1.0 - 0.45 * depth * depth, shade)
            base = torch.where(isInner[:, None], wall[None], base)
        normal = torch.where(isB[:, None], _face_normal(RB, vB, iaB), normal)
        dist = torch.where(isB, _edge_distance((P - pB) @ RB, 0.5, iaB), dist)
        base = torch.where(isB[:, None], cube_b[None], base)

        light = tensor(p["camera"] @ light_cam)
        lambert = torch.clamp(normal @ light, min=0)
        colour = base * ((AMBIENT + DIFFUSE * lambert) * shade)[:, None]
        edge = torch.clamp(dist / (width * t) - 0.5, 0, 1)
        edge = edge * edge * (3 - 2 * edge)
        colour = ink[None] + edge[:, None] * (colour - ink[None])
        cover = hit.float()
        colour = colour * cover[:, None]

        # Ghost edges: where A's entry or exit point was drilled away and nothing
        # nearer covers it, draw the original cube's edge faintly.
        if drilled:
            for t_face, axis in ((tnA, iaA), (tfA, None)):
                visible = hitA & (t_hit > t_face + 1e-4)
                Q = O[None] + t_face[:, None] * D
                if axis is None:
                    axis = (Q.abs() - 0.5).abs().argmin(-1)
                g = _edge_distance(Q, 0.5, axis) / (width * t_face) - 0.5
                g = GHOST * (1 - torch.clamp(g, 0, 1)) * visible.float()
                colour = colour * (1 - g[:, None]) + g[:, None] * cube_a[None]
                cover = cover + g * (1 - cover)
        return colour, cover

    rows_all = torch.arange(HEIGHT * SS, device=device, dtype=torch.float32)
    cols_all = torch.arange(WIDTH * SS, device=device, dtype=torch.float32)

    def render(frame):
        p = plan[frame]
        y0, y1, x0, x1 = _crop(p)
        colour, cover = trace(
            p, rows_all[y0 * SS : y1 * SS], cols_all[x0 * SS : x1 * SS]
        )
        rgba = torch.cat([colour, cover[:, None]], 1)
        rgba = rgba.reshape((y1 - y0) * SS, (x1 - x0) * SS, 4).permute(2, 0, 1)[None]
        rgba = F.avg_pool2d(rgba, SS)[0].permute(1, 2, 0)
        image = background.clone()
        region = image[y0:y1, x0:x1]
        region.mul_(1 - rgba[..., 3:]).add_(255 * rgba[..., :3])
        s = p["s"]
        for (alpha, colour), a in (
            (question, 1 - eased(s, *QUESTION_OUT)),
            (answer, eased(s, *ANSWER_IN)),
        ):
            if a > 0:
                image = image * (1 - a * alpha) + a * colour
        out = float(smoothstep((DURATION - s) / FADE))
        if out < 1:
            image = background + out * (image - background)
        return image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag
    track = (
        soundtrack(tag, settings.audio_rng)
        if tag in STYLES
        else ambient(settings.audio_rng)
    )
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


def ambient(rng):
    """
    Drone & pad on the timeline of the picture (the posted soundtrack). No beat:
    pad chords that breathe, a soft "u" choir, and three bell tones at the
    moments that matter. vi while the tunnel is drilled (its lowpass opens with
    the hole), IV as the camera looks through and B flies round, a suspended V
    from first contact (the filter opening as B advances), then I the instant B
    is clear, ringing to the end in a long room. One-shot: nothing wraps.
    """
    root = int(rng.choice(AMBIENT_ROOTS))
    scale = Scale(root, "major")
    track = audio.silence(DURATION)
    contact, clear = slide_time(-(REACH + 0.5)), slide_time(REACH + 0.5)
    through = THROUGH[0]
    sections = [  # (degree, start, end, velocity, lowpass from, lowpass to)
        (5, 0.0, through + 0.6, 0.30, 500.0, 1100.0),
        (3, through - 0.3, contact + 0.6, 0.30, 900.0, 1400.0),
        (4, contact - 0.2, clear + 0.4, 0.34, 1000.0, 2600.0),
        (0, clear - 0.1, DURATION, 0.40, 1800.0, 1800.0),
    ]
    for degree, start, end, velocity, cut0, cut1 in sections:
        layer = audio.silence(DURATION)
        notes = scale.chord(degree, 4 if degree == 4 else 3)
        if degree == 4:  # sus4: hold the tension until B is through
            notes[1] = scale.note(degree + 3)
        for i, midi in enumerate(notes):
            hz = float(audio.midi_to_hz(midi))
            attack = 0.5 if start == 0.0 else 1.6
            held = audio.pad(hz, end - start, velocity, attack=attack, release=2.4)
            audio.add(
                layer, audio.pan(held[:, 0], 0.5 * (i - 1)), start, circular=False
            )
        bass = audio.pad(float(audio.midi_to_hz(notes[0] - 12)), end - start, 0.3)
        audio.add(layer, bass, start, circular=False)
        # The filter opens through the section: lowpass at a few cutoffs, crossfaded.
        n = len(layer)
        t = np.clip(
            (np.arange(n) / audio.SAMPLE_RATE - start) / max(end - start, 1e-3), 0, 1
        )
        dark = audio.lowpass(layer, cut0, circular=False)
        bright = audio.lowpass(layer, cut1, circular=False)
        track += dark * (1 - t)[:, None] + bright * t[:, None]
    for midi in scale.chord(0, 3):
        voice = audio.choir(
            float(audio.midi_to_hz(midi)),
            DURATION - 0.5,
            0.16,
            attack=3.0,
            release=3.0,
            vowel="u",
        )
        audio.add(track, voice, 0.2, circular=False)
    for when, degrees, velocity in (
        (0.05, (4,), 0.22),
        (through, (0, 4), 0.32),
        (contact, (4,), 0.26),
        (clear, (0, 2, 4), 0.38),
    ):
        for j, degree in enumerate(degrees):
            tone = audio.bell(
                float(audio.midi_to_hz(scale.note(degree, 1))), velocity, length=6.0
            )
            audio.add(
                track,
                audio.pan(tone, 0.35 * (j - (len(degrees) - 1) / 2)),
                when + 0.04 * j,
                circular=False,
            )
    track = audio.highpass(track, 55.0, circular=False)
    track = audio.delay(track, 0.45, feedback=0.4, wet=0.22, circular=False)
    track = audio.reverb(track, rng, rt60=5.0, wet=0.45, tone=4000.0, circular=False)
    logger.info(
        f"Ambient soundtrack: root {root}, contact {contact:.2f} s, clear {clear:.2f} s"
    )
    return audio.master(track, loop=False)


def soundtrack(style_name, rng):
    """
    Rise & resolve on the timeline of the picture (see generate()). One-shot: nothing
    wraps (circular=False) and master(loop=False).
    """
    style = STYLES[style_name]
    root = int(rng.choice(style.roots))
    melody = Scale(root, style.mode)
    harmony = Scale(root - 12, "major")
    track = audio.silence(DURATION)

    def strike(degree, when, velocity, position=0.0, octave=0):
        note = style.voice(melody.hz(degree, octave), velocity)
        audio.add(track, audio.pan(note, position), when, circular=False)

    def bass(degree, when, velocity=0.5):
        audio.add(
            track, style.bass(melody.hz(degree, -1), velocity), when, circular=False
        )

    def chord(degree, start, end, velocity):
        for midi in harmony.chord(degree, 3):
            held = audio.strings(float(audio.midi_to_hz(midi)), end - start, velocity)
            audio.add(track, held, start, circular=False)

    # The drill: a note each time the tunnel passes k/8 of its size.
    strike(0, 0.05, 0.45, -0.3)
    bass(0, 0.05)
    for k in range(1, DRILL_NOTES + 1):
        strike(k, hole_time(k / DRILL_NOTES), 0.3 + 0.04 * k, -0.5 + k / DRILL_NOTES)
    through = THROUGH[0]  # the camera looks straight down the tunnel
    strike(0, through, 0.55, -0.2, octave=1)
    strike(4, through, 0.5, 0.2, octave=1)
    bass(0, through)

    # The flight round the back: a falling figure.
    for i, degree in enumerate((7, 5, 3)):
        when = FLIGHT[0] + (i + 0.5) * (FLIGHT[1] - FLIGHT[0]) / 3.5
        strike(degree, when, 0.4, 0.4 - 0.4 * i)

    # The slide: tension from first contact until B is clear, then home.
    contact, clear = slide_time(-(REACH + 0.5)), slide_time(REACH + 0.5)
    chord(3, 0.0, contact, 0.22)
    chord(4, contact, clear, 0.3)
    swell = audio.riser(clear - contact, rng, 180.0, 1400.0, 0.22)
    audio.add(track, audio.pan(swell, 0.0), contact, circular=False)
    bass(4, contact)
    steps = np.arange(-(REACH + 0.5) + SLIDE_STEP, REACH + 0.5 - 1e-6, SLIDE_STEP)
    for i, x in enumerate(steps):
        strike(i % 5 + 2, slide_time(x), 0.28 + 0.2 * i / len(steps), 0.6 * x / 1.4)
    chord(0, clear, DURATION - 0.2, 0.42)
    for degree, position in ((0, -0.3), (2, 0.0), (4, 0.3), (5, 0.0)):
        strike(degree, clear, 0.55, position, octave=1 if degree == 5 else 0)
    bass(0, clear, 0.65)
    logger.info(
        f"Soundtrack ({style_name}): root {root}, contact {contact:.2f} s, "
        f"clear {clear:.2f} s, {len(steps)} slide steps"
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


if __name__ == "__main__":
    generate()
