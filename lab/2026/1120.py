"""
Any Four Sides Tile the Floor

Draw any quadrilateral, even a lopsided arrowhead with a dent in it, and
copies of it tile the floor with no gaps. Here one tile flips over its edges
until the screen is paved, then keeps changing shape while the tiling never
cracks.

The secret is in the coloured corners. Each flip turns a tile half a turn
about the middle of an edge, so the tile's four different angles meet around
every corner of the floor, and the angles of any quadrilateral add up to
exactly 360°. Five sides are harder: only fifteen kinds of pentagon tile the
plane, the last found in 2015.

Each ring of tiles lands on a kalimba note, a little higher every time; when
the shape settles, its four angles ring as a chord.

Made with Python, NumPy & OpenCV. Sound synthesized in NumPy.
#tessellation #geometry #mathart #generativeart #randomplots
"""

import gc
from collections import deque

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
    render_frames,
    smoothstep,
)

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)
FADE = 0.4  # s of fade to black at the end

TILE_PX = 175.0  # sqrt of the tile's area, output px
GROW_START, GROW_END = 0.45, 6.2  # s: the rings of flips
RING_SLOWDOWN = 2.2  # first ring lasts this many times the last one
LIFT = 0.07  # a flipping tile grows by this much at mid-flip (lifted off the floor)
MORPHS = ((6.4, 7.8), (8.3, 9.7), (10.2, 11.6), (12.1, 13.4))  # s: shape changes
ROTATION = 0.45  # rad the whole floor turns over the video
WEDGE = 0.34  # corner wedge radius, as a fraction of the shortest edge
WEDGE_MAX = 0.2  # ... and at most this, in units of sqrt(tile area)
ZOOM_START, ZOOM_END = 1.9, 4.5  # opening close-up, and when the pull-back ends (s)
MIN_ANGLE = np.radians(24)  # no corner sharper than this, at any time

SS = 2
SHIFT = 4
FLOOR = hex_to_bgr("#15120f")  # the bare floor and the grout
GROUT = hex_to_bgr("#1d1915")
TILE_TONES = [hex_to_bgr("#efe5cf"), hex_to_bgr("#d7c29b")]  # the two orientations
CORNERS = [
    hex_to_bgr("#c4502f"),  # terracotta
    hex_to_bgr("#e9a73a"),  # saffron
    hex_to_bgr("#23827a"),  # teal
    hex_to_bgr("#2e3f7f"),  # indigo
]
HOME_LINE = hex_to_bgr("#fff6e0")  # outline of the first tile

# common.music.STYLES; an --audio-only tag naming one overrides it
STYLE = "kalimba"
RISE = (0, 1, 2, 3, 4)  # each ring climbs this pattern, starting one degree higher


# --- Geometry --------------------------------------------------------------------


def area(q):
    """Signed area of a quadrilateral (4, 2), positive when counter-clockwise."""
    x, y = q[:, 0], q[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def centroid(q):
    """Area centroid of a simple polygon."""
    x, y = q[:, 0], q[:, 1]
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * c.sum()
    return np.array(
        [((x + np.roll(x, -1)) * c).sum(), ((y + np.roll(y, -1)) * c).sum()]
    ) / (6 * a)


def normalise(q):
    """Centroid at the origin, area 1."""
    q = q - centroid(q)
    return q / np.sqrt(abs(area(q)))


def angles(q):
    """Interior angles of a counter-clockwise quadrilateral, and each corner's start
    direction (along the next edge); the interior sweeps counter-clockwise from it."""
    nxt = np.roll(q, -1, 0) - q
    prv = np.roll(q, 1, 0) - q
    a_next = np.arctan2(nxt[:, 1], nxt[:, 0])
    a_prev = np.arctan2(prv[:, 1], prv[:, 0])
    return np.mod(a_prev - a_next, 2 * np.pi), a_next


def crosses(p1, p2, p3, p4):
    """Do the segments p1p2 and p3p4 cross?"""

    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return (
        side(p1, p2, p3) * side(p1, p2, p4) < 0
        and side(p3, p4, p1) * side(p3, p4, p2) < 0
    )


def is_good(q, min_angle):
    """Simple, counter-clockwise, no corner too sharp, no edge too short."""
    if (
        area(q) <= 0
        or crosses(q[0], q[1], q[2], q[3])
        or crosses(q[1], q[2], q[3], q[0])
    ):
        return False
    a, _ = angles(q)
    edges = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
    return (
        a.min() >= min_angle
        and a.max() <= 2 * np.pi - 2.2 * min_angle
        and edges.max() / edges.min() < 2.8
    )


def irregular(q, min_skew=np.radians(14)):
    """No two opposite sides anywhere near parallel: clearly not a parallelogram,
    a trapezoid or a rectangle."""
    e = np.roll(q, -1, 0) - q
    heading = np.arctan2(e[:, 1], e[:, 0])
    for i in (0, 1):
        skew = np.mod(heading[i] - heading[i + 2], np.pi)  # parallel: 0 or pi
        if min(skew, np.pi - skew) < min_skew:
            return False
    return True


def random_quad(rng, dent=None):
    """
    A random quadrilateral with corner i near the direction 45 + 90 i degrees.
    With `dent` = k, corner k is pushed inside the triangle of the other three,
    making a dart (an arrowhead) with its reflex angle at k.
    """
    while True:
        theta = np.pi / 4 + np.arange(4) * np.pi / 2 + rng.uniform(-0.4, 0.4, 4)
        r = rng.uniform(0.75, 1.3, 4)
        q = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
        if dent is not None:
            k = dent
            mid = (q[(k - 1) % 4] + q[(k + 1) % 4]) / 2
            q[k] = mid + rng.uniform(0.2, 0.45) * (q[(k + 2) % 4] - mid)
        q = normalise(q)
        if is_good(q, np.radians(32)) and (dent is not None or irregular(q)):
            return q


def lerp_quad(a, b, x):
    return normalise((1 - x) * a + x * b)


def keyframes(rng):
    """The shapes: a plain convex tile while the floor grows, then a convex kite-like
    one, a dart, a skewed convex one, and a dart dented at another corner to end."""
    dents = [None, None, int(rng.integers(4)), None]
    dents.append(int((dents[2] + rng.choice([1, 2, 3])) % 4))
    while True:
        shapes = [random_quad(rng, d) for d in dents]
        ok = all(
            is_good(lerp_quad(a, b, x), MIN_ANGLE)
            for a, b in zip(shapes, shapes[1:])
            for x in np.linspace(0, 1, 41)
        )
        if ok:
            return shapes


def shape_at(shapes, t):
    """The tile at time t (seconds), normalised: a keyframe, or eased between two."""
    for i, (start, end) in enumerate(MORPHS):
        if t < start:
            return shapes[i]
        if t < end:
            return lerp_quad(
                shapes[i], shapes[i + 1], smoothstep((t - start) / (end - start))
            )
    return shapes[-1]


def rotation_at(t):
    x = np.clip(t / DURATION, 0, 1)
    return ROTATION * (1 - (1 - x) ** 2)


# --- The tiling as a graph -----------------------------------------------------------


def neighbours(m, n, s):
    """The four tiles across the edges of tile (m, n, s), in edge order AB, BC, CD, DA.
    s = 0: the tile translated by m u + n v (u = C - A, v = D - B); s = 1: its half
    turn about the middle of AB, translated the same way."""
    if s == 0:
        return [(m, n, 1), (m + 1, n, 1), (m + 1, n + 1, 1), (m, n + 1, 1)]
    return [(m, n, 0), (m - 1, n, 0), (m - 1, n - 1, 0), (m, n - 1, 0)]


def build_tiles(reach, rng):
    """Breadth-first search from tile (0, 0, 0): every tile's ring (distance in flips)
    and the parent it flips out of (a random one of its neighbours a ring closer)."""
    dist = {(0, 0, 0): 0}
    queue = deque([(0, 0, 0)])
    while queue:
        tile = queue.popleft()
        for nb in neighbours(*tile):
            if max(abs(nb[0]), abs(nb[1])) <= reach and nb not in dist:
                dist[nb] = dist[tile] + 1
                queue.append(nb)
    tiles = sorted(dist, key=lambda k: (dist[k], k))
    index = {k: i for i, k in enumerate(tiles)}
    parent = np.zeros(len(tiles), np.int64)
    for i, tile in enumerate(tiles):
        if dist[tile] == 0:
            continue
        options = [
            index[nb] for nb in neighbours(*tile) if dist.get(nb) == dist[tile] - 1
        ]
        parent[i] = options[int(rng.integers(len(options)))]
    mns = np.array(tiles, np.int64)
    ring = np.array([dist[k] for k in tiles], np.int64)
    return mns[:, 0], mns[:, 1], mns[:, 2], ring, parent


def place(q, m, n, s):
    """Corners (N, 4, 2) of every tile for the normalised tile q (math axes)."""
    u, v = q[2] - q[0], q[3] - q[1]
    base = np.where(s[:, None, None] == 0, q[None], (q[0] + q[1])[None, None] - q[None])
    return base + (m[:, None] * u + n[:, None] * v)[:, None, :]


def zoom_at(t):
    """The camera starts close on the first tile and pulls back as the floor grows."""
    return 1 + (ZOOM_START - 1) * (1 - smoothstep(t / ZOOM_END))


def to_screen(points, t):
    """Math coordinates (tile area 1) -> output pixels at time t: turned, zoomed,
    y down, the first tile centred."""
    rot, size = rotation_at(t), TILE_PX * zoom_at(t)
    c, s = np.cos(rot), np.sin(rot)
    x = points[..., 0] * c - points[..., 1] * s
    y = points[..., 0] * s + points[..., 1] * c
    return np.stack([WIDTH / 2 + size * x, HEIGHT / 2 - size * y], -1)


def visible(px, margin=8.0):
    lo, hi = px.min(1), px.max(1)
    return (
        (hi[:, 0] > -margin)
        & (lo[:, 0] < WIDTH + margin)
        & (hi[:, 1] > -margin)
        & (lo[:, 1] < HEIGHT + margin)
    )


def ring_schedule(rings):
    """Start time and duration of each ring's flip (ring 0 is there from the start);
    the rings speed up, the first lasting RING_SLOWDOWN times the last."""
    w = np.linspace(RING_SLOWDOWN, 1.0, rings)
    dur = (GROW_END - GROW_START) * w / w.sum()
    start = GROW_START + np.concatenate([[0.0], np.cumsum(dur)[:-1]])
    return np.concatenate([[-1.0], start]), np.concatenate([[1e-6], dur])


# --- Sound -------------------------------------------------------------------------


def soundtrack(landings, sizes, chords, dents, style_name, rng):
    """
    Kalimba (by default). Every ring of flips lands with a plucked note: ring d plays
    degree RISE[d % 5] + d // 5 of the scale, so the growth climbs in little runs,
    louder as the rings hold more tiles, alternating left and right; a bass bar
    marks every fifth ring. When a morph settles, its four interior angles sound
    as a quick arpeggio, one note per corner colour: the larger the angle, the
    lower the note (degree 9 - 7 angle / pi), panned by the corner's x. A corner
    turning reflex (a dent appearing) strikes a low bass note. Under the morphs a
    soft pad holds a chord that moves with each new shape; the last shape's
    arpeggio is followed by the home chord, which rings out over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    for d, (when, size) in enumerate(zip(landings, sizes), start=1):
        degree = RISE[(d - 1) % len(RISE)] + (d - 1) // len(RISE)
        velocity = 0.3 + 0.35 * np.clip(np.log1p(size) / np.log1p(40), 0, 1)
        side = 0.35 if d % 2 else -0.35
        audio.add(
            track,
            audio.pan(style.voice(scale.hz(degree - 2), velocity), side),
            when,
            **kw,
        )
        if d % 5 == 1:
            audio.add(track, style.bass(scale.hz(degree, -2), 0.5), when, **kw)

    pad_degrees = (0, 3, 5, 4, 0)
    starts = [GROW_START] + [end for _, end in MORPHS]
    for i, (when, corners) in enumerate(chords):
        for j, (angle, x) in sorted(enumerate(corners), key=lambda c: -c[1][0]):
            degree = int(round(9 - 7 * angle / np.pi))
            note = audio.pan(style.voice(scale.hz(degree), 0.42), 0.8 * x)
            audio.add(track, note, when + 0.07 * j, **kw)
        if i < len(pad_degrees):
            length = (starts[i + 1] if i + 1 < len(starts) else DURATION) - starts[i]
            for k, note in enumerate(scale.chord(pad_degrees[i], 3, -1)):
                pad = audio.pad(
                    float(audio.midi_to_hz(note)),
                    length,
                    0.09 + 0.03 * i,
                    1.2,
                    1.8,
                    0.35,
                )
                audio.add(track, pad, max(starts[i] - 0.4, 0.0), **kw)
    for when in dents:
        audio.add(track, style.bass(scale.hz(0, -2), 0.6), when, **kw)

    final = MORPHS[-1][1] + 0.45
    for k, note in enumerate(scale.chord(0, 4)):
        voice = style.voice(float(audio.midi_to_hz(note)), 0.4)
        audio.add(track, audio.pan(voice, -0.3 + 0.2 * k), final + 0.05 * k, **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.6), final, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


# --- The video -----------------------------------------------------------------------


def generate(settings: ImageProcessingSettings = None):
    """
    Every quadrilateral tiles the plane. Take a simple quadrilateral ABCD (convex or
    not); the half turn about the midpoint of an edge maps it onto a neighbour
    sharing that edge, and two such half turns compose to translations by
    u = C - A and v = D - B. So the tiles are Q + m u + n v and (A + B - Q) + m u +
    n v, and the tile graph (four neighbours per tile) is fixed whatever the shape.

    Five keyframe shapes are drawn from settings.rng (corner i near 45 + 90 i
    degrees, area 1, centroid 0; two of them darts with a dent at random corners),
    rejected until every eased interpolation between consecutive keyframes stays
    simple with all angles >= MIN_ANGLE. The tile at time t is a pure function of
    t (smoothstep between keyframes in MORPHS, renormalised), as are the global
    rotation and the camera pulling back from ZOOM_START to 1 by ZOOM_END.

    Growth: a BFS from tile (0, 0, 0) gives every tile a ring and a parent a ring
    closer (random among the options). Ring d flips during its slot of
    [GROW_START, GROW_END] (slots shrink linearly by RING_SLOWDOWN): the tile is
    its parent rotated by angle +-pi * smoothstep(progress) about the midpoint of
    their shared edge (= half the sum of any corresponding corners), which lands
    exactly on the child. The number of rings is the largest ring of any tile
    visible at any time, so the floor is complete when growth ends. A flipping
    tile is drawn above the rest, scaled up by LIFT sin(pi p) about its centre and
    over a blurred offset shadow.

    Each corner of each tile carries a sector of radius WEDGE x the shortest edge
    (at most WEDGE_MAX),
    coloured by which corner (A, B, C, D) it is; at every vertex of the floor the
    four sectors close into a full disc, the angle sum made visible. Drawn with
    cv2 at 2x on the thread pool, grout lines on top, a vignette, fade at the end.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    shapes = keyframes(rng)
    reach = 26
    m, n, s, ring, parent = build_tiles(reach, rng)
    spin = rng.choice([-1.0, 1.0], len(m))
    tone_jitter = rng.uniform(-0.035, 0.035, len(m))

    # How many rings the screen needs, at any moment.
    samples = np.linspace(0, DURATION, 121)
    seen = np.zeros(len(m), bool)
    for t in samples:
        seen |= visible(to_screen(place(shape_at(shapes, t), m, n, s), t))
    rings = int(ring[seen].max())
    assert rings < ring.max() - 2, "increase reach"
    keep = ring <= rings
    old_to_new = np.cumsum(keep) - 1
    m, n, s, ring = m[keep], n[keep], s[keep], ring[keep]
    parent, spin, tone_jitter = old_to_new[parent[keep]], spin[keep], tone_jitter[keep]
    ring_start, ring_dur = ring_schedule(rings)
    logger.info(
        f"{len(m)} tiles in {rings} rings, {ring_dur[-1]:.2f}-{ring_dur[1]:.2f} s each"
    )

    fills = np.array(
        [np.array(TILE_TONES[k], np.float64) * (1 + j) for k, j in zip(s, tone_jitter)]
    )
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    r2 = ((xs - WIDTH / 2) / (0.75 * WIDTH)) ** 2 + (
        (ys - HEIGHT / 2) / (0.62 * HEIGHT)
    ) ** 2
    vignette = (1.0 - 0.32 * np.clip(r2, 0, 1.6) ** 1.3)[..., None].astype(np.float32)
    arc = np.linspace(0, 1, 18)
    scale_big = SS * (1 << SHIFT)

    def to_int(px):
        return np.round(px * scale_big).astype(np.int32)

    def render(frame):
        t = frame / (FRAME_COUNT - 1) * DURATION
        q = shape_at(shapes, t)
        a, a0 = angles(q)
        world = place(q, m, n, s)  # (N, 4, 2), math axes

        progress = smoothstep((t - ring_start[ring]) / ring_dur[ring])
        progress[ring == 0] = 1.0
        shown = progress > 0
        flipping = shown & (progress < 1)
        settled = progress >= 1

        # Flipping tiles: the parent's corners turned about the shared edge's middle.
        idx = np.nonzero(flipping)[0]
        if len(idx):
            pc = world[parent[idx]]
            mid = (pc + world[idx]) / 2
            phi = np.pi * progress[idx] * spin[idx]
            c, sn = np.cos(phi)[:, None], np.sin(phi)[:, None]
            d = pc - mid
            turned = mid + np.stack(
                [c * d[..., 0] - sn * d[..., 1], sn * d[..., 0] + c * d[..., 1]], -1
            )
            centre = turned.mean(1, keepdims=True)
            grow = 1 + LIFT * np.sin(np.pi * progress[idx])[:, None, None]
            world[idx] = centre + grow * (turned - centre)
        px = to_screen(world, t)
        shown &= visible(px)
        settled &= shown
        flipping &= shown

        edges = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
        wedge_r = min(WEDGE * edges.min(), WEDGE_MAX)
        lift = np.ones(len(m))
        lift[idx] = 1 + LIFT * np.sin(np.pi * progress[idx])

        def layer(sel, lifted=False):
            """Tiles in `sel` at 2x: fills, corner sectors, grout (and an alpha mask)."""
            big = np.zeros((HEIGHT * SS, WIDTH * SS, 3), np.uint8)
            mask = np.zeros((HEIGHT * SS, WIDTH * SS), np.uint8)
            if not lifted:
                big[:] = FLOOR
            ids = np.nonzero(sel)[0]
            if not len(ids):
                return big, mask
            polys = to_int(px[ids])
            # Corner sectors, from each tile's own corners (math axes, all tiles are
            # counter-clockwise: half turns and rotations keep the orientation).
            corners = world[ids]
            nxt = np.roll(corners, -1, 1) - corners
            start = np.arctan2(nxt[..., 1], nxt[..., 0])  # (T, 4)
            theta = start[..., None] + a[None, :, None] * arc  # (T, 4, arc)
            r = (wedge_r * lift[ids])[:, None, None]
            rim = corners[:, :, None, :] + np.stack(
                [r * np.cos(theta), r * np.sin(theta)], -1
            )
            sectors = to_int(
                to_screen(np.concatenate([corners[:, :, None, :], rim], 2), t)
            )  # (T, 4, arc + 1, 2)
            if lifted:
                # One tile at a time: several can start from the same parent, and
                # overlapping polygons in one fillPoly call cancel out (even-odd).
                for j, i in enumerate(ids):
                    colour = fills[i] + (255 - fills[i]) * 0.18 * np.sin(
                        np.pi * progress[i]
                    )
                    cv2.fillPoly(
                        big, [polys[j]], tuple(map(float, colour)), cv2.LINE_AA, SHIFT
                    )
                    cv2.fillPoly(mask, [polys[j]], 255, cv2.LINE_AA, SHIFT)
                    for k in range(4):
                        cv2.fillPoly(
                            big, [sectors[j, k]], CORNERS[k], cv2.LINE_AA, SHIFT
                        )
                    cv2.polylines(
                        big, [polys[j]], True, GROUT, 3 * SS, cv2.LINE_AA, SHIFT
                    )
                return big, mask
            for i, poly in zip(ids, polys):
                cv2.fillPoly(
                    big, [poly], tuple(map(float, fills[i])), cv2.LINE_AA, SHIFT
                )
            for k in range(4):  # settled sectors never overlap
                cv2.fillPoly(big, list(sectors[:, k]), CORNERS[k], cv2.LINE_AA, SHIFT)
            cv2.polylines(big, list(polys), True, GROUT, 3 * SS, cv2.LINE_AA, SHIFT)
            if not lifted and ids[0] == 0:
                cv2.polylines(
                    big, [polys[0]], True, HOME_LINE, 2 * SS, cv2.LINE_AA, SHIFT
                )
            return big, mask

        base, _ = layer(settled)
        image = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        if flipping.any():
            top, mask = layer(flipping, lifted=True)
            top = cv2.resize(top, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            alpha = (
                cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(
                    np.float32
                )
                / 255
            )
            shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
            for i in np.nonzero(flipping)[0]:
                h = np.sin(np.pi * progress[i])
                off = np.array([10.0, 16.0]) * h
                poly = np.round((px[i] + off) * (1 << SHIFT)).astype(np.int32)
                cv2.fillPoly(shadow, [poly], int(140 * h), cv2.LINE_AA, SHIFT)
            shadow = cv2.GaussianBlur(shadow.astype(np.float32) / 255, (0, 0), 9)
            image *= (1 - shadow)[..., None]
            image = image * (1 - alpha[..., None]) + top * alpha[..., None]
        image *= vignette
        image *= float(smoothstep((DURATION - t) / FADE))
        return image

    render_frames(settings, render, FRAME_COUNT)

    # Sound events, from the same closed forms.
    landings = ring_start[1:] + ring_dur[1:]
    sizes = [int(((ring == d)).sum()) for d in range(1, rings + 1)]
    chords = []
    for i, when in enumerate([GROW_START] + [end for _, end in MORPHS]):
        q = shape_at(shapes, when)
        a, _ = angles(q)
        x = to_screen(q, when)[:, 0]
        chords.append(
            (when if i else 0.05, list(zip(a, (x - WIDTH / 2) / (WIDTH / 2))))
        )
    grid = np.linspace(0, DURATION, 3001)
    reflex = np.array([angles(shape_at(shapes, t))[0].max() > np.pi for t in grid])
    dents = grid[1:][reflex[1:] & ~reflex[:-1]]
    logger.info(f"Dents appear at {np.round(dents, 2)} s")
    style = settings.audio_tag if settings.audio_tag in STYLES else STYLE
    track = soundtrack(landings, sizes, chords, dents, style, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
