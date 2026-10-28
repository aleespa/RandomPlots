"""
Seven Neighbours

Six thousand starlings pour into a dusk sky and fuse into one murmuration
that folds, thins and darkens like smoke. Then a falcon cuts through it.

No bird leads and no bird sees the whole flock. Each one watches only its
seven nearest neighbours, however near or far they happen to be, and does
three things: matches their heading, keeps a little room, and stays close.
Field studies of real starlings found exactly this: each bird tracks six or
seven others, whatever the distance. That one local rule is enough to turn
thousands of birds as a single body. When the falcon dives, the birds
nearest it swerve, their neighbours copy them, and a dark wave of turning
ripples across the whole flock in a fraction of a second.

The dark bands that sweep through the cloud are the birds banking together:
wings seen edge-on almost vanish, wings seen flat fill the sky.

The music is the flock steering. Every time it swings round another few
degrees a note sounds, higher when the flock is high in the sky, under the
rush of wings that swells as it turns. Low strings rise as the falcon closes
in, and the home chord lands as the birds pour down to roost.

Made with Python & PyTorch. Sound synthesized in NumPy.
#murmuration #starlings #emergence #mathart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames, smoothstep

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)

# --- The flock (metres, seconds) ---------------------------------------------------
BIRDS = 6000
NEIGHBOURS = 7  # topological: the k nearest, whatever their distance
STEPS_PER_SECOND = 120  # fixed simulation dt; steps per frame follow FPS
SPEED = 10.0  # cruising speed
ALIGN, COHERE, SEPARATE = 6.0, 2.0, 40.0  # gains of the three neighbour rules
FLOCK_PULL = (
    0.03  # 1/s^2: a weak pull toward the whole flock's centre, the visible mass
)
SPACING = 1.1  # separation acts on neighbours closer than this
SPEED_GAIN = 1.5  # relaxes |v| to SPEED
NOISE = 0.7  # random acceleration, m/s^2 per sqrt(s)
MAX_ACCEL = 30.0
BANK_TIME = 0.15  # s: smoothing of the acceleration that sets each bird's bank

# --- The roost: a soft fence round a wandering point -------------------------------
ALTITUDE = 48.0
ROOST_RADIUS, ROOST_GAIN = 35.0, 0.25  # horizontal pull, only beyond the radius
ALT_BAND, ALT_GAIN = 8.0, 0.3  # vertical pull, only outside the band
WANDER = ((16.0, 17.0), (6.0, 11.0), (14.0, 13.0))  # (amplitude, period) per axis
DIVE_START, DIVE_TIME, DIVE_DEPTH = 12.0, 1.6, -12.0  # the roost drops out of view
DIVE_SPEED, DIVE_PULL = 1.8, 6.0  # during the dive: faster, and pulled down harder

# --- The falcon --------------------------------------------------------------------
FALCON_ON, FALCON_OFF = 8.2, 12.2  # s: in the air
FALCON_HIT = 9.8  # s: about when it reaches the flock
FALCON_SPEED = 28.0
FEAR_RADIUS, FEAR_GAIN = 12.0, 45.0

# --- Opening streams ---------------------------------------------------------------
STREAMS = 4
STREAM_LENGTH, STREAM_WIDTH, STREAM_GAP = 40.0, 2.5, 22.0  # gap: head to the roost
STREAM_FAN = 1.4  # rad: the streams arrive from one side, spread over this angle

# --- Camera ------------------------------------------------------------------------
FOCAL = 2400.0  # px
CENTRE_Y = 0.48 * HEIGHT  # the view axis on screen
PITCH = 0.6  # rad: looking up at the flock
CAM_ALT = 1.5
YAW_SPAN = 0.6  # rad of slow pan over the clip
FOLLOW_SIGMA = 1.0  # s: smoothing of the flock centre the camera follows
FOLLOW_STOP = (11.6, 2.4)  # s: the camera eases to a stop over this span

# --- Look --------------------------------------------------------------------------
SKY = ["#1b1f3b", "#3d3a63", "#5a4a78", "#c27c88", "#f2b880"]  # top to bottom
SUN_COLOUR, SUN_AZIMUTH, SUN_WIDTH = "#ffd8a0", 0.25, 0.35
SUN_FALLOFF = 420.0  # px: the glow rises this far from the bottom edge
DUSK = "#3a3f66"  # the sky is multiplied toward this by the end
BIRD = "#0d0d12"
SPAN, FALCON_SPAN = 0.36, 2.2  # wingspans; the falcon larger than life
FLAP_HZ, FALCON_FLAP_HZ = 7.0, 3.0
FOG_NEAR, FOG_FAR = 80.0, 220.0  # birds fade between these distances...
FOG_MIN = 0.45  # ...down to this opacity
DEPTH_BUCKETS = 5

# --- Sound -------------------------------------------------------------------------
STYLE = (
    "bamboo_flute"  # common.music.STYLES; an --audio-only tag naming one overrides it
)
NOTE_TURN = 0.12  # rad: the flock's heading turns this much per melody note
NOTE_GAP = (0.2, 0.7)  # s: shortest and longest gap between melody notes
NOTE_SPAN = 7  # scale degrees from the bottom of the frame to the top
PAD_CHORDS = ((0.0, 0), (4.6, 3), (8.6, 4))  # (start s, degree) of the hummed chords
WHOOSH = 0.06  # loudness of the wing rush at the fastest turn
SWELL = 1.4  # s: the strings rise this long into the falcon strike


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])


def roost(seconds: float, phases: np.ndarray) -> np.ndarray:
    """The point the flock is fenced round: a slow Lissajous wander, then the dive."""
    (ax, px), (ay, py), (az, pz) = WANDER
    tau = 2 * np.pi * seconds
    dive = float(smoothstep((seconds - DIVE_START) / DIVE_TIME))
    return np.array(
        [
            ax * np.sin(tau / px + phases[0]),
            ALTITUDE
            + ay * np.sin(tau / py + phases[1]) * (1 - dive)
            + (DIVE_DEPTH - ALTITUDE) * dive,
            az * np.sin(tau / pz + phases[2]),
        ]
    )


def yaw(seconds):
    return YAW_SPAN * (np.asarray(seconds) / DURATION - 0.5)


def basis(psi: float):
    """Camera forward, right and up for yaw psi at the fixed upward PITCH."""
    c, s = np.cos(PITCH), np.sin(PITCH)
    forward = np.array([np.sin(psi) * c, s, np.cos(psi) * c])
    right = np.array([np.cos(psi), 0.0, -np.sin(psi)])
    return forward, right, np.cross(forward, right)


def simulate(rng: np.random.Generator, device: torch.device) -> dict:
    """
    Step the flock at STEPS_PER_SECOND and keep the state at every frame time:
    positions, velocities and the smoothed acceleration (for banking), plus the
    falcon's position (NaN when it is not in the air).
    """
    to = dict(device=device, dtype=torch.float32)
    phases = rng.uniform(0, 2 * np.pi, 3)
    centre = roost(0.0, phases)

    # Streams fanning in toward the roost from one side of the first view.
    forward, right, up = basis(float(yaw(0.0)))
    flat = np.array([forward[0], 0.0, forward[2]])
    flat /= np.linalg.norm(flat)
    per = np.full(STREAMS, BIRDS // STREAMS)
    per[: BIRDS % STREAMS] += 1
    pos, vel = [], []
    side = rng.choice([-1.0, 1.0])  # arriving from the left or the right
    angles = STREAM_FAN * (np.arange(STREAMS) / (STREAMS - 1) - 0.5)
    for count, angle in zip(per, angles + rng.uniform(-0.15, 0.15, STREAMS)):
        heading = side * np.cos(angle) * right + 0.6 * np.sin(angle) * up
        heading += 0.5 * rng.uniform(-1, 1) * flat
        heading /= np.linalg.norm(heading)
        head = centre - STREAM_GAP * heading
        along = rng.uniform(0, STREAM_LENGTH, count)
        p = head - along[:, None] * heading + rng.normal(0, STREAM_WIDTH, (count, 3))
        v = SPEED * heading + rng.normal(0, 0.8, (count, 3))
        pos.append(p)
        vel.append(v)
    x = torch.tensor(np.concatenate(pos), **to)
    v = torch.tensor(np.concatenate(vel), **to)
    bank = torch.zeros_like(x)

    gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31)))
    dt = 1.0 / STEPS_PER_SECOND
    per_frame = max(1, round(STEPS_PER_SECOND / FPS))
    up_t = torch.tensor([0.0, 1.0, 0.0], **to)
    falcon = falcon_v = None
    frames = dict(x=[], v=[], bank=[], falcon=[], falcon_v=[])

    def record():
        frames["x"].append(x.cpu().numpy())
        frames["v"].append(v.cpu().numpy())
        frames["bank"].append(bank.cpu().numpy())
        flying = falcon is not None
        frames["falcon"].append(falcon.cpu().numpy() if flying else np.full(3, np.nan))
        frames["falcon_v"].append(falcon_v.cpu().numpy() if flying else np.zeros(3))

    for step in range(FRAME_COUNT * per_frame):
        if step % per_frame == 0:
            record()
        seconds = step * dt

        distance = torch.cdist(x, x)
        near = distance.topk(NEIGHBOURS + 1, largest=False).indices[:, 1:]
        offset = x[:, None, :] - x[near]  # (N, k, 3), from each neighbour to the bird
        gap = offset.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        a = ALIGN * (v[near].mean(1) - v) - COHERE * offset.mean(1)
        a += SEPARATE * ((SPACING - gap).clamp_min(0) / gap * offset).mean(1)
        dive = float(smoothstep((seconds - DIVE_START) / DIVE_TIME))
        a += FLOCK_PULL * (x.mean(0) - x)
        speed = v.norm(dim=1, keepdim=True).clamp_min(1e-3)
        cruise = SPEED * (1 + (DIVE_SPEED - 1) * dive)
        a += SPEED_GAIN * (cruise - speed) * v / speed

        target = torch.tensor(roost(seconds, phases), **to)
        toward = target - x
        horizontal = toward * torch.tensor([1.0, 0.0, 1.0], **to)
        reach = horizontal.norm(dim=1, keepdim=True).clamp_min(1e-3)
        radius = ROOST_RADIUS * (1 - 0.6 * dive)
        a += ROOST_GAIN * (reach - radius).clamp_min(0) * horizontal / reach
        height = toward[:, 1:2]
        outside = height.sign() * (height.abs() - ALT_BAND * (1 - dive)).clamp_min(0)
        a += ALT_GAIN * (1 + (DIVE_PULL - 1) * dive) * outside * up_t

        if FALCON_ON <= seconds < FALCON_OFF:
            middle = x.mean(0)
            if falcon is None:
                f, r, u = (torch.tensor(b, **to) for b in basis(float(yaw(FALCON_HIT))))
                start = -0.75 * r + 0.55 * u + 0.35 * f
                lead = FALCON_SPEED * (FALCON_HIT - FALCON_ON)
                falcon = middle + lead * start / start.norm()
                falcon_v = -FALCON_SPEED * start / start.norm()
            aim = middle - falcon
            if aim.norm() > 6.0 and seconds < FALCON_HIT:  # homes in, then flies on
                falcon_v = FALCON_SPEED * aim / aim.norm()
            falcon = falcon + falcon_v * dt
            away = x - falcon
            fear = away.norm(dim=1, keepdim=True).clamp_min(0.5)
            a += FEAR_GAIN * (1 - fear / FEAR_RADIUS).clamp_min(0) * away / fear
        elif seconds >= FALCON_OFF:
            falcon = None

        a += NOISE / np.sqrt(dt) * torch.randn(x.shape, generator=gen, **to)
        size = a.norm(dim=1, keepdim=True)
        a = a * (MAX_ACCEL / size.clamp_min(MAX_ACCEL))
        v = v + a * dt
        x = x + v * dt
        bank += (a - bank) * (dt / BANK_TIME)

    return {key: np.stack(value).astype(np.float32) for key, value in frames.items()}


def cameras(flock: np.ndarray) -> list:
    """
    Per frame (position, forward, right, up). The camera stands CAM_ALT above
    the ground, looks up at PITCH, pans slowly, and follows the smoothed flock
    centre horizontally, easing to a stop for the dive.
    """
    times = np.arange(FRAME_COUNT) / FPS
    centre = gaussian_filter1d(
        flock.mean(1), FOLLOW_SIGMA * FPS, axis=0, mode="nearest"
    )
    rate = 1 - smoothstep((times - FOLLOW_STOP[0]) / FOLLOW_STOP[1])
    tau = np.concatenate([[0.0], np.cumsum(rate[:-1]) / FPS])  # follow time, slowing
    follow = np.stack(
        [np.interp(tau * FPS, np.arange(FRAME_COUNT), centre[:, i]) for i in range(3)],
        1,
    )
    back = (ALTITUDE - CAM_ALT) / np.tan(PITCH)
    out = []
    for t, c in zip(times, follow):
        forward, right, up = basis(float(yaw(t)))
        flat = np.array([forward[0], 0.0, forward[2]])
        flat /= np.linalg.norm(flat)
        position = np.array([c[0], CAM_ALT, c[2]]) - back * flat
        out.append((position, forward, right, up))
    return out


def project(points, position, forward, right, up):
    """Screen (x, y) in px and depth along the view axis of world points."""
    d = points - position
    depth = d @ forward
    safe = np.maximum(depth, 1.0)
    sx = WIDTH / 2 + FOCAL * (d @ right) / safe
    sy = CENTRE_Y - FOCAL * (d @ up) / safe
    return np.stack([sx, sy], -1), depth


def soundtrack(data: dict, views: list, style_name: str, rng: np.random.Generator):
    """
    A one-shot that follows the flock, computed from the simulated states.

    - The melody: the flock's mean heading is tracked, and a note is played
      each time it has turned another NOTE_TURN radians (so a wheeling flock
      sings faster), at least NOTE_GAP[0] and at most NOTE_GAP[1] apart.
      Higher on screen is higher in the scale (NOTE_SPAN degrees over the
      frame); a repeated pitch steps up for a left turn and down for a right
      one. Louder the faster it turns, panned with the flock.
    - The wing rush: band-passed noise whose loudness follows the turning rate.
    - A hummed choir chord underneath (PAD_CHORDS). Melody and choir both
      soften as the flock spreads out and swell as it packs tight.
    - The falcon: low strings swell for SWELL seconds into its closest pass
      by the flock centre, then a low note and a burst of rush.
    - The dive: from DIVE_START a falling run to the root, then the home
      chord as the birds pour out of the frame, ringing to the end.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    sr = audio.SAMPLE_RATE

    # Flock signals on a 100 Hz grid, the same at any FPS.
    frames = np.arange(FRAME_COUNT) / FPS
    grid = np.arange(0.0, DURATION, 0.01)
    centre = data["x"].mean(1)
    heading = data["v"].mean(1)
    screen = np.array([project(c[None], *v)[0][0] for c, v in zip(centre, views)])
    spread = data["x"].std(1).sum(1)

    def fine(series):
        return np.interp(grid, frames, series)

    h = np.stack([fine(heading[:, i]) for i in range(3)], 1)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    dot = np.clip((h[1:] * h[:-1]).sum(1), -1, 1)
    turn = np.concatenate([[0.0], np.arccos(dot)])  # rad per grid step
    side = np.concatenate([[0.0], np.cross(h[:-1], h[1:])[:, 1]])  # > 0: a left turn
    rate = gaussian_filter1d(turn / 0.01, 15)  # rad/s
    level = np.clip(rate / np.percentile(rate, 98), 0, 1)
    sx, sy = fine(screen[:, 0]), fine(screen[:, 1])
    pan = np.clip((sx - WIDTH / 2) / (WIDTH / 2), -1, 1)
    height = np.clip(1 - sy / HEIGHT, 0, 1)

    def strike(degree, velocity, when, position, hold=None):
        if style.voice is audio.flute and hold is not None:
            note = style.voice(scale.hz(degree), velocity, length=hold)
        else:
            note = style.voice(scale.hz(degree), velocity)
        audio.add(track, audio.pan(note, position), when, **kw)

    # The melody, until the dive.
    notes, last, degree, turned = 0, -10.0, None, NOTE_TURN
    for i, when in enumerate(grid):
        if when >= DIVE_START:
            break
        turned += turn[i]
        gap = when - last
        if gap < NOTE_GAP[0] or (turned < NOTE_TURN and gap < NOTE_GAP[1]):
            continue
        new = int(round(height[i] * NOTE_SPAN))
        if new == degree:
            new += 1 if side[i] > 0 else -1
        degree = new
        hold = min(NOTE_GAP[1], max(0.5, 2.5 * min(gap, 1.0)))
        strike(degree, 0.3 + 0.35 * level[i], when, 0.7 * pan[i], hold)
        notes += 1
        last, turned = when, 0.0

    # The dive: a falling run from the last note down to the root, then home.
    start = degree if degree is not None else 4
    for k, d in enumerate(np.linspace(start, 0, 5)[1:].round().astype(int)):
        when = DIVE_START + 0.3 * k
        strike(int(d), 0.45, when, 0.5 * pan[min(len(grid) - 1, int(when * 100))], 0.8)
    home = DIVE_START + 1.4
    bass = scale.hz(0, -2)
    bass = 2 * bass if bass < 65 else bass  # phone speakers stop around 60 Hz
    audio.add(track, style.bass(bass, 0.55, length=4.0), home, **kw)
    for k, d in enumerate((0, 2, 4, 7)):
        strike(d, 0.4, home + 0.09 * k, 0.3 * (k - 1.5), DURATION - home)

    # The hummed chords, then the home chord; quieter as the flock spreads.
    edges = [c[0] for c in PAD_CHORDS] + [home]
    for (begin, d), end in zip(PAD_CHORDS, edges[1:]):
        for n in scale.chord(d, 3, octave=-1):
            hz = float(audio.midi_to_hz(n))
            audio.add(
                track, audio.choir(hz, end - begin, 0.12, 1.2, 1.6, "u"), begin, **kw
            )
    for n in scale.chord(0, 3, octave=-1):
        hz = float(audio.midi_to_hz(n))
        audio.add(
            track, audio.choir(hz, DURATION - home, 0.14, 0.8, 1.0, "u"), home, **kw
        )
    seconds = np.arange(len(track)) / sr
    close = np.interp(seconds, frames, spread.min() / spread)
    track *= (0.55 + 0.45 * close)[:, None]

    # The falcon: a low swell into its closest pass.
    flying = ~np.isnan(data["falcon"][:, 0])
    hit = None
    if flying.any():
        distance = np.linalg.norm(data["falcon"] - centre, axis=1)
        hit = float(frames[np.nanargmin(np.where(flying, distance, np.nan))])
        for n in (scale.note(0, -1), scale.note(2, -1)):
            hz = float(audio.midi_to_hz(n))
            swell = audio.strings(hz, SWELL, 0.3, SWELL, 1.2, 0.3)
            audio.add(track, swell, hit - SWELL, **kw)
        audio.add(track, style.bass(scale.hz(0, -1), 0.5, length=3.0), hit, **kw)

    # The wing rush.
    band = butter(2, (300.0, 2200.0), btype="bandpass", fs=sr, output="sos")
    rush = sosfilt(band, rng.standard_normal(len(track)))
    rush /= np.abs(rush).max()
    envelope = np.interp(seconds, grid, level) ** 1.5
    if hit is not None:
        envelope += 1.5 * np.exp(-(((seconds - hit) / 0.35) ** 2))
    envelope *= 1 - smoothstep((seconds - DIVE_START - 1.0) / 1.5)
    angle = (0.5 * np.interp(seconds, grid, pan) + 1) * np.pi / 4
    sides = np.stack([np.cos(angle), np.sin(angle)], 1)
    track += (WHOOSH * envelope * rush)[:, None] * sides

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} melody notes"
        + (f", falcon at {hit:.2f} s" if hit is not None else "")
    )
    track = style.room(track, rng, circular=False)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A murmuration of BIRDS starlings as 3D boids with topological neighbours.

    Each bird looks at its NEIGHBOURS nearest flockmates (by distance, k fixed,
    as Ballerini et al. 2008 measured in starling flocks) and accelerates to
    match their mean velocity (ALIGN), toward their mean position (COHERE) and
    away from any closer than SPACING (SEPARATE), plus a weak pull toward the
    whole flock's centre (FLOCK_PULL, the dark mass every bird can see, after
    Pearce et al. 2014) that fuses sub-flocks; its speed relaxes to SPEED,
    and white noise jitters it. A soft fence keeps the flock over the roost:
    a horizontal pull only beyond ROOST_RADIUS of a slowly wandering point, a
    vertical one only outside ALT_BAND of ALTITUDE. At FALCON_ON a falcon
    appears up-left of the view, homes on the flock centre at FALCON_SPEED and
    flies on through; birds within FEAR_RADIUS are pushed away, and alignment
    carries the turn across the flock. At DIVE_START the roost point drops
    below the ground and the flock pours down out of the bottom of the frame.

    The simulation steps at a fixed dt on the GPU (cdist + topk for the
    neighbours, constant cost per step) and keeps every frame's state; frames
    are then drawn in parallel with cv2. The camera stands on the ground,
    pitched up, pans YAW_SPAN over the clip and follows the Gaussian-smoothed
    flock centre horizontally. Each bird is a wing stroke through its body:
    the wing axis is v x L, where the lift L leans with the smoothed
    acceleration (gravity plus turning), so birds banking together turn edge-on
    together and the dark bands sweep through the flock; the wings flap with a
    per-bird phase. Birds are inked on a 2x mask in depth buckets (far ones
    fainter), downsampled, and composited over a full-frame dusk gradient
    with a sun glow rising from the bottom edge (its azimuth pans with yaw),
    no horizon in view. The posted render uses --seed 0 (the default), with the
    bamboo_flute soundtrack.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()
    data = simulate(rng, device)
    logger.info(f"Simulated {BIRDS} birds over {FRAME_COUNT} frames")
    views = cameras(data["x"])
    flap_phase = rng.uniform(0, 2 * np.pi, BIRDS).astype(np.float32)
    flap_rate = rng.uniform(0.85, 1.15, BIRDS).astype(np.float32)
    dither = rng.uniform(-0.5, 0.5, (HEIGHT, WIDTH, 1)).astype(np.float32) / 255

    rows = np.arange(HEIGHT, dtype=np.float32)
    lut_y = rows / (HEIGHT - 1)
    sky = np.stack(
        [
            np.interp(lut_y, np.linspace(0, 1, len(SKY)), [_rgb(c)[i] for c in SKY])
            for i in range(3)
        ],
        1,
    ).astype(np.float32)
    glow_y = np.exp(-(HEIGHT - rows) / SUN_FALLOFF).astype(np.float32)
    sun = _rgb(SUN_COLOUR).astype(np.float32)
    bird = _rgb(BIRD).astype(np.float32)
    dusk = _rgb(DUSK).astype(np.float32)
    columns = np.arange(WIDTH, dtype=np.float32) - WIDTH / 2
    column_angle = np.arctan(columns / FOCAL)

    def strokes(ink, body, wing, lift, span, phase, view, thickness):
        """Ink V-shaped wing strokes (tip, body, tip) on the 2x mask, by depth."""
        flap = np.sin(phase)
        half = (0.5 * span * (0.75 + 0.25 * np.cos(phase)))[:, None] * wing
        dip = (0.18 * span * flap)[:, None] * lift
        points = np.stack([body - half + dip, body, body + half + dip], 1)
        screen, depth = project(points.reshape(-1, 3), *view)
        screen = screen.reshape(-1, 3, 2)
        depth = depth.reshape(-1, 3).min(1)
        keep = (depth > 2.0) & np.all(
            (screen > -50) & (screen < np.array([WIDTH + 50, HEIGHT + 50])), axis=(1, 2)
        )
        screen, depth = screen[keep], depth[keep]
        fog = np.clip((depth - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0, 1)
        bucket = np.minimum((fog * DEPTH_BUCKETS).astype(int), DEPTH_BUCKETS - 1)
        fixed = np.round(screen * 2 * 4).astype(np.int32)  # 2x canvas, 2 bits subpixel
        for b in range(DEPTH_BUCKETS - 1, -1, -1):  # far first
            chosen = fixed[bucket == b]
            if len(chosen):
                level = 1 - (1 - FOG_MIN) * (b + 0.5) / DEPTH_BUCKETS
                cv2.polylines(
                    ink,
                    list(chosen),
                    False,
                    int(255 * level),
                    thickness,
                    cv2.LINE_AA,
                    2,
                )

    def wings(velocity, acceleration):
        heading = velocity / np.linalg.norm(velocity, axis=1, keepdims=True).clip(1e-3)
        lift = acceleration + np.array([0.0, 9.8, 0.0], np.float32)
        lift -= (lift * heading).sum(1, keepdims=True) * heading
        lift /= np.linalg.norm(lift, axis=1, keepdims=True).clip(1e-3)
        return np.cross(heading, lift), lift

    def render(frame: int) -> np.ndarray:
        seconds = frame / FPS
        view = views[frame]
        psi = float(yaw(seconds))

        ink = np.zeros((2 * HEIGHT, 2 * WIDTH), np.uint8)
        wing, lift = wings(data["v"][frame], data["bank"][frame])
        phase = flap_phase + 2 * np.pi * FLAP_HZ * flap_rate * seconds
        strokes(ink, data["x"][frame], wing, lift, SPAN, phase, view, 2)
        falcon = data["falcon"][frame]
        if not np.isnan(falcon[0]):
            wing, lift = wings(data["falcon_v"][frame][None], np.zeros((1, 3)))
            phase = np.array([2 * np.pi * FALCON_FLAP_HZ * seconds])
            strokes(
                ink, falcon[None], wing, lift, np.array([FALCON_SPAN]), phase, view, 5
            )
        ink = cv2.resize(ink, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        alpha = (ink.astype(np.float32) / 255.0)[..., None]

        azimuth = psi + column_angle
        glow_x = np.exp(
            -((np.angle(np.exp(1j * (azimuth - SUN_AZIMUTH))) / SUN_WIDTH) ** 2)
        )
        fade = float(smoothstep((seconds - 9.0) / 6.0))
        dim = 1 - fade * (1 - dusk)
        image = (
            sky[:, None, :]
            + 0.35
            * glow_y[:, None, None]
            * glow_x[None, :, None].astype(np.float32)
            * sun
        )
        image = image * dim + dither
        image = image * (1 - alpha) + bird * alpha
        return (np.clip(image, 0, 1) * 255.0)[..., ::-1]

    render_frames(settings, render, FRAME_COUNT)
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(data, views, style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
