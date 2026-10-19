"""
Magnet Fall

Every pixel here is a magnetic pendulum: a weight on a string, let go from
rest above three magnets. Two million of them, each released from its own
spot, and each pixel takes the colour of the magnet its pendulum is closest
to right now.

At first the picture is three clean wedges: let go near a magnet and you
would bet on landing there. Then the swings tangle. Friction drains them,
each pendulum is caught by one magnet for good, and the map freezes into a
fractal. Wherever two colours meet, the third is there too, at every scale:
nudge the starting point by a hair and the pendulum ends on another magnet.
Dim pixels are still swinging; darker tones took longer to decide.

Each magnet is one note of a chord. Every time a wave of pendulums swings
over to a magnet, its note is struck: loud and low while the swings are
wild, softer and higher as they die down. A restless chord hums along with
the flipping, and when the last pendulum is caught the home chord lands.

ẍ = −kx − bẋ + Σ (mᵢ − x)/(|mᵢ − x|² + h²)^{3/2}

Made with Python & PyTorch. Sound synthesized in NumPy.
#magneticpendulum #chaostheory #fractal #mathart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.signal import find_peaks

from common import audio
from common.gpu import select_device
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import FPS, HEIGHT, WIDTH, frame_count, render_frames

DURATION = 15.0  # seconds
FRAME_COUNT = frame_count(DURATION)

# --- Physics -------------------------------------------------------------------
SPRING = 0.5  # k: the string pulls the weight back to the centre
DAMPING = 0.2  # b: friction; lower keeps the chaos alive longer and finer
HEIGHT_GAP = 0.25  # h: how far the weight swings above the magnets; lower is sharper
MAGNET_RADIUS = 1.0
MAGNETS = 3  # 4 or 5 work too, but the outer fractal mixes to mud
DT = 0.02  # simulation time step, fixed whatever the frame rate

# --- View ----------------------------------------------------------------------
HALF_WIDTH = 2.2  # the frame spans x in [-2.2, 2.2], y in proportion
SUPERSAMPLE = 2  # pendulums per pixel side, averaged down

# --- Timeline: simulation time tau as a function of video seconds ---------------
BASE_RATE = 1.5  # tau per second while the swings tangle
EXTRA_RATE = 11.0  # added rate once the map is decided, so friction finishes fast
SPEED_UP = (8.0, 12.0)  # seconds over which the extra rate eases in
SETTLE_END = 13.0  # the simulation stops here; the last 2 s rest on the fractal

# --- Look ----------------------------------------------------------------------
PALETTE = ["#e6e9ee", "#8a9bb4", "#3a4a6b"]  # moonlight: silver, steel, midnight
DECIDE_TAU = 15.0  # lock times are shaded on this scale
DECIDE_DEPTH = 0.55  # how dark the last-decided regions get
DECIDE_GAMMA = 0.55
MOVING_FLOOR = 0.58  # brightness of a pixel whose pendulum swings fast
MOVING_SPEED = 0.8  # speed at which the dimming sets in
GLOW_CORE, GLOW_HALO = 3.0, 16.0  # magnet glow radii, output pixels

# --- Sound ---------------------------------------------------------------------
STYLE = "harp"  # common.music.STYLES; an --audio-only tag naming one overrides it
SOUND_GRID = 4  # the soundtrack's own pendulums: one per 4x4 output pixels
SOUND_RATE = 60  # field samples per second, whatever RP_FPS is
CHORD = (4, 0, 2)  # scale degree of each magnet: the top one sings highest
WAVE_MIN = 1e-4  # a flip wave must move this fraction of the field to sound
WAVE_GAP = 0.25  # s, the closest two strikes of one magnet's note
FREEZE = 99.9  # percentile of lock times at which the map counts as decided


def tau(seconds: float) -> float:
    """Simulation time reached after `seconds` of video."""
    s = min(seconds, SETTLE_END)
    start, end = SPEED_UP
    x = (s - start) / (end - start)
    # integral of smoothstep: x^3 - x^4/2 on [0, 1], then linear
    ramp = 0.0 if x <= 0 else (x**3 - x**4 / 2 if x < 1 else 0.5 + (x - 1))
    return BASE_RATE * s + EXTRA_RATE * (end - start) * ramp


def magnets() -> np.ndarray:
    """(MAGNETS, 2) magnet positions on a circle, one straight up."""
    angles = np.pi / 2 + 2 * np.pi * np.arange(MAGNETS) / MAGNETS
    return MAGNET_RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def field_events(device) -> dict:
    """
    What the soundtrack hears: a coarse grid of the same pendulums on the
    same tau(s) timeline, sampled SOUND_RATE times a second. Per sample, the
    fraction of the field that swung into each magnet's region and the mean
    squared speed; per pendulum, the video second its nearest magnet last
    changed (the lock time).
    """
    mags = magnets()
    w, h = WIDTH // SOUND_GRID, HEIGHT // SOUND_GRID
    half_height = HALF_WIDTH * h / w
    xs = HALF_WIDTH * (2 * (torch.arange(w, device=device) + 0.5) / w - 1)
    ys = half_height * (1 - 2 * (torch.arange(h, device=device) + 0.5) / h)
    y, x = (a.reshape(-1) for a in torch.meshgrid(ys, xs, indexing="ij"))
    vx, vy = torch.zeros_like(x), torch.zeros_like(y)

    def nearest():
        d2 = torch.stack([(x - mx) ** 2 + (y - my) ** 2 for mx, my in mags])
        return d2.argmin(0)

    times = np.arange(int(DURATION * SOUND_RATE) + 1) / SOUND_RATE
    into = np.zeros((len(times), MAGNETS))
    speed2 = np.zeros(len(times))
    index, lock, steps = nearest(), torch.zeros_like(x), 0
    for i, seconds in enumerate(times):
        while steps < round(tau(seconds) / DT):
            ax, ay = -SPRING * x - DAMPING * vx, -SPRING * y - DAMPING * vy
            for mx, my in mags:
                dx, dy = mx - x, my - y
                inv = (dx * dx + dy * dy + HEIGHT_GAP**2) ** -1.5
                ax, ay = ax + dx * inv, ay + dy * inv
            vx, vy = vx + DT * ax, vy + DT * ay
            x, y = x + DT * vx, y + DT * vy
            steps += 1
        new = nearest()
        changed = new != index
        lock[changed] = seconds
        counts = torch.bincount(new[changed], minlength=MAGNETS)
        into[i] = counts.cpu().numpy() / len(x)
        speed2[i] = (vx * vx + vy * vy).mean().item()
        index = new
    return dict(times=times, into=into, speed2=speed2, lock=lock.cpu().numpy())


def soundtrack(events, style_name, rng):
    """
    A one-shot. The three magnets are the three notes of the home chord
    (CHORD degrees of the style's scale, the top magnet highest), panned to
    where they sit on screen.

    - Frame 0, three still wedges: the chord, softly, over a low root.
    - Flip waves: whenever the fraction of the field swinging into a magnet's
      region peaks (find_peaks on the coarse field, at least WAVE_MIN of it),
      that magnet's note is struck, louder for bigger waves (log size). The
      top third of waves sound in the middle octave, the rest an octave up,
      so the swings fade into high, quiet bells.
    - Underneath, a restless pad on the chord a step above home, its loudness
      following the smoothed flip rate.
    - The decision: once FREEZE % of the field has locked, a low root, the
      chord rolled upward an octave higher, and the home pad swelling while
      the last motion drains (up to SETTLE_END), then ringing over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    times, into = events["times"], events["into"]
    track = audio.silence(DURATION)
    n = len(track)
    kw = dict(circular=False)
    places = magnets()[:, 0] / MAGNET_RADIUS
    rolled = np.argsort(CHORD)  # magnets from the lowest note up
    freeze = float(np.percentile(events["lock"], FREEZE))

    # Frame 0: the chord, softly.
    audio.add(track, style.bass(scale.hz(0, octave=-2), 0.4), 0.0, **kw)
    for i, k in enumerate(rolled):
        note = audio.pan(style.voice(scale.hz(CHORD[k]), 0.3), 0.7 * places[k])
        audio.add(track, note, 0.05 * i, **kw)

    # Flip waves.
    waves = []
    for k in range(MAGNETS):
        peaks, info = find_peaks(
            into[:, k], height=WAVE_MIN, distance=int(WAVE_GAP * SOUND_RATE)
        )
        waves += [(times[p], k, h) for p, h in zip(peaks, info["peak_heights"])]
    sizes = np.log([h / WAVE_MIN for _, _, h in waves])
    sizes /= sizes.max()
    big = np.percentile(sizes, 67)
    struck = 0
    for (when, k, _), size in zip(waves, sizes):
        if when > freeze - 0.15:
            continue
        octave = 0 if size >= big else 1
        note = style.voice(scale.hz(CHORD[k], octave), 0.18 + 0.42 * size)
        audio.add(track, audio.pan(note, 0.7 * places[k]), when, **kw)
        struck += 1

    # The restless pad follows the flip rate.
    rate = into.sum(axis=1)
    kernel = np.hanning(int(0.8 * SOUND_RATE))
    rate = np.convolve(rate, kernel / kernel.sum(), mode="same")
    level = np.sqrt(rate / rate.max())
    envelope = np.interp(np.arange(n) / audio.SAMPLE_RATE, times, level)[:, None]
    restless = audio.silence(DURATION)
    for note in scale.chord(1, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        audio.add(restless, audio.pad(f, DURATION, 0.5, 0.3, 0.5, 0.35), 0.0, **kw)
    track += envelope * restless

    # The decision.
    hold = max(SETTLE_END - freeze, 1.0)
    bass = style.bass(scale.hz(0, octave=-2), 0.7, length=5.0)
    audio.add(track, bass, freeze, **kw)
    for i, k in enumerate(rolled):
        note = audio.pan(style.voice(scale.hz(CHORD[k], 1), 0.45), 0.7 * places[k])
        audio.add(track, note, freeze + 0.06 * i, **kw)
    for note in scale.chord(0, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        audio.add(track, audio.pad(f, hold, 0.4, hold, 3.0, 0.3), freeze, **kw)

    logger.info(
        f"Soundtrack: {style_name}, root MIDI {scale.root}, {struck} wave strikes, "
        f"decided at {freeze:.2f} s"
    )
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.8)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    A damped magnetic pendulum, one per (sub)pixel, integrated on the GPU.

    The bob moves in the plane under x'' = -k x - b x' + sum_i (m_i - x) /
    (|m_i - x|^2 + h^2)^(3/2): a linear spring to the centre, linear friction
    and inverse-square attraction to n magnets a height h below the swing
    plane. Every pendulum of the 2160x3840 grid (2x supersampled) starts at rest
    at its own position; all are stepped together with semi-implicit Euler at a
    fixed dt. Video time s maps to simulation time tau(s) (1.5 per second while
    the basins form, easing up to 12.5 per second after 8 s so friction
    finishes by 13 s), and each frame steps the grid to round(tau(s) / dt), so
    an RP_FPS preview shows the same motion.

    Colour is the magnet nearest the bob at that moment (so frame 0 is the
    Voronoi diagram of the magnets and the last frame is the basin map). It is
    darkened by the time the nearest magnet last changed (the "lock time",
    tracked per frame) and dimmed while the bob is fast. The supersampled image
    is averaged 2x2 and the magnets get a small additive glow.

    The soundtrack (see `soundtrack`) listens to a 4x coarser grid of the
    same pendulums on the same timeline, never to the rendered pixels.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    device = select_device()

    mags = magnets()
    order = rng.permutation(MAGNETS)  # the seed deals the colours to the magnets
    colours = torch.tensor(
        [[int(PALETTE[j][i : i + 2], 16) / 255.0 for i in (1, 3, 5)] for j in order],
        device=device,
    )

    width, height = WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE
    half_height = HALF_WIDTH * height / width
    xs = (torch.arange(width, device=device) + 0.5) / width
    ys = (torch.arange(height, device=device) + 0.5) / height
    y, x = torch.meshgrid(
        half_height * (1 - 2 * ys), HALF_WIDTH * (2 * xs - 1), indexing="ij"
    )
    x, y = x.contiguous(), y.contiguous()
    vx, vy = torch.zeros_like(x), torch.zeros_like(y)
    ax, ay = torch.empty_like(x), torch.empty_like(y)
    dx, dy, inv = torch.empty_like(x), torch.empty_like(y), torch.empty_like(x)

    def nearest():
        d2 = torch.stack([(x - mx) ** 2 + (y - my) ** 2 for mx, my in mags])
        return d2.argmin(0)

    def step():
        torch.mul(x, -SPRING, out=ax).add_(vx, alpha=-DAMPING)
        torch.mul(y, -SPRING, out=ay).add_(vy, alpha=-DAMPING)
        for mx, my in mags:
            torch.sub(float(mx), x, out=dx)
            torch.sub(float(my), y, out=dy)
            torch.addcmul(dx * dx, dy, dy, out=inv).add_(HEIGHT_GAP**2).rsqrt_()
            inv.mul_(inv * inv)
            ax.addcmul_(dx, inv)
            ay.addcmul_(dy, inv)
        vx.add_(ax, alpha=DT)
        vy.add_(ay, alpha=DT)
        x.add_(vx, alpha=DT)
        y.add_(vy, alpha=DT)

    # Magnet glow, in output pixels
    gx = torch.arange(WIDTH, device=device, dtype=torch.float32) + 0.5
    gy = torch.arange(HEIGHT, device=device, dtype=torch.float32) + 0.5
    glow = torch.zeros(HEIGHT, WIDTH, device=device)
    scale = WIDTH / (2 * HALF_WIDTH)
    for mx, my in mags:
        px, py = WIDTH / 2 + mx * scale, HEIGHT / 2 - my * scale
        r2 = (gx[None, :] - px) ** 2 + (gy[:, None] - py) ** 2
        glow += torch.exp(-r2 / (2 * GLOW_CORE**2)) + 0.18 * torch.exp(
            -r2 / (2 * GLOW_HALO**2)
        )
    glow = glow.clamp(max=1.0)[..., None]

    state = {"steps": 0, "index": nearest()}
    lock = torch.zeros_like(x)

    def render(frame):
        seconds = DURATION * frame / (FRAME_COUNT - 1)
        target = round(tau(seconds) / DT)
        while state["steps"] < target:
            step()
            state["steps"] += 1
        now = state["steps"] * DT

        index = nearest()
        lock.masked_fill_(index != state["index"], now)
        state["index"] = index

        decided = 1 - DECIDE_DEPTH * (lock / DECIDE_TAU).clamp(max=1) ** DECIDE_GAMMA
        speed2 = vx * vx + vy * vy
        calm = MOVING_FLOOR + (1 - MOVING_FLOOR) * torch.exp(-speed2 / MOVING_SPEED**2)
        image = colours[index] * (decided * calm)[..., None]
        image = F.avg_pool2d(image.permute(2, 0, 1)[None], SUPERSAMPLE)[0]
        image = image.permute(1, 2, 0)
        image = image + (1 - image) * glow
        return 255.0 * image

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    tag = settings.audio_tag if hasattr(settings, "audio_tag") else ""
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(field_events(device), style_name, settings.audio_rng)
    settings.save_video(fps=FPS, crf=18, audio=track)
    gc.collect()


if __name__ == "__main__":
    generate()
