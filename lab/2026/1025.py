"""
Coral Static

A storm of random noise locks into a labyrinth of glowing veins, like coral
grown overnight or a fingerprint drawn out of static. Then the camera pulls
back and time speeds up, ten times faster by the end, so you can watch the
maze keep creeping and straightening long after it forms.

This is a continuous cousin of Conway's Game of Life, in the Lenia / SmoothLife
family. Instead of pixels that are simply on or off, every point holds a smooth
value between 0 and 1. Two neighbourhood averages around each point -- how
filled-in it already is, and how dense its surroundings are -- feed one smooth
threshold rule that decides whether it strengthens or fades. No plan and no
players, just one rule applied everywhere, every frame, until plain noise
settles into a pattern that holds.

Listen: a handpan note sounds wherever the field is changing, panned to
where it happens and higher near the top of the screen, so the notes crowd
the growth and thin out as the maze settles. Strings swell as the camera
pulls back and land on the home chord the moment it stops.

Made with Python & PyTorch. Sound synthesized in NumPy.
#smoothlife #cellularautomata #mathart #generativeart #randomplots
"""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from colors.palettes import VERDANT_DECAY
from common import audio
from common.gpu import gaussian_kernel, select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    palette_lut,
    render_frames,
    smoothstep,
)

DURATION = 15.0
FRAME_COUNT = frame_count(DURATION)
CELL = 5  # screen pixels per simulation cell, at the tightest (opening) framing
_RES_SCALE = 6.0 / CELL  # vs. the original CELL=6 pass, for kernel/seed sizes below

# The simulated field is larger than one screenful: the opening frame shows
# only its centre at native 1:1 density (CELL px/cell, matching a plain
# screen-sized grid); ZOOM_OUT_FACTOR sets how much wider the field is than
# that, so pulling back at the end reveals the rest of it at a higher
# effective resolution (fewer screen pixels per cell), not a lower one.
BASE_SIM_WIDTH, BASE_SIM_HEIGHT = WIDTH // CELL, HEIGHT // CELL
ZOOM_OUT_FACTOR = 2.5
SIM_WIDTH = round(BASE_SIM_WIDTH * ZOOM_OUT_FACTOR)
SIM_HEIGHT = round(BASE_SIM_HEIGHT * ZOOM_OUT_FACTOR)
BASE_VISIBLE_FRACTION = 1.0 / ZOOM_OUT_FACTOR  # fraction of the full field shown at t=0
ZOOM_DURATION_FRACTION = (
    0.5  # the pull-back finishes this far into the clip, then holds
)

# Two neighbourhoods, SmoothLife-style: an inner disk (fill M, "is this point
# already part of something?") and a surrounding ring (density N, "how crowded
# is the neighbourhood?"). A smooth threshold on N decides the next value, and
# the threshold itself shifts depending on M -- a lenient one for empty points
# (birth) and a stricter one for filled points (survival). This two-field
# rule organises robustly from plain random noise, unlike a single peaked
# growth bump, which needs a hand-tuned seed creature to survive at all.
# Both radii are scaled by _RES_SCALE so the maze's feature size on screen
# stays the same as the original CELL=6 pass -- only the resolution rises.
RADIUS_INNER = round(6 * _RES_SCALE)
RADIUS_OUTER = round(18 * _RES_SCALE)
EDGE_BAND = 1.0 * _RES_SCALE  # cells of antialiasing at each kernel's edge

BIRTH = (0.278, 0.365)  # N-interval that fills an empty point
SURVIVE = (0.267, 0.445)  # N-interval that keeps a filled point filled
ALPHA_N = 0.028  # softness of the birth/survive interval edges
ALPHA_M = 0.147  # softness of the empty/filled blend on M

# Lenia time units per second of video at the start. REFERENCE_FPS fixes the
# step size dt itself (a nonlinear threshold rule like this one is sensitive
# to dt, not only to total elapsed time). Time then speeds up: the clock runs
# exponentially faster, SPEED_END times the starting rate by the last frame,
# so the slow settling of the maze stays visible. The steps taken by any
# instant are a closed form (`_sim_steps`), and a frame between two whole
# steps blends them, so a low-fps preview follows exactly the full render's
# trajectory, just sampled less often.
SIM_RATE = 3.0
REFERENCE_FPS = 60
SPEED_END = 10.0

SEED_BLOB_COUNT = round(
    60 * ZOOM_OUT_FACTOR**2
)  # scaled to keep the same areal density
SEED_BLOB_RADIUS = 9.0 * _RES_SCALE  # cells

# A touch of diffusion on the field itself each step, to damp the single-cell
# flicker explicit Euler on a sharp threshold otherwise leaves behind.
FIELD_SMOOTH_SIGMA = 0.8 * _RES_SCALE  # cells

PALETTE = VERDANT_DECAY
BLOOM_SIGMA = 22.0  # screen pixels
BLOOM_FACTOR = 4
BLOOM_GAIN = 0.55

# A render-time softening of the field before it is enlarged to the screen
# (bicubic) and only then coloured: colouring the coarse grid first and
# enlarging the colours showed the cells as pixels when zoomed in.
RENDER_SMOOTH_SIGMA = 0.8  # cells

# The rule settles towards a fixed maze, ever more slowly. The accelerating
# clock keeps that settling visible, and a pull-back from the centre crop to
# the full field keeps revealing new structure while it slows.

# --- Sound -----------------------------------------------------------------------
STYLE = "handpan"  # common.music.STYLES; an --audio-only tag naming one overrides it
SOUND_TILE = 30  # cells per side of the patches the field's change is measured on
NOTE_RATE = 7.0  # notes per second while the field changes fastest
NOTE_GAP = 0.12  # s, the closest two notes
NOTE_LOW, NOTE_SPAN = -2, 7  # scale degrees from the bottom of the screen to the top
HOME = ZOOM_DURATION_FRACTION * DURATION  # the pull-back ends: the chord resolves


def _sim_steps(t: float) -> float:
    """Simulation steps taken by clip fraction t, on the accelerating clock."""
    k = np.log(SPEED_END)
    return REFERENCE_FPS * DURATION * float(np.expm1(k * t)) / k


def _visible(t: float) -> float:
    """Fraction of the field's width and height on screen at clip fraction t."""
    zoom_t = min(1.0, t / ZOOM_DURATION_FRACTION)
    return BASE_VISIBLE_FRACTION + (1.0 - BASE_VISIBLE_FRACTION) * smoothstep(zoom_t)


def _antialiased_disk(r: torch.Tensor, radius: float, band: float) -> torch.Tensor:
    """Soft (0..1) disk mask: 1 well inside `radius`, 0 well outside, smooth at the edge."""
    return ((radius - r) / band + 0.5).clamp(0.0, 1.0)


def _neighbourhood_kernels(
    radius_inner: float, radius_outer: float, band: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised (1, 1, 2*radius_outer+1, 2*radius_outer+1) inner-disk and outer-ring kernels."""
    extent = int(radius_outer) + 1
    coords = torch.arange(-extent, extent + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(yy * yy + xx * xx)

    inner = _antialiased_disk(r, radius_inner, band)
    outer = _antialiased_disk(r, radius_outer, band) - inner
    return (inner / inner.sum())[None, None], (outer / outer.sum().clamp_min(1e-8))[
        None, None
    ]


def _circular_blur(field: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of a (H, W) field with toroidal (circular) borders."""
    radius = (kernel1d.numel() - 1) // 2
    x = F.pad(field[None, None], (radius, radius, 0, 0), mode="circular")
    x = F.conv2d(x, kernel1d.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="circular")
    x = F.conv2d(x, kernel1d.view(1, 1, -1, 1))
    return x[0, 0]


def _seed_blobs(
    height: int,
    width: int,
    count: int,
    radius: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Random soft Gaussian blobs on a dark field -- the first random breath."""
    cy = torch.rand(count, generator=generator, device=device) * height
    cx = torch.rand(count, generator=generator, device=device) * width
    amplitude = 0.6 + 0.4 * torch.rand(count, generator=generator, device=device)
    spread = radius * (
        0.6 + 0.8 * torch.rand(count, generator=generator, device=device)
    )

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    field = torch.zeros(height, width, device=device)
    for i in range(count):
        d2 = (yy - cy[i]) ** 2 + (xx - cx[i]) ** 2
        bump = amplitude[i] * torch.exp(-0.5 * d2 / (spread[i] ** 2))
        field = torch.maximum(field, bump)
    return field.clamp(0.0, 1.0)


class _LeniaRenderer:
    """A Lenia field on a torus: one real-valued cell per point, no discrete states."""

    def __init__(self, seed: int, device: torch.device):
        self.device = device
        generator = torch.Generator(device=device).manual_seed(seed)

        self.inner_kernel, self.outer_kernel = _neighbourhood_kernels(
            RADIUS_INNER, RADIUS_OUTER, EDGE_BAND, device
        )
        self.pad = int(RADIUS_OUTER) + 1
        self.smooth_kernel = gaussian_kernel(FIELD_SMOOTH_SIGMA, device)
        self.render_smooth_kernel = gaussian_kernel(RENDER_SMOOTH_SIGMA, device)
        lut = palette_lut(PALETTE, size=1024)
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)
        self.field = _seed_blobs(
            SIM_HEIGHT, SIM_WIDTH, SEED_BLOB_COUNT, SEED_BLOB_RADIUS, generator, device
        )
        self.dt = SIM_RATE / REFERENCE_FPS
        self.steps, self.previous, self.shown = 0, self.field, self.field
        # Per displayed frame, how much the shown field changed since the last
        # one, pooled to SOUND_TILE patches (for the soundtrack).
        self.change = []

    def _substep(self):
        padded = F.pad(self.field[None, None], (self.pad,) * 4, mode="circular")
        fill = F.conv2d(padded, self.inner_kernel)[0, 0]
        density = F.conv2d(padded, self.outer_kernel)[0, 0]

        empty_to_filled = torch.sigmoid((fill - 0.5) * (4.0 / ALPHA_M))
        lo = BIRTH[0] + (SURVIVE[0] - BIRTH[0]) * empty_to_filled
        hi = BIRTH[1] + (SURVIVE[1] - BIRTH[1]) * empty_to_filled
        target = torch.sigmoid((density - lo) * (4.0 / ALPHA_N)) * torch.sigmoid(
            (hi - density) * (4.0 / ALPHA_N)
        )

        self.field = (self.field + self.dt * (2.0 * target - 1.0)).clamp(0.0, 1.0)
        self.field = _circular_blur(self.field, self.smooth_kernel)

    def advance(self, t: float) -> torch.Tensor:
        """Step to clip fraction t and return the field shown then, blended between whole steps."""
        target = _sim_steps(t)
        whole = int(target)
        while self.steps <= whole:  # keep field = step whole + 1, previous = step whole
            self.previous = self.field
            self._substep()
            self.steps += 1
        shown = torch.lerp(self.previous, self.field, target - whole)
        change = F.avg_pool2d((shown - self.shown).abs()[None, None], SOUND_TILE)
        self.change.append(change[0, 0].cpu().numpy())
        self.shown = shown
        return shown

    def compose(self, field: torch.Tensor, t: float) -> torch.Tensor:
        """The field, softened, pulled back continuously, coloured and bloomed, as (H, W, 3) in 0..255."""
        field = _circular_blur(field, self.render_smooth_kernel)

        # A continuous (sub-cell) zoom via grid_sample, not an integer crop:
        # rounding the crop to whole cells each frame is what made the
        # pull-back visibly stair-step instead of gliding smoothly. The pull-
        # back itself finishes by ZOOM_DURATION_FRACTION of the clip and then
        # holds, rather than taking the full 15 s.
        visible = _visible(t)
        theta = torch.tensor(
            [[visible, 0.0, 0.0], [0.0, visible, 0.0]],
            device=self.device,
            dtype=torch.float32,
        )[None]
        grid = F.affine_grid(theta, (1, 1, HEIGHT, WIDTH), align_corners=False)
        value = F.grid_sample(
            field[None, None],
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )[0, 0]
        rgb = self.lut[(value.clamp(0.0, 1.0) * 1023.0).long()]  # (HEIGHT, WIDTH, 3)
        bloom = wide_blur(rgb, BLOOM_SIGMA, BLOOM_FACTOR)
        rgb = (rgb + BLOOM_GAIN * bloom).clamp(0.0, 1.0)
        return rgb * 255.0


def _sound_events(change: np.ndarray) -> dict:
    """
    Per displayed frame, from the pooled change: its mean over the patches on
    screen, how far the pull-back has got (0..1), and every patch's screen
    position (-1..1, y down) and change (zero when off screen).
    """
    frames, rows, cols = change.shape
    visible = np.array([_visible(f / (frames - 1)) for f in range(frames)])
    visible = visible[:, None, None]
    x = ((np.arange(cols) + 0.5) / cols * 2 - 1)[None, None, :] / visible
    y = ((np.arange(rows) + 0.5) / rows * 2 - 1)[None, :, None] / visible
    on = (np.abs(x) <= 1) & (np.abs(y) <= 1)
    count = on.sum((1, 2))
    weight = change * on
    return dict(
        time=DURATION * np.arange(frames) / (frames - 1),
        activity=weight.sum((1, 2)) / count,
        reveal=(visible[:, 0, 0] - BASE_VISIBLE_FRACTION) / (1 - BASE_VISIBLE_FRACTION),
        weight=weight.reshape(frames, -1),
        x=np.broadcast_to(x, on.shape).reshape(frames, -1),
        y=np.broadcast_to(y, on.shape).reshape(frames, -1),
    )


def soundtrack(events: dict, style_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    A one-shot following the field.

    - The first breath: the root over a low root as the blobs light up.
    - Notes wherever the field is changing. The change on screen falls about a
      hundredfold over the clip, so the note rate follows its logarithm (up
      to NOTE_RATE per second, at least NOTE_GAP apart); each sounds at a patch
      drawn in proportion to its change, panned to it, pitched by its height
      on screen and louder while the change is fast. They crowd the growth,
      thin out as the maze settles and fade away over the rest.
    - Strings on an unresolved chord (degree 4) that swell with the
      pull-back, resolving onto the home chord (low root, rolled chord,
      strings) at HOME, when the pull-back ends; it rings out over the rest.
    """
    style = STYLES[style_name]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    lift = 1 if scale.root < 60 else 0  # the landing chord: an octave up when low
    track = audio.silence(DURATION)
    kw = dict(circular=False)
    time = events["time"]

    # The first breath.
    audio.add(track, style.voice(scale.hz(0), 0.45), 0.0, **kw)
    audio.add(track, style.bass(scale.hz(0, -2), 0.45, length=3.0), 0.0, **kw)

    # Notes where the field changes.
    activity = events["activity"]
    low = activity[1:].min()  # the first frame has nothing to compare with
    level = np.log(np.maximum(activity, low) / low) / np.log(activity.max() / low)
    due = np.floor(np.cumsum(NOTE_RATE * level**0.7 * np.gradient(time)))
    last, notes = 0.0, 0
    for f in np.flatnonzero(np.diff(due, prepend=0) > 0):
        if time[f] - last < NOTE_GAP:
            continue
        weight = events["weight"][f]
        k = rng.choice(len(weight), p=weight / weight.sum())
        height = (1 - events["y"][f, k]) / 2  # 0 at the bottom, 1 at the top
        degree = NOTE_LOW + int(round(height * NOTE_SPAN))
        note = style.voice(scale.hz(degree), 0.18 + 0.32 * level[f])
        audio.add(track, audio.pan(note, 0.8 * events["x"][f, k]), time[f], **kw)
        last, notes = time[f], notes + 1

    # Strings: an open chord that grows with the pull-back...
    seconds = np.arange(len(track)) / audio.SAMPLE_RATE
    grow = np.interp(seconds, time, events["reveal"])
    grow *= 1 - smoothstep((seconds - HOME + 0.6) / 1.2)
    tension = audio.silence(DURATION)
    for note in scale.chord(4, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        audio.add(
            tension, audio.strings(f, HOME + 1.0, 0.65, 0.3, 0.5, 0.35), 0.0, **kw
        )
    track += grow[:, None] * tension

    # ...that resolves home as the pull-back ends.
    audio.add(track, style.bass(scale.hz(0, -2), 0.6, length=4.0), HOME, **kw)
    for i, note in enumerate(scale.chord(0, 3, octave=lift)):
        voice = style.voice(float(audio.midi_to_hz(note)), 0.4)
        audio.add(track, audio.pan(voice, 0.5 * (i - 1)), HOME + 0.06 * i, **kw)
    for note in scale.chord(0, 3, octave=-1):
        f = float(audio.midi_to_hz(note))
        held = audio.strings(f, DURATION - HOME - 2.0, 0.4, 0.8, 2.0, 0.35)
        audio.add(track, held, HOME - 0.4, **kw)

    logger.info(f"Soundtrack: {style_name}, root MIDI {scale.root}, {notes} notes")
    track = style.room(track, rng, circular=False)
    audio.fade(track, 0.0, 0.8)
    return audio.master(track, loop=False)


def generate(settings: ImageProcessingSettings = None):
    """A SmoothLife-family continuous automaton, a relative of Conway's Game of Life.

    The simulation lives on a toroidal grid of real numbers in [0, 1], sized
    ZOOM_OUT_FACTOR times wider than one screenful (CELL screen pixels per
    cell at native density). Each step reads two convolutions of the field:
    an inner-disk average (RADIUS_INNER, "is this point already filled in?")
    and a surrounding-ring average (RADIUS_OUTER, "how dense is the
    neighbourhood?"). A smooth sigmoid blends the BIRTH and SURVIVE intervals
    by the inner average, and a second sigmoid tests whether the outer density
    falls in that interval; the result nudges the field up or down by one
    Euler step of a fixed dt (SIM_RATE/REFERENCE_FPS). The clock accelerates:
    by clip fraction t the simulation has taken `_sim_steps(t)` steps, a rate
    growing exponentially from REFERENCE_FPS steps per second to SPEED_END
    times that, and a frame between two whole steps shows their linear blend,
    so the motion stays smooth and a low-fps preview reproduces the exact
    trajectory of the full 60 fps render.

    The field starts as scattered random Gaussian blobs (the "random breath").
    `compose` softens it with a small Gaussian blur (RENDER_SMOOTH_SIGMA),
    enlarges it to the screen bicubically and only then colours it through
    the VERDANT_DECAY palette (void -> growth -> bright core -> decay), so the
    cells never show as pixels however far it is zoomed in, and blooms it
    with a wide, cheap Gaussian blur for a soft, bioluminescent glow. The rule
    settles towards a fixed maze ever more slowly, so `compose` samples it
    through a continuously widening `affine_grid`/`grid_sample`
    window (by `common.video.smoothstep` of the frame fraction, rescaled so
    the pull-back itself finishes by ZOOM_DURATION_FRACTION of the clip and
    then holds, rather than an integer crop, which would resize by a
    slightly different whole-cell amount every few frames and read as a
    stair-stepping, shaky zoom instead of a smooth one). The pull-back keeps
    revealing new structure while the maze slows, at a rising rather than
    falling effective resolution. Stepping and composition
    run sequentially on the GPU via `common.video.render_frames(parallel=False)`;
    PNG writes run on a thread pool alongside it.

    Each frame also pools the change of the shown field onto SOUND_TILE-cell
    patches; `soundtrack` turns the patches on screen into notes and a string
    swell. `--audio-only` draws no frames, so it re-runs the steps alone.
    """
    settings = settings or ImageProcessingSettings(1)
    device = select_device()
    renderer = _LeniaRenderer(int(settings.rng.integers(0, 2**31)), device)

    def render(frame: int) -> torch.Tensor:
        t = frame / (FRAME_COUNT - 1)
        return renderer.compose(renderer.advance(t), t)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True)
    if settings.audio_only:  # no frames were drawn: take the same steps for the sound
        for frame in range(FRAME_COUNT):
            renderer.advance(frame / (FRAME_COUNT - 1))
    events = _sound_events(np.array(renderer.change))

    del renderer
    gc.collect()
    tag = settings.audio_tag
    style_name = tag if tag in STYLES else STYLE
    track = soundtrack(events, style_name, settings.audio_rng)
    settings.save_video(FPS, crf=18, audio=track)


if __name__ == "__main__":
    generate()
