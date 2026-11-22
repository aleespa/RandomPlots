"""
Euclid's Orchard

Stand in an orchard where a tree grows at every whole-number point (a, b),
and look around. Most trees are hidden behind others: the tree at (2, 4)
stands exactly behind the one at (1, 2). You see a tree only when a and b
share no common factor. Here the light of your gaze spreads out tree by tree:
gold trees are in sight, and the hidden ones are coloured by the smallest
prime they share: 2 teal, 3 blue, 5 violet, 7 or more rose.

How much of the orchard can you see? Count further and further out, and the
answer settles on 60.79…%, which is 6/π². The same number is the chance that
two random whole numbers have no common factor, and nobody drew a circle to
get it: π comes out of the primes, through 1 + 1/4 + 1/9 + 1/16 + … = π²/6.

Listen: a music-box bell rings as each new line of hidden trees appears, and
another as each digit of 6/π² locks into place.

Made with Python & PyTorch. Sound synthesized in NumPy.
#euclidsorchard #numbertheory #mathart #generativeart #randomplots
"""

import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import audio
from common.gpu import blur, gaussian_kernel, select_device, wide_blur
from common.image_processing import ImageProcessingSettings
from common.music import STYLES, Scale
from common.video import (
    FPS,
    HEIGHT,
    WIDTH,
    frame_count,
    latex_mask,
    render_frames,
    smoothstep,
)

# Opening line over the first 3 s, drawn by render_frames.
HOOK = 'Stand in an orchard. How many trees can you see?'
HOOK_Y = 240
HOOK_COLOUR = "#f4f0e7"

DURATION = 15.0  # seconds, one-shot
FRAME_COUNT = frame_count(DURATION)
GROW_END = 12.2  # s: the gaze stops spreading here, then the picture rests
R_START, R_END = 1.45, 100.0  # radius of the lit disc, in trees
SCREEN_START, SCREEN_END = 250.0, 1250.0  # the same radius on screen, px
M = 340  # the table of trees covers |a|, |b| <= M
SS = 2  # supersampling
FADE = 0.4  # s of fade to black at the end

BACKGROUND = "#06080f"
GOLD = "#ffc45a"  # trees in sight
SHARED = {2: "#1f9a93", 3: "#2f6fc0", 5: "#7652c4", 7: "#b8508f"}  # hidden: by prime
UNLIT = "#232a3d"
HIDDEN_LEVEL = 0.85  # brightness of the hidden trees
TEXT = "#efe9dc"

# Bells: when the first hidden tree on each low line of sight goes dark.
SPOKES = 40  # lines of sight (a, b) with 0 <= b <= a <= SPOKES
GAP = 0.42  # s between bells, at least
LAST_SPOKE = 10.6  # s
MELODY = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8)  # scale degrees


def rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)], np.float32)


def front(sec):
    """Radius of the lit disc in trees, and the pixels per tree, at time `sec`."""
    tau = np.clip(np.asarray(sec, dtype=float) / GROW_END, 0.0, 1.0)
    u = 1.0 - (1.0 - tau) ** 1.7  # ease-out in log-radius
    radius = R_START * (R_END / R_START) ** u
    screen = SCREEN_START + (SCREEN_END - SCREEN_START) * smoothstep(tau)
    return radius, screen / radius


def front_time(radius):
    """Inverse of `front`: the second at which the lit disc reaches `radius`."""
    u = np.log(np.asarray(radius) / R_START) / np.log(R_END / R_START)
    tau = 1.0 - (1.0 - np.clip(u, 0.0, 1.0)) ** (1.0 / 1.7)
    return tau * GROW_END


def orchard():
    """Every tree within M: coordinates, gcd and distance, as flat (2M+1)^2 arrays."""
    a, b = np.meshgrid(np.arange(-M, M + 1), np.arange(-M, M + 1), indexing="ij")
    g = np.gcd(a, b)
    return a, b, g, np.hypot(a, b)


def counts():
    """Sorted distances of all trees but the viewer's, and running counts of visible ones."""
    _, _, g, norm = orchard()
    keep = (norm > 0) & (norm <= M)
    order = np.argsort(norm[keep], kind="stable")
    norms = norm[keep][order]
    visible = np.cumsum(g[keep][order] == 1)
    return norms, visible


def in_sight(norms, visible, radius):
    """Fraction of the trees within `radius` that are in sight."""
    n = int(np.searchsorted(norms, radius, side="right"))
    return visible[n - 1] / n if n else 1.0


def smallest_prime(g):
    """Smallest prime factor of each entry of g >= 2 (g itself for primes)."""
    out = np.where(g >= 2, g, 0)
    for p in range(2, int(np.sqrt(g.max())) + 1):
        hit = (out == g) & (g % p == 0) & (g > p)
        out[hit] = p
    return out


class Renderer:
    def __init__(self):
        self.device = device = select_device()
        a, b, g, norm = orchard()
        prime = smallest_prime(g)
        colour = np.zeros(a.shape + (3,), np.float32)
        colour[g == 1] = rgb(GOLD)
        for p, hexa in SHARED.items():
            mask = (g >= 2) & ((prime == p) | ((p == 7) & (prime >= 7)))
            colour[mask] = HIDDEN_LEVEL * rgb(hexa)
        colour[g == 0] = 255.0  # the viewer
        radius = np.where(g == 1, 0.25, 0.2).astype(np.float32)
        radius[g == 0] = 0.3
        tensor = lambda x: torch.tensor(x, device=device).reshape(-1, *x.shape[2:])
        self.colour = tensor(colour)
        self.radius = tensor(radius)
        self.norm = tensor(norm.astype(np.float32))
        self.visible = tensor((g == 1).astype(np.float32))
        self.unlit = torch.tensor(rgb(UNLIT), device=device)
        self.background = torch.tensor(rgb(BACKGROUND), device=device)
        self.gold = torch.tensor(rgb(GOLD), device=device)
        self.cool = torch.tensor(rgb(SHARED[2]), device=device)
        self.white = torch.tensor(rgb(TEXT), device=device)

        # Pixel centres of the supersampled frame, relative to the viewer.
        ys = (torch.arange(HEIGHT * SS, device=device) + 0.5) / SS - HEIGHT / 2
        xs = (torch.arange(WIDTH * SS, device=device) + 0.5) / SS - WIDTH / 2
        self.y, self.x = torch.meshgrid(-ys, xs, indexing="ij")
        self.rho = torch.hypot(self.x, self.y)[::SS, ::SS]
        self.kernel = gaussian_kernel(2.5, device)

        # Lines of sight: every visible tree (primitive vector), sorted by distance.
        prim = (g == 1) & (norm <= 60)
        order = np.argsort(norm[prim])
        self.sight = np.stack([a[prim][order], b[prim][order]], 1).astype(np.float32)
        self.sight_norm = norm[prim][order]

        self.norms, self.cum = counts()
        self.text_setup()

        # Soft darkening behind the text bands, top and bottom.
        yy = np.arange(HEIGHT, dtype=np.float32)
        band = 0.8 * np.exp(-(((yy - 150) / 220) ** 2)) + 0.8 * np.exp(
            -(((yy - 1735) / 230) ** 2)
        )
        band = torch.tensor(1 - np.clip(band, 0, 0.88), device=device)
        self.band = band[:, None, None]

    def text_setup(self):
        def mask(tex, size):
            return torch.tensor(latex_mask(tex, size), device=self.device)

        self.label = mask(r"$\mathrm{trees\ in\ sight}$", 46)
        self.glyphs = {c: mask(f"${c}$", 104) for c in "0123456789."}
        self.glyphs["%"] = mask(r"$\%$", 104)
        self.advance = max(self.glyphs[c].shape[1] for c in "0123456789") - 4
        self.rule = mask(
            r"$(a{,}\,b)\ \mathrm{is\ hidden\ when\ } a \mathrm{\ and\ } b"
            r"\mathrm{\ share\ a\ factor}$",
            40,
        )
        self.shared = mask(r"$\mathrm{shared\ factor}{:}$", 36)
        self.keys = [
            (mask(f"${k}$", 40), torch.tensor(1.25 * rgb(h), device=self.device))
            for k, h in zip(("2", "3", "5", r"7{+}"), SHARED.values())
        ]
        self.formula = mask(r"$\frac{6}{\pi^2} = 60.79\ldots\%$", 88)
        self.series = mask(
            r"$1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \cdots"
            r" = \frac{\pi^2}{6}$",
            50,
        )

    @staticmethod
    def blit(image, mask, x, y, colour, alpha=1.0, centre=False):
        """Composite a text mask onto the (H, W, 3) tensor at top-left (x, y)."""
        h, w = mask.shape
        if centre:
            x -= w // 2
        a = alpha * mask[..., None]
        region = image[y : y + h, x : x + w]
        region.mul_(1 - a).add_(a * colour)

    def number(self, image, value, y):
        """A percentage like 60.8%, centred, digits on a fixed advance."""
        text = f"{100 * value:.1f}%"
        widths = [
            self.advance if c.isdigit() else self.glyphs[c].shape[1] for c in text
        ]
        x = WIDTH // 2 - sum(widths) // 2
        base = y + self.glyphs["0"].shape[0]
        for c, w in zip(text, widths):
            mask = self.glyphs[c]
            off = (w - mask.shape[1]) // 2 if c.isdigit() else 0
            yy = base - mask.shape[0] + (6 if c == "%" else 0)
            self.blit(image, mask, x + off, yy, self.gold)
            x += w

    def lines(self, radius, scale):
        """Lines of sight and the shadow lines behind them, as two uint8 planes."""
        strength = float(np.clip((scale - 18.0) / 50.0, 0.0, 1.0))
        n = int(np.searchsorted(self.sight_norm, radius))
        if strength < 0.03 or n == 0:
            return None
        cx, cy = WIDTH / 2, HEIGHT / 2
        k = 16.0  # cv2 fixed point, 4 fractional bits
        origin = (int(cx * k), int(cy * k))
        flash = np.exp(-np.log(radius / self.sight_norm[:n]) / 0.18)
        thick = 2 if scale > 70 else 1
        sight = np.zeros((HEIGHT, WIDTH), np.uint8)
        shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
        reach = min(radius, 2.5 * HEIGHT / scale)
        for (a, b), nrm, f in zip(self.sight[:n], self.sight_norm[:n], flash):
            p1 = (int((cx + a * scale) * k), int((cy - b * scale) * k))
            value = int(255 * (0.35 + 0.65 * f))
            cv2.line(sight, origin, p1, value, thick, cv2.LINE_AA, 4)
            if 2 * nrm < radius:  # a hidden tree stands behind this one
                x2 = cx + a / nrm * reach * scale
                y2 = cy - b / nrm * reach * scale
                p2 = (int(x2 * k), int(y2 * k))
                cv2.line(shadow, p1, p2, 128, thick, cv2.LINE_AA, 4)
        planes = torch.from_numpy(np.stack([sight, shadow])).to(self.device)
        planes = planes.float() * (strength / 255.0)
        gold = 0.42 * planes[0, ..., None] * self.gold
        return gold + 0.3 * planes[1, ..., None] * self.cool

    @torch.no_grad()
    def frame(self, sec):
        radius, scale = front(sec)
        radius, scale = float(radius), float(scale)
        x = self.x / scale
        y = self.y / scale
        a = torch.round(x).clamp(-M, M)
        b = torch.round(y).clamp(-M, M)
        d = torch.hypot(x - a, y - b) * scale  # px from the nearest tree's centre
        idx = ((a + M) * (2 * M + 1) + (b + M)).long()
        norm = self.norm[idx]
        soft = 0.35 + 0.08 * radius
        lit = ((radius - norm) / soft + 0.5).clamp(0, 1)
        lit = lit * lit * (3 - 2 * lit)
        vis = self.visible[idx]
        age = torch.log(radius / norm.clamp(min=0.5)).clamp(min=0)
        flash = vis * lit * torch.exp(-age / 0.16)
        colour = self.unlit + lit[..., None] * (self.colour[idx] - self.unlit)
        colour = colour * (1 + 1.4 * flash[..., None])
        r = (0.11 + lit * (self.radius[idx] - 0.11)) * scale
        cover = (r.clamp(min=1.0) - d + 0.5).clamp(0, 1)
        shade = 1 - 0.35 * (d / r).clamp(max=1) ** 2
        dots = (cover * shade)[..., None] * colour
        dots = F.avg_pool2d(dots.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)

        bloom = float(np.clip(scale / 50.0, 0.3, 1.0))
        glow = 0.9 * wide_blur(dots, 18.0, 6) + 0.6 * blur(dots, self.kernel)
        image = self.background + dots + bloom * glow
        ring = 0.16 * torch.exp(-(((self.rho - radius * scale) / 5.0) ** 2))
        image = image + ring[..., None] * self.white
        lines = self.lines(radius, scale)
        if lines is not None:
            image = image + lines
        image = (image * self.band).clamp(0, 255)
        self.text(image, sec, radius)
        out = float(smoothstep((DURATION - sec) / FADE))
        return image * out if out < 1 else image

    def text(self, image, sec, radius):
        white = self.white
        label = float(smoothstep(sec / 0.4))
        self.blit(image, self.label, WIDTH // 2, 70, white, label, True)
        self.number(image, in_sight(self.norms, self.cum, radius), 128)
        rule = float(smoothstep((sec - 2.0) / 0.6) * smoothstep((10.6 - sec) / 0.6))
        if rule > 0:
            self.blit(image, self.rule, WIDTH // 2, 1690, white, rule, True)
            widths = [self.shared.shape[1] + 30]
            widths += [m.shape[1] + 44 for m, _ in self.keys]
            x = WIDTH // 2 - sum(widths) // 2
            self.blit(image, self.shared, x, 1768, white, 0.8 * rule)
            x += widths[0]
            for (mask, colour), w in zip(self.keys, widths[1:]):
                self.blit(image, mask, x, 1764, colour, rule)
                x += w
        end = float(smoothstep((sec - 11.4) / 0.8))
        if end > 0:
            self.blit(image, self.formula, WIDTH // 2, 1645, self.gold, end, True)
            late = end * float(smoothstep((sec - 12.0) / 0.8))
            self.blit(image, self.series, WIDTH // 2, 1790, white, late, True)


# ------------------------------------------------------------------- the sound
STYLE = "music_box"  # the default soundtrack: a key of TAGS
# --audio-only tags: lead style and the bed under it.
TAGS = {
    "music_box": ("music_box", "pad"),
    "kalimba": ("kalimba", "pad"),
    "music_box_choir": ("music_box", "choir"),
}
BED = ((0, 0.0, 4.6), (5, 4.2, 8.4), (3, 8.0, 11.6), (0, 11.4, DURATION))  # chords


def spoke_events():
    """
    (second, height, pan) for lines of sight (a, b), 0 <= b <= a <= SPOKES, gcd 1:
    the moment the lit disc reaches the first hidden tree on the line, 2(a, b).
    Lowest lines (smallest a + b) first, keeping only events at least GAP s
    from one already kept, so the bells stay sparse as the lines multiply.
    """
    candidates = []
    for a in range(1, SPOKES + 1):
        for b in range(0, a + 1):
            if np.gcd(a, b) == 1:
                when = float(front_time(2 * np.hypot(a, b)))
                angle = np.arctan2(b, a) / (np.pi / 4)  # 0 (axis) .. 1 (diagonal)
                if when < LAST_SPOKE:
                    candidates.append((a + b, when, 0.6 * (2 * angle - 1)))
    events = []
    for height, when, pan in sorted(candidates):
        if all(abs(when - other) >= GAP for other, _, _ in events):
            events.append((when, height, pan))
    return sorted(events)


def digit_locks():
    """Seconds at which each leading digit of the counter (6, 60, 60.8) stops changing."""
    norms, cum = counts()
    secs = np.linspace(0.0, GROW_END, 4000)
    shown = [f"{100 * in_sight(norms, cum, float(front(s)[0])):.1f}" for s in secs]
    final = shown[-1]
    locks = []
    for n in (1, 2, 4):  # "6", "60", "60.8"
        wrong = [i for i, text in enumerate(shown) if text[:n] != final[:n]]
        locks.append(float(secs[wrong[-1] + 1]) if wrong else 0.0)
    return locks


def soundtrack(rng, tag=STYLE):
    """
    Ambient: a soft bed (pad or choir) holding I, vi, IV, I, with sparse struck
    accents from the maths. A bell (or kalimba) rings as each low line of sight
    gets its first hidden tree (the moment the lit disc reaches 2(a, b), from
    front_time), climbing the scale in order and panned by the line's angle;
    taller lines (larger a + b) are softer. Each time a leading digit of the
    counter locks for good (6, 60, 60.8, from the exact running fraction), a
    fifth is struck an octave up, and the home chord rings as 6/pi^2 appears.
    """
    lead, bed = TAGS.get(tag, TAGS[STYLE])
    style = STYLES[lead]
    scale = Scale(int(rng.choice(style.roots)), style.mode)
    hz = lambda note: float(audio.midi_to_hz(note))
    track = audio.silence(DURATION)
    kw = dict(circular=False)

    def ring(degree, octave, velocity, when, pan=0.0, length=3.5):
        note = style.voice(scale.hz(degree, octave), velocity, length)
        audio.add(track, audio.pan(note, pan), when, **kw)

    # The very first sight: the viewer opens their eyes on eight trees.
    ring(0, -1, 0.34, 0.05, 0.0, 4.0)
    for k, (when, height, pan) in enumerate(spoke_events()):
        degree = MELODY[k % len(MELODY)]
        ring(degree, -1, 0.36 - 0.12 * min(height, 60) / 60, when, pan)
    for k, when in enumerate(digit_locks()):
        ring(4 + k, 0, 0.3, when, -0.3 + 0.3 * k)
        ring(k, 1, 0.22, when + 0.12, 0.3 - 0.3 * k)
    home = 11.6
    for degree, octave, pan in ((0, 0, -0.2), (2, 0, 0.2), (4, 0, 0.0), (0, 1, 0.0)):
        ring(degree, octave, 0.3, home, pan, 5.0)

    for degree, start, stop in BED:
        for note in scale.chord(degree, 3, -2):
            if bed == "choir":
                voice = audio.choir(hz(note), stop - start, 0.14, 1.0, 1.6, "u")
            else:
                voice = audio.pad(hz(note), stop - start, 0.16, 1.4, 1.8, 0.3)
            audio.add(track, voice, start, **kw)
        root = scale.note(degree, -3)
        root += 12 if root < 40 else 0
        low = audio.pad(hz(root), stop - start, 0.14, 1.4, 1.8, 0.15)
        audio.add(track, low, start, **kw)
    return audio.master(style.room(track, rng, circular=False), loop=False)


def generate(settings: ImageProcessingSettings = None):
    """
    Euclid's orchard: the lattice points (a, b) seen from the origin.

    A tree at (a, b) is visible from (0, 0) iff gcd(a, b) = 1; otherwise it lies
    exactly behind (a/g, b/g). The fraction of visible trees within radius R
    tends to 1/zeta(2) = 6/pi^2 (with error O(log R / R)).

    Time drives one thing, the radius R(t) of the lit disc: log R eases out
    from 1.45 to 100 trees over 12.2 s, while its on-screen radius grows from
    250 to 1250 px, so the zoom (px per tree) falls from ~170 to 12.5. Each
    pixel finds its nearest lattice point by rounding, looks up that tree's
    gcd and distance in a precomputed (2M+1)^2 table and draws an anti-aliased
    disc: gold if visible, coloured by the smallest prime of the gcd if hidden,
    dim grey if still outside the disc. Visible trees flash as the front
    passes (decay in log R). Per-pixel on the GPU at 2x supersampling, with a
    two-scale bloom; while zoomed in, cv2 draws the lines of sight from the
    viewer to each visible tree and the shadow lines behind them. The counter
    is the exact running fraction of visible trees within R; the last 2.8 s
    rest on 6/pi^2 and the series 1 + 1/4 + 1/9 + ... = pi^2/6, with a 0.4 s
    fade. The picture uses no randomness; the seed only sets the soundtrack.
    """
    settings = settings or ImageProcessingSettings(1)
    renderer = None
    if not settings.audio_only:
        renderer = Renderer()

    def render(frame):
        return renderer.frame(frame / (FRAME_COUNT - 1) * DURATION)

    render_frames(settings, render, FRAME_COUNT, parallel=False, rgb=True, hook=HOOK, hook_y=HOOK_Y, hook_colour=HOOK_COLOUR)
    del renderer
    gc.collect()
    tag = settings.audio_tag
    track = soundtrack(settings.audio_rng, tag if tag in TAGS else STYLE)
    settings.save_video(fps=FPS, crf=18, audio=track)


if __name__ == "__main__":
    generate()
