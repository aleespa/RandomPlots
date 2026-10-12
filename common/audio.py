"""
Sound synthesis for animations, in plain NumPy/SciPy.

A soundtrack is a float array of shape (samples, 2) at SAMPLE_RATE, exactly as
long as the video, handed to `ImageProcessingSettings.save_video(audio=...)`.

Loops and one-shots: by default everything here treats the track as circular --
notes that ring past the end wrap around to the start, the reverb and delay are
circular, and filters are run over the wrap -- so a soundtrack built from
loop-periodic events loops as seamlessly as the frames. A one-shot passes
`circular=False` (and `loop=False` to `master`) so tails are cut at the end
instead of bleeding into the opening second.

Voices return mono arrays (except `pad`, which is stereo); place them with
`pan` and mix them in with `add`. Finish every track with `master`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48_000
PEAK = 0.89  # -1 dBFS headroom for the AAC encoder
LOUDNESS = -14.0  # LUFS, what Instagram and most platforms normalise towards


def silence(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """An empty stereo track of the given length in seconds."""
    return np.zeros((int(round(duration * sr)), 2))


def midi_to_hz(note) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(note, float) - 69.0) / 12.0)


# --- Voices --------------------------------------------------------------------
# Struck and plucked voices are cached on (rounded) arguments: a soundtrack
# repeats the same few pitches hundreds of times. Cached arrays are read-only;
# `add` and `pan` copy, so that is fine -- copy before modifying one in place.


def _seconds(length: float, sr: int) -> np.ndarray:
    return np.arange(int(length * sr)) / sr


def _cached(function):
    cached = lru_cache(maxsize=1024)(function)

    def voice(freq, velocity=0.6, *args, **kwargs):
        sound = cached(
            round(float(freq), 3), round(float(velocity), 3), *args, **kwargs
        )
        sound.flags.writeable = False
        return sound

    voice.__doc__ = function.__doc__
    voice.__name__ = function.__name__
    return voice


@_cached
def marimba(
    freq: float, velocity: float = 0.6, length: float = 2.5, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    One struck marimba bar, by modal synthesis. Warm, woody, round.

    A tuned marimba bar rings at roughly 1x, 4x and 10x its fundamental; the
    overtones die away much faster than the fundamental, low bars ring longer
    than high ones, and a softer stroke (lower velocity) excites less of the
    upper partials -- a darker, rounder tone. Returns a mono float array.
    """
    t = _seconds(length, sr)
    ring = np.clip(0.9 * (220.0 / freq) ** 0.6, 0.12, 1.6)  # fundamental decay, s
    note = np.zeros_like(t)
    for ratio, gain, softness in ((1.0, 1.0, 0), (3.98, 0.3, 1), (9.85, 0.1, 2)):
        if ratio * freq > 0.45 * sr:
            continue
        note += (
            gain
            * velocity**softness
            * np.sin(2.0 * np.pi * ratio * freq * t)
            * np.exp(-t / (ring / ratio**0.8))
        )
    attack = 1.0 - np.exp(-t / (0.0015 + 0.003 * (1.0 - velocity)))  # felt mallet
    return velocity * attack * note


@_cached
def bell(
    freq: float, velocity: float = 0.6, length: float = 3.0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    A small glassy bell (glockenspiel, celesta, music-box tine), by modal synthesis.

    A free metal bar rings at the inharmonic ratios 1 : 2.76 : 5.40 : 8.93,
    which is what makes it sound like metal rather than wood. A faint copy of
    the fundamental 0.6 Hz sharp gives a slow shimmer. Hard, quick attack;
    harder strokes (velocity) bring out the upper modes. Mono.
    """
    t = _seconds(length, sr)
    ring = np.clip(1.6 * (440.0 / freq) ** 0.35, 0.4, 3.0)
    note = 0.25 * np.sin(2.0 * np.pi * (freq + 0.6) * t) * np.exp(-t / ring)
    for ratio, gain, softness in (
        (1.0, 1.0, 0),
        (2.756, 0.35, 1),
        (5.404, 0.15, 1.5),
        (8.933, 0.06, 2),
    ):
        if ratio * freq > 0.45 * sr:
            continue
        note += (
            gain
            * velocity**softness
            * np.sin(2.0 * np.pi * ratio * freq * t)
            * np.exp(-t / (ring / ratio**0.9))
        )
    attack = 1.0 - np.exp(-t / 0.0006)
    return 0.8 * velocity * attack * note


@_cached
def pluck(
    freq: float,
    velocity: float = 0.6,
    length: float = 2.0,
    position: float = 0.18,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A plucked string (harp, koto, guitar-ish), by additive synthesis.

    A string plucked at `position` along its length (0..0.5) excites harmonic
    k with amplitude |sin(k pi position)| / k; high harmonics die out first, so
    the note starts bright and mellows, and a slight stiffness stretches the
    partials sharp. Softer plucks are darker. Exactly in tune at any pitch.
    Mono.
    """
    t = _seconds(length, sr)
    ring = np.clip(1.2 * (220.0 / freq) ** 0.5, 0.25, 2.5)
    note = np.zeros_like(t)
    for k in range(1, 25):
        partial = k * freq * np.sqrt(1.0 + 1e-4 * k * k)
        if partial > 0.45 * sr:
            break
        gain = abs(np.sin(k * np.pi * position)) / k * velocity ** (0.3 * (k - 1))
        note += gain * np.sin(2.0 * np.pi * partial * t) * np.exp(-t * k**0.8 / ring)
    attack = 1.0 - np.exp(-t / 0.0008)
    return 0.9 * velocity * attack * note


def pad(
    freq: float,
    duration: float,
    velocity: float = 0.5,
    attack: float = 1.5,
    release: float = 2.0,
    brightness: float = 0.5,
    detune: float = 7.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A soft sustained synth pad: three detuned band-limited sawtooths.

    Held for `duration` seconds after a raised-cosine swell of `attack`
    seconds, then fades over `release` (so the sound is duration + release
    long). `brightness` (0..1) sets how fast the harmonics fall away (0 is
    nearly a sine, 1 a full saw); `detune` (cents) spreads the outer voices,
    which are panned left and right for width. Returns STEREO (samples, 2).
    """
    t = _seconds(duration + release, sr)
    rolloff = 1.0 + 2.5 * (1.0 - brightness)
    channels = np.zeros((len(t), 2))
    for cents, side in ((-detune, 0), (0.0, None), (detune, 1)):
        f = freq * 2.0 ** (cents / 1200.0)
        voice = np.zeros_like(t)
        for k in range(1, 13):
            if k * f > 0.45 * sr:
                break
            voice += np.sin(2.0 * np.pi * k * f * t + k) / k**rolloff
        if side is None:
            channels += 0.5 * voice[:, None]
        else:
            channels[:, side] += 0.7 * voice
    swell = np.clip(t / max(attack, 1e-3), 0.0, 1.0)
    fade = np.clip((duration + release - t) / max(release, 1e-3), 0.0, 1.0)
    shape = (0.5 - 0.5 * np.cos(np.pi * swell)) * (0.5 - 0.5 * np.cos(np.pi * fade))
    return 0.35 * velocity * shape[:, None] * channels


def kick(velocity: float = 0.8, pitch: float = 48.0, sr: int = SAMPLE_RATE):
    """A soft electronic kick: a sine sweeping down to `pitch` Hz. Mono."""
    t = _seconds(0.6, sr)
    f = pitch * (1.0 + 2.5 * np.exp(-t / 0.03))
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    attack = 1.0 - np.exp(-t / 0.001)
    return velocity * attack * np.sin(phase) * np.exp(-t / 0.18)


def tick(
    rng: np.random.Generator,
    velocity: float = 0.4,
    decay: float = 0.03,
    tone: float = 7000.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A noise tick: closed hi-hat (decay ~0.03 s), shaker (~0.08 s) or brush
    (~0.2 s, tone ~3000). High-passed noise, so it needs `rng`. Mono.
    """
    t = _seconds(decay * 6.0, sr)
    sos = butter(2, tone, btype="highpass", fs=sr, output="sos")
    noise = sosfilt(sos, rng.standard_normal(len(t)))
    return 0.5 * velocity * noise * np.exp(-t / decay)


def riser(
    duration: float,
    rng: np.random.Generator,
    start: float = 200.0,
    end: float = 2000.0,
    velocity: float = 0.4,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A swell of tension for one-shots: a sine gliding exponentially from `start`
    to `end` Hz under a bed of airy noise, growing louder over `duration` s and
    stopping dead, so whatever lands next (a chord, a bass note) resolves it.
    Mono.
    """
    t = _seconds(duration, sr)
    u = t / duration
    f = start * (end / start) ** u
    glide = np.sin(2.0 * np.pi * np.cumsum(f) / sr)
    sos = butter(2, (2000.0, 9000.0), btype="bandpass", fs=sr, output="sos")
    air = sosfilt(sos, rng.standard_normal(len(t)))
    air /= max(np.abs(air).max(), 1e-12)
    return velocity * u**2 * (0.6 * glide + 0.4 * air)


# --- Placement and mixing ------------------------------------------------------


def pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Equal-power pan of a mono signal; position -1 (left) .. 1 (right)."""
    angle = (np.clip(position, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return mono[:, None] * np.array([np.cos(angle), np.sin(angle)])


def add(
    track: np.ndarray,
    sound: np.ndarray,
    start: float,
    circular: bool = True,
    sr: int = SAMPLE_RATE,
):
    """
    Mix `sound` (mono or stereo) into `track` in place at `start` seconds.
    Circular (loops): what rings past the end wraps to the start. Otherwise
    (one-shots) it is cut off at the end.
    """
    n = len(track)
    if sound.ndim == 1:
        sound = sound[:, None]
    i = int(round(start * sr))
    if circular:
        sound = sound[:n]
        i %= n
        head = min(len(sound), n - i)
        track[i : i + head] += sound[:head]
        track[: len(sound) - head] += sound[head:]
        return
    if i < 0:
        sound, i = sound[-i:], 0
    head = max(0, min(len(sound), n - i))
    track[i : i + head] += sound[:head]


def _filter(sos, track: np.ndarray, circular: bool, sr: int) -> np.ndarray:
    """IIR filter along time; circular runs it in over the end of the track first."""
    if not circular:
        return sosfilt(sos, track, axis=0)
    warm = min(len(track), sr)  # one second settles anything we use
    out = sosfilt(sos, np.concatenate([track[-warm:], track]), axis=0)
    return out[warm:]


def lowpass(track, cutoff: float, order: int = 2, circular=True, sr=SAMPLE_RATE):
    """Darken: remove content above `cutoff` Hz (gently, 12 dB/octave at order 2)."""
    return _filter(butter(order, cutoff, fs=sr, output="sos"), track, circular, sr)


def highpass(track, cutoff: float, order: int = 2, circular=True, sr=SAMPLE_RATE):
    """Thin out: remove content below `cutoff` Hz (rumble, muddy low end)."""
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return _filter(sos, track, circular, sr)


def delay(
    track: np.ndarray,
    time: float,
    feedback: float = 0.35,
    wet: float = 0.3,
    pingpong: bool = True,
    circular: bool = True,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Echoes every `time` seconds, each `feedback` times quieter. Sync it to the
    tempo (e.g. `music.beat(bpm) * 0.75`, a dotted eighth). `pingpong` bounces
    successive echoes between left and right.
    """
    n = len(track)
    echoes = np.zeros_like(track)
    gain, k = 1.0, 1
    while gain > 1e-3 and k * time * sr < 4 * n:
        shift = int(round(k * time * sr))
        if circular:
            echo = np.roll(track, shift, axis=0)
        else:
            echo = np.zeros_like(track)
            if shift < n:
                echo[shift:] = track[: n - shift]
        if pingpong and k % 2:
            echo = echo[:, ::-1]
        echoes += gain * echo
        gain *= feedback
        k += 1
    return track + wet * echoes


def reverb(
    track: np.ndarray,
    rng: np.random.Generator,
    rt60: float = 1.8,
    wet: float = 0.25,
    predelay: float = 0.015,
    tone: float = 5000.0,
    circular: bool = True,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A smooth room: convolution with an exponentially decaying, low-passed
    noise impulse response, independent per channel for width. `rt60` is how
    long the room rings (1 s small room, 2-3 s hall, 5+ s cathedral), `tone`
    how bright it is. Circular for loops; one-shots cut the tail at the end.
    """
    n = len(track)
    length = min(rt60 * 1.5, n / sr) if circular else rt60 * 1.5
    t = np.arange(int(length * sr)) / sr
    sos = butter(2, tone, fs=sr, output="sos")
    ir = sosfilt(sos, rng.standard_normal((2, len(t))), axis=-1)
    ir *= np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    ir = np.pad(ir, ((0, 0), (int(predelay * sr), 0)))
    ir /= np.sqrt((ir**2).sum(axis=-1, keepdims=True))
    size = n if circular else n + ir.shape[1]
    ir = ir[:, :size]
    spectrum = np.fft.rfft(track.T, n=size) * np.fft.rfft(ir, n=size)
    tail = np.fft.irfft(spectrum, n=size)[:, :n].T
    return (1.0 - wet) * track + wet * tail


# --- Finishing -----------------------------------------------------------------

# ITU-R BS.1770 K-weighting at 48 kHz: a high shelf (head acoustics) and a
# high-pass (the ear's insensitivity to the lowest frequencies).
_K_WEIGHTING = np.array(
    [
        [1.53512485958697, -2.69169618940638, 1.19839281085285]
        + [1.0, -1.69065929318241, 0.73248077421585],
        [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
    ]
)


def loudness(track: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770 / EBU R128, gated)."""
    if sr != 48_000:
        raise ValueError("loudness() has K-weighting coefficients for 48 kHz only")
    track = track if track.ndim == 2 else track[:, None]
    weighted = sosfilt(_K_WEIGHTING, track, axis=0)
    block, step = int(0.4 * sr), int(0.1 * sr)
    if len(weighted) < block:
        return float("-inf")
    starts = np.arange(0, len(weighted) - block + 1, step)
    power = np.cumsum(np.vstack([np.zeros((1, track.shape[1])), weighted**2]), axis=0)
    blocks = ((power[starts + block] - power[starts]) / block).sum(axis=1)
    level = -0.691 + 10.0 * np.log10(np.maximum(blocks, 1e-20))
    gated = blocks[level > -70.0]
    if len(gated) == 0:
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(gated.mean()) - 10.0
    gated = blocks[(level > -70.0) & (level > relative)]
    return float(-0.691 + 10.0 * np.log10(gated.mean()))


def soft_limit(track: np.ndarray, ceiling: float = PEAK, knee: float = 0.6):
    """Leave everything below knee * ceiling untouched; round peaks off smoothly."""
    k = knee * ceiling
    magnitude = np.abs(track)
    over = magnitude > k
    out = track.copy()
    out[over] = np.sign(track[over]) * (
        k + (ceiling - k) * np.tanh((magnitude[over] - k) / (ceiling - k))
    )
    return out


def fade(
    track: np.ndarray, fade_in: float = 0.0, fade_out: float = 0.0, sr=SAMPLE_RATE
):
    """Raised-cosine fades at the ends of a one-shot (in place, returns the track)."""
    for seconds, end in ((fade_in, False), (fade_out, True)):
        m = min(int(seconds * sr), len(track))
        if m:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, m))
            if end:
                track[-m:] *= ramp[::-1, None]
            else:
                track[:m] *= ramp[:, None]
    return track


def master(
    track: np.ndarray,
    loop: bool = True,
    target: float = LOUDNESS,
    low_cut: float = 35.0,
    max_limiting: float = 4.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    The last step of every soundtrack: clean the low end, bring it to
    `target` LUFS and keep true peaks under -1 dBFS, then log what it did.

    High-passes below `low_cut` (rumble phones can't play anyway), applies
    gain towards the target loudness but never pushes more than
    `max_limiting` dB into the soft limiter -- a sparse track ends up a bit
    under target rather than squashed. One-shots (`loop=False`) get 5 ms /
    60 ms fades so the ends don't click. Loops get a seam report: the jump
    from the last sample to the first should look like any other step.
    """
    track = highpass(track, low_cut, circular=loop, sr=sr)
    if not loop:
        fade(track, 0.005, 0.06, sr)
    before = loudness(track, sr)
    gain = 10.0 ** ((target - before) / 20.0)
    peak = np.abs(track).max() * gain
    limit = PEAK * 10.0 ** (max_limiting / 20.0)
    if peak > limit:
        gain *= limit / peak
    track = soft_limit(track * gain)
    after = loudness(track, sr)
    logger.info(
        f"Master: {after:.1f} LUFS (target {target:.0f}, was {before:.1f} before gain), "
        f"peak {20 * np.log10(np.abs(track).max()):.1f} dBFS"
    )
    if after < target - 3.0:
        logger.warning(
            f"Soundtrack is {target - after:.1f} LU under target: it's very sparse or "
            "spiky. Add sustain (longer notes, reverb, a pad) rather than more gain."
        )
    if loop:
        steps = np.abs(np.diff(track, axis=0)).max(axis=1)
        seam = np.abs(track[0] - track[-1]).max()
        typical = np.percentile(steps, 99.9)
        report = f"Loop seam step {seam:.4f}, 99.9th percentile step {typical:.4f}"
        (logger.warning if seam > 2.0 * typical else logger.info)(report)
    return track


def normalize(track: np.ndarray, peak: float = PEAK) -> np.ndarray:
    """Peak-normalise only. Prefer `master`, which also manages loudness."""
    return track * (peak / max(np.abs(track).max(), 1e-12))


def write_wav(path, track: np.ndarray, sr: int = SAMPLE_RATE):
    """Write a float track as a 32-bit float WAV."""
    wavfile.write(str(path), sr, np.asarray(track, np.float32))


def spectrogram_png(path, track: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Save a picture of the soundtrack to `path`: the level over time on top, a
    log-frequency spectrogram below. It is how an agent "listens": note
    density, register, gaps, harsh highs and muddy lows are all visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from scipy.signal import spectrogram

    mono = track.mean(axis=1) if track.ndim == 2 else track
    f, t, power = spectrogram(mono, sr, nperseg=4096, noverlap=3072)
    keep = (f >= 30.0) & (f <= 16_000.0)
    db = 10.0 * np.log10(power[keep] + 1e-12)
    hop = int(0.01 * sr)
    envelope = np.abs(mono[: len(mono) // hop * hop]).reshape(-1, hop).max(axis=1)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(14, 7), dpi=100, height_ratios=(1, 3), sharex=True
    )
    fig.patch.set_facecolor("#111111")
    for ax in (top, bottom):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#bbbbbb")
    top.fill_between(np.arange(len(envelope)) * hop / sr, envelope, color="#e8b04a")
    top.set_ylim(0, 1)
    top.set_ylabel("peak", color="#bbbbbb")
    bottom.pcolormesh(t, f[keep], db, vmin=db.max() - 80, vmax=db.max(), cmap="magma")
    bottom.set_yscale("log")
    bottom.set_ylabel("Hz", color="#bbbbbb")
    bottom.set_xlabel("s", color="#bbbbbb")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
