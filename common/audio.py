"""
Sound synthesis for animations, in plain NumPy/SciPy.

A soundtrack is a float array of shape (samples, 2) at SAMPLE_RATE, exactly as
long as the video, handed to `ImageProcessingSettings.save_video(audio=...)`.
Everything here treats the track as circular -- notes that ring past the end
wrap around to the start and the reverb is a circular convolution -- so a
soundtrack built from loop-periodic events loops as seamlessly as the frames.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48_000
PEAK = 0.89  # -1 dBFS headroom for the AAC encoder


def silence(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """An empty stereo track of the given length in seconds."""
    return np.zeros((int(round(duration * sr)), 2))


def midi_to_hz(note) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(note, float) - 69.0) / 12.0)


def marimba(
    freq: float, velocity: float = 0.6, length: float = 2.5, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    One struck marimba bar, by modal synthesis.

    A tuned marimba bar rings at roughly 1x, 4x and 10x its fundamental; the
    overtones die away much faster than the fundamental, low bars ring longer
    than high ones, and a softer stroke (lower velocity) excites less of the
    upper partials -- a darker, rounder tone. Returns a mono float array.
    """
    t = np.arange(int(length * sr)) / sr
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


def pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Equal-power pan of a mono signal; position -1 (left) .. 1 (right)."""
    angle = (np.clip(position, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return mono[:, None] * np.array([np.cos(angle), np.sin(angle)])


def add(track: np.ndarray, sound: np.ndarray, start: float, sr: int = SAMPLE_RATE):
    """Mix `sound` into `track` in place at `start` seconds, wrapping past the end."""
    n = len(track)
    sound = sound[:n]
    i = int(round(start * sr)) % n
    head = min(len(sound), n - i)
    track[i : i + head] += sound[:head]
    track[: len(sound) - head] += sound[head:]


def reverb(
    track: np.ndarray,
    rng: np.random.Generator,
    rt60: float = 1.8,
    wet: float = 0.25,
    predelay: float = 0.015,
    tone: float = 5000.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    A smooth room: circular convolution with an exponentially decaying,
    low-passed noise impulse response, independent per channel for width.
    """
    n = len(track)
    t = np.arange(int(min(rt60 * 1.5, n / sr) * sr)) / sr
    sos = butter(2, tone, fs=sr, output="sos")
    ir = sosfilt(sos, rng.standard_normal((2, len(t))), axis=-1)
    ir *= np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    ir = np.roll(np.pad(ir, ((0, 0), (0, n - len(t)))), int(predelay * sr), axis=-1)
    ir /= np.sqrt((ir**2).sum(axis=-1, keepdims=True))
    tail = np.fft.irfft(np.fft.rfft(track.T) * np.fft.rfft(ir), n=n).T
    return (1.0 - wet) * track + wet * tail


def normalize(track: np.ndarray, peak: float = PEAK) -> np.ndarray:
    return track * (peak / max(np.abs(track).max(), 1e-12))


def write_wav(path, track: np.ndarray, sr: int = SAMPLE_RATE):
    """Write a float track as a 32-bit float WAV."""
    wavfile.write(str(path), sr, np.asarray(track, np.float32))
