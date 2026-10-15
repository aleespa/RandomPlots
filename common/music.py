"""
Musical choices for soundtracks: scales, chords, tempo, and the event-voice
styles of the catalogue in docs/AGENT_INSTRUCTIONS.md ("Soundtrack").

`common.audio` makes sounds; this module decides which notes and when. Pitches
are MIDI numbers (60 = middle C, +12 = an octave up); convert with
`audio.midi_to_hz`. Degrees count steps of a scale from its root and may be
negative or run past the octave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from common import audio

# Semitones above the root. The mood of each, for non-musicians:
SCALES = {
    "major_pentatonic": (0, 2, 4, 7, 9),  # bright, open, can't sound wrong
    "minor_pentatonic": (0, 3, 5, 7, 10),  # wistful, bluesy
    "major": (0, 2, 4, 5, 7, 9, 11),  # plain happy
    "minor": (0, 2, 3, 5, 7, 8, 10),  # sad, serious
    "dorian": (0, 2, 3, 5, 7, 9, 10),  # mellow, minor but hopeful
    "lydian": (0, 2, 4, 6, 7, 9, 11),  # dreamy, floating, spacey
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),  # relaxed, folky
    "whole_tone": (0, 2, 4, 6, 8, 10),  # uncanny, weightless, no home note
    "hirajoshi": (0, 2, 3, 7, 8),  # Japanese, contemplative
    "pelog": (0, 1, 3, 7, 8),  # Balinese gamelan, mysterious
}

# Chord progressions as scale degrees of each chord's root (0 = the home chord),
# for 7-note scales. One chord per bar, repeated to fill the loop.
PROGRESSIONS = {
    "pop": (0, 4, 5, 3),  # I V vi IV -- uplifting
    "sentimental": (5, 3, 0, 4),  # vi IV I V -- bittersweet
    "jazz": (1, 4, 0, 0),  # ii V I I -- resolved, sophisticated
    "epic": (0, 5, 2, 6),  # in minor: i VI III VII -- cinematic
    "drift": (0, 3, 0, 4),  # I IV I V -- calm, rocking
    "suspended": (0, 0, 3, 3),  # barely moves -- ambient
}


@dataclass(frozen=True)
class Scale:
    """A scale on a root note: `Scale(57, "dorian").note(3)` -> MIDI 62."""

    root: int
    mode: str = "major_pentatonic"

    @property
    def steps(self) -> tuple[int, ...]:
        return SCALES[self.mode]

    def note(self, degree: int, octave: int = 0) -> int:
        """MIDI pitch of scale degree `degree` (any integer), shifted by octaves."""
        size = len(self.steps)
        return self.root + 12 * (octave + degree // size) + self.steps[degree % size]

    def hz(self, degree: int, octave: int = 0) -> float:
        return float(audio.midi_to_hz(self.note(degree, octave)))

    def chord(self, degree: int, size: int = 3, octave: int = 0) -> list[int]:
        """Stacked thirds on `degree`: a triad (3), seventh chord (4), ninth (5)."""
        return [self.note(degree + 2 * i, octave) for i in range(size)]


@dataclass(frozen=True)
class Style:
    """
    An event-voice style: which instrument strikes the notes and where it
    sits. Pick the root with `rng.choice(style.roots)` and build
    `Scale(root, style.mode)`; finish with `style.room(track, rng)`.
    """

    voice: Callable  # voice(freq, velocity) -> mono array
    mode: str
    roots: range  # melody root, MIDI
    bass: Callable  # voice for low notes, two octaves under the melody
    rt60: float  # reverb length, s
    wet: float  # reverb amount
    tone: float = 5000.0  # reverb brightness, Hz

    def room(self, track, rng, circular=True):
        return audio.reverb(
            track, rng, self.rt60, self.wet, tone=self.tone, circular=circular
        )


STYLES = {
    # Warm wooden bars, bright pentatonic: the 1006 / 1011 sound.
    "mallet_garden": Style(
        audio.marimba, "major_pentatonic", range(57, 65), audio.marimba, 2.2, 0.3
    ),
    # High glassy bells in a floating mode, a long bright room.
    "music_box": Style(
        audio.bell, "lydian", range(69, 77), audio.marimba, 3.0, 0.35, 7000.0
    ),
    # Plucked strings in a Japanese mode, a dry-ish room.
    "harp": Style(
        audio.pluck, "hirajoshi", range(55, 63), audio.pluck, 1.8, 0.25, 4500.0
    ),
    # Bells in a gamelan scale, long and dark.
    "gamelan": Style(
        audio.bell, "pelog", range(60, 68), audio.marimba, 3.5, 0.4, 3500.0
    ),
    # A soft piano in a plain major key, a warm hall: intimate, familiar.
    "felt_piano": Style(
        audio.piano, "major", range(57, 65), audio.piano, 2.4, 0.3, 4500.0
    ),
    # Electric piano in dorian, a small dark room: lo-fi, mellow, late-night.
    "lofi_keys": Style(
        audio.rhodes, "dorian", range(55, 63), audio.rhodes, 1.4, 0.22, 3500.0
    ),
    # Vibraphone with its motor on over a plucked bass: cool jazz.
    "vibes": Style(
        audio.vibraphone, "mixolydian", range(60, 68), audio.pluck, 2.0, 0.25
    ),
    # Thumb piano in the bright pentatonic, a dry room: playful, toy-like.
    "kalimba": Style(
        audio.kalimba, "major_pentatonic", range(62, 70), audio.marimba, 1.3, 0.2
    ),
    # Handpan in natural minor (the "Kurd" scale of most handpans): meditative.
    "handpan": Style(
        audio.handpan, "minor", range(57, 65), audio.handpan, 2.6, 0.32, 4500.0
    ),
    # Breathy flute in minor pentatonic, a long room: wistful, open-air.
    "bamboo_flute": Style(
        audio.flute, "minor_pentatonic", range(65, 72), audio.pluck, 2.8, 0.35, 4000.0
    ),
}


# --- Time ----------------------------------------------------------------------


def beat(bpm: float) -> float:
    """Seconds per beat."""
    return 60.0 / bpm


def loop_bpm(duration: float, bars: int, beats_per_bar: int = 4) -> float:
    """The tempo at which `duration` seconds is exactly `bars` bars."""
    return 60.0 * bars * beats_per_bar / duration


def tempo_choices(duration: float, low: float = 60.0, high: float = 140.0):
    """[(bpm, bars)] of 4/4 tempi that fit `duration` a whole number of bars."""
    return [
        (round(loop_bpm(duration, bars), 2), bars)
        for bars in range(1, 64)
        if low <= loop_bpm(duration, bars) <= high
    ]


def beat_times(duration: float, bpm: float, subdivision: int = 1, offset: float = 0.0):
    """Onsets of every beat (or 1/subdivision of a beat) in [0, duration)."""
    step = beat(bpm) / subdivision
    return np.arange(offset % step, duration - 1e-9, step)


def quantize(times, duration: float, bpm: float, subdivision: int = 4, amount=1.0):
    """
    Pull event times towards the nearest 1/subdivision beat (amount 1 = snap,
    0.5 = halfway: tighter but still organic). Wraps at `duration`.
    """
    step = beat(bpm) / subdivision
    times = np.asarray(times, float)
    snapped = np.round(times / step) * step
    return (times + amount * (snapped - times)) % duration
