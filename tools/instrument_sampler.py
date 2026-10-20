"""
Hear every voice in `common.audio`: each struck or plucked voice plays the same
short phrase in C major, then `pad`, `strings` and `choir` each hold a chord.

    uv run python -m tools.instrument_sampler   # -> outputs/instrument_sampler.wav

The order is printed with the time each voice starts, so you can tell them apart.
"""

from pathlib import Path

import numpy as np
from loguru import logger

from common import audio
from common.music import Scale

PHRASE = (0, 2, 4, 7, 4, 5, 3, 0)  # scale degrees
STRUCK = (
    "marimba",
    "bell",
    "pluck",
    "piano",
    "rhodes",
    "vibraphone",
    "kalimba",
    "handpan",
    "flute",
    "chip",
    "chip_bass",
)
SUSTAINED = ("pad", "strings", "choir")
SLOT, HOLD = 3.0, 4.0  # seconds per struck voice, per sustained voice


def main():
    scale = Scale(60, "major")
    track = audio.silence(SLOT * len(STRUCK) + HOLD * len(SUSTAINED))
    kw = dict(circular=False)
    for i, name in enumerate(STRUCK):
        logger.info(f"{i * SLOT:5.1f} s  {name}")
        voice = getattr(audio, name)
        for j, degree in enumerate(PHRASE):
            note = audio.pan(voice(scale.hz(degree), 0.55), -0.4 + 0.8 * j / 7)
            audio.add(track, note, i * SLOT + 0.3 * j, **kw)
    for i, name in enumerate(SUSTAINED):
        start = SLOT * len(STRUCK) + HOLD * i
        logger.info(f"{start:5.1f} s  {name}")
        for note in scale.chord(0, 3, octave=-1):
            sound = getattr(audio, name)(float(audio.midi_to_hz(note)), 2.5, 0.6)
            audio.add(track, sound, start, **kw)
    track = audio.reverb(track, np.random.default_rng(0), 2.2, 0.28, circular=False)
    track = audio.master(track, loop=False)
    path = Path("outputs") / "instrument_sampler.wav"
    path.parent.mkdir(exist_ok=True)
    audio.write_wav(path, track)
    logger.success(f"Wrote {path}")


if __name__ == "__main__":
    main()
