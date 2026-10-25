"""Asset Manager for audio sound effects, musical loops, and dynamic audio ducking."""

import os
import math
import wave
import struct
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AssetManager:
    """Manages audio SFX, background music, and dynamic ducking curves."""

    def __init__(
        self,
        sfx_dir: str = "data/sound_effects",
        music_dir: str = "data/music"
    ):
        self.sfx_dir = sfx_dir
        self.music_dir = music_dir
        os.makedirs(self.sfx_dir, exist_ok=True)
        os.makedirs(self.music_dir, exist_ok=True)
        self._ensure_default_assets()

    def get_sfx_path(self, sfx_name: str) -> str:
        """Returns the file path for a requested sound effect."""
        filename = f"{sfx_name}.wav"
        path = os.path.join(self.sfx_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(self.sfx_dir, "whoosh.wav")
        return path

    def get_bgm_path(self, tier: str = "younger", niche: str = "animal_secrets") -> str:
        """Returns the tier-specific background music track matching target age group."""
        tier = (tier or "younger").lower()
        if tier == "preschool":
            return os.path.join(self.music_dir, "preschool_nursery_bgm.wav")
        elif tier == "older":
            return os.path.join(self.music_dir, "older_mindblown_bgm.wav")
        else:
            return os.path.join(self.music_dir, "cheerful_bgm.wav")

    def get_default_bgm_path(self, tier: str = "younger") -> str:
        """Returns the background music track for the given tier (default: younger)."""
        return self.get_bgm_path(tier=tier)

    def _ensure_default_assets(self):
        """Synthesizes default crisp SFX and tier-specific background music if not present."""
        sfx_files = {
            "whoosh.wav": self._synth_whoosh,
            "pop.wav": self._synth_pop,
            "bell_ding.wav": self._synth_ding,
            "ding.wav": self._synth_ding,
            "chime.wav": self._synth_chime,
            "countdown_tick.wav": self._synth_tick
        }

        for fname, synth_fn in sfx_files.items():
            fpath = os.path.join(self.sfx_dir, fname)
            if not os.path.exists(fpath):
                synth_fn(fpath)

        bgm_files = {
            "cheerful_bgm.wav": self._synth_cheerful_bgm,
            "preschool_nursery_bgm.wav": self._synth_preschool_nursery_bgm,
            "older_mindblown_bgm.wav": self._synth_older_mindblown_bgm
        }

        for fname, synth_fn in bgm_files.items():
            bpath = os.path.join(self.music_dir, fname)
            if not os.path.exists(bpath):
                synth_fn(bpath)

    def _synth_whoosh(self, path: str):
        """Synthesizes an airy whoosh sound effect."""
        sample_rate = 44100
        duration = 0.45
        total_samples = int(sample_rate * duration)
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(total_samples):
                t = i / sample_rate
                # Exponential pitch sweep from 150Hz up to 900Hz and back down
                freq = 250 + 600 * math.sin(math.pi * (t / duration))
                # Bell curve envelope
                envelope = math.sin(math.pi * (t / duration)) ** 2
                sample = int(30000 * envelope * math.sin(2 * math.pi * freq * t))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_pop(self, path: str):
        """Synthesizes a clean bubble pop sound effect."""
        sample_rate = 44100
        duration = 0.12
        total_samples = int(sample_rate * duration)
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(total_samples):
                t = i / sample_rate
                freq = 800 * math.exp(-t * 25)
                envelope = math.exp(-t * 30)
                sample = int(28000 * envelope * math.sin(2 * math.pi * freq * t))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_ding(self, path: str):
        """Synthesizes a bright, crystalline bell ding."""
        sample_rate = 44100
        duration = 0.8
        total_samples = int(sample_rate * duration)
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(total_samples):
                t = i / sample_rate
                # Bell harmonics: A5 (880Hz) + E6 (1318.5Hz)
                envelope = math.exp(-t * 5.0)
                h1 = math.sin(2 * math.pi * 880 * t)
                h2 = 0.5 * math.sin(2 * math.pi * 1318.5 * t)
                sample = int(25000 * envelope * (h1 + h2))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_chime(self, path: str):
        """Synthesizes a pleasant ascending magic chime chord."""
        sample_rate = 44100
        duration = 1.0
        total_samples = int(sample_rate * duration)
        notes = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(total_samples):
                t = i / sample_rate
                sample_sum = 0
                for idx, freq in enumerate(notes):
                    note_start = idx * 0.12
                    if t >= note_start:
                        note_t = t - note_start
                        env = math.exp(-note_t * 6.0)
                        sample_sum += env * math.sin(2 * math.pi * freq * note_t)
                sample = int(18000 * sample_sum)
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_tick(self, path: str):
        """Synthesizes a crisp countdown tick-tock click."""
        sample_rate = 44100
        duration = 0.08
        total_samples = int(sample_rate * duration)
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(total_samples):
                t = i / sample_rate
                freq = 1200
                env = math.exp(-t * 80)
                sample = int(25000 * env * math.sin(2 * math.pi * freq * t))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_cheerful_bgm(self, path: str):
        """Synthesizes a cheerful, bouncy 45-second xylophone/kalimba melody loop."""
        sample_rate = 44100
        bpm = 124
        beat_duration = 60.0 / bpm
        melody_notes = [
            523.25, 659.25, 783.99, 880.00,  # C, E, G, A
            783.99, 659.25, 523.25, 587.33,  # G, E, C, D
            659.25, 783.99, 880.00, 1046.50, # E, G, A, C
            880.00, 783.99, 659.25, 523.25   # A, G, E, C
        ]
        total_seconds = 45.0
        total_samples = int(sample_rate * total_seconds)

        try:
            import numpy as np
            t = np.linspace(0, total_seconds, total_samples, endpoint=False)
            beat_sub = beat_duration * 0.5
            beat_indices = (t / beat_sub).astype(int) % len(melody_notes)
            beat_t = t % beat_sub
            
            freqs = np.array(melody_notes)[beat_indices]
            envelopes = np.exp(-beat_t * 9.0)
            audio = envelopes * (0.8 * np.sin(2 * np.pi * freqs * beat_t) + 0.2 * np.sin(4 * np.pi * freqs * beat_t))
            audio_int16 = (audio * 12000).astype(np.int16)

            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_int16.tobytes())
        except Exception:
            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                for i in range(total_samples):
                    t = i / sample_rate
                    beat_idx = int(t / (beat_duration * 0.5)) % len(melody_notes)
                    beat_t = t % (beat_duration * 0.5)
                    freq = melody_notes[beat_idx]
                    envelope = math.exp(-beat_t * 9.0)
                    val = envelope * (0.8 * math.sin(2 * math.pi * freq * beat_t) + 0.2 * math.sin(4 * math.pi * freq * beat_t))
                    sample = int(12000 * val)
                    w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_preschool_nursery_bgm(self, path: str):
        """Synthesizes a gentle, soothing 45-second music-box lullaby loop for preschoolers (96 BPM, F Major)."""
        sample_rate = 44100
        bpm = 96
        beat_duration = 60.0 / bpm
        melody_notes = [
            349.23, 440.00, 523.25, 587.33,  # F4, A4, C5, D5
            523.25, 440.00, 392.00, 349.23,  # C5, A4, G4, F4
            392.00, 440.00, 523.25, 440.00,  # G4, A4, C5, A4
            349.23, 392.00, 349.23, 261.63   # F4, G4, F4, C4
        ]
        total_seconds = 45.0
        total_samples = int(sample_rate * total_seconds)

        try:
            import numpy as np
            t = np.linspace(0, total_seconds, total_samples, endpoint=False)
            beat_sub = beat_duration * 0.75
            beat_indices = (t / beat_sub).astype(int) % len(melody_notes)
            beat_t = t % beat_sub
            
            freqs = np.array(melody_notes)[beat_indices]
            # Soft bell/music-box envelope with gentle decay
            envelopes = np.exp(-beat_t * 6.5)
            audio = envelopes * (0.85 * np.sin(2 * np.pi * freqs * beat_t) + 0.15 * np.sin(4 * np.pi * freqs * beat_t))
            audio_int16 = (audio * 11000).astype(np.int16)

            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_int16.tobytes())
        except Exception:
            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                for i in range(total_samples):
                    t = i / sample_rate
                    beat_sub = beat_duration * 0.75
                    beat_idx = int(t / beat_sub) % len(melody_notes)
                    beat_t = t % beat_sub
                    freq = melody_notes[beat_idx]
                    envelope = math.exp(-beat_t * 6.5)
                    val = envelope * (0.85 * math.sin(2 * math.pi * freq * beat_t) + 0.15 * math.sin(4 * math.pi * freq * beat_t))
                    sample = int(11000 * val)
                    w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

    def _synth_older_mindblown_bgm(self, path: str):
        """Synthesizes an upbeat 45-second electronic synth-bass science groove for tweens (132 BPM, D Minor)."""
        sample_rate = 44100
        bpm = 132
        beat_duration = 60.0 / bpm
        bass_notes = [
            146.83, 146.83, 174.61, 146.83,  # D3, D3, F3, D3
            220.00, 196.00, 174.61, 130.81,  # A3, G3, F3, C3
            146.83, 174.61, 220.00, 261.63,  # D3, F3, A3, C4
            220.00, 196.00, 174.61, 146.83   # A3, G3, F3, D3
        ]
        lead_notes = [
            293.66, 349.23, 440.00, 523.25,
            587.33, 440.00, 349.23, 293.66,
            440.00, 523.25, 587.33, 698.46,
            587.33, 523.25, 440.00, 293.66
        ]
        total_seconds = 45.0
        total_samples = int(sample_rate * total_seconds)

        try:
            import numpy as np
            t = np.linspace(0, total_seconds, total_samples, endpoint=False)
            beat_sub = beat_duration * 0.5
            b_indices = (t / beat_sub).astype(int) % len(bass_notes)
            l_indices = (t / (beat_sub * 0.5)).astype(int) % len(lead_notes)
            beat_t = t % beat_sub
            lead_t = t % (beat_sub * 0.5)

            b_freqs = np.array(bass_notes)[b_indices]
            l_freqs = np.array(lead_notes)[l_indices]

            bass_env = np.exp(-beat_t * 7.0)
            lead_env = np.exp(-lead_t * 12.0)

            # Punchy synth bass + crisp arpeggio
            bass_wave = bass_env * (0.7 * np.sin(2 * np.pi * b_freqs * beat_t) + 0.3 * np.sin(4 * np.pi * b_freqs * beat_t))
            lead_wave = lead_env * (0.6 * np.sin(2 * np.pi * l_freqs * lead_t) + 0.4 * np.sin(6 * np.pi * l_freqs * lead_t))

            audio = 0.65 * bass_wave + 0.35 * lead_wave
            audio_int16 = (audio * 12500).astype(np.int16)

            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_int16.tobytes())
        except Exception:
            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                for i in range(total_samples):
                    t = i / sample_rate
                    beat_sub = beat_duration * 0.5
                    b_idx = int(t / beat_sub) % len(bass_notes)
                    l_idx = int(t / (beat_sub * 0.5)) % len(lead_notes)
                    beat_t = t % beat_sub
                    lead_t = t % (beat_sub * 0.5)

                    b_freq = bass_notes[b_idx]
                    l_freq = lead_notes[l_idx]

                    bass_env = math.exp(-beat_t * 7.0)
                    lead_env = math.exp(-lead_t * 12.0)

                    b_val = bass_env * (0.7 * math.sin(2 * math.pi * b_freq * beat_t) + 0.3 * math.sin(4 * math.pi * b_freq * beat_t))
                    l_val = lead_env * (0.6 * math.sin(2 * math.pi * l_freq * lead_t) + 0.4 * math.sin(6 * math.pi * l_freq * lead_t))

                    sample = int(12500 * (0.65 * b_val + 0.35 * l_val))
                    w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))
