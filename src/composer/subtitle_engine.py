"""Kinetic Subtitle Engine for generating high-retention, bouncing, word-by-word highlighted captions."""

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SubtitleEngine:
    """Generates ASS (Advanced SubStation Alpha) and SRT captions with kinetic active word highlighting."""

    def __init__(
        self,
        font_name: str = "Arial Black",
        font_size: int = 68,
        primary_color: str = "&H0000FFFF",  # Bright Yellow in BGR format (&H00BBGGRR)
        highlight_color: str = "&H0066FF00", # Neon Green
        outline_color: str = "&H00000000",   # Deep Black
        outline_width: int = 5,
        vertical_pos_pct: float = 0.52       # Centered vertically at 52% of height
    ):
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.highlight_color = highlight_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.vertical_pos_pct = vertical_pos_pct

    def generate_ass_file(
        self,
        word_timestamps: List[Dict[str, Any]],
        output_ass_path: str,
        video_width: int = 1080,
        video_height: int = 1920
    ) -> str:
        """Generates an .ass file with dynamic word highlight effects in the safe zone."""
        os.makedirs(os.path.dirname(output_ass_path), exist_ok=True)
        margin_v = int(video_height * (1.0 - self.vertical_pos_pct))

        ass_header = f"""[Script Info]
Title: Viral Kids Shorts Kinetic Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KineticDefault,{self.font_name},{self.font_size},{self.primary_color},&H000000FF,{self.outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{self.outline_width},2,2,40,40,{margin_v},1
Style: KineticHighlight,{self.font_name},{int(self.font_size * 1.08)},{self.highlight_color},&H000000FF,{self.outline_color},&H80000000,-1,0,0,0,105,105,1,0,1,{self.outline_width + 1},3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogues = []

        # Batch words into groups of 2-3 words for high readability on mobile screens
        chunk_size = 2
        for i in range(0, len(word_timestamps), chunk_size):
            chunk = word_timestamps[i : i + chunk_size]
            if not chunk:
                continue
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]

            # Format start and end times for ASS (H:MM:SS.CC)
            start_str = self._format_ass_time(chunk_start)
            end_str = self._format_ass_time(chunk_end)

            # Highlight each word dynamically as it is spoken
            for word_idx, active_word in enumerate(chunk):
                w_start = self._format_ass_time(active_word["start"])
                w_end = self._format_ass_time(active_word["end"])
                
                # Assemble text with active word glowing neon green
                styled_parts = []
                for idx, w in enumerate(chunk):
                    w_text = w["word"].upper()
                    if idx == word_idx:
                        styled_parts.append(f"{{\\c{self.highlight_color}\\fscx108\\fscy108}}{w_text}{{\\r}}")
                    else:
                        styled_parts.append(f"{{\\c{self.primary_color}}}{w_text}{{\\r}}")

                line_text = " ".join(styled_parts)
                dialogues.append(f"Dialogue: 0,{w_start},{w_end},KineticDefault,,0,0,0,,{line_text}")

        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(dialogues))

        return output_ass_path

    def _format_ass_time(self, seconds: float) -> str:
        """Converts float seconds to ASS timestamp format H:MM:SS.cs."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        if cs >= 100:
            cs = 99
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
