"""Kinetic Subtitle Engine for generating high-retention, bouncing, word-by-word highlighted captions."""

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SubtitleEngine:
    """Generates ASS (Advanced SubStation Alpha) and SRT captions with kinetic active word highlighting."""

    def __init__(
        self,
        font_name: str = "Arial",
        font_size: int = 65,
        primary_color: str = "&H0000FFFF",  # Bright Yellow in BGR format (&H00BBGGRR)
        highlight_color: str = "&H00FFFFFF", # Bright White
        outline_color: str = "&H00000000",   # Deep Black
        outline_width: int = 6,
        vertical_pos_pct: float = 0.75       # Safe lower-middle: ~480px from bottom (clears characters and bottom badges)
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
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
        output_ass_path: str = "subtitles.ass",
        video_width: int = 1080,
        video_height: int = 1920,
        scene_narrations: Optional[List[str]] = None,
        scene_durations: Optional[List[float]] = None
    ) -> str:
        """Generates an .ass file with dynamic word highlight effects or synchronized nursery rhyme couplets."""
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

        # 1. If scene narrations and durations are provided, build rhythmic singalong couplets
        if scene_narrations and scene_durations:
            curr_time = 0.0
            for text, dur in zip(scene_narrations, scene_durations):
                lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
                # Strip out trailing ellipses for clean visual text
                clean_lines = [l.replace("...", "").strip() for l in lines]
                scene_start = curr_time
                scene_end = curr_time + dur

                if len(clean_lines) >= 2:
                    mid_time = scene_start + (dur * 0.50)
                    start_str1 = self._format_ass_time(scene_start)
                    mid_str = self._format_ass_time(mid_time)
                    end_str = self._format_ass_time(scene_end)

                    # Part 1: Line 1 highlighted in bright yellow, Line 2 in white
                    t1 = f"{{\\c{self.primary_color}}}{clean_lines[0]}{{\\r}}\\N{{\\c{self.highlight_color}}}{clean_lines[1]}{{\\r}}"
                    dialogues.append(f"Dialogue: 0,{start_str1},{mid_str},KineticDefault,,0,0,0,,{t1}")

                    # Part 2: Line 1 in white, Line 2 highlighted in bright yellow
                    t2 = f"{{\\c{self.highlight_color}}}{clean_lines[0]}{{\\r}}\\N{{\\c{self.primary_color}}}{clean_lines[1]}{{\\r}}"
                    dialogues.append(f"Dialogue: 0,{mid_str},{end_str},KineticDefault,,0,0,0,,{t2}")
                elif clean_lines:
                    start_str = self._format_ass_time(scene_start)
                    end_str = self._format_ass_time(scene_end)
                    t = f"{{\\c{self.primary_color}}}{clean_lines[0]}{{\\r}}"
                    dialogues.append(f"Dialogue: 0,{start_str},{end_str},KineticDefault,,0,0,0,,{t}")

                curr_time += dur

        # 2. Fallback to word-level timestamps if no scene narrations or empty
        elif word_timestamps:
            chunk_size = 2
            for i in range(0, len(word_timestamps), chunk_size):
                chunk = word_timestamps[i : i + chunk_size]
                if not chunk:
                    continue
                chunk_start = chunk[0]["start"]
                chunk_end = chunk[-1]["end"]

                for word_idx, active_word in enumerate(chunk):
                    w_start = self._format_ass_time(active_word["start"])
                    w_end = self._format_ass_time(active_word["end"])
                    
                    styled_parts = []
                    for idx, w in enumerate(chunk):
                        w_text = w["word"].upper().replace("...", "")
                        if idx == word_idx:
                            styled_parts.append(f"{{\\c{self.primary_color}\\fscx108\\fscy108}}{w_text}{{\\r}}")
                        else:
                            styled_parts.append(f"{{\\c{self.highlight_color}}}{w_text}{{\\r}}")

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
