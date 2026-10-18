"""Pattern Extractor for deconstructing trending kids Shorts into actionable viral formulas via fast Google AI Studio REST."""

import os
import json
import logging
import urllib.request
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PatternExtractor:
    """Uses Google AI Studio (gemini-3.8-flash REST) to deconstruct trending kids Shorts into high-retention archetypes."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    def extract_formula(self, title: str, description: str = "", transcript: str = "") -> Dict[str, Any]:
        """Analyzes a trending Short and returns structured viral mechanics with fast timeout."""
        if self.api_key:
            try:
                prompt = f"""
                You are a world-class viral YouTube Shorts retention analyst specializing in kids & family edutainment.
                Deconstruct this trending Short:
                Title: {title}
                Description: {description}
                Transcript: {transcript}

                Analyze and return ONLY a valid JSON object with these exact keys:
                {{
                    "hook_archetype": "string (e.g., visual_disbelief, interactive_countdown, curious_secret)",
                    "hook_text": "string (the exact or inferred 0-3 second opening hook)",
                    "estimated_retention_score": 96,
                    "pacing_seconds_per_scene": 2.5,
                    "retention_mechanic": "string (why kids watch to the end)",
                    "loop_strategy": "string (how the ending connects back to the start)",
                    "sfx_cues": ["whoosh", "ding"],
                    "visual_style": "string (e.g. 3D cartoon, bright neon vector, cinematic macro)"
                }}
                """
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={self.api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    text_out = (
                        res_data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "{}")
                    )
                    return json.loads(text_out.strip())
            except Exception as e:
                logger.warning(f"Gemini REST pattern extraction note: {e}. Using heuristic pattern extractor.")

        return self._heuristic_extraction(title, transcript)

    def _heuristic_extraction(self, title: str, transcript: str) -> Dict[str, Any]:
        """Extracts patterns using deterministic NLP rules when offline."""
        title_lower = title.lower()
        if "spot" in title_lower or "5s" in title_lower or "riddle" in title_lower or "guess" in title_lower:
            return {
                "hook_archetype": "interactive_countdown",
                "hook_text": f"Can you solve this in 5 seconds? {title}!",
                "estimated_retention_score": 98,
                "pacing_seconds_per_scene": 2.2,
                "retention_mechanic": "Active participation with countdown ticking sound",
                "loop_strategy": "Challenge viewer to watch again to see what they missed",
                "sfx_cues": ["countdown_tick", "bell_ding"],
                "visual_style": "vibrant 3D Pixar cartoon with bright yellow highlights"
            }
        elif "why" in title_lower or "what happens" in title_lower or "secret" in title_lower:
            return {
                "hook_archetype": "visual_disbelief_challenge",
                "hook_text": f"Wait till you find out: {title}!",
                "estimated_retention_score": 95,
                "pacing_seconds_per_scene": 2.5,
                "retention_mechanic": "Curiosity gap resolved with unexpected twist",
                "loop_strategy": "Last sentence grammatically leads into the opening question",
                "sfx_cues": ["whoosh", "pop", "chime"],
                "visual_style": "colorful 3D animated character, high contrast"
            }
        else:
            return {
                "hook_archetype": "cute_hero_in_trouble",
                "hook_text": f"You won't believe what happened in: {title}!",
                "estimated_retention_score": 93,
                "pacing_seconds_per_scene": 2.4,
                "retention_mechanic": "Emotional stakes and heartwarming payoff",
                "loop_strategy": "Seamless sentence continuation",
                "sfx_cues": ["whoosh", "ding"],
                "visual_style": "charming animated Disney/Pixar aesthetic"
            }
