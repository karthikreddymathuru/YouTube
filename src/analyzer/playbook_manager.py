"""Playbook Manager for maintaining and serving the persistent Viral Playbook knowledge base."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PlaybookManager:
    """Manages the persistent knowledge base of viral kids Shorts retention mechanisms."""

    def __init__(self, playbook_path: str = "data/viral_playbook.json"):
        self.playbook_path = playbook_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads the viral playbook from disk or initializes defaults."""
        if os.path.exists(self.playbook_path):
            try:
                with open(self.playbook_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load playbook from {self.playbook_path}: {e}")
        return self._default_playbook()

    def save(self):
        """Persists the playbook to disk."""
        os.makedirs(os.path.dirname(self.playbook_path), exist_ok=True)
        with open(self.playbook_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_winning_formula(self, niche: str = "animal_secrets") -> Dict[str, Any]:
        """Returns the highest scoring viral archetype and retention benchmarks for a niche."""
        archetypes = self.data.get("hook_archetypes", [])
        if not archetypes:
            archetypes = self._default_playbook()["hook_archetypes"]

        # Sort by engagement score descending
        sorted_archetypes = sorted(archetypes, key=lambda x: x.get("engagement_score", 0), reverse=True)
        chosen = sorted_archetypes[0]

        benchmarks = self.data.get("benchmarks", {
            "target_wpm": 150,
            "target_scene_duration_sec": 2.5,
            "ideal_total_duration_sec": 38,
            "target_loop_retention_pct": 115
        })

        return {
            "archetype": chosen,
            "benchmarks": benchmarks,
            "pacing": self.data.get("pacing_rules", {}),
            "loops": self.data.get("loop_archetypes", [])
        }

    def record_learning(self, insight: str, source: str = "Daily Trend Scout"):
        """Records a new learned pattern into the knowledge base."""
        learnings = self.data.setdefault("learned_insights", [])
        entry = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "insight": insight,
            "source": source
        }
        learnings.append(entry)
        self.data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        self.save()

    def update_from_trend_analysis(self, trend_results: List[Dict[str, Any]]):
        """Integrates newly extracted viral mechanics into the playbook."""
        current_archetypes = {a["id"]: a for a in self.data.get("hook_archetypes", [])}
        for item in trend_results:
            arch_id = item.get("hook_archetype", "custom_trend")
            score = item.get("estimated_retention_score", 90)
            if arch_id in current_archetypes:
                # Update rolling average score
                old_score = current_archetypes[arch_id].get("engagement_score", score)
                current_archetypes[arch_id]["engagement_score"] = int((old_score * 0.7) + (score * 0.3))
            else:
                current_archetypes[arch_id] = {
                    "id": arch_id,
                    "formula": item.get("hook_text", ""),
                    "example": item.get("hook_text", ""),
                    "engagement_score": score,
                    "sfx_cue": item.get("sfx_cues", ["whoosh"])[0] if item.get("sfx_cues") else "whoosh"
                }

        self.data["hook_archetypes"] = list(current_archetypes.values())
        self.save()

    def _default_playbook(self) -> Dict[str, Any]:
        return {
            "version": "1.0.0",
            "benchmarks": {
                "target_wpm": 150,
                "target_scene_duration_sec": 2.5,
                "ideal_total_duration_sec": 38,
                "target_loop_retention_pct": 115
            },
            "hook_archetypes": [
                {
                    "id": "interactive_5s_countdown",
                    "formula": "Can you spot the odd one out in 5 seconds?",
                    "example": "Which puppy is NOT real? 3... 2... 1...",
                    "engagement_score": 98,
                    "sfx_cue": "countdown_tick"
                },
                {
                    "id": "visual_disbelief_challenge",
                    "formula": "Did you know that [wild animal] can actually [impossible-sounding ability]?",
                    "example": "Did you know that octopuses have THREE hearts and blue blood?",
                    "engagement_score": 96,
                    "sfx_cue": "whoosh_pop"
                }
            ],
            "pacing_rules": {
                "visual_change_interval_sec": 2.4,
                "subtitles_words_per_batch": 2,
                "audio_dead_air_max_sec": 0.1
            },
            "loop_archetypes": [
                {
                    "id": "seamless_sentence_link",
                    "template_ending": "And if you think that's crazy, wait until you hear...",
                    "template_start": "...what scientists just discovered about..."
                }
            ],
            "learned_insights": []
        }
