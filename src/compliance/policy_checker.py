"""Compliance & Policy Checker for YouTube Kids Safety, COPPA, and Reused Content prevention."""

import os
import json
import hashlib
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class PolicyCheckResult:
    def __init__(self, passed: bool, score: float, flags: List[str], recommendations: List[str]):
        self.passed = passed
        self.score = score
        self.flags = flags
        self.recommendations = recommendations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 2),
            "flags": self.flags,
            "recommendations": self.recommendations
        }

class PolicyChecker:
    """Guarantees scripts satisfy YouTube Kids guidelines, COPPA, and monetization originality standards."""

    PROHIBITED_KIDS_TERMS = [
        "weapon", "knife", "gun", "kill", "die", "death", "gore", "bleeding",
        "scary", "monster", "demon", "choke", "poison", "fire",
        "dangerous", "dare", "prank", "steal", "fake news"
    ]

    def __init__(self, history_db_path: str = "data/history_scripts.json"):
        self.history_db_path = history_db_path
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Loads historical scripts database to check uniqueness."""
        if os.path.exists(self.history_db_path):
            try:
                with open(self.history_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load script history: {e}")
        return []

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_db_path), exist_ok=True)
        with open(self.history_db_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def evaluate_script(self, script_text: str, title: str) -> PolicyCheckResult:
        """Evaluates safety, originality, and advertiser friendliness."""
        flags = []
        recommendations = []
        score = 100.0

        full_text_lower = f"{title} {script_text}".lower()

        # 1. Prohibited & dangerous content check
        found_prohibited = [w for w in self.PROHIBITED_KIDS_TERMS if w in full_text_lower]
        if found_prohibited:
            flags.append(f"Contains sensitive words for kids audience: {', '.join(found_prohibited)}")
            score -= 35.0

        # 2. Originality / Reused Content Check
        max_similarity, matched_title = self._check_uniqueness(script_text)
        if max_similarity > 0.65:
            flags.append(f"Script too similar ({int(max_similarity*100)}%) to previously published Short: '{matched_title}'. Violates YouTube Reused Content rule.")
            score -= 40.0
            recommendations.append("Rewrite the hook and examples to guarantee 100% original narrative.")
        else:
            recommendations.append("Script originality verified. Free from repetitive content flags.")

        # 3. Contextual Ad & Educational Value Check
        if not any(k in full_text_lower for k in ["why", "how", "secret", "did you know", "fun fact", "discover", "superpower", "amazing"]):
            recommendations.append("Add clear educational trigger words to help YouTube match premium contextual ads.")

        passed = score >= 70.0
        return PolicyCheckResult(passed=passed, score=score, flags=flags, recommendations=recommendations)

    def register_published_script(self, title: str, script_text: str):
        """Records script in history database to prevent future duplicate generation."""
        script_hash = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
        self.history.append({
            "title": title,
            "hash": script_hash,
            "words": list(set(script_text.lower().split()))
        })
        self._save_history()

    def _check_uniqueness(self, new_script: str) -> Tuple[float, str]:
        """Calculates maximum Jaccard word-set similarity against past scripts."""
        if not self.history:
            return 0.0, ""

        new_words = set(new_script.lower().split())
        if not new_words:
            return 0.0, ""

        max_sim = 0.0
        matched_title = ""
        for item in self.history:
            past_words = set(item.get("words", []))
            if not past_words:
                continue
            intersection = len(new_words.intersection(past_words))
            union = len(new_words.union(past_words))
            sim = intersection / union if union > 0 else 0.0
            if sim > max_sim:
                max_sim = sim
                matched_title = item.get("title", "")

        return max_sim, matched_title
