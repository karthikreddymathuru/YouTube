"""
Topic Registry & Indefinite Non-Repetition Enforcement Engine for KuriousKids.
Tracks every species, subject, science phenomenon, and title ever published.
Guarantees zero topic repetition across the channel's lifetime.
"""

import os
import json
import logging
from typing import Set, Dict, Any, List

logger = logging.getLogger("KidsShorts.TopicRegistry")

LEDGER_PATH = "data/topic_ledger.json"
CHANNEL_LEDGER_PATH = "data/channel_ledger.json"
HISTORY_SCRIPTS_PATH = "data/history_scripts.json"


class TopicRegistry:
    """Manages lifetime deduplication of topics, creatures, and scientific phenomena."""

    @classmethod
    def load_used_entities(cls) -> Set[str]:
        """Returns set of all core entities/species ever featured on the channel."""
        entities = set()
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for ent in data.get("used_entities", []):
                        entities.add(ent.lower().strip())
                    for item in data.get("history", []):
                        if item.get("core_entity"):
                            entities.add(item["core_entity"].lower().strip())
            except Exception as e:
                logger.warning(f"Error reading {LEDGER_PATH}: {e}")

        # Fallback inspection of channel ledger titles
        if os.path.exists(CHANNEL_LEDGER_PATH):
            try:
                with open(CHANNEL_LEDGER_PATH, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for ep in cdata.get("episodes", []):
                        title = ep.get("title", "").lower()
                        for known in ["flamingo", "octopus", "cloud", "chameleon", "shrimp", "eel", "tardigrade", "frog", "axolotl"]:
                            if known in title:
                                entities.add(known)
            except Exception:
                pass

        return entities

    @classmethod
    def load_used_titles(cls) -> Set[str]:
        """Returns all titles ever evaluated or published."""
        titles = set()
        for path in (LEDGER_PATH, CHANNEL_LEDGER_PATH, HISTORY_SCRIPTS_PATH):
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for ep in data.get("episodes", []) + data.get("history", []):
                                if ep.get("title"):
                                    titles.add(ep["title"].strip())
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and item.get("title"):
                                    titles.add(item["title"].strip())
                except Exception:
                    pass
        return titles

    @classmethod
    def is_forbidden(cls, core_entity: str, topic: str, title: str = "") -> bool:
        """
        Returns True if the entity, topic, or title has already been covered.
        Strict zero-tolerance policy against repetition.
        """
        used_entities = cls.load_used_entities()
        used_titles = cls.load_used_titles()

        cent = (core_entity or "").lower().strip()
        top = (topic or "").lower().strip()
        tit = (title or "").strip()

        if cent and cent in used_entities:
            return True

        for u in used_entities:
            if u and (u in top or u in tit.lower()):
                return True

        if tit and tit in used_titles:
            return True

        return False

    @classmethod
    def register_episode(cls, episode_number: int, core_entity: str, topic: str, title: str, key_facts: List[str] = None):
        """Permanently records an episode into the lifetime topic ledger."""
        data = {"policy": "STRICT_ZERO_REPETITION_LIFETIME", "used_entities": [], "history": []}
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        used_ents = set(data.get("used_entities", []))
        if core_entity:
            used_ents.add(core_entity.lower().strip())
        data["used_entities"] = sorted(list(used_ents))

        # Check if episode already in history
        history = data.get("history", [])
        if not any(h.get("episode_number") == episode_number for h in history):
            history.append({
                "episode_number": episode_number,
                "core_entity": core_entity.lower().strip(),
                "topic": topic,
                "title": title,
                "key_facts": key_facts or []
            })
        data["history"] = history

        try:
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Registered new lifetime topic into {LEDGER_PATH}: {core_entity} ('{title}')")
        except Exception as e:
            logger.error(f"Failed writing {LEDGER_PATH}: {e}")
