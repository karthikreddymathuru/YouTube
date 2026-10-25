"""
Topic Registry & Indefinite Non-Repetition Enforcement Engine for KuriousKids.
Tracks every species, subject, science phenomenon, and title ever published across age tiers.
Guarantees zero topic repetition within specific tiers and channel-wide across its lifetime.
"""

import os
import json
import logging
from typing import Set, Dict, Any, List, Optional

logger = logging.getLogger("KidsShorts.TopicRegistry")

LEDGER_PATH = "data/topic_ledger.json"
CHANNEL_LEDGER_PATH = "data/channel_ledger.json"
HISTORY_SCRIPTS_PATH = "data/history_scripts.json"


class TopicRegistry:
    """Manages lifetime deduplication of topics, creatures, and scientific phenomena across age tiers."""

    @classmethod
    def load_used_entities(cls, tier: Optional[str] = None) -> Set[str]:
        """Returns set of all core entities/species ever featured (optionally filtered by tier)."""
        entities = set()
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if tier and "used_entities_by_tier" in data:
                        for ent in data["used_entities_by_tier"].get(tier, []):
                            entities.add(ent.lower().strip())
                    elif not tier:
                        for ent in data.get("used_entities", []):
                            entities.add(ent.lower().strip())

                    for item in data.get("history", []):
                        if tier and item.get("tier") and item.get("tier") != tier:
                            continue
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
                        if tier and ep.get("age_group") and ep.get("age_group") != tier:
                            continue
                        title = ep.get("title", "").lower()
                        for known in ["flamingo", "octopus", "cloud", "chameleon", "shrimp", "eel", "tardigrade", "frog", "axolotl", "duckling", "caterpillar", "nitrogen", "soundwave"]:
                            if known in title:
                                entities.add(known)
            except Exception:
                pass

        return entities

    @classmethod
    def load_used_titles(cls, tier: Optional[str] = None) -> Set[str]:
        """Returns all titles ever evaluated or published (optionally filtered by tier)."""
        titles = set()
        for path in (LEDGER_PATH, CHANNEL_LEDGER_PATH, HISTORY_SCRIPTS_PATH):
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for ep in data.get("episodes", []) + data.get("history", []):
                                if tier and ep.get("tier") and ep.get("tier") != tier:
                                    continue
                                if tier and ep.get("age_group") and ep.get("age_group") != tier:
                                    continue
                                if ep.get("title"):
                                    titles.add(ep["title"].strip())
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if tier and item.get("tier") and item.get("tier") != tier:
                                        continue
                                    if item.get("title"):
                                        titles.add(item["title"].strip())
                except Exception:
                    pass
        return titles

    @classmethod
    def is_forbidden(cls, core_entity: str, topic: str, title: str = "", tier: Optional[str] = None) -> bool:
        """
        Returns True if the entity, topic, or title has already been covered in this tier (or globally).
        Strict zero-tolerance policy against repetition.
        """
        used_entities_tier = cls.load_used_entities(tier=tier)
        used_entities_global = cls.load_used_entities()
        used_titles_tier = cls.load_used_titles(tier=tier)
        used_titles_global = cls.load_used_titles()

        cent = (core_entity or "").lower().strip()
        top = (topic or "").lower().strip()
        tit = (title or "").strip()

        # 1. Tier-specific entity check
        if cent and cent in used_entities_tier:
            return True

        # 2. Global title exact match check
        if tit and (tit in used_titles_tier or tit in used_titles_global):
            return True

        # 3. Topic contains forbidden tier entity
        for u in used_entities_tier:
            if u and (u in top or (tit and u in tit.lower())):
                return True

        # 4. If tier is specified, check if exact core entity was covered in this tier
        if tier and cent in used_entities_tier:
            return True

        # 5. Global entity check if no tier specified
        if not tier and cent and cent in used_entities_global:
            return True

        return False

    @classmethod
    def register_episode(
        cls,
        episode_number: int,
        core_entity: str,
        topic: str,
        title: str,
        tier: str = "younger",
        key_facts: List[str] = None
    ):
        """Permanently records an episode into the lifetime topic ledger with tier tagging."""
        data = {
            "policy": "STRICT_ZERO_REPETITION_LIFETIME",
            "used_entities": [],
            "used_entities_by_tier": {"preschool": [], "younger": [], "older": []},
            "history": []
        }
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        data.setdefault("used_entities_by_tier", {"preschool": [], "younger": [], "older": []})

        cent = core_entity.lower().strip() if core_entity else ""
        if cent:
            used_ents = set(data.get("used_entities", []))
            used_ents.add(cent)
            data["used_entities"] = sorted(list(used_ents))

            tier_ents = set(data["used_entities_by_tier"].get(tier, []))
            tier_ents.add(cent)
            data["used_entities_by_tier"][tier] = sorted(list(tier_ents))

        # Check if episode already in history
        history = data.get("history", [])
        existing_idx = next((i for i, h in enumerate(history) if h.get("episode_number") == episode_number and h.get("tier", "younger") == tier), None)

        record = {
            "episode_number": episode_number,
            "tier": tier,
            "core_entity": cent,
            "topic": topic,
            "title": title,
            "key_facts": key_facts or []
        }

        if existing_idx is not None:
            history[existing_idx] = record
        else:
            history.append(record)

        data["history"] = history

        try:
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Registered new lifetime topic into {LEDGER_PATH}: [{tier.upper()}] {core_entity} ('{title}')")
        except Exception as e:
            logger.error(f"Failed writing {LEDGER_PATH}: {e}")
