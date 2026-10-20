"""
Script and Storyboard Engine with Multi-Episode Library, Multi-Candidate Tournament Selection,
and 1:1 Audio-Visual Locking across all 4 YouTube Kids Niches.
"""

import os
import re
import json
import logging
import urllib.request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# Canonical 8-scene visual storyboard lock for Episode 1 (Flamingo Superpowers)
CANONICAL_SCENE_BLUEPRINT = [
    {
        "scene_id": 1,
        "topic_beat": "Flamingos drinking boiling hot volcanic geyser water without getting hurt",
        "required_keywords": ["boiling", "hot", "water", "drink", "sip", "geyser"],
        "character_expression": "shocked",
        "character_color_state": "pink",
        "visual_prop": "boiling_geyser",
        "visual_badge_text": "DRINKS 100°C WATER!",
        "sfx_cue": "whoosh",
    },
    {
        "scene_id": 2,
        "topic_beat": "Their tough armor-like beak and tongue acting like a heat shield to block boiling heat",
        "required_keywords": ["beak", "shield", "tough", "armor", "heat", "skin"],
        "character_expression": "happy_proud",
        "character_color_state": "pink",
        "visual_prop": "beak_shield",
        "visual_badge_text": "HEAT-PROOF BEAK SHIELD!",
        "sfx_cue": "ding",
    },
    {
        "scene_id": 3,
        "topic_beat": "Baby flamingos are not born pink at all—they hatch from eggs with fluffy gray feathers",
        "required_keywords": ["baby", "born", "hatch", "gray", "grey", "fluffy", "pink"],
        "character_expression": "baby_curious",
        "character_color_state": "gray_chick",
        "visual_prop": "hatching_egg",
        "visual_badge_text": "DAY 1: BORN GRAY!",
        "sfx_cue": "pop",
    },
    {
        "scene_id": 4,
        "topic_beat": "Flamingos scooping up thousands of glowing pink brine shrimp and red algae from the water",
        "required_keywords": ["shrimp", "algae", "eat", "munch", "scoop", "pink"],
        "character_expression": "eating_excited",
        "character_color_state": "gray_chick",
        "visual_prop": "glowing_shrimp",
        "visual_badge_text": "EATS PINK SHRIMP!",
        "sfx_cue": "pop",
    },
    {
        "scene_id": 5,
        "topic_beat": "The natural pink pigment inside the shrimp slowly dyeing their gray feathers bright neon pink",
        "required_keywords": ["pigment", "dye", "turn", "transform", "feather", "feathers", "pink", "neon"],
        "character_expression": "transforming_amazed",
        "character_color_state": "half_pink_transforming",
        "visual_prop": "color_transformation",
        "visual_badge_text": "GRAY TO NEON PINK!",
        "sfx_cue": "chime",
    },
    {
        "scene_id": 6,
        "topic_beat": "At night, flamingos falling fast asleep balanced on just ONE leg in freezing cold water",
        "required_keywords": ["sleep", "asleep", "one", "leg", "freezing", "ice", "icy", "water"],
        "character_expression": "sleeping_peaceful",
        "character_color_state": "pink",
        "visual_prop": "one_leg_ice",
        "visual_badge_text": "SLEEPS ON 1 LEG!",
        "sfx_cue": "chime",
    },
    {
        "scene_id": 7,
        "topic_beat": "A secret automatic bone lock clicks inside their ankle joint like gears so they never fall over",
        "required_keywords": ["lock", "bone", "ankle", "joint", "gear", "fall", "energy", "tired"],
        "character_expression": "xray_focused",
        "character_color_state": "pink",
        "visual_prop": "knee_gear_lock",
        "visual_badge_text": "AUTO-LOCK ANKLE GEAR!",
        "sfx_cue": "ding",
    },
    {
        "scene_id": 8,
        "topic_beat": "Asking the viewer which superpower they would pick: heat-proof beak or one-leg sleep lock",
        "required_keywords": ["which", "pick", "choose", "superpower", "beak", "leg", "sleep"],
        "character_expression": "waving_cheerful",
        "character_color_state": "pink",
        "visual_prop": "choice_card",
        "visual_badge_text": "PICK YOUR SUPERPOWER!",
        "sfx_cue": "whoosh",
    },
]

# Multi-Episode Storyboard Library across all 4 YouTube Kids Niches
EPISODE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "flamingo_superpowers": {
        "niche": "animal_secrets",
        "topic": "Flamingo Superpowers",
        "blueprint": CANONICAL_SCENE_BLUEPRINT,
        "candidates": [
            {
                "style_name": "Fast_Science_Adventure_Arc",
                "title": "3 Impossible Flamingo Superpowers That Are 100% REAL! 🦩🔥 #shorts",
                "description": "Can a bird really drink boiling water and sleep on one leg in ice? Meet Pip the Flamingo! #shorts #kids #animals #nature",
                "tags": ["shorts", "kidsshorts", "flamingos", "animals", "nature", "science"],
                "scene_narrations": [
                    "How can a flamingo sip boiling hot volcanic geyser water without burning its tongue?",
                    "Its tough beak acts like an armored heat shield that keeps boiling water completely safe!",
                    "And look at day one, baby flamingos actually hatch from eggs with fluffy gray feathers!",
                    "As they grow, they scoop up glowing pink brine shrimp and red algae from the lagoon!",
                    "The natural pink dye inside that shrimp food transforms their gray feathers bright neon pink!",
                    "At night, flamingos sleep standing on only one leg in freezing icy mountain water!",
                    "Their ankle joint has an automatic bone gear lock that holds them up using zero energy!",
                    "Which flamingo superpower would you choose, the boiling water beak or the one-leg sleep lock?"
                ]
            },
            {
                "style_name": "Ultimate_Nature_Mystery_Arc",
                "title": "The Flamingo Secret Nobody Tells You! 🦩⚡ #shorts",
                "description": "Discover why baby flamingos hatch gray, drink boiling geyser water, and lock their ankles to sleep! #shorts #kids #animals",
                "tags": ["shorts", "kidsshorts", "flamingos", "animals", "funfacts", "scienceforkids"],
                "scene_narrations": [
                    "Did you know a flamingo can drink boiling hot geyser water without feeling a thing?",
                    "Their tough leathery beak works like a heat shield that protects their mouth from boiling water!",
                    "Even more surprising, baby flamingos are never born pink, they hatch with fluffy gray feathers!",
                    "Every single day, they munch on thousands of tiny pink brine shrimp in the water!",
                    "Natural pink pigment from those shrimp slowly dyes every feather on their body neon pink!",
                    "When night falls, flamingos fall fast asleep balanced on one leg in freezing water!",
                    "Inside their ankle, a secret bone gear lock clicks shut so they never fall over!",
                    "Which superpower would you pick, the heat shield beak or the one-leg sleep lock?"
                ]
            }
        ]
    },
    "octopus_and_ocean_secrets": {
        "niche": "animal_secrets",
        "topic": "Octopus & Ocean Superpowers",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "Discovering that an octopus has THREE glowing hearts pumping inside its body",
                "required_keywords": ["three", "3", "hearts", "heart", "octopus", "pumping"],
                "character_expression": "shocked",
                "character_color_state": "coral",
                "visual_prop": "octopus_three_hearts",
                "visual_badge_text": "3 GLOWING HEARTS!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Octopus blood is bright neon blue instead of red to survive freezing deep ocean water",
                "required_keywords": ["blue", "blood", "ocean", "deep", "cold", "copper"],
                "character_expression": "happy_proud",
                "character_color_state": "coral",
                "visual_prop": "octopus_blue_blood",
                "visual_badge_text": "NEON BLUE BLOOD!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 3,
                "topic_beat": "Baby octopuses hatch from tiny floating eggs no bigger than a grain of rice",
                "required_keywords": ["baby", "hatch", "tiny", "egg", "eggs", "rice"],
                "character_expression": "baby_curious",
                "character_color_state": "coral",
                "visual_prop": "octopus_rice_eggs",
                "visual_badge_text": "RICE-SIZED EGGS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 4,
                "topic_beat": "Munching on glowing deep-sea shrimp and crabs using tentacles and a strong beak",
                "required_keywords": ["shrimp", "crab", "munch", "scoop", "eat", "glowing"],
                "character_expression": "eating_excited",
                "character_color_state": "coral",
                "visual_prop": "octopus_catching_shrimp",
                "visual_badge_text": "DEEP SEA SHRIMP!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 5,
                "topic_beat": "Instant color transformation cells letting them change color in 0.3 seconds",
                "required_keywords": ["color", "change", "transform", "camouflage", "neon"],
                "character_expression": "transforming_amazed",
                "character_color_state": "camouflage",
                "visual_prop": "octopus_03s_camouflage",
                "visual_badge_text": "0.3s COLOR SHIFT!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 6,
                "topic_beat": "Sleeping peacefully in deep ocean water while dreaming in bright glowing colors",
                "required_keywords": ["sleep", "dream", "icy", "freezing", "water", "night"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "coral",
                "visual_prop": "octopus_coral_sleep",
                "visual_badge_text": "COLOR DREAM SLEEP!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Nine mini brains—one central brain plus a smart nerve center inside each arm",
                "required_keywords": ["nine", "9", "brain", "brains", "arm", "smart", "gear"],
                "character_expression": "xray_focused",
                "character_color_state": "coral",
                "visual_prop": "octopus_nine_brains",
                "visual_badge_text": "9 SMART BRAINS!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would pick: 3 hearts or 0.3s color camouflage",
                "required_keywords": ["which", "pick", "choose", "superpower", "hearts", "color"],
                "character_expression": "waving_cheerful",
                "character_color_state": "coral",
                "visual_prop": "octopus_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Ocean_Superpower_Explorer",
                "title": "Why Octopuses Have 3 Hearts & 9 Brains! 🐙💙 #shorts",
                "description": "Discover why octopuses have 3 glowing hearts, neon blue blood, and 0.3-second color shift superpowers! #shorts #kids #animals #ocean",
                "tags": ["shorts", "kidsshorts", "octopus", "ocean", "animals", "scienceforkids"],
                "scene_narrations": [
                    "Did you know an octopus has three glowing hearts pumping inside its body at once?",
                    "Even crazier, its blood is bright neon blue to survive freezing deep ocean water!",
                    "On day one, baby octopuses hatch from tiny floating eggs the size of rice!",
                    "They grow fast by munching on glowing pink brine shrimp and deep sea crabs!",
                    "Special skin cells let them transform their color in less than half a second!",
                    "At night, they fall fast asleep in freezing water while dreaming in bright colors!",
                    "And inside their body, nine smart brains work together like interlocking golden gears!",
                    "Which superpower would you choose, three glowing hearts or instant color camouflage?"
                ]
            }
        ]
    },
    "cloud_and_lightning_science": {
        "niche": "science_curiosities",
        "topic": "Cloud & Weather Superpowers",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "How warm sun-heated lagoon water steams up at 100°C to form giant sky clouds",
                "required_keywords": ["warm", "hot", "water", "steam", "cloud", "sky"],
                "character_expression": "shocked",
                "character_color_state": "sky",
                "visual_prop": "cloud_steam_rise",
                "visual_badge_text": "STEAM BECOMES CLOUDS!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Earth's atmosphere acting like a giant protective shield holding a million pounds of cloud droplets",
                "required_keywords": ["shield", "air", "million", "float", "droplets", "safe"],
                "character_expression": "happy_proud",
                "character_color_state": "sky",
                "visual_prop": "cloud_atmosphere_shield",
                "visual_badge_text": "1,000,000 LB CLOUD!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 3,
                "topic_beat": "Every cloud starts as tiny invisible water droplets hatching in the sky turning fluffy gray and white",
                "required_keywords": ["tiny", "fluffy", "gray", "white", "cloud", "born"],
                "character_expression": "baby_curious",
                "character_color_state": "sky",
                "visual_prop": "cloud_micro_droplets",
                "visual_badge_text": "FLUFFY SKY DROPLETS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 4,
                "topic_beat": "Sunlight and ocean minerals scattering through water droplets",
                "required_keywords": ["sunlight", "pink", "glowing", "water", "lagoon", "scoop"],
                "character_expression": "eating_excited",
                "character_color_state": "sky",
                "visual_prop": "cloud_sunlight_scattering",
                "visual_badge_text": "SUNLIGHT & WATER!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 5,
                "topic_beat": "Sunset light transforming gray clouds into glowing neon pink and gold across the sky",
                "required_keywords": ["sunset", "transform", "gray", "neon", "pink", "sky"],
                "character_expression": "transforming_amazed",
                "character_color_state": "sky",
                "visual_prop": "cloud_sunset_neon",
                "visual_badge_text": "GRAY TO NEON SUNSET!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 6,
                "topic_beat": "High up at night, cloud droplets freeze into sparkling ice crystals in freezing air",
                "required_keywords": ["night", "freezing", "ice", "sleep", "leg", "cold"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "sky",
                "visual_prop": "cloud_ice_crystals",
                "visual_badge_text": "FREEZING SKY ICE!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Bumping ice crystals locking electric charges like gears to spark glowing lightning",
                "required_keywords": ["lock", "gear", "spark", "energy", "charge", "lightning"],
                "character_expression": "xray_focused",
                "character_color_state": "sky",
                "visual_prop": "lightning_electrostatic_lock",
                "visual_badge_text": "LIGHTNING CHARGE SPARK!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would pick: 1 million pound cloud or lightning spark",
                "required_keywords": ["which", "pick", "choose", "superpower", "cloud", "lightning"],
                "character_expression": "waving_cheerful",
                "character_color_state": "sky",
                "visual_prop": "cloud_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Sky_Science_Wonder_Arc",
                "title": "How Can A 1,000,000 Pound Cloud Float In The Sky? ☁️⚡ #shorts",
                "description": "Pip and Leo explain how steaming water rises into million-pound clouds, turns neon pink at sunset, and sparks lightning! #shorts #kids #science #weather",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "clouds", "weather", "funfacts"],
                "scene_narrations": [
                    "Did you know steaming hot water rises into the sky to build giant clouds?",
                    "Rising warm air acts like a giant shield that floats a million pounds of droplets!",
                    "Every cloud starts out as tiny droplets that puff up into fluffy gray shapes!",
                    "Down below, Pip scoops glowing pink shrimp while watching the clouds reflect the lagoon!",
                    "At sunset, golden light transforms those gray clouds into glowing neon pink cotton candy!",
                    "High up at night, the air is freezing cold while Pip sleeps on one leg!",
                    "Inside the cloud, ice crystals lock electric energy like gears to spark bright lightning!",
                    "Which superpower would you choose, floating on warm air or glowing neon pink?"
                ]
            }
        ]
    }
}

FORBIDDEN_META_PHRASES = [
    "read the following",
    "objective",
    "exercise",
    "storyteller voice",
    "lively pacing",
    "scene 1",
    "scene 2",
    "scene 3",
    "narration:",
    "voiceover:",
    "stage direction",
]


class CharacterSpec(BaseModel):
    name: str = Field(default="Pip the Flamingo", description="Recurring hero character name")
    species_or_role: str = Field(default="flamingo")
    face_traits: str = Field(
        default="Large glossy teal-blue expressive 3D eyes, cute curved pink-and-black smiling beak, soft rounded cheeks, signature tiny golden explorer star badge"
    )
    companion_name: str = Field(default="Leo the Explorer Kid")
    companion_traits: str = Field(
        default="7-year-old boy with warm tan skin, big brown Pixar eyes, curly dark brown hair, bright yellow explorer cap, teal hoodie"
    )


class Scene(BaseModel):
    scene_id: int
    duration_sec: float = Field(default=3.0)
    narration: str = Field(description="ONLY spoken words for this scene—zero instructions or labels")
    visual_prompt: str
    character_expression: str = Field(default="happy")
    character_color_state: str = Field(default="pink")
    visual_prop: str = Field(default="boiling_geyser")
    visual_badge_text: str = Field(default="100°C BOILING!")
    sfx_cue: str = Field(default="whoosh")


class ShortScript(BaseModel):
    title: str
    topic: str
    target_audience: str = "family_co_viewing"
    character_spec: CharacterSpec = Field(default_factory=CharacterSpec)
    description: str
    tags: List[str]
    hook_0_3s: str
    scenes: List[Scene]
    loop_ending: str
    estimated_duration_sec: float
    total_words: int
    tournament_report: Dict[str, Any] = Field(default_factory=dict)


def clean_spoken_narration(text: str) -> str:
    """Strips any accidental stage directions, scene labels, or prompt instructions so TTS ONLY speaks the story."""
    text = text.strip()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"^(scene\s*\d+|narration|voiceover|hook|script|objective|candidate\s*\w+)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^read the following.*?[:\n]+", "", text, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(text.split())


class ScriptEngine:
    """Generates and evaluates multiple candidate scripts across episode blueprints and picks the #1 highest-scoring script."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    def _select_episode_blueprint(self, topic: str, niche: str) -> Dict[str, Any]:
        """Selects the best matching episode blueprint or rotates to an unpublished episode automatically."""
        topic_lower = topic.lower()
        if "octopus" in topic_lower or "jellyfish" in topic_lower or "ocean" in topic_lower:
            return EPISODE_LIBRARY["octopus_and_ocean_secrets"]
        if "cloud" in topic_lower or "weather" in topic_lower or "lightning" in topic_lower:
            return EPISODE_LIBRARY["cloud_and_lightning_science"]

        # Check published history: if an episode has already had any candidate published, rotate to the next completely fresh episode!
        past_titles = self._get_published_titles()
        for ep_key, ep_data in EPISODE_LIBRARY.items():
            if not any(c["title"] in past_titles for c in ep_data["candidates"]):
                logger.info(f"Auto-selecting fresh unpublished episode blueprint: '{ep_key}' ({ep_data['topic']})")
                return ep_data

        # Second pass: find any episode that still has at least one unpublished candidate variation
        for ep_key, ep_data in EPISODE_LIBRARY.items():
            if any(c["title"] not in past_titles for c in ep_data["candidates"]):
                logger.info(f"Auto-rotating to episode blueprint with fresh candidate: '{ep_key}' ({ep_data['topic']})")
                return ep_data

        return EPISODE_LIBRARY["flamingo_superpowers"]

    def _get_published_titles(self) -> set:
        titles = set()
        for path in ("data/published_history.json", "data/history_scripts.json"):
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    for item in items:
                        if isinstance(item, dict) and item.get("title"):
                            titles.add(item["title"])
                except Exception:
                    pass
        if os.path.exists("data/channel_ledger.json"):
            try:
                with open("data/channel_ledger.json", "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                for ep in ledger.get("episodes", []):
                    if ep.get("title"):
                        titles.add(ep["title"])
            except Exception:
                pass
        return titles

    def generate_script(self, topic: str, formula: Dict[str, Any], niche: str = "animal_secrets") -> ShortScript:
        """
        Selects the active episode blueprint, generates/evaluates competing candidate scripts,
        scores each candidate on 5 viral & sync criteria, and returns the highest-scoring fresh script.
        """
        episode = self._select_episode_blueprint(topic, niche)
        active_blueprint = episode["blueprint"]
        active_topic = episode["topic"]
        candidates: List[Dict[str, Any]] = list(episode["candidates"])

        if self.api_key:
            ai_candidates = self._fetch_ai_studio_candidates_rest(active_topic, active_blueprint)
            if ai_candidates:
                candidates = ai_candidates + candidates

        scored_candidates = []
        for idx, cand in enumerate(candidates):
            cand_bp = cand.get("_blueprint", active_blueprint)
            cand_topic = cand.get("_topic", active_topic)
            score_breakdown = self._score_candidate(cand, cand_bp)
            scored_candidates.append({
                "candidate_id": cand.get("style_name", f"Candidate_{idx+1}"),
                "title": cand.get("title", "3 Impossible Animal Superpowers! 🦩✨ #shorts"),
                "topic": cand_topic,
                "blueprint": cand_bp,
                "total_score": score_breakdown["total_score"],
                "metrics": score_breakdown,
                "scene_narrations": [clean_spoken_narration(n) for n in cand.get("scene_narrations", [])],
                "description": cand.get("description", ""),
                "tags": cand.get("tags", ["shorts", "kidsshorts", "animals", "scienceforkids"])
            })

        scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        winner = scored_candidates[0]
        winner_bp = winner["blueprint"]

        logger.info(
            f"Daily Script Tournament Winner: '{winner['candidate_id']}' - '{winner['title']}' "
            f"(Score: {winner['total_score']}/100 across {len(scored_candidates)} candidates)"
        )

        scenes = []
        for idx, bp in enumerate(winner_bp):
            narr = winner["scene_narrations"][idx]
            scenes.append(
                Scene(
                    scene_id=bp["scene_id"],
                    duration_sec=3.2,
                    narration=narr,
                    visual_prompt=f"Vertical 9:16 3D Pixar animation of Pip the Flamingo: {bp['topic_beat']}",
                    character_expression=bp["character_expression"],
                    character_color_state=bp["character_color_state"],
                    visual_prop=bp["visual_prop"],
                    visual_badge_text=bp["visual_badge_text"],
                    sfx_cue=bp["sfx_cue"]
                )
            )

        total_words = sum(len(s.narration.split()) for s in scenes)
        total_dur = sum(s.duration_sec for s in scenes)

        tournament_report = {
            "winner_id": winner["candidate_id"],
            "winner_title": winner["title"],
            "winner_score": winner["total_score"],
            "candidates_evaluated": len(scored_candidates),
            "leaderboard": [
                {
                    "candidate_id": c["candidate_id"],
                    "title": c["title"],
                    "total_score": c["total_score"],
                    "metrics": c["metrics"]
                }
                for c in scored_candidates[:4]
            ]
        }

        return ShortScript(
            title=winner["title"],
            topic=winner["topic"],
            target_audience="family_co_viewing",
            character_spec=CharacterSpec(),
            description=winner["description"],
            tags=winner["tags"],
            hook_0_3s=scenes[0].narration,
            scenes=scenes,
            loop_ending=scenes[-1].narration,
            estimated_duration_sec=round(total_dur, 1),
            total_words=total_words,
            tournament_report=tournament_report
        )

    def _fetch_ai_studio_candidates_rest(
        self,
        topic: str,
        blueprint: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calls Google AI Studio `gemini-3.8-flash` via fast HTTPS REST to generate competing script candidates."""
        blueprint_desc = "\n".join(
            f"Scene {b['scene_id']}: MUST explain '{b['topic_beat']}' using at least one of {b['required_keywords']} "
            f"(Matches visual_prop='{b['visual_prop']}', badge='{b['visual_badge_text']}')"
            for b in blueprint
        )
        prompt = f"""
        Generate 1 high-retention candidate script for a kids YouTube Short about '{topic}'.
        STRICT RULES:
        1. Must have exactly 8 `scene_narrations` (10 to 14 words per scene).
        2. NEVER include any instructions, objectives, stage directions, or labels like "Scene 1:" in `scene_narrations`.
        3. Every scene narration MUST strictly match its visual prop in order:
        {blueprint_desc}

        Respond ONLY with valid JSON matching this schema:
        {{
            "candidates": [
                {{
                    "style_name": "AI_Studio_Fresh_Arc",
                    "title": "Unique High-CTR Title about {topic} #shorts",
                    "description": "Engaging 2-sentence kid-friendly description #shorts #kids #animals #science",
                    "tags": ["shorts", "kidsshorts", "animals", "scienceforkids"],
                    "scene_narrations": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
                }}
            ]
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
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text_out = (
                    res_data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}")
                )
                parsed = json.loads(text_out.strip())
                cands = parsed.get("candidates", [])
                valid = []
                for c in cands:
                    if len(c.get("scene_narrations", [])) == 8:
                        c["_blueprint"] = blueprint
                        c["_topic"] = topic
                        valid.append(c)
                return valid
        except Exception as e:
            logger.debug(f"AI Studio REST script candidate generation note: {e}")
            return []

    def _score_candidate(self, candidate: Dict[str, Any], blueprint: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scores a candidate 8-scene script on 5 strict metrics + originality against published history."""
        narrations = [clean_spoken_narration(n) for n in candidate.get("scene_narrations", [])]
        if len(narrations) != 8:
            return {"total_score": 0, "reason": "Must have exactly 8 scenes"}

        # 1. Hook Curiosity Power (25 pts)
        hook_lower = narrations[0].lower()
        hook_score = 18
        if "?" in narrations[0] or "!" in narrations[0]:
            hook_score += 4
        if any(w in hook_lower for w in ["did you know", "boiling", "secret", "superpower", "how", "why", "three", "hearts"]):
            hook_score += 3
        hook_score = min(hook_score, 25)

        # 2. 1:1 Audio-Visual Prop Alignment (25 pts)
        matched_scenes = 0
        for idx, bp in enumerate(blueprint):
            s_lower = narrations[idx].lower()
            if any(kw in s_lower for kw in bp["required_keywords"]):
                matched_scenes += 1
        av_sync_score = round((matched_scenes / 8.0) * 25, 1)

        # 3. Zero Meta/Objective Contamination (20 pts)
        full_lower = " ".join(narrations).lower()
        has_forbidden = any(fp in full_lower for fp in FORBIDDEN_META_PHRASES)
        clean_score = 0 if has_forbidden else 20

        # 4. Pacing & Word Density (15 pts: ideal 8–17 words per scene)
        good_length_scenes = sum(1 for n in narrations if 8 <= len(n.split()) <= 17)
        pacing_score = round((good_length_scenes / 8.0) * 15, 1)

        # 5. Interactive Loop Payoff & Originality vs History (15 pts)
        ending_lower = narrations[-1].lower()
        loop_score = 10
        if "?" in narrations[-1]:
            loop_score += 3
        if any(w in ending_lower for w in ["which", "pick", "choose", "you"]):
            loop_score += 2
        loop_score = min(loop_score, 15)

        # Deduct points if this exact title or script was already published in history
        history_penalty = 0.0
        past_titles = self._get_published_titles()
        if candidate.get("title", "") in past_titles:
            history_penalty = 15.0

        total = round(max(hook_score + av_sync_score + clean_score + pacing_score + loop_score - history_penalty, 0.0), 1)
        return {
            "total_score": total,
            "hook_curiosity_score_25": hook_score,
            "audio_visual_alignment_25": av_sync_score,
            "zero_meta_contamination_20": clean_score,
            "pacing_density_15": pacing_score,
            "interactive_loop_15": loop_score,
            "history_duplicate_penalty": history_penalty,
            "scenes_matched_8": matched_scenes
        }
