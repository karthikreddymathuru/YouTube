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
from src.compliance.topic_registry import TopicRegistry

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
        "core_entity": "flamingo",
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
        "core_entity": "octopus",
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
        "core_entity": "cloud",
        "niche": "science_curiosities",
        "topic": "Cloud & Weather Superpowers",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "How warm ocean water steams up into the sky to form giant clouds",
                "required_keywords": ["warm", "water", "steam", "cloud", "sky", "rises"],
                "character_expression": "shocked",
                "character_color_state": "sky",
                "visual_prop": "cloud_steam_rise",
                "visual_badge_text": "STEAM BECOMES CLOUDS!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Rising warm updrafts holding a million pounds of floating cloud droplets",
                "required_keywords": ["air", "million", "float", "droplets", "updraft", "weigh"],
                "character_expression": "happy_proud",
                "character_color_state": "sky",
                "visual_prop": "cloud_atmosphere_shield",
                "visual_badge_text": "1,000,000 LB CLOUD!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 3,
                "topic_beat": "Every cloud starts as billions of tiny invisible water droplets clustering together",
                "required_keywords": ["tiny", "fluffy", "droplets", "cloud", "billions", "air"],
                "character_expression": "baby_curious",
                "character_color_state": "sky",
                "visual_prop": "cloud_micro_droplets",
                "visual_badge_text": "BILLIONS OF DROPLETS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 4,
                "topic_beat": "Bright sunlight scattering through water droplets into dazzling white glow",
                "required_keywords": ["sunlight", "scatter", "white", "water", "light", "droplets"],
                "character_expression": "eating_excited",
                "character_color_state": "sky",
                "visual_prop": "cloud_sunlight_scattering",
                "visual_badge_text": "SCATTERING SUNLIGHT!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 5,
                "topic_beat": "Sunset light bending across the sky transforming white clouds into glowing neon pink and gold",
                "required_keywords": ["sunset", "transform", "neon", "pink", "gold", "sky"],
                "character_expression": "transforming_amazed",
                "character_color_state": "sky",
                "visual_prop": "cloud_sunset_neon",
                "visual_badge_text": "NEON SUNSET CLOUDS!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 6,
                "topic_beat": "High up in freezing air, cloud droplets freeze into sparkling ice crystals",
                "required_keywords": ["freezing", "ice", "crystals", "cold", "high", "droplets"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "sky",
                "visual_prop": "cloud_ice_crystals",
                "visual_badge_text": "FREEZING ICE CRYSTALS!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Bumping ice crystals generate static electricity like giant batteries to spark lightning",
                "required_keywords": ["bump", "spark", "energy", "charge", "lightning", "electric"],
                "character_expression": "xray_focused",
                "character_color_state": "sky",
                "visual_prop": "lightning_electrostatic_lock",
                "visual_badge_text": "LIGHTNING CHARGE SPARK!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would choose: floating a 1 million pound cloud or sparking bright lightning",
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
                "description": "Discover how steaming water rises to float 1,000,000-pound clouds, turns neon pink at sunset, and sparks electric lightning! #shorts #kids #science #weather",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "clouds", "weather", "lightning", "funfacts"],
                "scene_narrations": [
                    "Did you know warm ocean water steams up into the sky to create giant fluffy clouds?",
                    "A single fluffy cumulus cloud can weigh over one million pounds and still float!",
                    "That is because rising warm updrafts of air hold up billions of microscopic water droplets!",
                    "When bright sunlight hits those tiny water droplets, it scatters all colors into pure white!",
                    "At sunset, long light waves transform those white clouds into glowing neon pink and gold cotton candy!",
                    "High up where the air is freezing cold, water droplets turn into sparkling ice crystals!",
                    "When billions of ice crystals bump into each other, they build electric charge and spark lightning!",
                    "Which superpower would you choose, floating a million-pound cloud or sparking bright lightning?"
                ]
            }
        ]
    },
    "chameleon_crystal_color_science": {
        "core_entity": "chameleon",
        "niche": "animal_secrets",
        "topic": "Chameleon Color Superpowers",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "Chameleons do NOT change color to blend in with trees like camouflage",
                "required_keywords": ["chameleons", "color", "change", "not", "camouflage", "trees"],
                "character_expression": "shocked",
                "character_color_state": "green",
                "visual_prop": "chameleon_myth_busted",
                "visual_badge_text": "NOT FOR CAMOUFLAGE!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "They actually change color based on their mood and temperature like mood rings",
                "required_keywords": ["mood", "temperature", "color", "feel", "sunshine", "shift"],
                "character_expression": "happy_proud",
                "character_color_state": "green",
                "visual_prop": "chameleon_mood_ring_shift",
                "visual_badge_text": "MOOD & TEMPERATURE!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 3,
                "topic_beat": "Under their skin are millions of microscopic photonic crystals like tiny mirrors",
                "required_keywords": ["crystals", "skin", "mirrors", "microscopic", "tiny", "light"],
                "character_expression": "baby_curious",
                "character_color_state": "green",
                "visual_prop": "chameleon_nanocrystal_zoom",
                "visual_badge_text": "MICROSCOPIC CRYSTALS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 4,
                "topic_beat": "When relaxed, the crystals cluster tight together to reflect cool blue and emerald green",
                "required_keywords": ["relaxed", "crystals", "tight", "reflect", "green", "close"],
                "character_expression": "eating_excited",
                "character_color_state": "green",
                "visual_prop": "chameleon_relaxed_green",
                "visual_badge_text": "TIGHT CRYSTALS: GREEN!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 5,
                "topic_beat": "When excited or angry, skin stretches the crystals apart reflecting yellow and fiery red",
                "required_keywords": ["excited", "skin", "stretches", "crystals", "yellow", "red"],
                "character_expression": "transforming_amazed",
                "character_color_state": "fiery_red",
                "visual_prop": "chameleon_excited_neon_red",
                "visual_badge_text": "SPREAD CRYSTALS: RED!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 6,
                "topic_beat": "Their eyes are like 360-degree radar turrets looking in two different directions at once",
                "required_keywords": ["eyes", "swivel", "two", "different", "directions", "once"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "green",
                "visual_prop": "chameleon_radar_eyes",
                "visual_badge_text": "360° DUAL RADAR EYES!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Their spring-loaded tongue shoots out faster than a jet fighter at 60 miles per hour",
                "required_keywords": ["spring", "tongue", "shoots", "60", "miles", "jet"],
                "character_expression": "xray_focused",
                "character_color_state": "green",
                "visual_prop": "chameleon_ballistic_tongue",
                "visual_badge_text": "60 MPH ROCKET TONGUE!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking viewer which superpower they would choose: 360 degree dual eyes or 60 mph rocket tongue",
                "required_keywords": ["which", "pick", "choose", "superpower", "eyes", "tongue"],
                "character_expression": "waving_cheerful",
                "character_color_state": "green",
                "visual_prop": "chameleon_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Reptile_Superpower_Explorer",
                "title": "Why Chameleons ACTUALLY Change Color! 🦎🌈 #shorts",
                "description": "Discover why chameleons don't change color for camouflage, how nanocrystals bend light, and their 60 mph rocket tongue! #shorts #kids #animals #science",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "chameleon", "animals", "funfacts"],
                "scene_narrations": [
                    "Did you know chameleons do NOT change color to hide or camouflage in trees?",
                    "They actually shift colors to express their mood and warm up in the sunshine!",
                    "Under their skin are millions of tiny nanocrystals that act like microscopic mirrors!",
                    "When the chameleon is relaxed, crystals squeeze close together to reflect bright green!",
                    "When excited, its skin stretches the crystals apart to reflect fiery yellow and red!",
                    "Even crazier, their eyes can swivel in two completely different directions at once!",
                    "And their spring-loaded tongue shoots out at 60 miles per hour faster than a jet!",
                    "Which superpower would you choose, 360-degree dual radar eyes or a 60-mile-per-hour tongue?"
                ]
            }
        ]
    },
    "pistol_shrimp_plasma_blast": {
        "core_entity": "shrimp",
        "niche": "animal_secrets",
        "topic": "Pistol Shrimp Sonic Blast",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "The pistol shrimp possesses a giant claw that snaps shut faster than the blink of an eye",
                "required_keywords": ["pistol", "shrimp", "giant", "claw", "snaps", "blink"],
                "character_expression": "shocked",
                "character_color_state": "coral",
                "visual_prop": "shrimp_snap_claw",
                "visual_badge_text": "60 MPH SNAP CLAW!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "The snap shoots a high-speed jet of water creating a low-pressure cavitation bubble",
                "required_keywords": ["water", "shoots", "jet", "bubble", "speed", "pressure"],
                "character_expression": "happy_proud",
                "character_color_state": "coral",
                "visual_prop": "cavitation_plasma_bubble",
                "visual_badge_text": "CAVITATION BUBBLE!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "When the bubble collapses, it unleashes temperatures hotter than four thousand degrees",
                "required_keywords": ["collapses", "temperatures", "hot", "sun", "thousand", "heat"],
                "character_expression": "transforming_amazed",
                "character_color_state": "coral",
                "visual_prop": "heat_hotter_than_sun",
                "visual_badge_text": "4,700°C SUN HEAT!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 4,
                "topic_beat": "The collapse creates a tiny spark of light called sonoluminescence",
                "required_keywords": ["spark", "light", "glow", "tiny", "flash", "bright"],
                "character_expression": "eating_excited",
                "character_color_state": "coral",
                "visual_prop": "sonoluminescence_flash",
                "visual_badge_text": "LIGHT FROM SOUND!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 5,
                "topic_beat": "The deafening sonic shockwave instantly stuns passing crabs and small fish",
                "required_keywords": ["shockwave", "sonic", "stuns", "crabs", "fish", "instant"],
                "character_expression": "shocked",
                "character_color_state": "coral",
                "visual_prop": "shockwave_stun_fish",
                "visual_badge_text": "SONIC SHOCKWAVE!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "Colonies of pistol shrimp make so much noise they actually jam military submarine sonar",
                "required_keywords": ["colonies", "noise", "jam", "submarine", "sonar", "ocean"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "coral",
                "visual_prop": "submarine_sonar_jamming",
                "visual_badge_text": "JAMS SUBMARINE SONAR!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "If it ever loses its giant blaster claw, its smaller claw transforms into the new big gun",
                "required_keywords": ["loses", "claw", "transforms", "new", "grows", "magic"],
                "character_expression": "xray_focused",
                "character_color_state": "coral",
                "visual_prop": "claw_regeneration_swap",
                "visual_badge_text": "MAGIC CLAW SWAP!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would choose: sun-hot plasma blast or sonar-jamming claw",
                "required_keywords": ["which", "pick", "choose", "superpower", "plasma", "claw"],
                "character_expression": "waving_cheerful",
                "character_color_state": "coral",
                "visual_prop": "shrimp_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Plasma_Shockwave_Arc",
                "title": "The Shrimp That Shoots SUN-HOT Plasma! 🦐💥 #shorts",
                "description": "Discover how the tiny pistol shrimp snaps its claw to create plasma hotter than the sun! #shorts #animals #scienceforkids",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "pistolshrimp", "animals", "funfacts"],
                "scene_narrations": [
                    "Did you know a tiny shrimp in the ocean has a superpower stronger than a superhero?",
                    "Its giant claw snaps shut so fast it shoots a high-speed jet of water!",
                    "That snap creates a bubble that collapses with heat hotter than the sun!",
                    "The collapse even creates a flash of light making light out of sound!",
                    "The sonic shockwave travels through the water and stuns crabs instantly!",
                    "Pistol shrimp make so much crackling noise they actually jam submarine sonars!",
                    "And if it loses its snap claw, its other small claw transforms into the big one!",
                    "Which superpower would you choose, the sun-hot plasma blast or the sonic boom claw?"
                ]
            }
        ]
    },
    "axolotl_infinite_regeneration": {
        "core_entity": "axolotl",
        "niche": "science_curiosities",
        "topic": "Axolotl Infinite Regeneration",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "The axolotl is an adorable smiling creature known as the Mexican walking fish",
                "required_keywords": ["axolotl", "smiling", "walking", "fish", "mexico", "cute"],
                "character_expression": "happy_proud",
                "character_color_state": "pink",
                "visual_prop": "axolotl_smile_gills",
                "visual_badge_text": "SMILING WALKING FISH!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "If it loses an arm or leg, it can completely regrow the entire limb with zero scars",
                "required_keywords": ["loses", "limb", "regrow", "scars", "perfect", "weeks"],
                "character_expression": "transforming_amazed",
                "character_color_state": "pink",
                "visual_prop": "regrow_lost_limbs",
                "visual_badge_text": "REGROWS FULL LIMBS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "Even more incredible, it can regrow damaged parts of its heart and spinal cord",
                "required_keywords": ["heart", "spine", "spinal", "cord", "heals", "alive"],
                "character_expression": "shocked",
                "character_color_state": "pink",
                "visual_prop": "regrow_heart_spine",
                "visual_badge_text": "HEALS HEART & SPINE!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 4,
                "topic_beat": "Scientists discovered the axolotl can even regrow parts of its own brain",
                "required_keywords": ["brain", "regrows", "smart", "scientists", "miracle", "cells"],
                "character_expression": "eating_excited",
                "character_color_state": "pink",
                "visual_prop": "brain_regeneration_scan",
                "visual_badge_text": "REGROWS BRAIN CELLS!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 5,
                "topic_beat": "Its feathery pink external gills work like natural underwater oxygen filters",
                "required_keywords": ["feathery", "gills", "pink", "oxygen", "breathe", "water"],
                "character_expression": "happy_proud",
                "character_color_state": "pink",
                "visual_prop": "feathery_oxygen_gills",
                "visual_badge_text": "FEATHERY OXYGEN GILLS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "It stays in its adorable baby form forever through a superpower called neoteny",
                "required_keywords": ["baby", "forever", "neoteny", "grows", "never", "young"],
                "character_expression": "baby_curious",
                "character_color_state": "pink",
                "visual_prop": "eternal_baby_neoteny",
                "visual_badge_text": "ETERNAL BABY FORM!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Their immune system accepts tissue and organ transplants with zero rejection",
                "required_keywords": ["immune", "system", "accepts", "organs", "zero", "rejection"],
                "character_expression": "xray_focused",
                "character_color_state": "pink",
                "visual_prop": "organ_swap_zero_rejection",
                "visual_badge_text": "ZERO ORGAN REJECTION!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would choose: regrowing lost limbs or staying forever young",
                "required_keywords": ["which", "pick", "choose", "superpower", "limbs", "young"],
                "character_expression": "waving_cheerful",
                "character_color_state": "pink",
                "visual_prop": "axolotl_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Regeneration_Superpower_Arc",
                "title": "The Animal That Can Regrow Its Entire Brain! 🦎✨ #shorts",
                "description": "Meet the axolotl, the real-life superhero that can regrow limbs, heart, and brain tissue! #shorts #animals #scienceforkids",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "axolotl", "animals", "funfacts"],
                "scene_narrations": [
                    "Meet the axolotl, the cute smiling creature with the greatest healing power on Earth!",
                    "If an axolotl loses a leg, it can completely regrow the entire limb with zero scars!",
                    "Even crazier, it can regenerate damaged parts of its heart and spinal cord!",
                    "Scientists found that axolotls can even regrow parts of their own brain!",
                    "Those pretty pink crown feathers on its head are actually gills that breathe underwater!",
                    "And unlike other salamanders, it stays in its cute baby form forever and never grows up!",
                    "Its immune system is so gentle it can accept organ transplants with zero rejection!",
                    "Which superpower would you pick, regrowing your limbs or staying forever young?"
                ]
            }
        ]
    },
    "electric_eel_bio_battery": {
        "core_entity": "eel",
        "niche": "animal_secrets",
        "topic": "Electric Eel 860V Shock Science",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "Electric eels are not actually eels at all—they are giant South American knifefish",
                "required_keywords": ["electric", "eel", "not", "knifefish", "river", "amazon"],
                "character_expression": "shocked",
                "character_color_state": "yellow",
                "visual_prop": "not_an_eel_knifefish",
                "visual_badge_text": "NOT A REAL EEL!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Over eighty percent of its long body is packed with thousands of bio-battery cells",
                "required_keywords": ["body", "cells", "electrocytes", "batteries", "inside", "thousand"],
                "character_expression": "happy_proud",
                "character_color_state": "yellow",
                "visual_prop": "electrocyte_battery_cells",
                "visual_badge_text": "6,000 LIVING BATTERIES!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 3,
                "topic_beat": "It can unleash eight hundred and sixty volts of power—enough to run ten lightbulbs",
                "required_keywords": ["volts", "860", "power", "shock", "bulbs", "zap"],
                "character_expression": "transforming_amazed",
                "character_color_state": "yellow",
                "visual_prop": "voltage_860v_meter",
                "visual_badge_text": "860 VOLTS OF POWER!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 4,
                "topic_beat": "It uses low voltage electrical pulses like radar to navigate dark muddy river water",
                "required_keywords": ["radar", "radar", "muddy", "pulses", "navigate", "see"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "yellow",
                "visual_prop": "electric_radar_navigation",
                "visual_badge_text": "ELECTRIC RADAR VISION!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 5,
                "topic_beat": "When hunting, it curls its body around prey to double the electrical circuit shock",
                "required_keywords": ["curls", "body", "circuit", "double", "hunting", "shock"],
                "character_expression": "eating_excited",
                "character_color_state": "yellow",
                "visual_prop": "curling_voltage_booster",
                "visual_badge_text": "DOUBLE CIRCUIT CURL!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "Its vital organs and thick skin are specially insulated so it never shocks itself",
                "required_keywords": ["insulated", "skin", "never", "safe", "organs", "shield"],
                "character_expression": "xray_focused",
                "character_color_state": "yellow",
                "visual_prop": "self_insulation_armor",
                "visual_badge_text": "BUILT-IN INSULATION!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 7,
                "topic_beat": "A mother electric eel can light up the entire water to chase off giant hungry caimans",
                "required_keywords": ["mother", "lights", "water", "caiman", "scare", "zap"],
                "character_expression": "shocked",
                "character_color_state": "yellow",
                "visual_prop": "shock_predators_away",
                "visual_badge_text": "SCARES GIANT CAIMANS!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would choose: 860-volt lightning burst or electric radar vision",
                "required_keywords": ["which", "pick", "choose", "superpower", "volts", "radar"],
                "character_expression": "waving_cheerful",
                "character_color_state": "yellow",
                "visual_prop": "eel_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Electric_Shock_Arc",
                "title": "Why Electric Eels Are LIVING Batteries! ⚡🔋 #shorts",
                "description": "Discover how electric eels pack 6,000 biological batteries to blast 860 volts of electricity! #shorts #animals #scienceforkids",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "electriceel", "animals", "funfacts"],
                "scene_narrations": [
                    "Did you know an electric eel isn't actually an eel at all? It is a giant knifefish!",
                    "Over eighty percent of its body is filled with six thousand living battery cells!",
                    "When it discharges, it releases eight hundred and sixty volts—enough to power ten lightbulbs!",
                    "In dark muddy water, it sends out gentle electrical radar clicks to see everything around it!",
                    "To hunt, it curls its long body into a circle to double the electrical voltage!",
                    "And thanks to its thick rubbery skin, it is fully insulated and never shocks itself!",
                    "One big electric zap can even startle and chase away an entire giant hungry crocodile!",
                    "Which superpower would you choose, the 860-volt shock or the electric radar vision?"
                ]
            }
        ]
    },
    "tardigrade_water_bear": {
        "core_entity": "tardigrade",
        "niche": "science_curiosities",
        "topic": "Tardigrade Indestructible Astronauts",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "Tardigrades or water bears are microscopic eight-legged creatures that look like gummy bears",
                "required_keywords": ["tardigrade", "water", "bear", "microscopic", "gummy", "cute"],
                "character_expression": "baby_curious",
                "character_color_state": "blue",
                "visual_prop": "water_bear_microscope",
                "visual_badge_text": "INDESCRUCTIBLE GUMMY BEAR!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "They can survive freezing at near absolute zero minus two hundred and seventy-two degrees",
                "required_keywords": ["freezing", "absolute", "zero", "cold", "degrees", "survive"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "blue",
                "visual_prop": "absolute_zero_freeze",
                "visual_badge_text": "-272°C ABSOLUTE ZERO!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "They can also survive boiling inside volcanic hot springs up to one hundred and fifty degrees",
                "required_keywords": ["boiling", "volcanic", "hot", "springs", "heat", "survive"],
                "character_expression": "transforming_amazed",
                "character_color_state": "blue",
                "visual_prop": "volcano_heat_resist",
                "visual_badge_text": "150°C BOILING HEAT!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 4,
                "topic_beat": "Scientists sent water bears into outer space and they survived the freezing vacuum and radiation",
                "required_keywords": ["space", "outer", "astronaut", "vacuum", "radiation", "survived"],
                "character_expression": "shocked",
                "character_color_state": "blue",
                "visual_prop": "outer_space_vacuum_survive",
                "visual_badge_text": "SURVIVES OUTER SPACE!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 5,
                "topic_beat": "To survive extreme danger, they turn their body into pure bioglass through cryptobiosis",
                "required_keywords": ["bioglass", "cryptobiosis", "proteins", "protect", "glass", "sleep"],
                "character_expression": "happy_proud",
                "character_color_state": "blue",
                "visual_prop": "cryptobiosis_glass_proteins",
                "visual_badge_text": "TURNS INTO BIOGLASS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "They can sleep in this dry glass state for over thirty years without eating or drinking",
                "required_keywords": ["sleep", "thirty", "years", "food", "water", "survive"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "blue",
                "visual_prop": "thirty_year_sleep",
                "visual_badge_text": "30-YEAR SLEEP LOCK!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Add just a single drop of water and the little water bear wakes up alive and crawls away",
                "required_keywords": ["drop", "water", "wakes", "alive", "crawls", "magic"],
                "character_expression": "eating_excited",
                "character_color_state": "blue",
                "visual_prop": "drop_of_water_revive",
                "visual_badge_text": "REVIVES WITH 1 DROP!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the viewer which superpower they would choose: surviving outer space or the 30-year sleep lock",
                "required_keywords": ["which", "pick", "choose", "superpower", "space", "sleep"],
                "character_expression": "waving_cheerful",
                "character_color_state": "blue",
                "visual_prop": "tardigrade_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Cosmic_Microscope_Arc",
                "title": "The Immortal Animal That Survives OUTER SPACE! 🔬🚀 #shorts",
                "description": "Meet the tardigrade or water bear, the tiny microscopic beast that can survive outer space and volcanoes! #shorts #animals #science",
                "tags": ["shorts", "kidsshorts", "scienceforkids", "tardigrade", "animals", "funfacts"],
                "scene_narrations": [
                    "Under a microscope lives the toughest creature in the universe: the eight-legged water bear!",
                    "It can survive freezing at near absolute zero colder than the coldest ice on Earth!",
                    "It can even survive boiling in volcanic hot springs at three hundred degrees!",
                    "Scientists sent tardigrades into outer space, and they survived the airless vacuum and radiation!",
                    "When conditions get harsh, they turn their body into liquid glass through a trick called cryptobiosis!",
                    "They can stay asleep in this indestructible glass state for over thirty whole years!",
                    "Add just a single drop of water, and the tiny creature wakes up and crawls away completely fine!",
                    "Which superpower would you choose, surviving the vacuum of outer space or the 30-year sleep lock?"
                ]
            }
        ]
    },
    "baby_ducklings_counting_quack": {
        "core_entity": "duckling",
        "tier": "preschool",
        "niche": "nursery_rhyme_singalong",
        "topic": "Five Baby Ducklings Count & Splash",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "One tiny fluffy yellow baby duckling waddles cheerfully down to the green pond",
                "required_keywords": ["one", "1", "duckling", "fluffy", "yellow", "waddles"],
                "character_expression": "happy_proud",
                "character_color_state": "yellow",
                "visual_prop": "duckling_one_waddle",
                "visual_badge_text": "1 LITTLE DUCKLING!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Two baby ducklings splash and kick their tiny orange feet in the water",
                "required_keywords": ["two", "2", "ducklings", "splash", "water", "feet"],
                "character_expression": "happy_proud",
                "character_color_state": "yellow",
                "visual_prop": "duckling_two_splash",
                "visual_badge_text": "2 SPLASHING DUCKS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "Three baby ducklings open their tiny beaks and sing quack quack quack",
                "required_keywords": ["three", "3", "quack", "sing", "song", "beaks"],
                "character_expression": "eating_excited",
                "character_color_state": "yellow",
                "visual_prop": "duckling_three_quack",
                "visual_badge_text": "3 SINGING QUACKS!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 4,
                "topic_beat": "Four baby ducklings shake their soft fluffy feather tails with happy wiggles",
                "required_keywords": ["four", "4", "shake", "tails", "fluffy", "wiggle"],
                "character_expression": "happy_proud",
                "character_color_state": "yellow",
                "visual_prop": "duckling_four_shake",
                "visual_badge_text": "4 WIGGLY TAILS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 5,
                "topic_beat": "Five baby ducklings swim in a neat little line under the bright smiling sunshine",
                "required_keywords": ["five", "5", "swim", "line", "sunshine", "together"],
                "character_expression": "happy_proud",
                "character_color_state": "yellow",
                "visual_prop": "duckling_five_paddle",
                "visual_badge_text": "5 HAPPY DUCKLINGS!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 6,
                "topic_beat": "Mama duck calls them over for delicious crunchy green pondweed snacks",
                "required_keywords": ["mama", "duck", "calls", "snacks", "crunchy", "yummy"],
                "character_expression": "eating_excited",
                "character_color_state": "yellow",
                "visual_prop": "duckling_mama_snack",
                "visual_badge_text": "MAMA DUCK'S SNACK!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 7,
                "topic_beat": "All five tired ducklings cuddle close together for a warm cozy sleepy nap",
                "required_keywords": ["cuddle", "nap", "sleep", "cozy", "tired", "dreams"],
                "character_expression": "sleeping_peaceful",
                "character_color_state": "yellow",
                "visual_prop": "duckling_cuddle_sleep",
                "visual_badge_text": "COZY SLEEPY NAP!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the toddler which they like best: splashing in the pond or saying quack quack",
                "required_keywords": ["which", "love", "like", "best", "splash", "quack"],
                "character_expression": "waving_cheerful",
                "character_color_state": "yellow",
                "visual_prop": "duckling_choice_quack_splash",
                "visual_badge_text": "WHAT'S YOUR FAVORITE?",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Joyful_Preschool_Counting_Singalong",
                "title": "Five Little Ducklings Count & Quack! 🐥💛 #shorts",
                "description": "Count 1 to 5 with five cute fluffy baby ducklings splashing in the pond! Fun singalong for toddlers and preschoolers. #shorts #kids #preschool #ducklings #nurseryrhymes",
                "tags": ["shorts", "preschool", "toddlers", "nurseryrhymes", "counting", "ducklings", "kidslearning"],
                "scene_narrations": [
                    "One little duckling... yellow and bright!\nWaddling along... in the warm sunlight!",
                    "Two little ducklings... splash and play!\nKicking their feet... in the pond all day!",
                    "Three little ducklings... in a neat row!\nQuack, quack, quack... off they go!",
                    "Four little ducklings... shake their tails!\nWiggle and waggle... like little sails!",
                    "Five little ducklings... swim with pride!\nGliding along... side by side!",
                    "Mama Duck calls them... quack, quack, quack!\nCome eat your yummy... leafy snack!",
                    "Five little ducklings... time for bed!\nTuck in your feathers... rest your head!",
                    "Which do you love... when ducklings play?\nSplashing in water... or quacking all day?"
                ]
            }
        ]
    },
    "liquid_nitrogen_fire_showdown": {
        "core_entity": "nitrogen",
        "tier": "older",
        "niche": "mindblown_experiments",
        "topic": "Liquid Nitrogen vs 2,000°F Fire Showdown",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "Liquid nitrogen boils at an extreme cryogenic temperature of minus 320 degrees Fahrenheit",
                "required_keywords": ["liquid", "nitrogen", "boils", "minus", "320", "cryogenic"],
                "character_expression": "shocked",
                "character_color_state": "blue",
                "visual_prop": "nitrogen_minus_320f",
                "visual_badge_text": "-320°F CRYOGENIC LIQUID!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Submerging a bouncy rubber ball freezes it instantly so it shatters into glass-like shards",
                "required_keywords": ["ball", "rubber", "freezes", "shatters", "glass", "instant"],
                "character_expression": "transforming_amazed",
                "character_color_state": "blue",
                "visual_prop": "nitrogen_shatter_ball",
                "visual_badge_text": "RUBBER SHATTERS LIKE GLASS!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "The Leidenfrost effect creates a protective microscopic vapor cushion around warm surfaces",
                "required_keywords": ["leidenfrost", "vapor", "cushion", "protective", "boils", "layer"],
                "character_expression": "happy_proud",
                "character_color_state": "blue",
                "visual_prop": "leidenfrost_vapor_barrier",
                "visual_badge_text": "LEIDENFROST VAPOR SHIELD!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 4,
                "topic_beat": "What happens when minus 320°F cryogenic nitrogen collides directly with roaring 2,000°F fire?",
                "required_keywords": ["fire", "collides", "thousand", "flame", "degrees", "roaring"],
                "character_expression": "shocked",
                "character_color_state": "blue",
                "visual_prop": "nitrogen_vs_fire_standoff",
                "visual_badge_text": "-320°F VS 2,000°F FIRE!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 5,
                "topic_beat": "Liquid nitrogen violently expands seven hundred times its liquid volume into a massive gas cloud",
                "required_keywords": ["expands", "700", "seven", "hundred", "times", "cloud"],
                "character_expression": "transforming_amazed",
                "character_color_state": "blue",
                "visual_prop": "nitrogen_700x_expansion",
                "visual_badge_text": "700X GAS EXPANSION!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "The massive nitrogen gas cloud displaces all oxygen and instantly suffocates the roaring fire",
                "required_keywords": ["oxygen", "displaces", "suffocates", "flame", "extinguishes", "smoke"],
                "character_expression": "happy_proud",
                "character_color_state": "blue",
                "visual_prop": "fire_oxygen_starvation",
                "visual_badge_text": "FIRE SUFFOCATES INSTANTLY!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "Scientists use this cryogenic power to supercool quantum computers and liquid rocket propellant",
                "required_keywords": ["quantum", "rocket", "supercool", "computers", "space", "power"],
                "character_expression": "xray_focused",
                "character_color_state": "blue",
                "visual_prop": "cryo_quantum_rocket",
                "visual_badge_text": "POWERS QUANTUM & ROCKETS!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Asking the pre-teen viewer which superpower they would choose: flash freeze blast or vapor heat shield",
                "required_keywords": ["which", "pick", "choose", "superpower", "freeze", "shield"],
                "character_expression": "waving_cheerful",
                "character_color_state": "blue",
                "visual_prop": "nitrogen_superpower_choice",
                "visual_badge_text": "PICK YOUR SUPERPOWER!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Extreme_Thermodynamics_Showdown",
                "title": "Liquid Nitrogen (-320°F) VS 2,000°F Fire! ❄️🔥 #shorts",
                "description": "What happens when minus 320°F liquid nitrogen collides with roaring 2,000°F fire? Discover the Leidenfrost effect and quantum cooling! #shorts #science #physics #experiments",
                "tags": ["shorts", "science", "physics", "experiments", "liquidnitrogen", "mindblown", "teenscience"],
                "scene_narrations": [
                    "Liquid nitrogen boils at an extreme cryogenic temperature of minus 320 degrees Fahrenheit!",
                    "Dip a bouncy rubber ball inside, and it freezes so fast it shatters like fragile glass!",
                    "Even crazier, the Leidenfrost effect creates a microscopic vapor cushion protecting warm hands!",
                    "So what happens when minus 320-degree nitrogen is poured directly onto 2,000-degree fire?",
                    "The cryogenic liquid violently expands seven hundred times its volume in less than a second!",
                    "This massive nitrogen cloud starves the flame of oxygen and suffocates the fire instantly!",
                    "Today, engineers use this extreme cold to supercool quantum computers and liquid rocket engines!",
                    "Which superpower would you choose: the instant flash freeze blast or the Leidenfrost vapor shield?"
                ]
            }
        ]
    },
    "quantum_levitation_superconductors": {
        "core_entity": "superconductor",
        "niche": "interactive_riddles",
        "tier": "older",
        "topic": "Quantum Levitation & Superconductors",
        "blueprint": [
            {
                "scene_id": 1,
                "topic_beat": "A disc floats locked in mid-air and cannot be pushed down even by heavy weight",
                "required_keywords": ["floating", "mid-air", "push", "down", "gravity", "weight"],
                "character_expression": "shocked",
                "character_color_state": "blue",
                "visual_prop": "floating_quantum_disc",
                "visual_badge_text": "FLOATS IN MID-AIR!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 2,
                "topic_beat": "Quantum levitation happens when a superconductor ceramic disc is chilled with liquid nitrogen",
                "required_keywords": ["superconductor", "liquid", "nitrogen", "chilled", "degrees", "ceramic"],
                "character_expression": "transforming_amazed",
                "character_color_state": "blue",
                "visual_prop": "superconductor_liquid_cooling",
                "visual_badge_text": "CHILLED TO -320°F!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 3,
                "topic_beat": "Electrical resistance vanishes completely to zero allowing endless currents to circulate",
                "required_keywords": ["resistance", "zero", "current", "electrons", "energy", "flow"],
                "character_expression": "happy_proud",
                "character_color_state": "blue",
                "visual_prop": "zero_electrical_resistance",
                "visual_badge_text": "ZERO RESISTANCE!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 4,
                "topic_beat": "The Meissner effect expels magnetic fields, locking thousands of magnetic flux tubes in place",
                "required_keywords": ["meissner", "magnetic", "field", "flux", "expels", "lines"],
                "character_expression": "shocked",
                "character_color_state": "blue",
                "visual_prop": "meissner_magnetic_repulsion",
                "visual_badge_text": "MEISSNER MAGNETIC SHIELD!",
                "sfx_cue": "whoosh",
            },
            {
                "scene_id": 5,
                "topic_beat": "Flux pinning traps magnetic vortices inside tiny crystal defects creating an invisible anchor",
                "required_keywords": ["flux", "pinning", "vortices", "anchor", "trapped", "crystal"],
                "character_expression": "transforming_amazed",
                "character_color_state": "blue",
                "visual_prop": "flux_pinning_anchor",
                "visual_badge_text": "FLUX PINNING LOCK!",
                "sfx_cue": "pop",
            },
            {
                "scene_id": 6,
                "topic_beat": "You can flip the track upside down and the levitating disc stays suspended in space",
                "required_keywords": ["upside", "down", "suspended", "track", "spin", "space"],
                "character_expression": "happy_proud",
                "character_color_state": "blue",
                "visual_prop": "upside_down_levitation",
                "visual_badge_text": "LEVITATES UPSIDE DOWN!",
                "sfx_cue": "chime",
            },
            {
                "scene_id": 7,
                "topic_beat": "High-speed maglev trains use this frictionless quantum power to cruise at 600 kilometers per hour",
                "required_keywords": ["maglev", "trains", "speed", "frictionless", "bullet", "cruise"],
                "character_expression": "xray_focused",
                "character_color_state": "blue",
                "visual_prop": "quantum_maglev_bullet_train",
                "visual_badge_text": "600 KM/H MAGLEV TRAIN!",
                "sfx_cue": "ding",
            },
            {
                "scene_id": 8,
                "topic_beat": "Would you rather ride a 600km/h maglev bullet train or glide on a quantum hoverboard?",
                "required_keywords": ["would", "rather", "ride", "maglev", "train", "hoverboard"],
                "character_expression": "waving_cheerful",
                "character_color_state": "blue",
                "visual_prop": "maglev_vs_hoverboard",
                "visual_badge_text": "PICK YOUR RIDE!",
                "sfx_cue": "whoosh",
            },
        ],
        "candidates": [
            {
                "style_name": "Quantum_Levitation_Showdown",
                "title": "Quantum Levitation: Defying Gravity With Liquid Nitrogen! ⚡🛸 #shorts",
                "description": "Watch a frozen superconductor lock into mid-air magnetic levitation! Discover the Meissner effect and flux pinning. #shorts #science #quantum #physics #experiments",
                "tags": ["shorts", "science", "quantum", "physics", "superconductors", "mindblown", "experiments"],
                "scene_narrations": [
                    "Imagine an object floating in mid-air that you cannot push down even if you lean on it!",
                    "This is quantum levitation, unlocked when a ceramic superconductor is chilled with liquid nitrogen!",
                    "Under extreme cold, electrical resistance drops completely to zero with infinite current flow!",
                    "The Meissner effect expels magnetic fields, locking thousands of magnetic flux tubes into place!",
                    "Scientists call this flux pinning, trapping the disc in three-dimensional space like an invisible cage!",
                    "You can spin it, flip it upside down on a magnetic track, and it floats without touching!",
                    "Engineers use this quantum magic to build frictionless bullet trains flying at six hundred kilometers per hour!",
                    "Which would you ride: a six hundred kilometer per hour maglev train or a quantum hoverboard?"
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
    tier: str = Field(default="younger")


class ShortScript(BaseModel):
    title: str
    topic: str
    tier: str = "younger"
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

    def _select_episode_blueprint(self, topic: str, niche: str, tier: str = "younger") -> Dict[str, Any]:
        """
        Selects a 100% brand-new, never-before-covered episode blueprint for the given age tier.
        Strictly enforces indefinite zero-repetition across entities, topics, and titles within the tier.
        """
        used_entities_tier = TopicRegistry.load_used_entities(tier=tier)
        used_titles_tier = TopicRegistry.load_used_titles(tier=tier)
        topic_lower = (topic or "").lower()

        logger.info(f"Enforcing Zero-Repetition Registry for [{tier.upper()}]: {len(used_entities_tier)} entities locked out: {sorted(list(used_entities_tier))}")

        # If a specific topic was requested, verify it is NOT in used entities or past titles for this tier
        if topic_lower:
            for ep_key, ep_data in EPISODE_LIBRARY.items():
                if ep_data.get("tier", "younger") != tier:
                    continue
                ent = ep_data.get("core_entity", ep_key).lower()
                if ent in topic_lower or ep_key in topic_lower or ep_data["topic"].lower() in topic_lower:
                    if not TopicRegistry.is_forbidden(ent, ep_data["topic"], tier=tier):
                        if not any(c["title"] in used_titles_tier for c in ep_data["candidates"]):
                            logger.info(f"Selected requested fresh episode blueprint: '{ep_key}' ({ep_data['topic']}) for [{tier}]")
                            return ep_data
                    else:
                        logger.warning(f"Requested topic '{topic}' in tier '{tier}' was already used. Bypassing to enforce zero repetition!")

        # Find all blueprints in EPISODE_LIBRARY matching this tier whose core entity and topic have NEVER been used in this tier
        fresh_blueprints = []
        for ep_key, ep_data in EPISODE_LIBRARY.items():
            if ep_data.get("tier", "younger") != tier:
                continue
            ent = ep_data.get("core_entity", ep_key).lower()
            if TopicRegistry.is_forbidden(ent, ep_data["topic"], tier=tier):
                continue
            if any(c["title"] in used_titles_tier for c in ep_data["candidates"]):
                continue
            fresh_blueprints.append((ep_key, ep_data))

        if fresh_blueprints:
            niche_matches = [b for b in fresh_blueprints if b[1].get("niche") == niche]
            selected_key, selected_ep = niche_matches[0] if niche_matches else fresh_blueprints[0]
            logger.info(f"Auto-selected fresh non-repeating blueprint for [{tier}]: '{selected_key}' ({selected_ep['topic']})")
            return selected_ep

        # If static library for this tier is exhausted, dynamically generate a novel blueprint with AI Studio!
        logger.info(f"All static blueprints for tier [{tier}] exhausted. Dynamically generating novel blueprint via AI Studio...")
        novel_ep = self._generate_novel_blueprint_with_ai(niche, used_entities_tier, tier=tier, topic=topic)
        if novel_ep:
            return novel_ep

        raise RuntimeError(f"FATAL ZERO-REPETITION LOCK: All {len(used_entities_tier)} topics in tier [{tier}] have been used. No fresh blueprint available.")

    def _generate_novel_blueprint_with_ai(self, niche: str, used_entities: set, tier: str = "younger", topic: str = "") -> Optional[Dict[str, Any]]:
        """Dynamically synthesizes a brand-new, never-before-covered 8-scene blueprint using Gemini tailored to age tier."""
        if not self.api_key:
            return None
        forbidden_str = ", ".join(sorted(list(used_entities)))

        tier_instructions = {
            "preschool": (
                "TARGET AUDIENCE: Preschool & Toddlers (Ages 4 & under).\n"
                "TONE & STYLE: Joyful, gentle, rhythmic, catchy rhyming singalong.\n"
                "TOPIC THEMES: Counting 1-5, primary colors, baby animal sounds, simple nature wonders.\n"
                "VOCABULARY: Maximum 10-12 simple phonetic words per scene. Zero complex words."
            ),
            "younger": (
                "TARGET AUDIENCE: Early Elementary Kids (Ages 5–8).\n"
                "TONE & STYLE: Curious, playful, energetic, mystery & superpower showdown.\n"
                "TOPIC THEMES: Animal superpowers, nature secrets, cool body facts, interactive 2-choice questions.\n"
                "VOCABULARY: 14-18 lively words per scene."
            ),
            "older": (
                "TARGET AUDIENCE: Older Kids & Tweens (Ages 9–12).\n"
                "TONE & STYLE: Fast-paced, high-energy, intellectual curiosity, mind-blown, mythbusting, ZERO baby talk.\n"
                "TOPIC THEMES: Extreme physics, cryogenic chemistry, space phenomena, engineering hacks.\n"
                "VOCABULARY: 18-22 punchy, intellectually stimulating words per scene."
            )
        }
        tier_prompt_info = tier_instructions.get(tier, tier_instructions["younger"])

        prompt = (
            f"You are the Lead Curriculum Showrunner for KuriousKids YouTube Channel.\n"
            f"Task: Generate a completely new 8-scene Short blueprint for niche '{niche}' and age tier '{tier}'.\n"
            f"{tier_prompt_info}\n"
            f"CRITICAL MANDATE: The following entities and topics have ALREADY been produced in this tier and are STRICTLY FORBIDDEN from repeating: {forbidden_str}.\n"
            f"You MUST select a completely different, unmentioned animal or scientific concept.\n"
            f"Return JSON with keys: core_entity, topic, niche, candidates (list of dict with title, description, tags, scene_narrations), "
            f"blueprint (8 scenes with scene_id, topic_beat, required_keywords, character_expression, visual_prop, visual_badge_text, sfx_cue)."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                novel_data = json.loads(text)
                if novel_data.get("blueprint") and len(novel_data["blueprint"]) == 8:
                    novel_data["tier"] = tier
                    logger.info(f"AI Studio synthesized brand-new blueprint for [{tier}]: {novel_data.get('topic')}")
                    return novel_data
        except Exception as e:
            logger.warning(f"Failed to generate dynamic blueprint via AI Studio: {e}")
        return self._generate_procedural_novel_blueprint(niche, used_entities, tier, requested_topic=topic)

    def _generate_procedural_novel_blueprint(self, niche: str, used_entities: set, tier: str = "younger", requested_topic: str = "") -> Optional[Dict[str, Any]]:
        """Procedural zero-crash engine generating fresh, compliant 8-scene blueprints for any tier."""
        tier = (tier or "younger").lower()
        topic_seeds = [
            # 1. PRESCHOOL (Ages 4 & Under): Simple rhymes, joyful counting, animal sounds
            {
                "core_entity": "frog",
                "tier": "preschool",
                "topic": "Five Little Frogs Hop & Splash",
                "title": "Five Little Frogs Hop & Splash! 🐸💚 #shorts",
                "desc": "Count 1 to 5 with five cheerful green tree frogs hopping across lily pads! Fun singalong for toddlers. #shorts #kids #preschool #counting",
                "tags": ["shorts", "preschool", "toddlers", "counting", "frogs", "nurseryrhymes"],
                "badge": "HOP HOP SPLASH!",
                "badges": [
                    "FROG ON LILY PAD",
                    "BIG LEG STRETCH",
                    "COOL POND DIVE",
                    "RIBBIT SINGALONG",
                    "WATER RIPPLES",
                    "YUMMY BUG SNACK",
                    "COZY LEAF BED",
                    "HOPPING SHOWDOWN"
                ],
                "prop": "tree_frog_lily_pad",
                "props": [
                    "frog_on_lily_pad",
                    "frog_legs_jumping",
                    "frog_swimming_pond",
                    "frog_croaking",
                    "frog_watching_ripples",
                    "frog_eating_insect",
                    "frog_sleeping_under_leaf",
                    "frogs_splashing"
                ],
                "narrations": [
                    "One little froggy... sitting in the sun!\nBouncing on a lily pad... having so much fun!",
                    "Two little frogs... stretch their legs so high!\nCatching yummy flies... as they flutter by!",
                    "Three little frogs... jump into the pool!\nSwimming in the pond... feeling nice and cool!",
                    "Four little frogs... sing a happy song!\nRibbit, ribbit, ribbit... singing all day long!",
                    "Five little frogs... sitting side by side!\nWatching tiny ripples... on the water wide!",
                    "Mama Frog calls out... time to come and eat!\nCrunchy little snacks... yummy and so sweet!",
                    "Five sleepy frogs... rest their tired head!\nTucked in cozy leaves... time to go to bed!",
                    "Which do you love best... when the frogs play?\nJumping up so high... or splashing all day?"
                ]
            },
            {
                "core_entity": "puppy",
                "tier": "preschool",
                "topic": "Three Happy Puppies Wag & Play",
                "title": "Three Happy Puppies Wag & Play! 🐶🎾 #shorts",
                "desc": "Play with three cute fluffy puppies running in the park! Fun puppy song for toddlers. #shorts #kids #preschool #puppy #nurseryrhymes",
                "tags": ["shorts", "preschool", "toddlers", "puppies", "dogs", "nurseryrhymes"],
                "badge": "PUPPY PLAYTIME!",
                "badges": [
                    "SWEET PUPPY PAWS",
                    "CHASING THE BALL",
                    "WAGGING TAILS",
                    "FETCH THE STICK",
                    "PUPPY TREAT TIME",
                    "SPINNING IN LEAVES",
                    "PUPPY PILE SNOOZE",
                    "SUNNY PUPPY FUN"
                ],
                "prop": "cute_playful_puppy",
                "props": [
                    "puppy_walking_street",
                    "puppies_chasing_ball",
                    "puppies_wagging_tails",
                    "puppy_fetching_stick",
                    "puppy_catching_treat",
                    "puppy_tumbling_leaves",
                    "puppies_sleeping_pile",
                    "puppies_running_sun"
                ],
                "narrations": [
                    "One little puppy... fluffy and so sweet!\nPattering his paws... down the sunny street!",
                    "Two little puppies... chasing round a ball!\nRolling in the grass... having such a ball!",
                    "Three little puppies... wag their tails with glee!\nBarking happy barks... under the shade tree!",
                    "Puppy number one... loves to fetch a stick!\nRunning super fast... clever and so quick!",
                    "Puppy number two... jumps up for a treat!\nCatching little biscuits... crunchy and so sweet!",
                    "Puppy number three... spins in circles round!\nTumbling in leaves... right onto the ground!",
                    "Now all three puppies... cuddle in a heap!\nYawning tiny yawns... falling fast asleep!",
                    "Which do you love best... when puppies run?\nChasing after balls... or cuddling in the sun?"
                ]
            },
            # 2. YOUNGER (Ages 5–8): Easy animal superpowers and nature questions
            {
                "core_entity": "cat",
                "tier": "younger",
                "topic": "Why Do Cats Always Land on Their Feet?",
                "title": "Why Do Cats Always Land on Their Feet? 🐱🤸 #shorts",
                "desc": "Discover the incredible gymnastic flip that helps cats land on their paws every time! #shorts #animals #cats #science #kids",
                "tags": ["shorts", "kids", "animals", "cats", "science", "facts"],
                "badge": "CAT NINJA FLIP!",
                "badges": [
                    "NINJA CAT REFLEX",
                    "INNER EAR BALANCE",
                    "180° FRONT TWIST",
                    "FLEXIBLE SPINE",
                    "SHOCK ABSORBING PAWS",
                    "KITTEN ACROBATICS",
                    "TAIL COUNTERBALANCE",
                    "CAT SKILL SHOWDOWN"
                ],
                "prop": "cat_midair_twist",
                "props": [
                    "cat_midair_jump",
                    "cat_ear_balance_system",
                    "cat_front_legs_twist",
                    "cat_arched_spine_rotation",
                    "cat_paw_pads_cushion",
                    "playful_kittens_jumping",
                    "cat_tail_balance_pole",
                    "cat_climbing_tree"
                ],
                "narrations": [
                    "Ever wonder how cats... always land on their feet?\nIt's a secret superpower... super cool and neat!",
                    "Inside their ears... there is a balance guide!\nIt tells their clever brain... which way is up and wide!",
                    "First their flexible shoulders... twist their front legs round!\nSpinning like a gymnast... high above the ground!",
                    "Then their bendy backbone... snaps the back paws tight!\nLining up all four pads... ready for the flight!",
                    "Their soft paw cushions... act like trampoline springs!\nAbsorbing all the landing bounce... better than anything!",
                    "Even tiny kittens... learn this ninja trick!\nBy seven weeks of age... they are lightning quick!",
                    "Their long fluffy tail... balances like a pole!\nHelping them land gently... in total calm control!",
                    "Would you rather have... ninja cat landing feet?\nOr run up trees like a cat... down the sunny street?"
                ]
            },
            {
                "core_entity": "dog",
                "tier": "younger",
                "topic": "Why Do Dogs Wag Their Tails?",
                "title": "Why Do Dogs Wag Their Tails? 🐕❤️ #shorts",
                "desc": "Find out the secret language of dog tail wags and what your furry best friend is saying! #shorts #animals #dogs #pets #facts",
                "tags": ["shorts", "kids", "dogs", "pets", "animals", "facts"],
                "badge": "TAIL WAG SECRETS!",
                "badges": [
                    "SECRET TAIL CODE",
                    "DOORWAY EXCITEMENT",
                    "RIGHT VS LEFT WAG",
                    "HELICOPTER TAIL",
                    "PUPPY FRIENDSHIP",
                    "SCENT MESSAGING",
                    "TAIL LANGUAGE",
                    "DOG TAIL SHOWDOWN"
                ],
                "prop": "happy_wagging_dog",
                "props": [
                    "dog_wagging_tail_intro",
                    "dog_happy_at_front_door",
                    "dog_wagging_tail_direction",
                    "dog_spinning_tail_circle",
                    "puppies_wagging_together",
                    "dog_tail_scent_communication",
                    "dog_calm_tail_rest",
                    "happy_dog_outdoor_run"
                ],
                "narrations": [
                    "Have you ever wondered... why dogs wag their tails?\nIt's a secret language... with cheerful happy tales!",
                    "When a dog is excited... and sees you at the door!\nTheir tail wags back and forth... sweeping on the floor!",
                    "Wagging to the right side... means feeling super glad!\nWagging to the left side... means nervous or a little sad!",
                    "A high spinning tail... is like a helicopter blade!\nSaying let's play fetch... in the backyard shade!",
                    "Even tiny puppies... learn this friendly sign!\nTo tell all their puppy friends... everything is fine!",
                    "Their wagging tail spreads... a friendly happy scent!\nShowing every friend... what their warm feelings meant!",
                    "A soft gentle wag... says I love you so!\nGiving you big puppy hugs... everywhere you go!",
                    "Would you rather speak... the secret language of dogs?\nOr talk to friendly birds... chirping on the logs?"
                ]
            },
            # 3. OLDER (Ages 9–12): Easy everyday science, fun myths, food hacks
            {
                "core_entity": "soda",
                "tier": "older",
                "topic": "Why Does Soda Explode With Mentos?",
                "title": "Why Does Soda Explode With Mentos? 🥤💥 #shorts",
                "desc": "The real science behind the famous Mentos and Diet Coke backyard fountain explosion! #shorts #science #experiment #diy #coolfacts",
                "tags": ["shorts", "science", "experiment", "soda", "coolfacts", "diy"],
                "badge": "10-FT SODA GEYSER!",
                "badges": [
                    "10-FT SODA GEYSER",
                    "PHYSICAL REACTION",
                    "MICROSCOPIC PITS",
                    "NUCLEATION SITES",
                    "BILLION BUBBLE BLAST",
                    "FOAM SPOUT ERUPTION",
                    "DIET SODA SECRET",
                    "SCIENCE SHOWDOWN"
                ],
                "prop": "soda_geyser_explosion",
                "props": [
                    "soda_bottle_candies_drop",
                    "soda_carbon_gas_bubbles",
                    "microscopic_candy_pits",
                    "nucleation_sites_bubbles",
                    "candy_sinking_upward_blast",
                    "foaming_geyser_blast",
                    "diet_soda_fountain",
                    "backyard_science_showdown"
                ],
                "narrations": [
                    "Drop a single candy into soda... and boom!\nA giant fizzy geyser... shoots straight across the room!",
                    "Is it a chemical explosion? ... Nope, it's physical science!\nSoda is packed with carbon gas... looking for defiance!",
                    "A candy surface looks smooth... but zoom in super close!\nThere are millions of microscopic pits... on every single dose!",
                    "Scientists call these... tiny nucleation sites!\nWhere trapped carbon gas bubbles... gather for their flights!",
                    "The heavy candy sinks to the bottom... lightning fast!\nReleasing billions of gas bubbles... in a massive upward blast!",
                    "All that foaming pressure... rushes toward the spout!\nBlasting ten feet into the air... with a fizzy shout!",
                    "Diet soda works best... because aspartame lowers tension!\nCreating the ultimate... backyard science invention!",
                    "Would you rather launch... a ten-foot soda fountain blast?\nOr build a water rocket... that zooms super fast?"
                ]
            },
            {
                "core_entity": "brain_freeze",
                "tier": "older",
                "topic": "What Actually Causes Brain Freeze?",
                "title": "What Actually Causes Brain Freeze? 🍦🧠 #shorts",
                "desc": "Why does eating ice cream too fast give you a crazy frozen headache? The secret science explained! #shorts #science #icecream #brainfreeze #facts",
                "tags": ["shorts", "science", "icecream", "facts", "brainfreeze", "cool"],
                "badge": "BRAIN FREEZE HACK!",
                "badges": [
                    "BRAIN FREEZE OUCH!",
                    "ROOF OF MOUTH TRICK",
                    "COLD SENSOR ALARM",
                    "RAPID BLOOD SURGE",
                    "TRIGEMINAL NERVE",
                    "TONGUE PRESS HACK",
                    "5-SECOND RECOVERY",
                    "COLD RUSH SHOWDOWN"
                ],
                "prop": "ice_cream_brain_freeze",
                "props": [
                    "eating_ice_cream_cold_bite",
                    "human_mouth_palate_diagram",
                    "cold_temperature_nerve_signal",
                    "expanding_blood_vessels_rush",
                    "trigeminal_nerve_forehead",
                    "warm_tongue_pressing_palate",
                    "blood_vessels_calming_down",
                    "frozen_slushy_dessert"
                ],
                "narrations": [
                    "Take a giant bite of ice cream... and suddenly ouch!\nA massive frozen headache... knocks you onto the couch!",
                    "Is your brain really freezing? ... No, it's an optical trick!\nYour mouth is playing games... super weird and quick!",
                    "When freezing cold food... hits the roof of your mouth!\nNerves trigger an emergency alarm... heading north and south!",
                    "Blood vessels instantly expand... rushing warm blood fast!\nTrying to heat your palate... from the icy cold blast!",
                    "The trigeminal nerve... gets confused by the sudden pain!\nAnd tells your forehead... there is an ice attack again!",
                    "To stop the brain freeze instantly... here is the secret trick!\nPress your warm tongue against the roof... super firm and quick!",
                    "Within five seconds... blood vessels calm right down!\nMelting the icy freeze... into a happy smile from a frown!",
                    "What gives you brain freeze fastest... a giant slushy drink?\nOr eating ice cream super quick... before you even think?"
                ]
            }
        ]

        sel = None
        if requested_topic:
            req = requested_topic.lower()
            matching = [s for s in topic_seeds if s["core_entity"] in req or s["topic"].lower() in req or any(w in req for w in s["core_entity"].split()) or any(w in req for w in s["topic"].lower().split() if len(w) > 3)]
            if matching:
                sel = matching[0]

        if not sel:
            candidates = [s for s in topic_seeds if s["tier"] == tier and s["core_entity"] not in used_entities]
            if not candidates:
                # Fallback to any unused entity regardless of tier tags
                candidates = [s for s in topic_seeds if s["core_entity"] not in used_entities]
            if not candidates:
                candidates = topic_seeds
            sel = candidates[0]
        logger.info(f"Procedural engine auto-crafted fresh novel blueprint for [{tier}]: {sel['topic']}")

        # Build 8-scene canonical blueprint with unique, section-specific beats and props
        bp_scenes = []
        badges = sel.get("badges") or [sel["badge"]] * 8
        props = sel.get("props") or [sel["prop"]] * 8
        narr_list = sel.get("narrations") or []

        for i in range(1, 9):
            idx = i - 1
            narr_chunk = narr_list[idx] if idx < len(narr_list) else sel["topic"]
            # Extract first meaningful concept line from narration as the specific section beat
            first_line = narr_chunk.split("\n")[0].replace("...", "").strip()
            beat = first_line if len(first_line) > 5 else f"{sel['topic']} part {i}"

            bp_scenes.append({
                "scene_id": i,
                "topic_beat": beat,
                "required_keywords": [sel["core_entity"], "power", "science", "discovery"],
                "character_expression": "happy_proud" if i % 2 == 0 else "shocked",
                "character_color_state": "blue" if tier == "older" else "pink",
                "visual_prop": props[idx % len(props)],
                "visual_badge_text": badges[idx % len(badges)],
                "sfx_cue": "whoosh" if i in (1, 8) else "pop"
            })

        return {
            "core_entity": sel["core_entity"],
            "tier": tier,
            "topic": sel["topic"],
            "niche": niche,
            "blueprint": bp_scenes,
            "candidates": [
                {
                    "style_name": "Procedural_Adventure_Arc",
                    "title": sel["title"],
                    "description": sel["desc"],
                    "tags": sel["tags"],
                    "scene_narrations": sel["narrations"]
                }
            ]
        }

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

    def generate_script(
        self,
        topic: str = "",
        formula: Dict[str, Any] = None,
        niche: str = "animal_secrets",
        tier: str = "younger"
    ) -> ShortScript:
        """
        Selects the active episode blueprint for the given age tier, generates/evaluates competing
        candidate scripts, scores each candidate, and returns the highest-scoring fresh script.
        """
        episode = self._select_episode_blueprint(topic, niche, tier=tier)
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
            f"(Score: {winner['total_score']}/100 across {len(scored_candidates)} candidates) [Tier: {tier}]"
        )

        topic_low = winner["topic"].lower()
        if "duck" in topic_low:
            char_name = "Ducky the Fluffy Duckling"
            char_spec = CharacterSpec(
                name="Ducky the Fluffy Duckling",
                species_or_role="duckling",
                face_traits="Cute round baby yellow duckling, fluffy cheeks, tiny smiling orange bill, sparkly black eyes",
                companion_name="Mama Duck"
            )
        elif "nitrogen" in topic_low:
            char_name = "Dr. Nova & Leo's Lab"
            char_spec = CharacterSpec(
                name="Dr. Nova",
                species_or_role="scientist",
                face_traits="Sleek safety goggles, futuristic white lab coat, cryogenic frost mist aura",
                companion_name="Robo-Drone Vector"
            )
        elif "shrimp" in topic_low:
            char_name = "Perry the Pistol Shrimp & Leo"
            char_spec = CharacterSpec(
                name="Perry the Pistol Shrimp",
                species_or_role="shrimp",
                face_traits="Cute coral shell, bright inquisitive eyes, signature giant glowing blast claw, golden explorer star badge",
                companion_name="Leo the Explorer Kid"
            )
        elif "axolotl" in topic_low:
            char_name = "Axel the Axolotl & Leo"
            char_spec = CharacterSpec(
                name="Axel the Axolotl",
                species_or_role="axolotl",
                face_traits="Smiling pink baby face, feathery crown gills, sparkly eyes, golden explorer badge",
                companion_name="Leo the Explorer Kid"
            )
        elif "eel" in topic_low:
            char_name = "Volt the Electric Eel & Leo"
            char_spec = CharacterSpec(
                name="Volt the Electric Eel",
                species_or_role="eel",
                face_traits="Sleek golden knifefish body, friendly smiling jaw, sparkling electric aura, golden explorer badge",
                companion_name="Leo the Explorer Kid"
            )
        elif "tardigrade" in topic_low or "water bear" in topic_low:
            char_name = "Barnaby the Water Bear & Leo"
            char_spec = CharacterSpec(
                name="Barnaby the Water Bear",
                species_or_role="tardigrade",
                face_traits="Chubby 8-legged microscopic gummy bear body, shiny space helmet, golden explorer star",
                companion_name="Leo the Explorer Kid"
            )
        elif "chameleon" in topic_low:
            char_name = "Cam the Chameleon & Leo"
            char_spec = CharacterSpec(
                name="Cam the Chameleon",
                species_or_role="chameleon",
                face_traits="Large expressive 360 turret eyes, cute smiling snout, lime-green and ruby color-shifting scales, signature golden star badge",
                companion_name="Leo the Explorer Kid"
            )
        elif "cloud" in topic_low or "weather" in topic_low:
            char_name = "Cumulus the Cloud & Leo"
            char_spec = CharacterSpec(
                name="Cumulus the Cloud",
                species_or_role="cloud",
                face_traits="Fluffy soft white cheeks, friendly glowing eyes, golden sunshine halo",
                companion_name="Leo the Explorer Kid"
            )
        elif "octopus" in topic_low:
            char_name = "Inky the Octopus & Leo"
            char_spec = CharacterSpec(
                name="Inky the Octopus",
                species_or_role="octopus",
                face_traits="Cute coral-orange tentacles, large bright glossy eyes, golden explorer badge",
                companion_name="Leo the Explorer Kid"
            )
        elif "flamingo" in topic_low:
            char_name = "Pip the Flamingo & Leo"
            char_spec = CharacterSpec()
        else:
            char_name = "Leo the Explorer Kid"
            char_spec = CharacterSpec(
                name="Leo the Explorer Kid",
                species_or_role="explorer",
                face_traits="Warm smile, yellow explorer hat, golden star badge",
                companion_name="Curious Squad"
            )

        # Determine real animal/subject for 100% photorealistic wildlife visual prompt
        species = char_spec.species_or_role.lower()
        if species == "eel" or "eel" in topic_low:
            real_subject = "electric eel"
        elif species == "duckling" or "duck" in topic_low:
            real_subject = "baby duckling"
        elif species == "flamingo" or "flamingo" in topic_low:
            real_subject = "flamingo"
        elif species == "octopus" or "octopus" in topic_low:
            real_subject = "octopus"
        elif species == "axolotl" or "axolotl" in topic_low:
            real_subject = "axolotl"
        elif species == "chameleon" or "chameleon" in topic_low:
            real_subject = "chameleon"
        elif species == "shrimp" or "shrimp" in topic_low:
            real_subject = "mantis shrimp"
        elif "soda" in topic_low or "mentos" in topic_low:
            real_subject = "backyard soda bottle eruption science experiment"
        elif "ice cream" in topic_low or "brain freeze" in topic_low:
            real_subject = "ice cream and cold sensation science"
        elif "nitrogen" in topic_low:
            real_subject = "cryogenic liquid nitrogen laboratory experiment"
        else:
            real_subject = species or topic_low

        scenes = []
        for idx, bp in enumerate(winner_bp):
            narr = winner["scene_narrations"][idx]
            scenes.append(
                Scene(
                    scene_id=bp["scene_id"],
                    duration_sec=3.2,
                    narration=narr,
                    visual_prompt=f"Stunning photorealistic photograph of real {real_subject}, {bp['topic_beat']}. National Geographic documentary photography, f/11 deep focus aperture, pin-sharp focus on entire subject, crisp textures, natural cinematic daylight, zero motion blur, no shallow depth of field, no blurry background, zero cartoon, zero animation, zero CGI, zero 3D render.",
                    character_expression=bp["character_expression"],
                    character_color_state=bp["character_color_state"],
                    visual_prop=bp["visual_prop"],
                    visual_badge_text=bp["visual_badge_text"],
                    sfx_cue=bp["sfx_cue"],
                    tier=tier
                )
            )

        total_words = sum(len(s.narration.split()) for s in scenes)
        total_dur = sum(s.duration_sec for s in scenes)

        tournament_report = {
            "winner_id": winner["candidate_id"],
            "winner_title": winner["title"],
            "winner_score": winner["total_score"],
            "tier": tier,
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
            tier=tier,
            target_audience="preschool" if tier == "preschool" else ("older_kids" if tier == "older" else "family_co_viewing"),
            character_spec=char_spec,
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
