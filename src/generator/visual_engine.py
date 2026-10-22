"""
Visual Asset Engine powered by Google AI Studio Multimodal Reference Conditioning
and a Deterministic Character-Consistent 3D-Shaded Rig Engine ("Similar Faces" + Silent Visual Storytelling).
"""

import os
import io
import math
import json
import base64
import random
import urllib.request
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VisualEngine:
    """
    Generates 1080x1920 vertical story scenes with 100% character face consistency ("similar faces")
    and clear visual cause-and-effect storytelling that passes the Silent Watch Test (muted audio).
    """

    def __init__(
        self,
        output_dir: str = "data/output/visuals",
        width: int = 1080,
        height: int = 1920,
        api_key: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        os.makedirs(self.output_dir, exist_ok=True)
        self.reference_sheet_path = os.path.join(self.output_dir, "master_character_sheet.png")
        self._ensure_master_character_sheet()

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        font_candidates = [
            "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf"
        ]
        for fp in font_candidates:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _ensure_master_character_sheet(self) -> str:
        """
        Renders the Master Character Reference Sheet showing the exact facial geometry,
        eye color, beak shape, and proportions of Pip the Flamingo and Leo the Explorer Kid.
        Used both as multimodal `inlineData` input to Google AI Studio and as our canonical visual spec.
        """
        img = Image.new("RGB", (1080, 1080), (24, 32, 56))
        draw = ImageDraw.Draw(img, "RGBA")

        # Header banner
        draw.rounded_rectangle([40, 30, 1040, 130], radius=25, fill=(40, 55, 95, 255), outline=(0, 230, 255, 255), width=4)
        title_font = self._get_font(38)
        sub_font = self._get_font(26)
        draw.text((540, 80), "MASTER CHARACTER IDENTITY SHEET (SIMILAR FACES)", fill=(255, 255, 255), font=title_font, anchor="mm")

        # Left Panel: Pip the Flamingo (Adult Pink & Baby Gray)
        draw.rounded_rectangle([40, 160, 520, 1040], radius=30, fill=(35, 48, 82, 255), outline=(255, 105, 180, 255), width=4)
        draw.text((280, 210), "HERO: PIP THE FLAMINGO", fill=(255, 150, 205), font=sub_font, anchor="mm")
        self._draw_pip_character(
            draw, cx=280, cy=520, scale=0.82,
            color_state="pink", expression="happy_proud", one_leg=False
        )
        draw.text((280, 960), "Teal 3D Eyes • Curved Beak • Gold Star", fill=(200, 235, 255), font=sub_font, anchor="mm")

        # Right Panel: Leo the Explorer Kid
        draw.rounded_rectangle([560, 160, 1040, 1040], radius=30, fill=(35, 48, 82, 255), outline=(255, 215, 0, 255), width=4)
        draw.text((800, 210), "COMPANION: LEO THE EXPLORER", fill=(255, 225, 100), font=sub_font, anchor="mm")
        self._draw_leo_character(draw, cx=800, cy=560, scale=1.1, expression="happy")
        draw.text((800, 960), "Yellow Cap • Big Brown Eyes • Teal Hoodie", fill=(200, 235, 255), font=sub_font, anchor="mm")

        img.save(self.reference_sheet_path, "PNG")
        return self.reference_sheet_path

    def generate_scene_image(self, scene: Any, filename_prefix: str = "scene") -> str:
        """
        Generates a 1080x1920 scene frame with consistent character faces and clear visual storytelling.
        1. Attempts Google AI Studio Multimodal Image Generation (`gemini-3.1-flash-image`) conditioned on
           `master_character_sheet.png` so character faces stay identical.
        2. If free-tier image quota is 0 (HTTP 429), renders our deterministic Character-Consistent 3D-Shaded
           Rig & Visual Storyboard scene so faces and story props are 100% accurate and watermark-free.
        """
        scene_id = getattr(scene, "scene_id", 1) if not isinstance(scene, int) else scene
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_{scene_id}.png")

        # Extract scene directives
        if isinstance(scene, int):
            prompt = "Vibrant 3D Pixar cartoon scene with Pip the Flamingo"
            expression = "happy_proud"
            color_state = "pink"
            visual_prop = "boiling_geyser"
            badge_text = f"SCENE {scene_id}"
            narration = ""
        else:
            prompt = getattr(scene, "visual_prompt", "")
            expression = getattr(scene, "character_expression", "happy_proud")
            color_state = getattr(scene, "character_color_state", "pink")
            visual_prop = getattr(scene, "visual_prop", "boiling_geyser")
            badge_text = getattr(scene, "visual_badge_text", "FUN FACT!")
            narration = getattr(scene, "narration", "")

        # 1. Try Google AI Studio Multimodal Reference-Conditioned Image Generation
        if self.api_key:
            ai_studio_ok = self._generate_via_ai_studio_multimodal(
                scene_id=scene_id,
                prompt=prompt,
                expression=expression,
                color_state=color_state,
                visual_prop=visual_prop,
                badge_text=badge_text,
                output_path=output_path
            )
            if ai_studio_ok and os.path.exists(output_path):
                return output_path

        # 2. Deterministic Character-Consistent 3D-Shaded Rig & Storyboard Renderer
        return self._render_consistent_character_scene(
            scene_id=scene_id,
            expression=expression,
            color_state=color_state,
            visual_prop=visual_prop,
            badge_text=badge_text,
            narration=narration,
            output_path=output_path
        )

    def generate_all_scenes_parallel(self, scenes: List[Any], filename_prefix: str = "scene") -> List[str]:
        """Generates all scene illustrations in parallel while preserving character identity across every frame."""
        results = [None] * len(scenes)

        def _task(idx, scene_obj):
            path = self.generate_scene_image(scene=scene_obj, filename_prefix=filename_prefix)
            return idx, path

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_task, i, s) for i, s in enumerate(scenes)]
            for fut in as_completed(futures):
                try:
                    idx, path = fut.result()
                    results[idx] = path
                except Exception as e:
                    logger.error(f"Error generating scene image in thread: {e}")

        for i in range(len(results)):
            if results[i] is None:
                results[i] = self.generate_scene_image(scene=scenes[i], filename_prefix=filename_prefix)

        return results

    def _generate_via_ai_studio_multimodal(
        self,
        scene_id: int,
        prompt: str,
        expression: str,
        color_state: str,
        visual_prop: str,
        badge_text: str,
        output_path: str
    ) -> bool:
        """
        Calls Google AI Studio `gemini-3.1-flash-image` with `master_character_sheet.png` passed as
        multimodal `inlineData` so every generated frame keeps the exact same character faces.
        """
        if not os.path.exists(self.reference_sheet_path):
            self._ensure_master_character_sheet()

        with open(self.reference_sheet_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode("utf-8")

        conditioned_prompt = (
            "Using the attached Master Character Identity Sheet, generate a vertical 9:16 (1080x1920) "
            "3D Pixar-style animated scene featuring the EXACT SAME face, big glossy teal-blue eyes, "
            "curved pink-and-black beak, and golden star badge of Pip the Flamingo (and Leo the Explorer Kid where applicable). "
            f"Scene Action: {prompt}. "
            f"Character Expression: {expression}. Feather Color State: {color_state}. "
            f"Key Visual Story Prop: {visual_prop}. Top Callout Badge: '{badge_text}'. "
            "Zero watermarks, clean studio lighting, crystal clear visual storytelling."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": ref_b64
                            }
                        },
                        {"text": conditioned_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"]
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    for part in candidates[0].get("content", {}).get("parts", []):
                        inline = part.get("inlineData")
                        if inline and inline.get("data"):
                            img_bytes = base64.b64decode(inline["data"])
                            with Image.open(io.BytesIO(img_bytes)) as im:
                                self._format_and_save_9_16(im, output_path)
                            logger.info(f"Generated AI Studio multimodal reference-locked image for scene {scene_id}.")
                            return True
        except Exception as e:
            logger.debug(f"AI Studio image model quota/status note for scene {scene_id}: {e}")
        return False

    def _format_and_save_9_16(self, img: Image.Image, output_path: str):
        img = img.convert("RGB")
        w, h = img.size
        target_ratio = self.width / self.height
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            cropped = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            cropped = img.crop((0, top, w, top + new_h))

        final_img = cropped.resize((self.width, self.height), Image.Resampling.LANCZOS)
        final_img.save(output_path, "PNG", quality=95)

    # =========================================================================
    # CHARACTER-CONSISTENT 3D-SHADED RIG & VISUAL STORYTELLER ENGINE
    # =========================================================================

    def _render_consistent_character_scene(
        self,
        scene_id: int,
        expression: str,
        color_state: str,
        visual_prop: str,
        badge_text: str,
        narration: str,
        output_path: str,
        anim_phase: float = 0.0
    ) -> str:
        """
        Renders a 1080x1920 rich storyboard scene featuring Pip the Flamingo and Leo the Explorer Kid
        with 100% consistent facial features across every scene and unmistakable visual story props.
        """
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img, "RGBA")

        # 1. Scene-specific environment background that matches the story beat
        self._draw_scene_environment(img, draw, visual_prop, scene_id)

        # 2. Render the specific visual story action & consistent characters
        low_narr = (narration or "").lower()
        is_octopus = "octopus" in visual_prop.lower() or "octopus" in low_narr
        is_cloud = "cloud" in visual_prop.lower() or "lightning" in visual_prop.lower() or "weather" in low_narr or "cloud" in low_narr
        is_chameleon = "chameleon" in visual_prop.lower() or "lizard" in visual_prop.lower() or "chameleon" in low_narr or "nanocrystal" in visual_prop.lower()

        # --- OCTOPUS & OCEAN EPISODES ---
        if is_octopus:
            if scene_id == 8 or visual_prop in ("octopus_superpower_choice", "choice_card"):
                # Scene 8: Choice Card / Loop Finale with Octopus and Leo waving together
                self._draw_octopus_character(
                    draw, cx=360, cy=1020, scale=0.98, color_state="coral",
                    expression="happy", anim_phase=anim_phase
                )
                self._draw_leo_character(
                    draw, cx=780, cy=1060, scale=1.15, expression="waving", anim_phase=anim_phase
                )
                self._draw_superpower_vs_cards(draw, y=520, narration=narration)

            elif scene_id == 1 or visual_prop in ("octopus_three_hearts", "three_hearts_xray"):
                # Scene 1: Octopus transparent body with 3 glowing hearts pumping + Leo reaction
                self._draw_octopus_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="coral",
                    expression="happy", hearts_visible=True, anim_phase=anim_phase
                )
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="LEO: 3 HEARTS?!")

            elif scene_id == 2 or visual_prop in ("octopus_blue_blood", "blue_blood_tube"):
                # Scene 2: Octopus in cold ocean with neon-blue copper blood circulation + test tube
                self._draw_octopus_character(
                    draw, cx=440, cy=980, scale=1.1, color_state="coral",
                    expression="happy", blue_blood=True, anim_phase=anim_phase
                )
                self._draw_blue_blood_tube_prop(draw, cx=760, cy=740, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="BLUE BLOOD!")

            elif scene_id == 3 or visual_prop in ("octopus_rice_eggs",):
                # Scene 3: Translucent ocean seaweed cradle with tiny rice-sized baby octopuses hatching!
                self._draw_octopus_rice_eggs_prop(draw, cx=540, cy=960, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="RICE-SIZED!")

            elif scene_id == 4 or visual_prop in ("octopus_catching_shrimp",):
                # Scene 4: Octopus using tentacles to scoop up glowing deep-sea shrimp!
                self._draw_octopus_character(
                    draw, cx=420, cy=980, scale=1.05, color_state="coral",
                    expression="happy", anim_phase=anim_phase
                )
                self._draw_glowing_shrimp_school(draw, cx=780, cy=980, anim_phase=anim_phase)
                self._draw_pigment_arrows(draw, start_xy=(730, 960), end_xy=(510, 960))
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="SHRIMP FEAST!")

            elif scene_id == 5 or visual_prop in ("octopus_03s_camouflage",):
                # Scene 5: Instant 0.3s color transformation from coral orange to neon tiger camouflage!
                self._draw_transformation_aura(draw, cx=540, cy=940)
                self._draw_octopus_character(
                    draw, cx=540, cy=960, scale=1.15, color_state="camouflage",
                    expression="happy", anim_phase=anim_phase
                )
                self._draw_color_progress_meter(draw, cx=540, y=1440, narration=narration)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="0.3s SHIFT!")

            elif scene_id == 6 or visual_prop in ("octopus_coral_sleep",):
                # Scene 6: Octopus curled up fast asleep in glowing coral reef dreaming in neon colors!
                self._draw_coral_reef_bed(draw, cy=1320)
                self._draw_octopus_character(
                    draw, cx=540, cy=980, scale=1.1, color_state="sleep_blue",
                    expression="sleeping", anim_phase=anim_phase
                )
                self._draw_zzz_bubbles(draw, start_x=680, start_y=640, anim_phase=anim_phase)
                self._draw_color_dream_thought_bubble(draw, cx=740, cy=460)

            elif scene_id == 7 or visual_prop in ("octopus_nine_brains",):
                # Scene 7: Full Octopus showing 1 Central Brain + 8 Tentacle Mini-Brains!
                self._draw_octopus_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="coral",
                    expression="happy", nine_brains=True, anim_phase=anim_phase
                )
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="9 BRAINS SYNC!")

            else:
                self._draw_octopus_character(
                    draw, cx=540, cy=960, scale=1.15, color_state="coral",
                    expression="happy", anim_phase=anim_phase
                )

        # --- CLOUD & WEATHER EPISODES ---
        elif is_cloud:
            if scene_id == 8 or visual_prop in ("cloud_superpower_choice", "choice_card"):
                # Scene 8: Choice Card / Loop Finale with Fluffy Cloud and Leo waving together
                self._draw_fluffy_cloud(
                    draw, cx=320, cy=1020, scale=1.05,
                    color=(255, 255, 255), expression="happy", anim_phase=anim_phase
                )
                self._draw_leo_character(
                    draw, cx=770, cy=1060, scale=1.15, expression="waving", anim_phase=anim_phase
                )
                self._draw_superpower_vs_cards(draw, y=520, narration=narration)

            elif scene_id == 1 or visual_prop in ("cloud_steam_rise",):
                # Scene 1: Ocean water at bottom, steaming updraft particles rising into the sky to build cloud
                self._draw_steam_rise_prop(draw, cx=540, cy=1220, anim_phase=anim_phase)
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=740, scale=1.1,
                    color=(255, 255, 255), expression="shocked", anim_phase=anim_phase
                )
                self._draw_thermometer_prop(draw, x=150, y=520, temp_text="100°C STEAM")
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="STEAM RISES!")

            elif scene_id == 2 or visual_prop in ("cloud_atmosphere_shield",):
                # Scene 2: Atmosphere aerodynamic shield holding up 1,000,000 lbs of water droplets
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=860, scale=1.35,
                    color=(255, 255, 255), expression="happy", anim_phase=anim_phase
                )
                self._draw_atmosphere_shield_prop(draw, cx=540, cy=860, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="1,000,000 LBS!")

            elif scene_id == 3 or visual_prop in ("cloud_micro_droplets",):
                # Scene 3: Cute fluffy cloud with magnifying bubble showing billions of micro water droplets
                self._draw_fluffy_cloud(
                    draw, cx=360, cy=880, scale=1.0,
                    color=(255, 255, 255), expression="happy", anim_phase=anim_phase
                )
                self._draw_cloud_micro_droplets_prop(draw, cx=740, cy=880, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="TINY DROPLETS!")

            elif scene_id == 4 or visual_prop in ("cloud_sunlight_scattering",):
                # Scene 4: Golden radiant sun scattering prism light waves through droplets into pure white glow
                self._draw_cloud_sunlight_scattering_prop(draw, cx=260, cy=600, anim_phase=anim_phase)
                self._draw_fluffy_cloud(
                    draw, cx=580, cy=960, scale=1.15,
                    color=(255, 255, 255), expression="happy", anim_phase=anim_phase
                )
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="PRISM LIGHT!")

            elif scene_id == 5 or visual_prop in ("cloud_sunset_neon",):
                # Scene 5: Sunset turning cloud into glowing neon pink and golden cotton candy!
                self._draw_cloud_sunset_neon_prop(draw, cx=540, cy=900, anim_phase=anim_phase)
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=900, scale=1.25,
                    color=(255, 110, 180), outline=(255, 220, 120),
                    expression="happy", anim_phase=anim_phase
                )
                self._draw_color_progress_meter(draw, cx=540, y=1440, narration=narration)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="NEON SUNSET!")

            elif scene_id == 6 or visual_prop in ("cloud_ice_crystals",):
                # Scene 6: Freezing cold air turning droplets into sparkling geometric ice crystals
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=880, scale=1.15,
                    color=(215, 238, 255), outline=(130, 185, 245),
                    expression="sleeping", anim_phase=anim_phase
                )
                self._draw_cloud_ice_crystals_prop(draw, cx=540, cy=880, anim_phase=anim_phase)
                self._draw_thermometer_prop(draw, x=150, y=520, temp_text="-40°C FREEZE")
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="ICE CRYSTALS!")

            elif scene_id == 7 or visual_prop in ("lightning_electrostatic_lock",):
                # Scene 7: Stormy thundercloud bumping ice crystals with static charges sparking electric lightning!
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=780, scale=1.25,
                    color=(68, 72, 98), outline=(145, 155, 200),
                    expression="shocked", anim_phase=anim_phase
                )
                self._draw_lightning_bolt_prop(draw, cx=540, cy=980, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="LIGHTNING SPARK!")

            else:
                self._draw_fluffy_cloud(
                    draw, cx=540, cy=920, scale=1.2,
                    color=(255, 255, 255), expression="happy", anim_phase=anim_phase
                )

        # --- CHAMELEON & COLOR EPISODES ---
        elif is_chameleon:
            if scene_id == 8 or visual_prop in ("chameleon_superpower_choice", "choice_card"):
                # Scene 8: Choice Card / Loop Finale with Chameleon and Leo waving together
                self._draw_tree_branch(draw, cy=1140)
                self._draw_chameleon_character(
                    draw, cx=360, cy=1020, scale=1.05, color_state="green", anim_phase=anim_phase
                )
                self._draw_leo_character(
                    draw, cx=780, cy=1060, scale=1.15, expression="waving", anim_phase=anim_phase
                )
                self._draw_superpower_vs_cards(draw, y=520, narration=narration)

            elif scene_id == 1 or visual_prop in ("chameleon_myth_busted",):
                # Scene 1: Chameleon on branch + big NOT CAMOUFLAGE badge
                self._draw_tree_branch(draw, cy=1120)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="green", anim_phase=anim_phase
                )
                self._draw_comparison_callout(draw, x=540, y=540, title="NOT CAMOUFLAGE!", subtitle="THEY DON'T HIDE IN TREES", color=(255, 60, 90))
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="NOT CAMO?!")

            elif scene_id == 2 or visual_prop in ("chameleon_mood_ring_shift",):
                # Scene 2: Mood ring shift based on temperature and feelings
                self._draw_tree_branch(draw, cy=1120)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="half_orange", anim_phase=anim_phase
                )
                self._draw_thermometer_prop(draw, x=150, y=520, temp_text="WARM SUN")
                self._draw_comparison_callout(draw, x=750, y=540, title="MOOD & HEAT!", subtitle="SHIFTS LIKE A MOOD RING", color=(255, 180, 50))
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="MOOD SHIFT!")

            elif scene_id == 3 or visual_prop in ("chameleon_nanocrystal_zoom",):
                # Scene 3: Magnifying glass showing grid of photonic nanocrystals like tiny mirrors
                self._draw_tree_branch(draw, cy=1160)
                self._draw_chameleon_character(
                    draw, cx=330, cy=1040, scale=0.98, color_state="green", anim_phase=anim_phase
                )
                self._draw_nanocrystals_zoom_prop(draw, cx=740, cy=920, anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="NANOCRYSTALS!")

            elif scene_id == 4 or visual_prop in ("chameleon_relaxed_green",):
                # Scene 4: Relaxed green chameleon with tight crystal lattice
                self._draw_tree_branch(draw, cy=1120)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="green", anim_phase=anim_phase
                )
                self._draw_crystal_spacing_diagram(draw, cx=540, y=540, state="tight", label="TIGHT CRYSTALS -> BLUE/GREEN")
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="TIGHT = GREEN!")

            elif scene_id == 5 or visual_prop in ("chameleon_excited_neon_red",):
                # Scene 5: Excited chameleon transformed into fiery yellow and red stripes!
                self._draw_tree_branch(draw, cy=1120)
                self._draw_transformation_aura(draw, cx=540, cy=960)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.2, color_state="fiery_red", anim_phase=anim_phase
                )
                self._draw_crystal_spacing_diagram(draw, cx=540, y=540, state="spread", label="SPREAD CRYSTALS -> RED/YELLOW")
                self._draw_color_progress_meter(draw, cx=540, y=1440, narration=narration)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="FIERY RED!")

            elif scene_id == 6 or visual_prop in ("chameleon_radar_eyes",):
                # Scene 6: Chameleon close up with 360 dual radar eyes
                self._draw_tree_branch(draw, cy=1140)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.3, color_state="green", eye_mode="dual_radar", anim_phase=anim_phase
                )
                self._draw_radar_eye_arcs(draw, cx=540, cy=960)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="360° EYES!")

            elif scene_id == 7 or visual_prop in ("chameleon_ballistic_tongue",):
                # Scene 7: Ballistic spring tongue shooting at 60 mph!
                self._draw_tree_branch(draw, cy=1140)
                self._draw_chameleon_character(
                    draw, cx=320, cy=1020, scale=1.05, color_state="green", tongue=True, anim_phase=anim_phase
                )
                self._draw_ballistic_tongue_prop(draw, start_xy=(535, 1015), end_xy=(880, 880), anim_phase=anim_phase)
                self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="60 MPH JET!")

            else:
                self._draw_tree_branch(draw, cy=1120)
                self._draw_chameleon_character(
                    draw, cx=540, cy=960, scale=1.15, color_state="green", anim_phase=anim_phase
                )

        # --- FLAMINGO EPISODES ---
        elif visual_prop == "boiling_geyser":
            self._draw_boiling_geyser_prop(draw, cx=330, cy=1220, anim_phase=anim_phase)
            self._draw_pip_character(
                draw, cx=660, cy=980, scale=1.05,
                color_state="pink", expression="shocked", one_leg=False, anim_phase=anim_phase
            )
            self._draw_thermometer_prop(draw, x=150, y=520, temp_text="100°C / 212°F")
            self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="shocked", label="LEO: BOILING?!")

        elif visual_prop == "beak_shield":
            self._draw_pip_character(
                draw, cx=540, cy=1020, scale=1.28,
                color_state="pink", expression="happy_proud", one_leg=False, anim_phase=anim_phase
            )
            self._draw_beak_shield_prop(draw, cx=580, cy=780)
            self._draw_leo_reaction_corner(draw, cx=220, cy=1480, expression="happy", label="ARMOR BEAK!")

        elif visual_prop == "hatching_egg":
            self._draw_nest_and_egg_bottom(draw, cx=540, cy=1220)
            self._draw_pip_character(
                draw, cx=540, cy=980, scale=0.88,
                color_state="gray_chick", expression="baby_curious", one_leg=False, is_chick=True, anim_phase=anim_phase
            )
            self._draw_cracked_eggshell_top(draw, cx=540, cy=640)
            self._draw_comparison_callout(draw, x=830, y=560, title="NOT PINK!", subtitle="100% GRAY CHICK", color=(180, 195, 210))

        elif visual_prop == "glowing_shrimp":
            self._draw_pip_character(
                draw, cx=420, cy=980, scale=1.0,
                color_state="gray_chick", expression="eating_excited", one_leg=False, anim_phase=anim_phase
            )
            self._draw_glowing_shrimp_school(draw, cx=770, cy=1050, anim_phase=anim_phase)
            self._draw_pigment_arrows(draw, start_xy=(720, 960), end_xy=(480, 960))

        elif visual_prop == "color_transformation":
            self._draw_transformation_aura(draw, cx=540, cy=960)
            self._draw_pip_character(
                draw, cx=540, cy=980, scale=1.12,
                color_state="half_pink_transforming", expression="transforming_amazed", one_leg=False, anim_phase=anim_phase
            )
            self._draw_color_progress_meter(draw, cx=540, y=1440, narration=narration)

        elif visual_prop == "one_leg_ice":
            self._draw_ice_floes(draw, cy=1340)
            self._draw_pip_character(
                draw, cx=540, cy=980, scale=1.08,
                color_state="pink", expression="sleeping_peaceful", one_leg=True, anim_phase=anim_phase
            )
            self._draw_zzz_bubbles(draw, start_x=680, start_y=620, anim_phase=anim_phase)
            self._draw_one_leg_pointer(draw, target_x=520, target_y=1240, narration=narration)

        elif visual_prop == "knee_gear_lock":
            self._draw_pip_character(
                draw, cx=340, cy=980, scale=1.0,
                color_state="pink", expression="xray_focused", one_leg=True, anim_phase=anim_phase
            )
            self._draw_xray_gear_lock_diagram(draw, target_leg_xy=(330, 1210), zoom_center_xy=(740, 960), narration=narration)

        else:
            # Universal Choice Card / Finale
            self._draw_pip_character(
                draw, cx=330, cy=1020, scale=0.95,
                color_state="pink", expression="waving_cheerful", one_leg=True, anim_phase=anim_phase
            )
            self._draw_leo_character(
                draw, cx=770, cy=1060, scale=1.15, expression="waving", anim_phase=anim_phase
            )
            self._draw_superpower_vs_cards(draw, y=520, narration=narration)

        # 3. Draw Top Storyboard Header & Visual Fact Callout Badge (so muted viewers always follow the story)
        self._draw_top_story_banner(draw, scene_id=scene_id, total_scenes=8, badge_text=badge_text)

        img.save(output_path, "PNG")
        return output_path

    def _draw_scene_environment(self, img: Image.Image, draw: ImageDraw.ImageDraw, visual_prop: str, scene_id: int):
        """Draws a clean, rich 3D-styled environment gradient, water/ground, and atmospheric lighting."""
        palettes = {
            "boiling_geyser": ((18, 45, 92), (0, 168, 204), (0, 128, 170)),
            "beak_shield": ((42, 24, 88), (220, 75, 120), (35, 95, 145)),
            "hatching_egg": ((25, 95, 130), (130, 220, 165), (65, 155, 95)),
            "glowing_shrimp": ((10, 50, 110), (0, 190, 235), (0, 110, 185)),
            "color_transformation": ((55, 20, 110), (255, 90, 180), (45, 30, 95)),
            "one_leg_ice": ((12, 22, 58), (35, 75, 140), (20, 55, 115)),
            "knee_gear_lock": ((15, 38, 78), (0, 150, 195), (12, 65, 115)),
            "three_hearts_xray": ((8, 32, 78), (0, 145, 215), (10, 52, 115)),
            "blue_blood_tube": ((10, 25, 68), (0, 175, 245), (12, 60, 128)),
            "choice_card": ((35, 25, 95), (255, 120, 90), (30, 80, 140)),
            "octopus_three_hearts": ((6, 22, 60), (0, 110, 195), (4, 45, 95)),
            "octopus_blue_blood": ((8, 25, 75), (0, 155, 235), (6, 50, 105)),
            "octopus_rice_eggs": ((12, 35, 70), (0, 140, 180), (10, 65, 100)),
            "octopus_catching_shrimp": ((10, 30, 80), (0, 160, 210), (8, 55, 110)),
            "octopus_03s_camouflage": ((25, 15, 65), (0, 185, 200), (20, 30, 80)),
            "octopus_coral_sleep": ((5, 12, 45), (15, 45, 110), (8, 25, 75)),
            "octopus_nine_brains": ((10, 20, 65), (0, 140, 210), (8, 45, 100)),
            "octopus_superpower_choice": ((15, 25, 75), (0, 165, 230), (12, 50, 110)),
            "cloud_steam_rise": ((18, 45, 95), (0, 180, 225), (10, 85, 150)),
            "cloud_atmosphere_shield": ((20, 70, 140), (70, 190, 255), (20, 120, 180)),
            "cloud_micro_droplets": ((18, 60, 125), (50, 175, 245), (15, 100, 165)),
            "cloud_sunlight_scattering": ((30, 80, 160), (120, 210, 255), (40, 140, 190)),
            "cloud_sunset_neon": ((55, 18, 90), (255, 110, 130), (70, 30, 95)),
            "cloud_ice_crystals": ((8, 20, 60), (45, 95, 165), (15, 45, 95)),
            "lightning_electrostatic_lock": ((12, 12, 38), (35, 30, 75), (18, 18, 48)),
            "cloud_superpower_choice": ((20, 50, 115), (0, 175, 235), (20, 110, 175)),
            "chameleon_myth_busted": ((15, 65, 45), (40, 180, 110), (25, 95, 55)),
            "chameleon_mood_ring_shift": ((20, 60, 50), (255, 175, 40), (30, 90, 60)),
            "chameleon_nanocrystal_zoom": ((10, 35, 75), (0, 195, 230), (12, 60, 100)),
            "chameleon_relaxed_green": ((15, 70, 40), (50, 210, 115), (20, 100, 50)),
            "chameleon_excited_neon_red": ((70, 20, 30), (255, 95, 50), (90, 30, 40)),
            "chameleon_radar_eyes": ((18, 55, 45), (45, 175, 120), (22, 85, 55)),
            "chameleon_ballistic_tongue": ((20, 45, 60), (255, 110, 130), (25, 75, 85)),
            "chameleon_superpower_choice": ((15, 55, 50), (0, 180, 160), (20, 85, 75)),
        }
        sky_top, sky_bot, ground_col = palettes.get(visual_prop, ((20, 60, 120), (0, 180, 220), (0, 120, 170)))

        horizon_y = int(self.height * 0.68)
        for y in range(horizon_y):
            t = y / float(horizon_y)
            r = int(sky_top[0] * (1 - t) + sky_bot[0] * t)
            g = int(sky_top[1] * (1 - t) + sky_bot[1] * t)
            b = int(sky_top[2] * (1 - t) + sky_bot[2] * t)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        # Water / Lagoon surface below horizon
        for y in range(horizon_y, self.height):
            t = (y - horizon_y) / float(self.height - horizon_y)
            r = int(ground_col[0] * (1 - 0.35 * t))
            g = int(ground_col[1] * (1 - 0.35 * t))
            b = int(ground_col[2] * (1 - 0.35 * t))
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        # Water ripples
        rng = random.Random(scene_id * 101)
        for _ in range(14):
            rx = rng.randint(120, self.width - 120)
            ry = rng.randint(horizon_y + 40, self.height - 80)
            rw = rng.randint(90, 220)
            draw.arc([rx - rw, ry - 12, rx + rw, ry + 12], start=0, end=180, fill=(255, 255, 255, 90), width=4)

        # Soft clouds or stars
        if visual_prop in ("one_leg_ice", "color_transformation"):
            for _ in range(35):
                sx = rng.randint(40, self.width - 40)
                sy = rng.randint(240, horizon_y - 60)
                sr = rng.randint(3, 7)
                draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 255, 220, 220))
            if visual_prop == "one_leg_ice":
                # Glowing crescent moon
                draw.ellipse([120, 320, 260, 460], fill=(255, 248, 190, 255))
                draw.ellipse([160, 305, 280, 435], fill=sky_top)
        else:
            for cx, cy, cw, ch in [(220, 380, 140, 55), (840, 420, 160, 60), (540, 330, 130, 50)]:
                draw.ellipse([cx - cw, cy - ch, cx + cw, cy + ch], fill=(255, 255, 255, 165))
                draw.ellipse([cx - cw // 2, cy - int(ch * 1.3), cx + cw // 2, cy + ch // 2], fill=(255, 255, 255, 185))

    def _draw_pip_character(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        scale: float = 1.0,
        color_state: str = "pink",
        expression: str = "happy_proud",
        one_leg: bool = False,
        is_chick: bool = False,
        anim_phase: float = 0.0
    ):
        """
        Draws "Pip the Flamingo" with 100% consistent facial structure, big glossy teal-blue 3D eyes,
        signature curved pink-and-black beak, rosy cheeks, and golden explorer star badge.
        """
        bob_y = int(math.sin(anim_phase * 2 * math.pi) * 10 * scale)
        cy = cy + bob_y

        # Determine feather palette from color_state
        if color_state == "gray_chick":
            body_main = (195, 205, 218, 255)
            body_shadow = (155, 168, 184, 255)
            wing_col = (218, 226, 236, 255)
            outline_col = (105, 118, 135, 255)
        elif color_state == "half_pink_transforming":
            body_main = (255, 95, 165, 255)
            body_shadow = (180, 190, 205, 255)
            wing_col = (255, 145, 198, 255)
            outline_col = (255, 235, 100, 255)
        else:
            body_main = (255, 88, 158, 255)
            body_shadow = (224, 52, 124, 255)
            wing_col = (255, 138, 190, 255)
            outline_col = (145, 24, 78, 255)

        leg_col = (255, 145, 75, 255)
        joint_col = (255, 95, 130, 255)

        # 1. Legs (Standing on two legs OR sleeping/locked on ONE leg)
        leg_top_y = cy + int(75 * scale)
        foot_y = cy + int(310 * scale)
        knee_y = cy + int(185 * scale)

        # Left standing leg (always down)
        lx = cx - int(28 * scale) if not one_leg else cx - int(8 * scale)
        draw.line([(lx, leg_top_y), (lx, foot_y)], fill=leg_col, width=max(int(16 * scale), 6))
        # Webbed foot
        draw.polygon(
            [(lx, foot_y), (lx - int(45 * scale), foot_y + int(18 * scale)), (lx + int(45 * scale), foot_y + int(18 * scale))],
            fill=leg_col, outline=outline_col
        )
        # Prominent ankle/knee joint circle
        jr = int(16 * scale)
        draw.ellipse([lx - jr, knee_y - jr, lx + jr, knee_y + jr], fill=joint_col, outline=outline_col, width=3)

        # Right leg (either standing or tucked up into triangle for 1-leg balance!)
        rx = cx + int(38 * scale)
        if one_leg:
            # Tucked leg bent sharply into a figure-4!
            tuck_knee_x = cx + int(95 * scale)
            tuck_knee_y = cy + int(155 * scale)
            draw.line([(rx, leg_top_y), (tuck_knee_x, tuck_knee_y)], fill=leg_col, width=max(int(15 * scale), 6))
            draw.line([(tuck_knee_x, tuck_knee_y), (lx + int(10 * scale), knee_y - int(15 * scale))], fill=leg_col, width=max(int(14 * scale), 6))
            draw.ellipse([tuck_knee_x - jr, tuck_knee_y - jr, tuck_knee_x + jr, tuck_knee_y + jr], fill=joint_col, outline=outline_col, width=3)
        else:
            draw.line([(rx, leg_top_y), (rx, foot_y)], fill=leg_col, width=max(int(16 * scale), 6))
            draw.polygon(
                [(rx, foot_y), (rx - int(45 * scale), foot_y + int(18 * scale)), (rx + int(45 * scale), foot_y + int(18 * scale))],
                fill=leg_col, outline=outline_col
            )
            draw.ellipse([rx - jr, knee_y - jr, rx + jr, knee_y + jr], fill=joint_col, outline=outline_col, width=3)

        # 2. Body Oval (Plump 3D shaded body)
        bw = int((145 if not is_chick else 125) * scale)
        bh = int((110 if not is_chick else 105) * scale)
        body_box = [cx - bw, cy - bh // 2, cx + bw, cy + bh]
        draw.ellipse(body_box, fill=body_main, outline=outline_col, width=5)

        # Half-gray / half-pink split effect for Scene 5 transformation
        if color_state == "half_pink_transforming":
            draw.chord(body_box, start=90, end=270, fill=(190, 202, 216, 255), outline=outline_col, width=4)
            # Glowing vertical transformation line
            draw.line([(cx, cy - int(320 * scale)), (cx, cy + int(160 * scale))], fill=(255, 255, 120, 255), width=8)

        # Fluffy tail feathers on left
        for angle_offset in (-25, 0, 25):
            tx = cx - bw - int(25 * scale)
            ty = cy + int(angle_offset * scale)
            draw.ellipse([tx - int(35 * scale), ty - int(22 * scale), tx + int(35 * scale), ty + int(22 * scale)], fill=body_shadow, outline=outline_col, width=4)

        # Expressive Wing
        if expression == "waving_cheerful":
            # Raised waving wing!
            wing_box = [cx - int(40 * scale), cy - int(95 * scale), cx + int(95 * scale), cy + int(25 * scale)]
        else:
            wing_box = [cx - int(85 * scale), cy - int(20 * scale), cx + int(65 * scale), cy + int(68 * scale)]
        draw.ellipse(wing_box, fill=wing_col, outline=outline_col, width=4)

        # 3. Graceful Curved Neck (shorter for baby chick, tall S-curve for adult)
        head_x = cx + int((55 if not is_chick else 20) * scale)
        head_y = cy - int((225 if not is_chick else 145) * scale)
        neck_points = [
            (cx + int(55 * scale), cy - int(15 * scale)),
            (cx + int(95 * scale), cy - int(95 * scale)),
            (head_x - int(15 * scale), cy - int(165 * scale)),
            (head_x, head_y)
        ]
        if is_chick:
            draw.line([(cx + int(15 * scale), cy - int(25 * scale)), (head_x, head_y)], fill=body_main, width=max(int(46 * scale), 12))
        else:
            draw.line(neck_points, fill=body_main, width=max(int(42 * scale), 12), joint="curve")

        # 4. SIGNATURE GOLDEN EXPLORER STAR BADGE on chest/neck base (Immutable Character Identity Marker)
        badge_cx = cx + int(62 * scale)
        badge_cy = cy - int(10 * scale)
        self._draw_gold_star_badge(draw, badge_cx, badge_cy, radius=int(26 * scale))

        # 5. CONSISTENT HEAD & FACE RIG (Identical facial geometry across ALL scenes)
        hr = int(82 * scale)
        head_box = [head_x - hr, head_y - hr, head_x + hr, head_y + hr]
        draw.ellipse(head_box, fill=body_main, outline=outline_col, width=5)

        # Fluffy crown tuft (always 3 cute feathers on top of Pip's head)
        for dx, dy in [(-18, -hr - 18), (0, -hr - 28), (18, -hr - 16)]:
            fx = head_x + int(dx * scale)
            fy = head_y + int(dy * scale)
            draw.ellipse([fx - int(12 * scale), fy - int(22 * scale), fx + int(12 * scale), fy + int(22 * scale)], fill=wing_col, outline=outline_col, width=3)

        # 6. SIGNATURE CURVED TWO-TONE FLAMINGO BEAK (Soft cream upper bridge, pink middle, glossy black tip)
        beak_start_x = head_x + int(52 * scale)
        beak_start_y = head_y - int(8 * scale)
        beak_mid_x = head_x + int(138 * scale)
        beak_mid_y = head_y + int(12 * scale)
        beak_tip_x = head_x + int(122 * scale)
        beak_tip_y = head_y + int(78 * scale)

        # Main pink/cream upper beak
        draw.polygon(
            [
                (beak_start_x, beak_start_y - int(28 * scale)),
                (beak_mid_x, beak_mid_y - int(15 * scale)),
                (beak_tip_x, beak_tip_y),
                (beak_start_x, beak_start_y + int(30 * scale))
            ],
            fill=(255, 185, 210, 255),
            outline=outline_col
        )
        # Signature black curved beak tip
        draw.polygon(
            [
                (beak_mid_x - int(25 * scale), beak_mid_y + int(5 * scale)),
                (beak_mid_x, beak_mid_y - int(5 * scale)),
                (beak_tip_x, beak_tip_y),
                (beak_start_x + int(32 * scale), beak_start_y + int(30 * scale))
            ],
            fill=(32, 36, 48, 255),
            outline=(15, 18, 25, 255)
        )

        # Open mouth / smiling beak line based on expression
        if expression in ("shocked", "eating_excited", "waving_cheerful"):
            draw.pieslice(
                [beak_start_x - int(10 * scale), beak_start_y + int(5 * scale), beak_start_x + int(55 * scale), beak_start_y + int(48 * scale)],
                start=0, end=180, fill=(220, 45, 95, 255), outline=outline_col, width=3
            )
        else:
            draw.arc(
                [beak_start_x - int(15 * scale), beak_start_y - int(5 * scale), beak_start_x + int(60 * scale), beak_start_y + int(30 * scale)],
                start=10, end=155, fill=(40, 20, 35, 255), width=4
            )

        # Rosy Pixar Cheeks
        cheek_r = int(20 * scale)
        for cx_off in (-32, 28):
            chx = head_x + int(cx_off * scale)
            chy = head_y + int(24 * scale)
            draw.ellipse([chx - cheek_r, chy - int(cheek_r * 0.65), chx + cheek_r, chy + int(cheek_r * 0.65)], fill=(255, 65, 130, 145))

        # 7. SIGNATURE LARGE GLOSSY TEAL-BLUE 3D EYES (Identical across all scenes)
        eye_y = head_y - int(12 * scale)
        left_eye_x = head_x - int(24 * scale)
        right_eye_x = head_x + int(26 * scale)
        er = int((26 if not is_chick else 30) * scale)

        for ex in (left_eye_x, right_eye_x):
            if expression == "sleeping_peaceful":
                # Peaceful closed curved eyelids with cute eyelashes
                draw.arc([ex - er, eye_y - er // 2, ex + er, eye_y + er // 2], start=10, end=170, fill=(40, 20, 45, 255), width=6)
            else:
                # White sclera
                draw.ellipse([ex - er, eye_y - er, ex + er, eye_y + er], fill=(255, 255, 255, 255), outline=(45, 30, 50, 255), width=4)
                # Glossy Teal-Blue Iris (Signature Identity Trait)
                ir = int(er * (0.72 if expression != "shocked" else 0.58))
                draw.ellipse([ex - ir + 3, eye_y - ir, ex + ir + 3, eye_y + ir], fill=(0, 185, 215, 255))
                # Deep Navy Pupil
                pr = int(ir * 0.58)
                draw.ellipse([ex - pr + 4, eye_y - pr, ex + pr + 4, eye_y + pr], fill=(15, 22, 42, 255))
                # Triple 3D Pixar Specular Catchlights
                hl1 = max(int(er * 0.28), 4)
                hl2 = max(int(er * 0.14), 2)
                draw.ellipse([ex - int(er * 0.35), eye_y - int(er * 0.45), ex - int(er * 0.35) + hl1, eye_y - int(er * 0.45) + hl1], fill=(255, 255, 255, 255))
                draw.ellipse([ex + int(er * 0.22), eye_y + int(er * 0.18), ex + int(er * 0.22) + hl2, eye_y + int(er * 0.18) + hl2], fill=(255, 255, 255, 255))

            # Expressive Eyebrows
            brow_y = eye_y - er - int(10 * scale)
            if expression == "shocked":
                draw.arc([ex - er, brow_y - 12, ex + er, brow_y + 12], start=190, end=350, fill=(60, 25, 45, 255), width=5)
            elif expression == "happy_proud":
                draw.line([(ex - er + 4, brow_y + 4), (ex + er - 4, brow_y - 4)], fill=(60, 25, 45, 255), width=5)
            else:
                draw.arc([ex - er, brow_y - 6, ex + er, brow_y + 10], start=200, end=340, fill=(60, 25, 45, 255), width=4)

    def _draw_octopus_character(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        scale: float = 1.0,
        color_state: str = "coral",
        expression: str = "happy",
        hearts_visible: bool = False,
        blue_blood: bool = False,
        nine_brains: bool = False,
        anim_phase: float = 0.0
    ):
        """
        Draws a vibrant, cute 3D-shaded Pixar-style Octopus hero with 8 animated wavy tentacles,
        big glossy eyes, suction cups, and optional anatomical layers (3 glowing hearts, blue blood,
        or 9 synapsing brain nodes).
        """
        bob_y = int(math.sin(anim_phase * 2 * math.pi) * 12 * scale)
        cy = cy + bob_y

        if color_state == "camouflage":
            body_col = (0, 225, 205, 255)
            shadow_col = (0, 145, 135, 255)
            highlight_col = (255, 235, 90, 255)
            sucker_col = (255, 245, 140, 255)
        elif color_state == "sleep_blue":
            body_col = (95, 115, 205, 255)
            shadow_col = (55, 75, 145, 255)
            highlight_col = (155, 175, 245, 255)
            sucker_col = (185, 205, 255, 255)
        else:
            body_col = (255, 110, 80, 255)
            shadow_col = (205, 65, 45, 255)
            highlight_col = (255, 170, 135, 255)
            sucker_col = (255, 220, 190, 255)

        tentacle_configs = [
            (-320, 260, -0.4), (-220, 310, -0.2), (-120, 335, -0.05), (-35, 345, 0.05),
            (35, 345, -0.05), (120, 335, 0.05), (220, 310, 0.2), (320, 260, 0.4)
        ]
        tentacle_base_y = cy + int(75 * scale)

        for i, (tx_off, ty_off, curve) in enumerate(tentacle_configs):
            wave = math.sin((anim_phase + i * 0.12) * 2 * math.pi) * 20 * scale
            base_x = cx + int(tx_off * 0.28 * scale)
            end_x = cx + int((tx_off + wave) * scale)
            end_y = cy + int(ty_off * scale)
            mid_x = (base_x + end_x) // 2 + int(curve * 140 * scale)
            mid_y = (tentacle_base_y + end_y) // 2

            w = max(int(26 * scale), 8)
            draw.line([(base_x, tentacle_base_y), (mid_x, mid_y)], fill=shadow_col, width=w + 4)
            draw.line([(mid_x, mid_y), (end_x, end_y)], fill=shadow_col, width=w + 4)
            draw.line([(base_x, tentacle_base_y), (mid_x, mid_y)], fill=body_col, width=w)
            draw.line([(mid_x, mid_y), (end_x, end_y)], fill=body_col, width=w)

            for frac in (0.35, 0.65, 0.9):
                sx = int(base_x * (1 - frac) + end_x * frac + curve * 25 * scale)
                sy = int(tentacle_base_y * (1 - frac) + end_y * frac)
                sr = int(8 * scale)
                draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=sucker_col, outline=shadow_col, width=2)

            if nine_brains:
                draw.ellipse([mid_x - 14, mid_y - 14, mid_x + 14, mid_y + 14], fill=(0, 255, 220, 255), outline=(255, 255, 255, 255), width=3)
                draw.line([(mid_x, mid_y), (cx, cy - int(40 * scale))], fill=(255, 215, 0, 220), width=4)

        hw = int(145 * scale)
        hh = int(170 * scale)
        draw.ellipse([cx - hw - 6, cy - hh - 6, cx + hw + 6, cy + int(90 * scale)], fill=shadow_col)
        draw.ellipse([cx - hw, cy - hh, cx + hw, cy + int(85 * scale)], fill=body_col)
        draw.ellipse([cx - int(hw * 0.7), cy - hh + int(12 * scale), cx + int(hw * 0.7), cy - int(hh * 0.3)], fill=highlight_col)

        if color_state == "camouflage":
            for sy_off in (-100, -50, 0, 50):
                py = cy + int(sy_off * scale)
                draw.arc([cx - hw + 15, py - 20, cx + hw - 15, py + 20], start=10, end=170, fill=(255, 230, 80, 230), width=int(12 * scale))

        if hearts_visible:
            draw.rounded_rectangle([cx - int(105 * scale), cy - int(155 * scale), cx + int(105 * scale), cy - int(45 * scale)], radius=int(25 * scale), fill=(10, 25, 65, 210), outline=(0, 240, 255, 255), width=4)
            pulse = int(math.sin(anim_phase * 4 * math.pi) * 6 * scale)
            for hx_off in (-50, 50):
                hx = cx + int(hx_off * scale)
                hy = cy - int(95 * scale)
                hr = int(20 * scale) + pulse
                draw.ellipse([hx - hr - 4, hy - hr - 4, hx + hr + 4, hy + hr + 4], fill=(255, 40, 100, 110))
                draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 45, 105, 255), outline=(255, 255, 255, 255), width=3)
            cx_h = cx
            cy_h = cy - int(120 * scale)
            chr = int(24 * scale) + pulse
            draw.ellipse([cx_h - chr - 5, cy_h - chr - 5, cx_h + chr + 5, cy_h + chr + 5], fill=(255, 200, 0, 120))
            draw.ellipse([cx_h - chr, cy_h - chr, cx_h + chr, cy_h + chr], fill=(255, 215, 0, 255), outline=(255, 255, 255, 255), width=3)
            hfont = self._get_font(18)
            draw.text((cx, cy - int(60 * scale)), "3 HEARTS", fill=(255, 255, 255), font=hfont, anchor="mm")

        if blue_blood:
            for bx_off in (-65, 0, 65):
                bx = cx + int(bx_off * scale)
                draw.line([(bx, cy - int(110 * scale)), (bx + int(bx_off * 0.4), cy + int(70 * scale))], fill=(0, 235, 255, 240), width=int(10 * scale))
                for bub_y in (-80, -30, 20):
                    draw.ellipse([bx - 6, cy + int(bub_y * scale) - 6, bx + 6, cy + int(bub_y * scale) + 6], fill=(220, 250, 255, 255))

        if nine_brains:
            cb_y = cy - int(85 * scale)
            draw.ellipse([cx - int(38 * scale), cb_y - int(38 * scale), cx + int(38 * scale), cb_y + int(38 * scale)], fill=(255, 215, 0, 255), outline=(255, 255, 255, 255), width=4)
            bfont = self._get_font(22)
            draw.text((cx, cb_y), "MAIN BRAIN", fill=(20, 30, 60), font=bfont, anchor="mm")

        eye_y = cy - int(10 * scale)
        for ex_off in (-52, 52):
            ex = cx + int(ex_off * scale)
            er = int(32 * scale)

            if expression == "sleeping":
                draw.arc([ex - er, eye_y - er // 2, ex + er, eye_y + er // 2], start=190, end=350, fill=(35, 25, 45, 255), width=int(6 * scale))
            else:
                draw.ellipse([ex - er, eye_y - er, ex + er, eye_y + er], fill=(255, 255, 255, 255), outline=shadow_col, width=4)
                ir = int(er * 0.72)
                draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir], fill=(0, 165, 195, 255))
                pw = int(ir * 0.75)
                ph = int(ir * 0.38)
                draw.ellipse([ex - pw, eye_y - ph, ex + pw, eye_y + ph], fill=(15, 20, 35, 255))
                draw.ellipse([ex - int(er * 0.4), eye_y - int(er * 0.45), ex - int(er * 0.4) + int(10 * scale), eye_y - int(er * 0.45) + int(10 * scale)], fill=(255, 255, 255, 255))

        for ch_off in (-78, 78):
            draw.ellipse([cx + int(ch_off * scale) - 18, cy + int(18 * scale) - 12, cx + int(ch_off * scale) + 18, cy + int(18 * scale) + 12], fill=(255, 90, 120, 140))

        if expression != "sleeping":
            draw.arc([cx - int(24 * scale), cy + int(16 * scale), cx + int(24 * scale), cy + int(42 * scale)], start=20, end=160, fill=(45, 20, 30, 255), width=int(5 * scale))

    def _draw_leo_character(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        scale: float = 1.0,
        expression: str = "happy",
        anim_phase: float = 0.0
    ):
        """
        Draws "Leo the Explorer Kid" (7-year-old boy with warm tan skin, curly dark brown hair,
        bright yellow explorer cap, big brown 3D eyes, and teal explorer hoodie) with 100% face consistency.
        """
        skin_col = (238, 182, 138, 255)
        skin_outline = (145, 92, 58, 255)
        hair_col = (58, 36, 24, 255)
        cap_col = (255, 210, 25, 255)
        hoodie_col = (0, 175, 195, 255)

        # 1. Teal Explorer Hoodie Torso
        tw = int(95 * scale)
        th = int(125 * scale)
        draw.rounded_rectangle([cx - tw, cy + int(55 * scale), cx + tw, cy + int(55 * scale) + th], radius=int(35 * scale), fill=hoodie_col, outline=(0, 105, 125, 255), width=5)
        # Golden star explorer badge on hoodie
        self._draw_gold_star_badge(draw, cx, cy + int(115 * scale), radius=int(22 * scale))

        # 2. Head Circle
        hr = int(78 * scale)
        draw.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=skin_col, outline=skin_outline, width=5)

        # Ears
        er_ear = int(20 * scale)
        draw.ellipse([cx - hr - er_ear // 2, cy - er_ear, cx - hr + er_ear // 2, cy + er_ear], fill=skin_col, outline=skin_outline, width=4)
        draw.ellipse([cx + hr - er_ear // 2, cy - er_ear, cx + hr + er_ear // 2, cy + er_ear], fill=skin_col, outline=skin_outline, width=4)

        # 3. Curly Dark Brown Hair puffs
        for dx, dy, r_puff in [(-55, -52, 26), (-25, -66, 28), (15, -66, 28), (52, -52, 26)]:
            px = cx + int(dx * scale)
            py = cy + int(dy * scale)
            pr = int(r_puff * scale)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=hair_col)

        # 4. Signature Bright Yellow Explorer Cap
        draw.pieslice([cx - hr - 6, cy - int(128 * scale), cx + hr + 6, cy - int(25 * scale)], start=180, end=360, fill=cap_col, outline=(180, 130, 0, 255), width=4)
        draw.rounded_rectangle([cx - hr - int(15 * scale), cy - int(82 * scale), cx + hr + int(35 * scale), cy - int(62 * scale)], radius=10, fill=(255, 190, 0, 255), outline=(180, 130, 0, 255), width=3)

        # 5. Big Expressive Brown 3D Pixar Eyes
        eye_y = cy - int(8 * scale)
        for ex_off in (-30, 30):
            ex = cx + int(ex_off * scale)
            er = int(22 * scale)
            draw.ellipse([ex - er, eye_y - er, ex + er, eye_y + er], fill=(255, 255, 255, 255), outline=(55, 35, 25, 255), width=3)
            ir = int(er * (0.68 if expression != "shocked" else 0.55))
            draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir], fill=(115, 62, 28, 255))
            pr = int(ir * 0.55)
            draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(20, 15, 12, 255))
            # Catchlight
            hl = max(int(er * 0.28), 4)
            draw.ellipse([ex - int(er * 0.35), eye_y - int(er * 0.4), ex - int(er * 0.35) + hl, eye_y - int(er * 0.4) + hl], fill=(255, 255, 255, 255))

        # Cute nose & rosy cheeks
        draw.ellipse([cx - int(8 * scale), cy + int(6 * scale), cx + int(8 * scale), cy + int(18 * scale)], fill=(215, 150, 108, 255))
        for ch_off in (-44, 44):
            draw.ellipse([cx + int(ch_off * scale) - int(16 * scale), cy + int(14 * scale), cx + int(ch_off * scale) + int(16 * scale), cy + int(32 * scale)], fill=(255, 105, 115, 130))

        # Mouth expression
        if expression == "shocked":
            mr = int(18 * scale)
            draw.ellipse([cx - mr, cy + int(28 * scale), cx + mr, cy + int(28 * scale) + int(mr * 1.4)], fill=(185, 45, 65, 255), outline=skin_outline, width=3)
        else:
            draw.pieslice([cx - int(28 * scale), cy + int(18 * scale), cx + int(28 * scale), cy + int(56 * scale)], start=0, end=180, fill=(220, 60, 80, 255), outline=skin_outline, width=3)

    def _draw_leo_reaction_corner(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, expression: str, label: str):
        """Draws a circular Picture-in-Picture reaction cam of Leo the Explorer Kid so both recurring characters appear."""
        r = 135
        draw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=(255, 215, 0, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(32, 48, 88, 255))
        self._draw_leo_character(draw, cx=cx, cy=cy - 10, scale=0.72, expression=expression)
        draw.rounded_rectangle([cx - 130, cy + r - 32, cx + 130, cy + r + 22], radius=16, fill=(255, 60, 110, 255), outline=(255, 255, 255, 255), width=3)
        font = self._get_font(26)
        draw.text((cx, cy + r - 5), label, fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_gold_star_badge(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int = 24):
        """Draws the signature 5-point golden star badge worn by Pip and Leo in every scene."""
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = radius if i % 2 == 0 else int(radius * 0.45)
            pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
        draw.polygon(pts, fill=(255, 215, 0, 255), outline=(185, 125, 0, 255))

    # =========================================================================
    # VISUAL STORYTELLING PROPS & DIAGRAMS (FOR 100% SILENT COMPREHENSION)
    # =========================================================================

    def _draw_boiling_geyser_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws a bubbling volcanic hot spring geyser with rising steam clouds and bubbles."""
        # Volcanic rock rim
        draw.ellipse([cx - 210, cy - 55, cx + 210, cy + 75], fill=(95, 88, 98, 255), outline=(55, 48, 58, 255), width=5)
        # Glowing hot turquoise-and-orange bubbling water
        draw.ellipse([cx - 175, cy - 38, cx + 175, cy + 52], fill=(255, 115, 55, 255), outline=(255, 210, 80, 255), width=4)
        # Boiling bubbles & rising steam puffs
        shift = int(anim_phase * 35)
        for bx, by, br in [(-90, -10, 22), (-25, 12, 28), (45, -8, 24), (95, 10, 20), (0, -65 - shift, 35), (-55, -135 - shift, 45), (40, -195 - shift, 52)]:
            draw.ellipse([cx + bx - br, cy + by - br, cx + bx + br, cy + by + br], fill=(255, 245, 235, 185), outline=(255, 255, 255, 230), width=3)

    def _draw_thermometer_prop(self, draw: ImageDraw.ImageDraw, x: int, y: int, temp_text: str):
        """Draws a giant clear 100°C boiling thermometer icon so muted viewers immediately see 'BOILING HOT'."""
        # Outer white tube
        draw.rounded_rectangle([x - 36, y, x + 36, y + 320], radius=36, fill=(245, 250, 255, 255), outline=(40, 55, 80, 255), width=6)
        draw.ellipse([x - 68, y + 275, x + 68, y + 410], fill=(245, 250, 255, 255), outline=(40, 55, 80, 255), width=6)
        # Red boiling mercury all the way to the top
        draw.rounded_rectangle([x - 20, y + 25, x + 20, y + 310], radius=20, fill=(255, 35, 65, 255))
        draw.ellipse([x - 50, y + 292, x + 50, y + 392], fill=(255, 35, 65, 255))
        # Tick marks
        for ty in range(y + 50, y + 270, 35):
            draw.line([(x + 22, ty), (x + 48, ty)], fill=(40, 55, 80, 255), width=5)
        # Callout tag
        draw.rounded_rectangle([x - 110, y - 85, x + 190, y - 15], radius=18, fill=(255, 35, 65, 255), outline=(255, 255, 255, 255), width=4)
        font = self._get_font(32)
        draw.text((x + 40, y - 50), temp_text, fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_beak_shield_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws a glowing golden energy shield around Pip's beak deflecting heat waves."""
        for r, alpha in [(165, 70), (135, 120), (105, 190)]:
            draw.arc([cx - r, cy - r, cx + r, cy + r], start=280, end=85, fill=(255, 225, 50, alpha), width=14)
        # Deflected red heat arrows bouncing off shield
        for ay in (cy - 70, cy, cy + 70):
            draw.line([(cx + 260, ay), (cx + 155, ay)], fill=(255, 75, 55, 255), width=10)
            draw.polygon([(cx + 150, ay), (cx + 180, ay - 18), (cx + 180, ay + 18)], fill=(255, 75, 55, 255))

    def _draw_nest_and_egg_bottom(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws a warm woven twig nest and cracked white lower eggshell."""
        draw.ellipse([cx - 280, cy - 30, cx + 280, cy + 120], fill=(165, 110, 55, 255), outline=(105, 65, 28, 255), width=6)
        # Lower eggshell with zigzag crack
        pts = [(cx - 165, cy - 90)]
        for i in range(9):
            px = cx - 165 + i * 41
            py = cy - 90 + (28 if i % 2 == 1 else -12)
            pts.append((px, py))
        pts.extend([(cx + 165, cy + 45), (cx - 165, cy + 45)])
        draw.polygon(pts, fill=(250, 248, 242, 255), outline=(160, 155, 145, 255))

    def _draw_cracked_eggshell_top(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws the popped-off top cap of the white eggshell above baby gray Pip's head."""
        draw.pieslice([cx - 125, cy - 95, cx + 125, cy + 85], start=180, end=360, fill=(250, 248, 242, 255), outline=(160, 155, 145, 255), width=5)

    def _draw_comparison_callout(self, draw: ImageDraw.ImageDraw, x: int, y: int, title: str, subtitle: str, color: Tuple[int, int, int]):
        draw.rounded_rectangle([x - 180, y - 75, x + 180, y + 75], radius=24, fill=(28, 42, 75, 240), outline=color, width=5)
        f1 = self._get_font(34)
        f2 = self._get_font(25)
        draw.text((x, y - 22), title, fill=(255, 110, 140), font=f1, anchor="mm")
        draw.text((x, y + 26), subtitle, fill=(255, 255, 255), font=f2, anchor="mm")

    def _draw_glowing_shrimp_school(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws 4 cute neon-pink glowing brine shrimp in a magnified lagoon circle."""
        draw.ellipse([cx - 210, cy - 210, cx + 210, cy + 210], fill=(10, 85, 145, 230), outline=(255, 105, 185, 255), width=8)
        for sx, sy in [(-75, -75), (65, -60), (-60, 65), (70, 70)]:
            px, py = cx + sx, cy + sy
            # Glowing pink aura
            draw.ellipse([px - 52, py - 36, px + 52, py + 36], fill=(255, 80, 165, 95))
            # Curled pink shrimp body
            draw.arc([px - 36, py - 26, px + 36, py + 26], start=30, end=290, fill=(255, 95, 165, 255), width=18)
            draw.ellipse([px + 14, py - 14, px + 26, py - 2], fill=(20, 20, 30, 255))
        font = self._get_font(30)
        draw.rounded_rectangle([cx - 175, cy + 175, cx + 175, cy + 235], radius=16, fill=(255, 65, 150, 255), outline=(255, 255, 255, 255), width=3)
        draw.text((cx, cy + 205), "PINK BRINE SHRIMP", fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_pigment_arrows(self, draw: ImageDraw.ImageDraw, start_xy: Tuple[int, int], end_xy: Tuple[int, int]):
        """Draws glowing neon-pink arrows pointing from the shrimp directly into Pip's feathers."""
        sx, sy = start_xy
        ex, ey = end_xy
        draw.line([(sx, sy), (ex, ey)], fill=(255, 230, 80, 255), width=12)
        draw.polygon([(ex, ey), (ex + 35, ey - 22), (ex + 35, ey + 22)], fill=(255, 230, 80, 255))

    def _draw_transformation_aura(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        for r, col in [(360, (255, 80, 175, 55)), (290, (255, 140, 210, 85)), (220, (255, 230, 120, 110))]:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    def _draw_color_progress_meter(self, draw: ImageDraw.ImageDraw, cx: int, y: int, narration: str = ""):
        """Draws a visual color transformation / camouflage meter at the bottom of Scene 5."""
        low = (narration or "").lower()
        if "chameleon" in low or "stretches" in low or "crystal" in low or "neon" in low and "red" in low:
            left_txt, right_txt = "RELAXED: GREEN", "EXCITED: RED!"
        elif "skin" in low or "second" in low or "camouflage" in low:
            left_txt, right_txt = "NORMAL SKIN", "0.3s CAMOUFLAGE!"
        elif "sunset" in low or "cotton candy" in low or "pink" in low or "neon" in low:
            left_txt, right_txt = "WHITE CLOUD", "NEON SUNSET!"
        elif "lightning" in low or "sun" in low:
            left_txt, right_txt = "COOL CLOUD", "5X SUN HEAT!"
        else:
            left_txt, right_txt = "BABY: GRAY", "SHRIMP DYE: PINK!"
        draw.rounded_rectangle([cx - 380, y - 45, cx + 380, y + 45], radius=25, fill=(25, 35, 65, 240), outline=(255, 255, 255, 255), width=4)
        draw.rounded_rectangle([cx - 360, y - 28, cx, y + 28], radius=18, fill=(180, 192, 208, 255))
        draw.rounded_rectangle([cx, y - 28, cx + 360, y + 28], radius=18, fill=(255, 65, 155, 255))
        font = self._get_font(28)
        draw.text((cx - 180, y), left_txt, fill=(30, 40, 55), font=font, anchor="mm")
        draw.text((cx + 180, y), right_txt, fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_ice_floes(self, draw: ImageDraw.ImageDraw, cy: int):
        for ix, iw in [(240, 130), (540, 190), (830, 140)]:
            draw.polygon(
                [(ix - iw, cy), (ix - iw // 2, cy - 28), (ix + iw // 2, cy - 28), (ix + iw, cy), (ix + iw // 2, cy + 24), (ix - iw // 2, cy + 24)],
                fill=(205, 242, 255, 230), outline=(255, 255, 255, 255)
            )

    def _draw_zzz_bubbles(self, draw: ImageDraw.ImageDraw, start_x: int, start_y: int, anim_phase: float = 0.0):
        for i, (dx, dy, sz) in enumerate([(0, 0, 36), (55, -65, 48), (120, -140, 62)]):
            font = self._get_font(sz)
            draw.text((start_x + dx, start_y + dy), "Z", fill=(255, 245, 130, 255), font=font, anchor="mm")

    def _draw_one_leg_pointer(self, draw: ImageDraw.ImageDraw, target_x: int, target_y: int, narration: str = ""):
        low = (narration or "").lower()
        if "dream" in low or "color" in low:
            label = "COLOR DREAMS!"
        elif "crystal" in low or "snow" in low:
            label = "ICE CRYSTALS!"
        else:
            label = "ONLY 1 LEG!"
        draw.rounded_rectangle([95, target_y - 45, 430, target_y + 45], radius=20, fill=(0, 200, 255, 240), outline=(255, 255, 255, 255), width=4)
        font = self._get_font(28)
        draw.text((262, target_y), label, fill=(15, 25, 45), font=font, anchor="mm")
        draw.line([(430, target_y), (target_x - 25, target_y)], fill=(255, 255, 255, 255), width=8)

    def _draw_xray_gear_lock_diagram(self, draw: ImageDraw.ImageDraw, target_leg_xy: Tuple[int, int], zoom_center_xy: Tuple[int, int], narration: str = ""):
        """Draws a glowing cyan X-ray magnifying callout zooming in on Pip's ankle bone-lock or 9 Brains gear mechanism."""
        tx, ty = target_leg_xy
        zx, zy = zoom_center_xy
        low = (narration or "").lower()
        if "brain" in low or "nine" in low:
            bottom_label = "9 SMART BRAINS SYNCED!"
        elif "updraft" in low or "wind" in low:
            bottom_label = "UPDRAFT WIND LOCK!"
        else:
            bottom_label = "AUTO-LOCK = 0% ENERGY!"

        # Small target ring on Pip's leg joint
        draw.ellipse([tx - 42, ty - 42, tx + 42, ty + 42], outline=(0, 245, 255, 255), width=6)
        # Connector lines to giant X-ray circle
        draw.line([(tx + 35, ty - 25), (zx - 170, zy - 110)], fill=(0, 245, 255, 220), width=6)
        draw.line([(tx + 35, ty + 25), (zx - 170, zy + 110)], fill=(0, 245, 255, 220), width=6)

        # Giant X-Ray Zoom Circle
        zr = 210
        draw.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=(12, 32, 68, 245), outline=(0, 245, 255, 255), width=8)
        # Upper and lower bone shafts inside X-ray
        draw.line([(zx, zy - 175), (zx, zy - 30)], fill=(235, 245, 255, 255), width=34)
        draw.line([(zx, zy + 30), (zx, zy + 175)], fill=(235, 245, 255, 255), width=34)
        # Interlocking Golden Puzzle/Gear Joint in center of X-ray
        draw.ellipse([zx - 65, zy - 65, zx + 65, zy + 65], fill=(255, 205, 0, 255), outline=(255, 255, 255, 255), width=5)
        draw.rounded_rectangle([zx - 32, zy - 22, zx + 32, zy + 22], radius=8, fill=(255, 85, 55, 255), outline=(255, 255, 255, 255), width=3)

        font = self._get_font(28)
        draw.rounded_rectangle([zx - 205, zy + zr - 25, zx + 205, zy + zr + 42], radius=16, fill=(0, 225, 160, 255), outline=(255, 255, 255, 255), width=3)
        draw.text((zx, zy + zr + 8), bottom_label, fill=(12, 25, 45), font=font, anchor="mm")

    def _draw_three_hearts_xray_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws a glowing cyan X-ray ocean scanner showing a friendly coral octopus with 3 glowing hearts and 9 mini-brains."""
        zr = 225
        draw.ellipse([cx - zr, cy - zr, cx + zr, cy + zr], fill=(10, 32, 72, 245), outline=(0, 245, 255, 255), width=8)
        # Cute 3D coral-orange octopus head silhouette inside X-ray scanner
        draw.ellipse([cx - 135, cy - 155, cx + 135, cy + 65], fill=(255, 115, 85, 195), outline=(255, 210, 140, 255), width=5)
        # 3 Glowing Hearts inside the octopus! (2 gill hearts + 1 central systemic heart)
        pulse = int(math.sin(anim_phase * 2 * math.pi) * 6)
        heart_coords = [(cx - 72, cy - 35, (255, 55, 105, 255)), (cx + 72, cy - 35, (255, 55, 105, 255)), (cx, cy - 85, (255, 215, 0, 255))]
        for hx, hy, hcol in heart_coords:
            hr = 28 + pulse
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=hcol, outline=(255, 255, 255, 255), width=4)
        # 8 glowing mini-brain nodes along tentacles at bottom of X-ray
        for i in range(-3, 5):
            tx = cx + i * 34
            ty = cy + 115 + (abs(i) % 2) * 20
            draw.line([(cx + i * 18, cy + 55), (tx, ty)], fill=(255, 130, 95, 255), width=12)
            draw.ellipse([tx - 11, ty - 11, tx + 11, ty + 11], fill=(0, 255, 220, 255), outline=(255, 255, 255, 255), width=2)
        font = self._get_font(28)
        draw.rounded_rectangle([cx - 205, cy + zr - 25, cx + 205, cy + zr + 42], radius=16, fill=(255, 60, 115, 255), outline=(255, 255, 255, 255), width=3)
        draw.text((cx, cy + zr + 8), "3 HEARTS + 9 BRAINS!", fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_blue_blood_tube_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws a glowing neon-blue copper blood science tube next to a red iron blood comparison badge."""
        zr = 215
        draw.ellipse([cx - zr, cy - zr, cx + zr, cy + zr], fill=(12, 34, 78, 245), outline=(0, 225, 255, 255), width=8)
        # Giant glowing blue test tube
        draw.rounded_rectangle([cx - 58, cy - 145, cx + 58, cy + 125], radius=45, fill=(225, 245, 255, 255), outline=(255, 255, 255, 255), width=6)
        draw.rounded_rectangle([cx - 44, cy - 65, cx + 44, cy + 112], radius=36, fill=(0, 185, 255, 255))
        # Glowing cyan oxygen bubbles inside blue blood
        for bx, by, br in [(-15, 65, 14), (18, 20, 16), (-10, -25, 12)]:
            draw.ellipse([cx + bx - br, cy + by - br, cx + bx + br, cy + by + br], fill=(165, 245, 255, 255))
        font = self._get_font(28)
        draw.rounded_rectangle([cx - 205, cy + zr - 25, cx + 205, cy + zr + 42], radius=16, fill=(0, 175, 255, 255), outline=(255, 255, 255, 255), width=3)
        draw.text((cx, cy + zr + 8), "COPPER = BLUE BLOOD!", fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_octopus_rice_eggs_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws glowing underwater seaweed cluster with clusters of tiny translucent rice-grain sized eggs hatching baby octopuses."""
        for kx_off, kcol in [(-220, (18, 95, 65, 220)), (-140, (28, 135, 85, 230)), (140, (28, 135, 85, 230)), (220, (18, 95, 65, 220))]:
            draw.line([(cx + kx_off, cy + 320), (cx + kx_off + int(math.sin(anim_phase * 2 * math.pi) * 30), cy - 220)], fill=kcol, width=32)

        zr = 220
        draw.ellipse([cx - zr, cy - zr, cx + zr, cy + zr], fill=(12, 38, 78, 245), outline=(0, 245, 255, 255), width=8)

        draw.line([(cx, cy - zr + 40), (cx, cy + 40)], fill=(180, 225, 210, 220), width=8)
        egg_offsets = [
            (-95, -80), (-45, -110), (0, -70), (55, -100), (95, -75),
            (-75, -20), (-25, -35), (35, -25), (75, -15),
            (-50, 45), (0, 30), (50, 40)
        ]
        for ex, ey in egg_offsets:
            draw.ellipse([cx + ex - 18, cy + ey - 32, cx + ex + 18, cy + ey + 32], fill=(235, 248, 255, 240), outline=(0, 215, 255, 255), width=3)
            draw.ellipse([cx + ex - 8, cy + ey - 4, cx + ex - 2, cy + ey + 4], fill=(20, 35, 55, 255))
            draw.ellipse([cx + ex + 2, cy + ey - 4, cx + ex + 8, cy + ey + 4], fill=(20, 35, 55, 255))

        bx, by = cx + 85, cy + 120
        draw.ellipse([bx - 32, by - 36, bx + 32, by + 16], fill=(255, 115, 85, 255), outline=(255, 220, 150, 255), width=3)
        draw.ellipse([bx - 14, by - 16, bx - 2, by - 2], fill=(255, 255, 255, 255))
        draw.ellipse([bx + 2, by - 16, bx + 14, by - 2], fill=(255, 255, 255, 255))
        draw.ellipse([bx - 10, by - 12, bx - 4, by - 6], fill=(10, 25, 45, 255))
        draw.ellipse([bx + 4, by - 12, bx + 10, by - 6], fill=(10, 25, 45, 255))
        for i in (-18, -6, 6, 18):
            draw.line([(bx + i, by + 14), (bx + i * 2, by + 42)], fill=(255, 115, 85, 255), width=6)

        font = self._get_font(28)
        draw.rounded_rectangle([cx - 215, cy + zr - 25, cx + 215, cy + zr + 42], radius=16, fill=(0, 225, 185, 255), outline=(255, 255, 255, 255), width=3)
        draw.text((cx, cy + zr + 8), "TINY RICE-SIZED EGGS!", fill=(12, 30, 50), font=font, anchor="mm")

    def _draw_coral_reef_bed(self, draw: ImageDraw.ImageDraw, cy: int):
        """Draws colorful glowing deep-sea coral reef at bottom of screen."""
        coral_colors = [(255, 85, 125, 255), (145, 75, 225, 255), (0, 215, 205, 255), (255, 195, 55, 255)]
        for idx, (cx_off, cr, col) in enumerate([(-380, 160, coral_colors[0]), (-180, 190, coral_colors[1]), (60, 175, coral_colors[2]), (320, 210, coral_colors[3])]):
            bx = 540 + cx_off
            draw.pieslice([bx - cr, cy - cr, bx + cr, cy + cr], start=180, end=360, fill=col, outline=(255, 255, 255, 180), width=4)
            for py in range(cy - cr + 40, cy, 35):
                draw.ellipse([bx - 12, py - 8, bx + 12, py + 8], fill=(255, 255, 255, 160))

    def _draw_color_dream_thought_bubble(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws a dreamy cloud bubble showing the sleeping octopus dreaming in shifting rainbow neon colors."""
        draw.ellipse([cx - 180, cy - 110, cx + 180, cy + 110], fill=(25, 15, 60, 240), outline=(255, 120, 240, 255), width=5)
        draw.ellipse([cx - 140, cy + 130, cx - 115, cy + 155], fill=(255, 120, 240, 230))
        draw.ellipse([cx - 175, cy + 175, cx - 160, cy + 190], fill=(255, 120, 240, 200))
        for r, col in [(85, (255, 65, 115, 255)), (65, (255, 205, 55, 255)), (45, (0, 240, 220, 255))]:
            draw.arc([cx - r, cy - r, cx + r, cy + r], start=0, end=360, fill=col, width=10)
        font = self._get_font(26)
        draw.text((cx, cy), "COLOR DREAMS!", fill=(255, 255, 255), font=font, anchor="mm")


    def _draw_superpower_vs_cards(self, draw: ImageDraw.ImageDraw, y: int, narration: str = ""):
        """Draws two visual superpower choice cards above Pip and Leo in Scene 8."""
        font = self._get_font(30)
        low = (narration or "").lower()
        if "chameleon" in low or "eyes" in low or "tongue" in low or "360" in low or "radar" in low or "60" in low:
            card1_text = "360° RADAR EYES"
            card2_text = "60 MPH TONGUE"
        elif "heart" in low or "brain" in low or "camouflage" in low or "blue" in low:
            card1_text = "3 GLOWING HEARTS"
            card2_text = "0.3s CAMOUFLAGE"
        elif "cloud" in low or "lightning" in low:
            card1_text = "1M POUND CLOUD"
            card2_text = "5X SUN LIGHTNING"
        else:
            card1_text = "100°C HEAT BEAK"
            card2_text = "1-LEG SLEEP LOCK"

        # Left Card
        draw.rounded_rectangle([80, y - 80, 490, y + 80], radius=24, fill=(255, 65, 95, 245), outline=(255, 255, 255, 255), width=5)
        draw.text((285, y - 22), "SUPERPOWER #1", fill=(255, 235, 130), font=font, anchor="mm")
        draw.text((285, y + 25), card1_text, fill=(255, 255, 255), font=font, anchor="mm")

        # Right Card
        draw.rounded_rectangle([590, y - 80, 1000, y + 80], radius=24, fill=(0, 165, 235, 245), outline=(255, 255, 255, 255), width=5)
        draw.text((795, y - 22), "SUPERPOWER #2", fill=(255, 235, 130), font=font, anchor="mm")
        draw.text((795, y + 25), card2_text, fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_top_story_banner(self, draw: ImageDraw.ImageDraw, scene_id: int, total_scenes: int, badge_text: str):
        """Draws a clean, kid-friendly visual story progress bar & fact badge at the top of the 9:16 frame."""
        # Story progress pills at very top (y=90)
        seg_w = (self.width - 160) // total_scenes
        for i in range(total_scenes):
            x0 = 80 + i * seg_w + 4
            x1 = x0 + seg_w - 8
            col = (0, 245, 200, 255) if (i + 1) <= scene_id else (255, 255, 255, 85)
            draw.rounded_rectangle([x0, 75, x1, 95], radius=8, fill=col)

        # Big High-Contrast Fact Badge (y=130..245)
        draw.rounded_rectangle(
            [90, 125, self.width - 90, 245],
            radius=32,
            fill=(22, 34, 68, 240),
            outline=(255, 215, 0, 255),
            width=6
        )
        badge_font = self._get_font(46)
        draw.text((self.width // 2, 185), badge_text.upper(), fill=(255, 235, 85), font=badge_font, anchor="mm")

    def _draw_fluffy_cloud(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        outline: Tuple[int, int, int] = (190, 215, 245),
        expression: str = "happy",
        anim_phase: float = 0.0
    ):
        """Draws a cute 3D-shaded cumulus cloud character ('Nimbus') with big Pixar eyes and smile."""
        puffs = [
            (-140, 20, 95),
            (-80, -35, 115),
            (10, -60, 130),
            (95, -30, 110),
            (145, 25, 90),
            (-50, 45, 100),
            (55, 45, 95)
        ]
        # 1. Drop shadow / bottom shading for 3D depth
        shadow_col = (max(0, color[0] - 40), max(0, color[1] - 35), max(0, color[2] - 25), 255)
        for dx, dy, r in puffs:
            px = cx + int(dx * scale)
            py = cy + int((dy + 14) * scale)
            pr = int(r * scale)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=shadow_col)

        # 2. Outer stroke / outline
        for dx, dy, r in puffs:
            px = cx + int(dx * scale)
            py = cy + int(dy * scale)
            pr = int((r + 4) * scale)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=outline)

        # 3. Main cloud body puffs
        fill_col = (color[0], color[1], color[2], 255)
        for dx, dy, r in puffs:
            px = cx + int(dx * scale)
            py = cy + int(dy * scale)
            pr = int(r * scale)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=fill_col)

        # 4. Top highlight puffs for fluffy 3D sunlight sheen
        highlight_col = (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30), 180)
        for dx, dy, r in [(-60, -45, 80), (15, -70, 95), (85, -40, 75)]:
            px = cx + int(dx * scale)
            py = cy + int(dy * scale)
            pr = int(r * scale)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=highlight_col)

        # 5. Cute Face (in center of cloud)
        face_y = cy + int(10 * scale)
        eye_y = face_y - int(10 * scale)

        # Rosy cheeks
        blush_col = (255, 120, 160, 160) if color[0] > 100 else (255, 160, 180, 140)
        draw.ellipse([cx - int(70 * scale), face_y, cx - int(35 * scale), face_y + int(24 * scale)], fill=blush_col)
        draw.ellipse([cx + int(35 * scale), face_y, cx + int(70 * scale), face_y + int(24 * scale)], fill=blush_col)

        # Big Pixar Eyes
        for ex_off in (-40, 40):
            ex = cx + int(ex_off * scale)
            er = int(22 * scale)
            if expression == "sleeping":
                draw.arc([ex - er, eye_y - er // 2, ex + er, eye_y + er // 2], start=0, end=180, fill=(35, 45, 65, 255), width=int(5 * scale))
            else:
                draw.ellipse([ex - er, eye_y - er, ex + er, eye_y + er], fill=(255, 255, 255, 255), outline=(35, 45, 65, 255), width=3)
                ir = int(er * (0.65 if expression != "shocked" else 0.52))
                iris_col = (30, 80, 140, 255) if color[0] > 100 else (100, 180, 255, 255)
                draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir], fill=iris_col)
                pr = int(ir * 0.55)
                draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(15, 20, 30, 255))
                # Sparkle catchlight
                draw.ellipse([ex - int(er * 0.35), eye_y - int(er * 0.4), ex - int(er * 0.1), eye_y - int(er * 0.15)], fill=(255, 255, 255, 255))

        # Mouth
        if expression == "sleeping":
            draw.arc([cx - int(12 * scale), face_y + int(12 * scale), cx + int(12 * scale), face_y + int(24 * scale)], start=0, end=180, fill=(35, 45, 65, 255), width=3)
        elif expression == "shocked":
            draw.ellipse([cx - int(16 * scale), face_y + int(12 * scale), cx + int(16 * scale), face_y + int(36 * scale)], fill=(210, 60, 80, 255), outline=(35, 45, 65, 255), width=3)
        else:
            draw.pieslice([cx - int(24 * scale), face_y + int(8 * scale), cx + int(24 * scale), face_y + int(36 * scale)], start=0, end=180, fill=(230, 65, 95, 255), outline=(35, 45, 65, 255), width=3)

    def _draw_steam_rise_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws warm wavy steam ribbons and rising vapor bubbles floating from lagoon water into sky."""
        for off_x in (-160, -80, 0, 80, 160):
            pts = []
            for i in range(12):
                y = cy - i * 45
                x = cx + off_x + int(math.sin((i * 0.8) + (off_x * 0.05) + anim_phase * 6.28) * 22)
                pts.append((x, y))
            for i in range(len(pts) - 1):
                alpha = int(180 * (1.0 - (i / len(pts))))
                draw.line([pts[i], pts[i + 1]], fill=(255, 255, 255, alpha), width=6)

        rng = random.Random(42)
        for _ in range(25):
            bx = cx + rng.randint(-220, 220)
            by = cy - rng.randint(40, 520)
            br = rng.randint(8, 22)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 255, 255, 120), outline=(200, 230, 255, 200), width=2)

        draw.polygon([(cx - 25, cy - 320), (cx + 25, cy - 320), (cx + 25, cy - 250), (cx - 25, cy - 250)], fill=(0, 245, 160, 220))
        draw.polygon([(cx - 50, cy - 320), (cx + 50, cy - 320), (cx, cy - 380)], fill=(0, 245, 160, 220))

    def _draw_atmosphere_shield_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws protective glowing atmosphere shield dome and 1,000,000 LB weight callout."""
        for r_off in (30, 45, 60):
            draw.arc([cx - 360 - r_off, cy - 220 - r_off, cx + 360 + r_off, cy + 220 + r_off], start=160, end=380, fill=(0, 225, 255, 140), width=6)

        for ax in (cx - 220, cx, cx + 220):
            draw.polygon([(ax - 20, cy + 260), (ax + 20, cy + 260), (ax + 20, cy + 180), (ax - 20, cy + 180)], fill=(0, 255, 175, 230))
            draw.polygon([(ax - 40, cy + 180), (ax + 40, cy + 180), (ax, cy + 130)], fill=(0, 255, 175, 230))

        draw.rounded_rectangle([cx - 220, cy + 300, cx + 220, cy + 410], radius=24, fill=(20, 32, 65, 245), outline=(255, 215, 0, 255), width=5)
        font = self._get_font(34)
        sub_font = self._get_font(24)
        draw.text((cx, cy + 335), "1,000,000 LBS!", fill=(255, 225, 80), font=font, anchor="mm")
        draw.text((cx, cy + 380), "HELD UP BY WARM AIR", fill=(255, 255, 255), font=sub_font, anchor="mm")

    def _draw_cloud_micro_droplets_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws magnifying glass zoom bubble showing cute water droplets clustering together."""
        r = 170
        draw.line([(cx + int(r * 0.7), cy + int(r * 0.7)), (cx + r + 90, cy + r + 90)], fill=(120, 80, 50, 255), width=24)
        draw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=(255, 215, 0, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 55, 110, 245))

        droplet_coords = [
            (-70, -60, 22), (0, -80, 26), (70, -50, 20),
            (-80, 0, 24), (-20, -15, 28), (45, -5, 25), (90, 15, 20),
            (-60, 60, 24), (5, 50, 28), (70, 65, 22), (0, 105, 18)
        ]
        for dx, dy, dr in droplet_coords:
            px = cx + dx
            py = cy + dy
            draw.ellipse([px - dr, py - dr // 2, px + dr, py + dr], fill=(0, 210, 255, 255), outline=(255, 255, 255, 255), width=2)
            draw.polygon([(px - dr + 2, py), (px + dr - 2, py), (px, py - dr - 6)], fill=(0, 210, 255, 255))
            draw.ellipse([px - dr // 2, py - dr // 3, px - dr // 4, py], fill=(255, 255, 255, 220))
            draw.ellipse([px - 4, py + 2, px - 1, py + 5], fill=(15, 25, 45, 255))
            draw.ellipse([px + 1, py + 2, px + 4, py + 5], fill=(15, 25, 45, 255))

        draw.rounded_rectangle([cx - 150, cy + r - 30, cx + 150, cy + r + 25], radius=16, fill=(255, 60, 110, 255), outline=(255, 255, 255, 255), width=3)
        font = self._get_font(24)
        draw.text((cx, cy + r - 2), "TINY DROPLETS!", fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_cloud_sunlight_scattering_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws radiant golden sun and prism light rays scattering into rainbow spectrum."""
        sr = 90
        draw.ellipse([cx - sr - 15, cy - sr - 15, cx + sr + 15, cy + sr + 15], fill=(255, 240, 120, 140))
        draw.ellipse([cx - sr, cy - sr, cx + sr, cy + sr], fill=(255, 215, 0, 255), outline=(255, 255, 255, 255), width=5)

        for i in range(12):
            ang = (i / 12.0) * 2 * math.pi
            bx1 = cx + int(math.cos(ang) * (sr + 10))
            by1 = cy + int(math.sin(ang) * (sr + 10))
            bx2 = cx + int(math.cos(ang) * (sr + 65))
            by2 = cy + int(math.sin(ang) * (sr + 65))
            draw.line([(bx1, by1), (bx2, by2)], fill=(255, 235, 80, 220), width=6)

        colors = [
            (255, 60, 60),
            (255, 160, 40),
            (255, 230, 40),
            (40, 220, 100),
            (0, 205, 255),
            (80, 110, 255),
            (180, 70, 255)
        ]
        target_x = cx + 550
        for idx, col in enumerate(colors):
            y_off = (idx - 3) * 28
            draw.line([(cx + sr, cy), (target_x, cy + 200 + y_off)], fill=(col[0], col[1], col[2], 180), width=8)

        draw.line([(cx + sr, cy), (target_x, cy + 200)], fill=(255, 255, 255, 240), width=12)

    def _draw_cloud_sunset_neon_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws sunset golden hour glow, neon pink cotton candy aura, and floating stars."""
        draw.pieslice([cx - 240, 1150, cx + 240, 1550], start=180, end=360, fill=(255, 180, 40, 255), outline=(255, 230, 120, 255), width=5)

        for r_off in (35, 65, 95):
            draw.ellipse([cx - 320 - r_off, cy - 140 - r_off, cx + 320 + r_off, cy + 140 + r_off], fill=(255, 80, 160, 40))

        rng = random.Random(88)
        for _ in range(25):
            sx = rng.randint(80, 1000)
            sy = rng.randint(450, 1100)
            sr = rng.randint(4, 9)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 240, 160, 220))

    def _draw_cloud_ice_crystals_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws sparkling geometric hexagonal snowflakes and ice crystals inside freezing cloud."""
        def draw_snowflake(scx: int, scy: int, radius: int):
            for i in range(6):
                ang = (i / 6.0) * math.pi
                dx = int(math.cos(ang) * radius)
                dy = int(math.sin(ang) * radius)
                draw.line([(scx - dx, scy - dy), (scx + dx, scy + dy)], fill=(255, 255, 255, 240), width=3)
                for frac in (0.4, 0.75):
                    bx = scx + int(dx * frac)
                    by = scy + int(dy * frac)
                    bdx = int(math.cos(ang + 0.6) * (radius * 0.3))
                    bdy = int(math.sin(ang + 0.6) * (radius * 0.3))
                    draw.line([(bx, by), (bx + bdx, by + bdy)], fill=(200, 235, 255, 230), width=2)
                    draw.line([(bx, by), (bx - bdx, by - bdy)], fill=(200, 235, 255, 230), width=2)
            draw.ellipse([scx - 4, scy - 4, scx + 4, scy + 4], fill=(255, 255, 255, 255))

        flake_locs = [
            (cx - 180, cy - 60, 32),
            (cx - 70, cy - 90, 38),
            (cx + 80, cy - 70, 36),
            (cx + 190, cy - 50, 30),
            (cx - 120, cy + 70, 34),
            (cx + 10, cy + 90, 42),
            (cx + 140, cy + 60, 32),
        ]
        for fx, fy, fr in flake_locs:
            draw_snowflake(fx, fy, fr)

    def _draw_lightning_bolt_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws electrostatic positive/negative charges and bright jagged electric lightning bolt."""
        charges = [
            (cx - 180, cy - 120, "+", (255, 80, 80)),
            (cx - 90, cy - 140, "+", (255, 80, 80)),
            (cx + 40, cy - 130, "+", (255, 80, 80)),
            (cx + 160, cy - 110, "+", (255, 80, 80)),
            (cx - 150, cy + 120, "-", (60, 180, 255)),
            (cx - 50, cy + 140, "-", (60, 180, 255)),
            (cx + 80, cy + 130, "-", (60, 180, 255)),
            (cx + 180, cy + 110, "-", (60, 180, 255)),
        ]
        font = self._get_font(28)
        for ch_x, ch_y, ch_sym, ch_col in charges:
            draw.ellipse([ch_x - 18, ch_y - 18, ch_x + 18, ch_y + 18], fill=ch_col, outline=(255, 255, 255, 255), width=2)
            draw.text((ch_x, ch_y - 2), ch_sym, fill=(255, 255, 255), font=font, anchor="mm")

        bolt_pts = [
            (cx - 40, cy - 120),
            (cx + 25, cy - 20),
            (cx - 30, cy + 10),
            (cx + 60, cy + 110),
            (cx + 10, cy + 130),
            (cx + 80, cy + 240),
            (cx + 20, cy + 260),
            (cx + 110, cy + 390)
        ]
        for i in range(len(bolt_pts) - 1):
            draw.line([bolt_pts[i], bolt_pts[i + 1]], fill=(0, 235, 255, 180), width=28)
        for i in range(len(bolt_pts) - 1):
            draw.line([bolt_pts[i], bolt_pts[i + 1]], fill=(255, 245, 60, 245), width=14)
        for i in range(len(bolt_pts) - 1):
            draw.line([bolt_pts[i], bolt_pts[i + 1]], fill=(255, 255, 255, 255), width=6)

        strike_x, strike_y = bolt_pts[-1]
        for spk_ang in (0.3, 1.1, 2.2, 3.4, 4.5, 5.6):
            sx2 = strike_x + int(math.cos(spk_ang) * 45)
            sy2 = strike_y + int(math.sin(spk_ang) * 45)
            draw.line([(strike_x, strike_y), (sx2, sy2)], fill=(255, 255, 120, 220), width=4)

    # =========================================================================
    # CHAMELEON & COLOR SCIENCE RENDERING METHODS
    # =========================================================================

    def _draw_tree_branch(self, draw: ImageDraw.ImageDraw, cy: int):
        """Draws a sturdy jungle tree branch with lush green leaves across the frame."""
        # Main wooden branch
        branch_pts = [
            (0, cy + 20),
            (260, cy + 5),
            (540, cy - 10),
            (820, cy + 8),
            (self.width, cy - 15),
            (self.width, cy + 75),
            (820, cy + 95),
            (540, cy + 80),
            (260, cy + 90),
            (0, cy + 105),
        ]
        draw.polygon(branch_pts, fill=(110, 72, 38, 255), outline=(75, 48, 22, 255))
        # Bark texture ridges
        for bx, by, bw in [(140, cy + 35, 120), (420, cy + 25, 140), (720, cy + 30, 160)]:
            draw.arc([bx, by, bx + bw, by + 30], start=15, end=165, fill=(80, 52, 26, 200), width=5)

        # Fresh jungle leaves sprouting along branch
        leaf_clusters = [
            (90, cy + 5, -25),
            (220, cy - 20, 30),
            (470, cy + 75, 40),
            (760, cy - 15, -35),
            (960, cy + 65, 25),
        ]
        for lx, ly, rot in leaf_clusters:
            # Leaf pair
            draw.ellipse([lx - 35, ly - 18, lx + 35, ly + 18], fill=(46, 204, 113, 240), outline=(230, 255, 230, 255), width=2)
            draw.line([(lx - 30, ly), (lx + 30, ly)], fill=(30, 140, 75, 255), width=3)

    def _draw_chameleon_character(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        scale: float = 1.0,
        color_state: str = "green",
        eye_mode: str = "normal",
        tongue: bool = False,
        anim_phase: float = 0.0,
    ):
        """Draws the adorable hero chameleon perched on the branch with animated states and color shifts."""
        s = scale
        # Determine color palette based on state
        if color_state == "fiery_red":
            body_col = (245, 55, 55, 255)
            belly_col = (255, 165, 30, 255)
            stripe_col = (255, 230, 50, 255)
            crest_col = (210, 25, 45, 255)
        elif color_state == "half_orange":
            body_col = (245, 135, 35, 255)
            belly_col = (255, 205, 75, 255)
            stripe_col = (46, 204, 113, 255)
            crest_col = (220, 95, 20, 255)
        else:  # "green"
            body_col = (46, 204, 113, 255)
            belly_col = (130, 235, 160, 255)
            stripe_col = (26, 160, 110, 255)
            crest_col = (22, 145, 90, 255)

        # 1. Prehensile Spiral Curled Tail (left of body)
        tail_cx = cx - int(150 * s)
        tail_cy = cy + int(10 * s)
        draw.arc([tail_cx - int(75 * s), tail_cy - int(75 * s), tail_cx + int(75 * s), tail_cy + int(75 * s)], start=90, end=360, fill=body_col, width=int(26 * s))
        draw.arc([tail_cx - int(45 * s), tail_cy - int(45 * s), tail_cx + int(45 * s), tail_cy + int(45 * s)], start=0, end=270, fill=body_col, width=int(20 * s))
        draw.ellipse([tail_cx - int(15 * s), tail_cy - int(15 * s), tail_cx + int(15 * s), tail_cy + int(15 * s)], fill=body_col)

        # 2. Chameleon Legs gripping the branch (zygodactylous mitten feet)
        # Back foot
        bf_x, bf_y = cx - int(70 * s), cy + int(70 * s)
        draw.line([(cx - int(60 * s), cy + int(20 * s)), (bf_x, bf_y)], fill=body_col, width=int(22 * s))
        draw.ellipse([bf_x - int(24 * s), bf_y - int(12 * s), bf_x + int(24 * s), bf_y + int(12 * s)], fill=crest_col)
        # Front foot
        ff_x, ff_y = cx + int(70 * s), cy + int(70 * s)
        draw.line([(cx + int(50 * s), cy + int(20 * s)), (ff_x, ff_y)], fill=body_col, width=int(22 * s))
        draw.ellipse([ff_x - int(24 * s), ff_y - int(12 * s), ff_x + int(24 * s), ff_y + int(12 * s)], fill=crest_col)

        # 3. Main Body (arched back)
        bw, bh = int(145 * s), int(85 * s)
        draw.ellipse([cx - bw, cy - bh, cx + bw, cy + bh], fill=body_col, outline=(255, 255, 255, 120), width=int(3 * s))
        # Soft belly
        draw.ellipse([cx - int(100 * s), cy - int(25 * s), cx + int(100 * s), cy + int(75 * s)], fill=belly_col)

        # Decorative dorsal spines / crest ridges along back
        for i in range(-4, 5):
            sp_x = cx + int(i * 26 * s)
            sp_y = cy - int(math.cos(i * 0.28) * 85 * s)
            draw.polygon([(sp_x - int(8 * s), sp_y), (sp_x, sp_y - int(16 * s)), (sp_x + int(8 * s), sp_y)], fill=crest_col)

        # Tiger / mood stripes across flank
        for st_x in (-60, -20, 20, 60):
            sx = cx + int(st_x * s)
            draw.line([(sx, cy - int(55 * s)), (sx + int(15 * s), cy + int(35 * s))], fill=stripe_col, width=int(9 * s))

        # 4. Chameleon Head and Casque (tall helmet crest behind head)
        head_cx = cx + int(135 * s)
        head_cy = cy - int(20 * s)
        # Casque crest
        casque_pts = [
            (head_cx - int(45 * s), head_cy - int(30 * s)),
            (head_cx - int(25 * s), head_cy - int(95 * s)),
            (head_cx + int(15 * s), head_cy - int(55 * s)),
        ]
        draw.polygon(casque_pts, fill=crest_col)
        # Head dome
        draw.ellipse([head_cx - int(60 * s), head_cy - int(50 * s), head_cx + int(65 * s), head_cy + int(50 * s)], fill=body_col)

        # Cute Snout & Mouth
        snout_x = head_cx + int(75 * s)
        snout_y = head_cy + int(15 * s)
        draw.pieslice([snout_x - int(40 * s), snout_y - int(30 * s), snout_x + int(25 * s), snout_y + int(30 * s)], start=280, end=80, fill=body_col)

        if tongue:
            # Mouth opened wide for tongue shot!
            draw.pieslice([head_cx + int(20 * s), head_cy, snout_x + int(25 * s), snout_y + int(30 * s)], start=340, end=80, fill=(180, 20, 50, 255))
        else:
            # Sweet friendly lizard smile
            draw.arc([snout_x - int(30 * s), snout_y - int(10 * s), snout_x + int(15 * s), snout_y + int(20 * s)], start=20, end=150, fill=(20, 45, 25, 255), width=int(4 * s))

        # 5. Turret Cone Eyes
        if eye_mode == "dual_radar":
            # Dual 360 radar eyes looking in completely opposite directions!
            # Left Eye turret pointing UP and BACK
            le_x, le_y = head_cx - int(15 * s), head_cy - int(35 * s)
            draw.ellipse([le_x - int(28 * s), le_y - int(28 * s), le_x + int(28 * s), le_y + int(28 * s)], fill=body_col, outline=crest_col, width=int(4 * s))
            draw.ellipse([le_x - int(14 * s), le_y - int(24 * s), le_x + int(4 * s), le_y - int(6 * s)], fill=(255, 255, 255, 255))
            draw.ellipse([le_x - int(10 * s), le_y - int(20 * s), le_x, le_y - int(10 * s)], fill=(15, 25, 45, 255))

            # Right Eye turret pointing FORWARD and DOWN
            re_x, re_y = head_cx + int(25 * s), head_cy - int(10 * s)
            draw.ellipse([re_x - int(28 * s), re_y - int(28 * s), re_x + int(28 * s), re_y + int(28 * s)], fill=body_col, outline=crest_col, width=int(4 * s))
            draw.ellipse([re_x + int(4 * s), re_y + int(2 * s), re_x + int(22 * s), re_y + int(20 * s)], fill=(255, 255, 255, 255))
            draw.ellipse([re_x + int(8 * s), re_y + int(6 * s), re_x + int(18 * s), re_y + int(16 * s)], fill=(15, 25, 45, 255))
        else:
            # Cute big cartoon turret eye
            eye_cx = head_cx + int(10 * s)
            eye_cy = head_cy - int(20 * s)
            er = int(34 * s)
            # Turret ring
            draw.ellipse([eye_cx - er, eye_cy - er, eye_cx + er, eye_cy + er], fill=body_col, outline=crest_col, width=int(5 * s))
            # Sclera
            draw.ellipse([eye_cx - int(22 * s), eye_cy - int(22 * s), eye_cx + int(22 * s), eye_cy + int(22 * s)], fill=(255, 255, 255, 255))
            # Big curious pupil
            draw.ellipse([eye_cx - int(8 * s), eye_cy - int(14 * s), eye_cx + int(18 * s), eye_cy + int(12 * s)], fill=(15, 30, 45, 255))
            # Catchlight
            draw.ellipse([eye_cx + int(2 * s), eye_cy - int(10 * s), eye_cx + int(12 * s), eye_cy], fill=(255, 255, 255, 255))

    def _draw_nanocrystals_zoom_prop(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, anim_phase: float = 0.0):
        """Draws magnifying glass zoom bubble showing a microscopic grid of photonic nanocrystals like tiny mirrors."""
        r = 175
        # Wooden handle
        draw.line([(cx + int(r * 0.7), cy + int(r * 0.7)), (cx + r + 85, cy + r + 85)], fill=(130, 85, 45, 255), width=24)
        # Gold brass frame
        draw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=(255, 215, 0, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(12, 28, 56, 245))

        # Geometric lattice of sparkling diamond nanocrystals
        grid_offsets = [
            (-90, -70), (-30, -70), (30, -70), (90, -70),
            (-60, -20), (0, -20), (60, -20),
            (-90, 30), (-30, 30), (30, 30), (90, 30),
            (-60, 80), (0, 80), (60, 80)
        ]
        pulse = int(math.sin(anim_phase * 2 * math.pi) * 3)
        for gx, gy in grid_offsets:
            px = cx + gx
            py = cy + gy
            cr = 16 + pulse
            # Diamond crystal
            diamond = [(px, py - cr), (px + cr, py), (px, py + cr), (px - cr, py)]
            draw.polygon(diamond, fill=(0, 240, 255, 240), outline=(255, 255, 255, 255))
            # Inner gleam
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 255, 255, 255))

        # Bouncing light beam reflection
        draw.line([(cx - 120, cy - 120), (cx, cy - 20)], fill=(255, 240, 100, 220), width=6)
        draw.line([(cx, cy - 20), (cx + 120, cy - 110)], fill=(0, 255, 180, 240), width=6)

        # Label pill
        draw.rounded_rectangle([cx - 165, cy + r - 30, cx + 165, cy + r + 25], radius=16, fill=(0, 205, 160, 255), outline=(255, 255, 255, 255), width=3)
        font = self._get_font(23)
        draw.text((cx, cy + r - 2), "NANOCRYSTAL MIRRORS!", fill=(10, 30, 45), font=font, anchor="mm")

    def _draw_crystal_spacing_diagram(self, draw: ImageDraw.ImageDraw, cx: int, y: int, state: str = "tight", label: str = ""):
        """Draws a comparison diagram illustrating tight crystal spacing (blue/green) vs spread crystal spacing (red/yellow)."""
        w, h = 420, 52
        draw.rounded_rectangle([cx - w, y - h, cx + w, y + h], radius=22, fill=(20, 32, 60, 245), outline=(255, 255, 255, 255), width=4)

        # Draw lattice dots on left side
        if state == "tight":
            # Packed tight crystals
            dot_col = (0, 240, 220, 255)
            for i in range(7):
                dx = cx - 360 + i * 22
                draw.ellipse([dx - 8, y - 8, dx + 8, y + 8], fill=dot_col, outline=(255, 255, 255, 255), width=2)
            # Arrow
            draw.line([(cx - 200, y), (cx - 170, y)], fill=(255, 255, 255, 255), width=4)
        else:
            # Spread wide crystals
            dot_col = (255, 75, 65, 255)
            for i in range(4):
                dx = cx - 360 + i * 46
                draw.ellipse([dx - 12, y - 12, dx + 12, y + 12], fill=dot_col, outline=(255, 230, 100, 255), width=2)
            # Stretched arrows between them
            draw.line([(cx - 200, y), (cx - 160, y)], fill=(255, 215, 0, 255), width=4)

        font = self._get_font(26)
        draw.text((cx + 90, y), label or ("TIGHT = GREEN" if state == "tight" else "SPREAD = RED"), fill=(255, 255, 255), font=font, anchor="mm")

    def _draw_radar_eye_arcs(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws 360-degree radar vision arcs illustrating independent turret eye movement."""
        # Eye center points
        re_x, re_y = cx + int(135 * 1.3), cy - int(20 * 1.3)
        # Radar scan rings
        for r_arc, col in [(90, (0, 245, 255, 180)), (140, (0, 245, 255, 120)), (190, (0, 245, 255, 70))]:
            draw.arc([re_x - r_arc, re_y - r_arc, re_x + r_arc, re_y + r_arc], start=290, end=90, fill=col, width=4)
        for r_arc, col in [(90, (255, 215, 0, 180)), (140, (255, 215, 0, 120)), (190, (255, 215, 0, 70))]:
            draw.arc([re_x - r_arc, re_y - r_arc, re_x + r_arc, re_y + r_arc], start=120, end=260, fill=col, width=4)

        # Radar callout badge
        draw.rounded_rectangle([cx - 220, cy - 310, cx + 220, cy - 240], radius=18, fill=(0, 220, 200, 245), outline=(255, 255, 255, 255), width=3)
        font = self._get_font(28)
        draw.text((cx, cy - 275), "360° INDEPENDENT RADAR!", fill=(15, 30, 50), font=font, anchor="mm")

    def _draw_ballistic_tongue_prop(
        self,
        draw: ImageDraw.ImageDraw,
        start_xy: Tuple[int, int],
        end_xy: Tuple[int, int],
        anim_phase: float = 0.0,
    ):
        """Draws the lightning-fast 60 mph ballistic chameleon tongue shooting across the screen."""
        sx, sy = start_xy
        ex, ey = end_xy
        # Accordion spring tongue shaft
        num_coils = 8
        pts = [(sx, sy)]
        for i in range(1, num_coils):
            frac = i / float(num_coils)
            tx = sx + int((ex - sx) * frac)
            ty = sy + int((ey - sy) * frac) + (22 if i % 2 == 1 else -22)
            pts.append((tx, ty))
        pts.append((ex, ey))

        # Outer tongue
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(255, 60, 120, 240), width=18)
        # Inner tongue highlight
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(255, 175, 200, 255), width=8)

        # Bulbous sticky suction-cup tip
        draw.ellipse([ex - 28, ey - 28, ex + 28, ey + 28], fill=(255, 30, 85, 255), outline=(255, 255, 255, 255), width=4)
        # Impact starburst
        for ang in (0.0, 0.78, 1.57, 2.35, 3.14, 3.92, 4.71, 5.49):
            spx = ex + int(math.cos(ang) * 44)
            spy = ey + int(math.sin(ang) * 44)
            draw.line([(ex, ey), (spx, spy)], fill=(255, 235, 80, 240), width=5)

        # Speed badge
        draw.rounded_rectangle([cx if 'cx' in locals() else (sx + ex)//2 - 190, ey - 100, (sx + ex)//2 + 190, ey - 45], radius=16, fill=(255, 50, 75, 245), outline=(255, 255, 255, 255), width=3)
        font = self._get_font(26)
        draw.text(((sx + ex)//2, ey - 72), "60 MPH IN 0.01s!", fill=(255, 255, 255), font=font, anchor="mm")


