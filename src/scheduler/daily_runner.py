"""Daily Pipeline Runner orchestrating the end-to-end automated generation, AI Studio voice/vision audit, and publishing flow."""

import os
import yaml
import logging
import datetime
from typing import Dict, Any, Optional

from src.analyzer.trend_scout import TrendScout
from src.analyzer.pattern_extractor import PatternExtractor
from src.analyzer.playbook_manager import PlaybookManager
from src.generator.script_engine import ScriptEngine
from src.generator.voice_engine import VoiceEngine
from src.generator.visual_engine import VisualEngine
from src.generator.asset_manager import AssetManager
from src.composer.subtitle_engine import SubtitleEngine
from src.composer.video_composer import VideoComposer
from src.compliance.policy_checker import PolicyChecker
from src.compliance.visual_auditor import VisualAuditor
from src.publisher.youtube_uploader import YouTubeUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DailyRunner")


class DailyRunner:
    """End-to-end automated orchestrator for daily YouTube Shorts generation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.scout = TrendScout()
        self.extractor = PatternExtractor()
        self.playbook = PlaybookManager(self.config.get("paths", {}).get("playbook_file", "data/viral_playbook.json"))
        self.script_engine = ScriptEngine()

        audio_cfg = self.config.get("audio", {})
        self.voice_engine = VoiceEngine(
            voice=audio_cfg.get("primary_voice", "en-US-AnaNeural"),
            ai_studio_voice=audio_cfg.get("ai_studio_voice", "Aoede"),
            rate=audio_cfg.get("speech_rate", "+5%"),
            pitch=audio_cfg.get("speech_pitch", "+3Hz"),
            voice_gender=audio_cfg.get("voice_gender", "female")
        )

        self.visual_engine = VisualEngine(
            output_dir=os.path.join(self.config.get("paths", {}).get("output_dir", "data/output"), "visuals")
        )
        self.asset_manager = AssetManager(
            sfx_dir=self.config.get("paths", {}).get("sound_effects_dir", "data/sound_effects"),
            music_dir=self.config.get("paths", {}).get("music_dir", "data/music")
        )
        self.subtitle_engine = SubtitleEngine()
        self.composer = VideoComposer(
            output_dir=os.path.join(self.config.get("paths", {}).get("output_dir", "data/output"), "videos")
        )
        self.policy_checker = PolicyChecker()
        self.visual_auditor = VisualAuditor()
        self.uploader = YouTubeUploader()

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def run_daily_pipeline(
        self,
        niche: Optional[str] = None,
        auto_publish: bool = False,
        tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the full automated video generation cycle for today."""
        start_time = datetime.datetime.now()
        timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
        logger.info(f"=== Starting Daily YouTube Shorts Generator Pipeline [{timestamp_str}] ===")

        # 1. Resolve age tier (Option A schedule rotation fallback)
        if not tier:
            tiers_path = os.path.join("data", "age_tiers.json")
            if os.path.exists(tiers_path):
                try:
                    import json
                    with open(tiers_path, "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                    weekday_str = str(start_time.weekday())
                    tier = tdata.get("weekly_rotation_schedule", {}).get(weekday_str, "younger")
                except Exception:
                    tier = "younger"
            else:
                tier = "younger"
        logger.info(f"[1/9] Active Age Tier: {tier.upper()}")

        # 2. Select niche
        active_niches = self.config.get("niches", {}).get("active", ["animal_secrets"])
        selected_niche = niche or active_niches[0]
        logger.info(f"[2/9] Selected Niche: {selected_niche}")

        # 3. Scout trending shorts & update Playbook (/learn)
        logger.info("[3/9] Scouting trending shorts & analyzing retention mechanics...")
        trending_items = self.scout.scout_trending_topics(niche=selected_niche, max_results=3)
        analyzed_patterns = []
        for item in trending_items:
            formula = self.extractor.extract_formula(
                title=item.get("title", ""),
                description=item.get("snippet", ""),
                transcript=item.get("hook", "")
            )
            analyzed_patterns.append(formula)

        self.playbook.update_from_trend_analysis(analyzed_patterns)
        winning_formula = self.playbook.get_winning_formula(niche=selected_niche)
        logger.info(f"Playbook updated. Active winning archetype: {winning_formula['archetype']['id']}")

        # 4. Generate character-consistent script using winning formula & tier
        today_topic = trending_items[0].get("topic", "Flamingo Superpowers") if trending_items else "Flamingo Superpowers"
        logger.info(f"[4/9] Generating script for topic: '{today_topic}' [Tier: {tier}]...")
        script = self.script_engine.generate_script(topic=today_topic, formula=winning_formula, niche=selected_niche, tier=tier)
        logger.info(f"Script generated: '{script.title}' ({len(script.scenes)} scenes) [Tier: {tier}]")

        # 4. Compliance & Originality Pre-flight Check
        logger.info("[4/9] Running COPPA and Reused Content pre-flight check...")
        full_script_text = " ".join([s.narration for s in script.scenes])
        policy_res = self.policy_checker.evaluate_script(script_text=full_script_text, title=script.title)
        if not policy_res.passed:
            logger.warning(f"Policy flags detected: {policy_res.flags}. Adjusting script parameters.")

        # 5. Synthesize Google AI Studio (`gemini-3.8-flash-tts`) voiceover & lock scene durations
        logger.info("[5/9] Synthesizing studio voiceover via Google AI Studio TTS & locking scene timings...")
        audio_dir = os.path.join(self.config.get("paths", {}).get("output_dir", "data/output"), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        narration_wav = os.path.join(audio_dir, f"narration_{timestamp_str}.wav")
        _, word_timestamps, scene_durations = self.voice_engine.synthesize_scenes(
            scenes=script.scenes,
            output_path=narration_wav
        )
        logger.info(
            f"Voiceover complete ({round(sum(scene_durations), 2)}s total). "
            f"Extracted {len(word_timestamps)} word timestamps locked to {len(scene_durations)} scenes."
        )

        # 6. Generate consistent-face visual scene assets
        logger.info(f"[6/9] Generating {len(script.scenes)} character-consistent 9:16 scene visuals...")
        scene_images = self.visual_engine.generate_all_scenes_parallel(
            scenes=script.scenes,
            filename_prefix=f"short_{timestamp_str}"
        )
        scene_sfx = [s.sfx_cue for s in script.scenes]

        # 7. Run Automated "Silent Watch Audit" using Google AI Studio (`gemini-3.8-flash` Vision)
        logger.info("[7/9] Running Silent Watch Visual Audit (Audio Muted Test) via Gemini Vision...")
        audit_report_path = os.path.join(
            self.config.get("paths", {}).get("output_dir", "data/output"),
            "visuals",
            f"silent_audit_{timestamp_str}.json"
        )
        visual_audit = self.visual_auditor.audit_silent_visual_story(
            scene_images=scene_images,
            intended_topic=today_topic,
            scenes=script.scenes,
            word_timestamps=word_timestamps,
            tournament_report=getattr(script, "tournament_report", {}),
            output_report_path=audit_report_path
        )
        logger.info(
            f"Silent Watch Audit Result: Comprehension={visual_audit.get('silent_story_comprehension_score')}/100, "
            f"Face Consistency={visual_audit.get('character_face_consistency_score')}/100, "
            f"AV Sync={visual_audit.get('audio_visual_sync_score')}/100, "
            f"Watermark-Free={visual_audit.get('watermark_free')}"
        )

        # 8. Generate kinetic subtitles & assemble final video
        logger.info("[8/9] Generating kinetic subtitles and assembling final 1080x1920 MP4...")
        ass_path = os.path.join(
            self.config.get("paths", {}).get("output_dir", "data/output"),
            "subtitles",
            f"subs_{timestamp_str}.ass"
        )
        self.subtitle_engine.generate_ass_file(
            word_timestamps=word_timestamps,
            output_ass_path=ass_path,
            scene_narrations=[s.narration for s in script.scenes],
            scene_durations=scene_durations
        )

        bgm_path = self.asset_manager.get_bgm_path(tier=tier, niche=selected_niche)
        final_mp4 = self.composer.assemble_short(
            scene_images=scene_images,
            scene_durations=scene_durations,
            scene_sfx=scene_sfx,
            narration_audio_path=narration_wav,
            bgm_audio_path=bgm_path,
            subtitle_ass_path=ass_path,
            output_filename=f"short_{timestamp_str}.mp4"
        )
        mp4_sync_check = self.visual_auditor.verify_rendered_mp4_sync(final_mp4, sum(scene_durations))
        visual_audit["mp4_stream_sync_verification"] = mp4_sync_check
        logger.info(f"Final video successfully rendered & sync-verified: {final_mp4} ({mp4_sync_check})")

        # 9. Record in history & upload / stage package
        self.policy_checker.register_published_script(title=script.title, script_text=full_script_text)
        privacy = self.config.get("publishing", {}).get("default_privacy", "unlisted")
        made_for_kids = self.config.get("monetization", {}).get("default_made_for_kids", False)

        upload_res = self.uploader.publish_short(
            video_path=final_mp4,
            title=script.title,
            description=script.description,
            tags=script.tags,
            privacy_status=privacy,
            made_for_kids=made_for_kids,
            scene_images=scene_images,
            character_sheet_path=self.visual_engine.reference_sheet_path,
            visual_audit_report=visual_audit,
            tier=tier
        )
        logger.info(f"[9/9] Distribution result: {upload_res.message}")

        return {
            "status": "success",
            "timestamp": timestamp_str,
            "title": script.title,
            "topic": today_topic,
            "video_path": final_mp4,
            "upload_result": upload_res.to_dict(),
            "policy_result": policy_res.to_dict(),
            "visual_audit": visual_audit,
            "duration_sec": round(sum(scene_durations), 2)
        }
