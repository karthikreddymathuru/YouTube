"""Unit and Integration Tests for the Automated Kids YouTube Shorts Generator."""

import os
import sys
import unittest
import tempfile
import shutil

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.publisher.youtube_uploader import YouTubeUploader

class TestKidsShortsPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_yt_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_trend_scout_and_playbook(self):
        scout = TrendScout()
        topics = scout.scout_trending_topics(niche="animal_secrets", max_results=2)
        self.assertGreaterEqual(len(topics), 1)
        self.assertIn("title", topics[0])

        extractor = PatternExtractor()
        formula = extractor.extract_formula(
            title=topics[0]["title"],
            description="Amazing facts about animals for kids",
            transcript="Did you know flamingos can drink boiling water?"
        )
        self.assertIn("hook_archetype", formula)
        self.assertIn("estimated_retention_score", formula)

        playbook_path = os.path.join(self.test_dir, "test_playbook.json")
        playbook = PlaybookManager(playbook_path=playbook_path)
        playbook.update_from_trend_analysis([formula])
        self.assertTrue(os.path.exists(playbook_path))

        winning = playbook.get_winning_formula()
        self.assertIn("archetype", winning)
        self.assertIn("benchmarks", winning)

    def test_02_script_engine(self):
        engine = ScriptEngine()
        formula = {
            "archetype": {
                "id": "visual_disbelief_challenge",
                "example": "Did you know that octopuses have 3 hearts?",
                "sfx_cue": "whoosh"
            }
        }
        script = engine.generate_script(topic="Octopus Superpowers", formula=formula, niche="animal_secrets")
        self.assertIsNotNone(script.title)
        self.assertGreaterEqual(len(script.scenes), 3)
        self.assertTrue(any("3D" in s.visual_prompt for s in script.scenes))
        self.assertIsNotNone(script.loop_ending)

    def test_03_voice_engine(self):
        voice_eng = VoiceEngine()
        audio_path = os.path.join(self.test_dir, "test_narration.wav")
        res_path, word_timestamps = voice_eng.synthesize("Did you know that octopuses have three hearts?", audio_path)
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(len(word_timestamps), 0)
        self.assertIn("word", word_timestamps[0])
        self.assertIn("start", word_timestamps[0])
        self.assertIn("end", word_timestamps[0])

    def test_04_visual_engine(self):
        visual_eng = VisualEngine(output_dir=self.test_dir, width=540, height=960) # Scaled for fast test
        img_path = visual_eng.generate_scene_image(
            scene_id=1,
            prompt="Cute cartoon octopus swimming in glowing blue water, 3D Pixar style",
            filename_prefix="test_scene"
        )
        self.assertTrue(os.path.exists(img_path))
        from PIL import Image
        with Image.open(img_path) as im:
            self.assertEqual(im.size, (540, 960))

    def test_05_asset_manager(self):
        sfx_dir = os.path.join(self.test_dir, "sfx")
        music_dir = os.path.join(self.test_dir, "music")
        mgr = AssetManager(sfx_dir=sfx_dir, music_dir=music_dir)

        bgm = mgr.get_default_bgm_path()
        whoosh = mgr.get_sfx_path("whoosh")
        ding = mgr.get_sfx_path("bell_ding")

        self.assertTrue(os.path.exists(bgm))
        self.assertTrue(os.path.exists(whoosh))
        self.assertTrue(os.path.exists(ding))

    def test_06_subtitle_engine(self):
        sub_eng = SubtitleEngine()
        ass_path = os.path.join(self.test_dir, "captions.ass")
        words = [
            {"word": "Did", "start": 0.0, "end": 0.25},
            {"word": "you", "start": 0.25, "end": 0.45},
            {"word": "know?", "start": 0.45, "end": 0.85}
        ]
        out_ass = sub_eng.generate_ass_file(words, ass_path, video_width=540, video_height=960)
        self.assertTrue(os.path.exists(out_ass))
        with open(out_ass, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("KineticHighlight", content)
        self.assertIn("DID", content)

    def test_07_policy_checker(self):
        history_path = os.path.join(self.test_dir, "history.json")
        checker = PolicyChecker(history_db_path=history_path)

        # 1. Clean script should pass
        res_pass = checker.evaluate_script(
            "Did you know why puppies wag their tails when they are excited?",
            "Secret of Puppies #shorts"
        )
        self.assertTrue(res_pass.passed)

        # 2. Dangerous words should fail
        res_fail = checker.evaluate_script(
            "Kids should play with this dangerous fire and knife dare!",
            "Scary Prank"
        )
        self.assertFalse(res_fail.passed)
        self.assertTrue(len(res_fail.flags) > 0)

        # 3. Duplicate check should trigger
        checker.register_published_script("Secret of Puppies #shorts", "Did you know why puppies wag their tails when they are excited?")
        res_dup = checker.evaluate_script(
            "Did you know why puppies wag their tails when they are excited?",
            "Secret of Puppies #shorts"
        )
        self.assertFalse(res_dup.passed)
        self.assertTrue(any("similar" in f for f in res_dup.flags))

    def test_08_end_to_end_assembly(self):
        # Test rendering a mini 2-scene 9:16 vertical video
        composer = VideoComposer(output_dir=self.test_dir, width=540, height=960, fps=24)
        visual_eng = VisualEngine(output_dir=self.test_dir, width=540, height=960)
        img1 = visual_eng.generate_scene_image(1, "Scene 1", "test_v")
        img2 = visual_eng.generate_scene_image(2, "Scene 2", "test_v")

        voice_eng = VoiceEngine()
        audio_path = os.path.join(self.test_dir, "test_audio.wav")
        _, timestamps = voice_eng.synthesize("Octopuses have three hearts and blue blood!", audio_path)

        sub_eng = SubtitleEngine()
        ass_path = os.path.join(self.test_dir, "test.ass")
        sub_eng.generate_ass_file(timestamps, ass_path, video_width=540, video_height=960)

        asset_mgr = AssetManager(sfx_dir=os.path.join(self.test_dir, "sfx"), music_dir=os.path.join(self.test_dir, "music"))

        final_mp4 = composer.assemble_short(
            scene_images=[img1, img2],
            scene_durations=[2.0, 2.0],
            scene_sfx=["whoosh", "pop"],
            narration_audio_path=audio_path,
            bgm_audio_path=asset_mgr.get_default_bgm_path(),
            subtitle_ass_path=ass_path,
            output_filename="test_render.mp4"
        )

        self.assertTrue(os.path.exists(final_mp4))
        self.assertGreater(os.path.getsize(final_mp4), 1000)

if __name__ == "__main__":
    unittest.main()
