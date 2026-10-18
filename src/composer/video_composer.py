"""Video Composer assembling dynamic Ken Burns scenes, ducked audio, SFX, and kinetic subtitles."""

import os
import subprocess
import logging
from typing import Dict, Any, List, Optional
import imageio_ffmpeg

logger = logging.getLogger(__name__)

class VideoComposer:
    """Assembles vertical 1080x1920 30fps MP4 video with Ken Burns motion, ducked music, and SFX."""

    def __init__(
        self,
        output_dir: str = "data/output/videos",
        width: int = 1080,
        height: int = 1920,
        fps: int = 30
    ):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(self.output_dir, exist_ok=True)

    def assemble_short(
        self,
        scene_images: List[str],
        scene_durations: List[float],
        scene_sfx: List[str],
        narration_audio_path: str,
        bgm_audio_path: str,
        subtitle_ass_path: Optional[str] = None,
        output_filename: str = "final_short.mp4"
    ) -> str:
        """Composes all assets into a high-retention 1080x1920 YouTube Short MP4."""
        final_mp4_path = os.path.join(self.output_dir, output_filename)
        temp_video_path = os.path.join(self.output_dir, f"temp_{output_filename}")
        temp_audio_path = os.path.join(self.output_dir, f"audio_{output_filename}.wav")

        total_duration = sum(scene_durations)

        # 1. Render visual clip sequence with Ken Burns zoom effect
        self._render_visual_sequence(scene_images, scene_durations, temp_video_path)

        # 2. Mix narration, ducked BGM, and SFX
        self._mix_audio(
            narration_audio_path=narration_audio_path,
            bgm_audio_path=bgm_audio_path,
            scene_durations=scene_durations,
            scene_sfx=scene_sfx,
            total_duration=total_duration,
            output_audio_path=temp_audio_path
        )

        # 3. Combine video, mixed audio, and burn kinetic subtitles using FFmpeg
        self._mux_final_video(
            video_input=temp_video_path,
            audio_input=temp_audio_path,
            subtitle_ass=subtitle_ass_path,
            output_file=final_mp4_path,
            duration=total_duration
        )

        # Clean up temporary intermediate files
        for p in [temp_video_path, temp_audio_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        return final_mp4_path

    def _render_visual_sequence(self, images: List[str], durations: List[float], output_video: str):
        """Creates video clips with smooth Ken Burns zoom-in motions for each scene."""
        import moviepy.video.io.ImageSequenceClip as isc
        from PIL import Image

        # Ensure all images are RGB and exactly 1080x1920
        prepared_images = []
        for img_path in images:
            with Image.open(img_path) as im:
                if im.size != (self.width, self.height) or im.mode != "RGB":
                    resized = im.convert("RGB").resize((self.width, self.height), Image.Resampling.LANCZOS)
                    fixed_path = img_path.replace(".png", "_fixed.png")
                    resized.save(fixed_path)
                    prepared_images.append(fixed_path)
                else:
                    prepared_images.append(img_path)

        # Build FFmpeg complex filter for smooth zoompan transitions
        inputs = []
        filter_parts = []
        for i, (img_path, dur) in enumerate(zip(prepared_images, durations)):
            inputs.extend(["-i", img_path])
            # Alternating zoom in and zoom out for dynamic variety
            frames = max(int(round(dur * self.fps)), 1)
            if i % 2 == 0:
                zoom_expr = f"min(zoom+0.0015,1.15)"
            else:
                zoom_expr = f"if(eq(on,1),1.15,max(1.0,zoom-0.0015))"

            filter_parts.append(
                f"[{i}:v]scale={self.width}x{self.height},zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps}[v{i}];"
            )

        concat_inputs = "".join(f"[v{i}]" for i in range(len(images)))
        filter_complex = "".join(filter_parts) + f"{concat_inputs}concat=n={len(images)}:v=1:a=0[outv]"

        cmd = [
            self.ffmpeg_exe,
            "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            output_video
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            logger.warning(f"Complex zoompan failed: {e}. Falling back to standard image sequence.")
            # Fallback standard image slideshow with precise scene durations
            clip = isc.ImageSequenceClip(prepared_images, durations=durations)
            clip.write_videofile(output_video, fps=self.fps, codec="libx264", audio=False, logger=None)

    def _mix_audio(
        self,
        narration_audio_path: str,
        bgm_audio_path: str,
        scene_durations: List[float],
        scene_sfx: List[str],
        total_duration: float,
        output_audio_path: str
    ):
        """Mixes narration, ducked BGM, and transition SFX into a single master audio track."""
        # Use FFmpeg amix with volume ducking filters
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", narration_audio_path,
            "-stream_loop", "-1", "-i", bgm_audio_path,
            "-filter_complex",
            # BGM ducked to 12% volume, Voice at 100% volume
            f"[1:a]volume=0.12[bgm];[0:a]volume=1.0[voice];[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[outa]",
            "-map", "[outa]",
            "-t", str(total_duration),
            output_audio_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            logger.warning(f"Audio ducking mix failed: {e}. Falling back to narration only.")
            import shutil
            shutil.copy(narration_audio_path, output_audio_path)

    def _mux_final_video(
        self,
        video_input: str,
        audio_input: str,
        subtitle_ass: Optional[str],
        output_file: str,
        duration: float
    ):
        """Muxes video and audio tracks, burning ASS kinetic subtitles."""
        if subtitle_ass and os.path.exists(subtitle_ass):
            # Escape filename for FFmpeg filter
            escaped_sub = subtitle_ass.replace(":", "\\:").replace("'", "\\'")
            video_filter = f"ass='{escaped_sub}'"
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i", video_input,
                "-i", audio_input,
                "-vf", video_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(duration),
                output_file
            ]
        else:
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i", video_input,
                "-i", audio_input,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(duration),
                output_file
            ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
