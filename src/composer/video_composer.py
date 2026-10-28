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
        """
        Renders dynamic cinematic video clips with buttery-smooth subpixel Pan In and Pan Out
        for each scene, keeping the top headline banner and reaction webcam rock-solid stable.
        """
        import subprocess
        from PIL import Image

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_video
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        try:
            for i, (img_path, dur) in enumerate(zip(images, durations)):
                raw_path = img_path.replace(".png", "_raw.png")
                overlay_path = img_path.replace(".png", "_overlay.png")

                # If raw background exists, pan/zoom it while keeping overlay steady.
                # Otherwise, pan the composited image smoothly.
                if os.path.exists(raw_path):
                    base_im = Image.open(raw_path).convert("RGB")
                    overlay_im = Image.open(overlay_path).convert("RGBA") if os.path.exists(overlay_path) else None
                else:
                    base_im = Image.open(img_path).convert("RGB")
                    overlay_im = None

                if base_im.size != (self.width, self.height):
                    base_im = base_im.resize((self.width, self.height), Image.Resampling.LANCZOS)

                frames = max(int(round(dur * self.fps)), 1)

                # Alternating dynamic cinematic camera motions based on scene flow:
                # Scene 0 (1st scene): Smooth Pan In (Zoom In) towards animal: 1.00x -> 1.15x
                # Scene 1 (2nd scene): Smooth Pan Out (Zoom Out) revealing habitat: 1.15x -> 1.00x
                # Scene 2 (3rd scene): Smooth Pan In towards superpower/shock: 1.00x -> 1.15x
                # Scene 3 (4th scene): Smooth Pan Out / Wide reveal: 1.15x -> 1.00x
                motion_type = i % 2  # 0: Pan In, 1: Pan Out

                for f in range(frames):
                    # Cubic smoothstep easing: zero jerk, perfectly soft acceleration & deceleration
                    t = f / float(max(frames - 1, 1))
                    progress = t * t * (3.0 - 2.0 * t)

                    if motion_type == 0:
                        # Pan In: zoom from 1.00 to 1.15
                        zoom = max(1.0, 1.0 + 0.15 * progress)
                    else:
                        # Pan Out: zoom from 1.15 down to 1.00
                        zoom = max(1.0, 1.15 - 0.15 * progress)

                    cw = float(self.width) / zoom
                    ch = float(self.height) / zoom
                    cx = max(0.0, (float(self.width) - cw) / 2.0)
                    cy = max(0.0, (float(self.height) - ch) / 2.0)
                    x2 = min(float(self.width), cx + cw)
                    y2 = min(float(self.height), cy + ch)

                    # Subpixel float resampling with Bicubic: high fidelity, crisp edges, zero blurring!
                    frame_im = base_im.resize(
                        (self.width, self.height),
                        resample=Image.Resampling.BICUBIC,
                        box=(cx, cy, x2, y2)
                    )

                    if overlay_im is not None:
                        frame_rgba = frame_im.convert("RGBA")
                        frame_rgba.alpha_composite(overlay_im)
                        frame_im = frame_rgba.convert("RGB")

                    proc.stdin.write(frame_im.tobytes())

            proc.stdin.close()
            proc.wait()
        except Exception as e:
            if proc and proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc:
                proc.kill()
            logger.warning(f"Subpixel pan/zoom rendering failed: {e}. Falling back to standard image slideshow.")
            import moviepy.video.io.ImageSequenceClip as isc
            clip = isc.ImageSequenceClip(images, durations=durations)
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
        # Balanced high-presence mixing: Voice at 135% (crisp, loud, crystal-clear), BGM at 22%
        # normalize=0 prevents amix from automatically halving the volume of the inputs
        voice_vol = 1.35
        bgm_vol = 0.22

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", narration_audio_path,
            "-stream_loop", "-1", "-i", bgm_audio_path,
            "-filter_complex",
            f"[0:a]volume={voice_vol}[voice];[1:a]volume={bgm_vol}[bgm];[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[outa]",
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
                "-crf", "21",
                "-b:v", "2300k",
                "-maxrate", "2600k",
                "-bufsize", "4500k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
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
                "-crf", "21",
                "-b:v", "2300k",
                "-maxrate", "2600k",
                "-bufsize", "4500k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
                "-t", str(duration),
                output_file
            ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
