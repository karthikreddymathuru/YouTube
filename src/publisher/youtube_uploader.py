"""YouTube Uploader for automating Shorts uploads via Google YouTube Data API v3."""

import os
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
import shutil
import imageio_ffmpeg

logger = logging.getLogger(__name__)


class UploadResult:
    def __init__(self, success: bool, video_id: str, video_url: str, mode: str, message: str, package_dir: Optional[str] = None):
        self.success = success
        self.video_id = video_id
        self.video_url = video_url
        self.mode = mode
        self.message = message
        self.package_dir = package_dir

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "video_id": self.video_id,
            "video_url": self.video_url,
            "mode": self.mode,
            "message": self.message,
            "package_dir": self.package_dir
        }


class YouTubeUploader:
    """Manages YouTube Data API v3 uploads and staging export packages."""

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly"
    ]

    def __init__(
        self,
        client_secrets_path: Optional[str] = None,
        staging_dir: str = "data/ready_for_upload"
    ):
        self.client_secrets_path = client_secrets_path or self._find_credentials_file()
        self.staging_dir = staging_dir
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(self.staging_dir, exist_ok=True)

    def _find_credentials_file(self) -> Optional[str]:
        """Discovers credentials.json or client_secret.json in workspace root or data/."""
        import glob
        candidates = [
            "credentials.json",
            "client_secret.json",
            "data/credentials.json",
            "data/client_secret.json",
        ]
        candidates.extend(glob.glob("client_secret*.json"))
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def authenticate(self, force: bool = False):
        """Authenticates with YouTube Data API v3 and saves/refreshes data/token.json."""
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        token_path = "data/token.json"
        creds = None
        if os.path.exists(token_path) and not force:
            try:
                creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
            except Exception as e:
                logger.debug(f"Could not load existing token: {e}")

        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(token_path, "w") as tf:
                    tf.write(creds.to_json())
                logger.info(f"Refreshed YouTube OAuth token at {token_path}")
                return creds
            except Exception as e:
                logger.warning(f"Token refresh failed ({e}); re-authenticating...")

        creds_file = self._find_credentials_file()
        if not creds_file:
            raise FileNotFoundError(
                "Neither 'credentials.json' nor 'client_secret.json' found. "
                "Please place your OAuth client JSON file in the project root."
            )

        flow = InstalledAppFlow.from_client_secrets_file(creds_file, self.SCOPES)
        creds = flow.run_local_server(port=8080)
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as tf:
            tf.write(creds.to_json())
        logger.info(f"Successfully generated YouTube OAuth token at {token_path}")
        return creds

    def publish_short(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: List[str],
        privacy_status: str = "unlisted",
        made_for_kids: bool = False,
        category_id: str = "27",
        scene_images: Optional[List[str]] = None,
        character_sheet_path: Optional[str] = None,
        visual_audit_report: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """Stages the video package locally and uploads to YouTube via API v3 if authenticated."""
        stage_res = self._stage_local_package(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            privacy_status=privacy_status,
            made_for_kids=made_for_kids,
            scene_images=scene_images,
            character_sheet_path=character_sheet_path,
            visual_audit_report=visual_audit_report
        )

        creds_file = self._find_credentials_file()
        token_path = "data/token.json"

        if os.path.exists(token_path) or creds_file:
            try:
                staged_video = os.path.join(stage_res.package_dir, os.path.basename(video_path))
                api_res = self._upload_via_api(
                    video_path=staged_video if os.path.exists(staged_video) else video_path,
                    title=title,
                    description=description,
                    tags=tags,
                    privacy_status=privacy_status,
                    made_for_kids=made_for_kids,
                    category_id=category_id
                )
                api_res.package_dir = stage_res.package_dir
                return api_res
            except Exception as e:
                logger.error(f"YouTube API upload error: {e}. Video remains staged locally at {stage_res.package_dir}.")

        return stage_res

    def _upload_via_api(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: List[str],
        privacy_status: str,
        made_for_kids: bool,
        category_id: str
    ) -> UploadResult:
        """Executes authorized upload using googleapiclient."""
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = self.authenticate()
        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": title[:100],
                "description": description,
                "tags": tags,
                "categoryId": category_id
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": made_for_kids
            }
        }

        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()

        video_id = response.get("id", "")
        video_url = f"https://youtube.com/shorts/{video_id}"

        return UploadResult(
            success=True,
            video_id=video_id,
            video_url=video_url,
            mode="youtube_api_v3",
            message=f"Successfully uploaded to YouTube as {privacy_status}!"
        )

    def _stage_local_package(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: List[str],
        privacy_status: str,
        made_for_kids: bool,
        scene_images: Optional[List[str]] = None,
        character_sheet_path: Optional[str] = None,
        visual_audit_report: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """Creates a ready-to-upload package with video, muted test video, character sheet, scene images, and audit."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = os.path.join(self.staging_dir, f"package_{timestamp}")
        os.makedirs(package_dir, exist_ok=True)

        # 1. Copy final video
        dest_video = os.path.join(package_dir, os.path.basename(video_path))
        shutil.copy(video_path, dest_video)

        # 2. Generate muted video copy (`muted_silent_watch_test.mp4`) for direct visual storytelling inspection
        muted_video_path = os.path.join(package_dir, "muted_silent_watch_test.mp4")
        try:
            subprocess.run(
                [self.ffmpeg_exe, "-y", "-i", dest_video, "-c:v", "copy", "-an", muted_video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except Exception as e:
            logger.warning(f"Could not create muted test video copy: {e}")

        # 3. Copy Master Character Reference Sheet
        if character_sheet_path and os.path.exists(character_sheet_path):
            shutil.copy(character_sheet_path, os.path.join(package_dir, "master_character_sheet.png"))

        # 4. Copy story scene images & create thumbnail
        scenes_dir = os.path.join(package_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)
        staged_scenes = []

        if scene_images:
            for idx, img_p in enumerate(scene_images, 1):
                if os.path.exists(img_p):
                    dest_img = os.path.join(scenes_dir, f"scene_{idx}.png")
                    shutil.copy(img_p, dest_img)
                    staged_scenes.append(f"scenes/scene_{idx}.png")

            if scene_images and os.path.exists(scene_images[0]):
                thumb_dest = os.path.join(package_dir, "thumbnail.png")
                shutil.copy(scene_images[0], thumb_dest)

        # 5. Save Silent Watch Audit report
        if visual_audit_report:
            with open(os.path.join(package_dir, "silent_visual_audit.json"), "w", encoding="utf-8") as af:
                json.dump(visual_audit_report, af, indent=2, ensure_ascii=False)

        manifest = {
            "title": title,
            "description": description,
            "tags": tags,
            "privacy_status": privacy_status,
            "self_declared_made_for_kids": made_for_kids,
            "video_file": os.path.basename(video_path),
            "muted_visual_test_file": "muted_silent_watch_test.mp4" if os.path.exists(muted_video_path) else None,
            "master_character_sheet": "master_character_sheet.png" if os.path.exists(os.path.join(package_dir, "master_character_sheet.png")) else None,
            "thumbnail_file": "thumbnail.png" if os.path.exists(os.path.join(package_dir, "thumbnail.png")) else None,
            "scenes_folder": "scenes",
            "scene_images_count": len(staged_scenes),
            "silent_visual_audit": visual_audit_report,
            "generated_at": timestamp,
            "ready_for_upload": True
        }

        with open(os.path.join(package_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return UploadResult(
            success=True,
            video_id="staged_local",
            video_url=f"file://{os.path.abspath(dest_video)}",
            mode="staged_local_package",
            message=f"Video, muted visual test, character sheet, and {len(staged_scenes)} consistent-face story scenes staged in {package_dir}.",
            package_dir=os.path.abspath(package_dir)
        )
