"""
Playlist Manager for KuriousKids YouTube Channel.
Manages age-tiered playlists:
1. Preschool (Ages 4 & Under)
2. Younger (Ages 5–8)
3. Older (Ages 9–12)
Creates, caches, and automatically assigns Shorts to their dedicated playlists.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("KidsShorts.PlaylistManager")

PLAYLISTS_DB_PATH = "data/playlists.json"
AGE_TIERS_PATH = "data/age_tiers.json"


class PlaylistManager:
    """Handles playlist lookup, persistence, and video insertion for age-tiered Shorts."""

    DEFAULT_PLAYLIST_METADATA = {
        "preschool": {
            "title": "Preschool Fun & Learning (Ages 4 & Under) 🎈 | Kurious Kids",
            "description": "Joyful nursery rhymes, colorful animal sounds, and playful early learning adventures for babies, toddlers, and preschoolers!",
            "privacy": "public"
        },
        "younger": {
            "title": "Curious Explorers: Animal & Science Superpowers (Ages 5–8) 🔍 | Kurious Kids",
            "description": "Incredible animal superpowers, nature mysteries, and fun science adventures with Pip the Flamingo and Leo the Explorer Kid!",
            "privacy": "public"
        },
        "older": {
            "title": "Mind-Blown Science, Tech & Challenges (Ages 9–12) ⚡ | Kurious Kids",
            "description": "Crazy science experiments, space mysteries, engineering hacks, and brain-bending challenges for pre-teens and curious minds!",
            "privacy": "public"
        }
    }

    def __init__(self, db_path: str = PLAYLISTS_DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            initial_data = {
                "channel_name": "Kurious Kids",
                "playlists": {
                    "preschool": {
                        "playlist_id": None,
                        "title": self.DEFAULT_PLAYLIST_METADATA["preschool"]["title"],
                        "url": None,
                        "videos": []
                    },
                    "younger": {
                        "playlist_id": None,
                        "title": self.DEFAULT_PLAYLIST_METADATA["younger"]["title"],
                        "url": None,
                        "videos": []
                    },
                    "older": {
                        "playlist_id": None,
                        "title": self.DEFAULT_PLAYLIST_METADATA["older"]["title"],
                        "url": None,
                        "videos": []
                    }
                }
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def load_playlists(self) -> Dict[str, Any]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"playlists": {}}

    def save_playlists(self, data: Dict[str, Any]):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed saving playlists database: {e}")

    def sync_playlists_from_youtube(self, youtube_client) -> Dict[str, Any]:
        """Queries YouTube Data API to discover matching playlists on the channel."""
        data = self.load_playlists()
        try:
            res = youtube_client.playlists().list(part="snippet,status", mine=True, maxResults=50).execute()
            items = res.get("items", [])
            for p in items:
                p_id = p["id"]
                p_title = p["snippet"]["title"].lower()
                for tier in ["preschool", "younger", "older"]:
                    meta_title = self.DEFAULT_PLAYLIST_METADATA[tier]["title"].lower()
                    if tier in p_title or meta_title[:20] in p_title:
                        data["playlists"][tier]["playlist_id"] = p_id
                        data["playlists"][tier]["title"] = p["snippet"]["title"]
                        data["playlists"][tier]["url"] = f"https://www.youtube.com/playlist?list={p_id}"
                        logger.info(f"Discovered YouTube playlist for '{tier}': {p_id}")
            self.save_playlists(data)
        except Exception as e:
            logger.debug(f"Could not list playlists from YouTube API: {e}")
        return data

    def get_or_create_playlist(self, youtube_client, tier: str) -> Optional[str]:
        """Gets existing playlist ID or creates a new playlist on YouTube if permitted."""
        data = self.load_playlists()
        current = data.get("playlists", {}).get(tier, {})
        if current.get("playlist_id"):
            return current["playlist_id"]

        # Try to sync from channel first
        self.sync_playlists_from_youtube(youtube_client)
        data = self.load_playlists()
        current = data.get("playlists", {}).get(tier, {})
        if current.get("playlist_id"):
            return current["playlist_id"]

        # Attempt creation via API
        meta = self.DEFAULT_PLAYLIST_METADATA.get(tier, self.DEFAULT_PLAYLIST_METADATA["younger"])
        try:
            body = {
                "snippet": {
                    "title": meta["title"][:150],
                    "description": meta["description"]
                },
                "status": {
                    "privacyStatus": meta.get("privacy", "public")
                }
            }
            res = youtube_client.playlists().insert(part="snippet,status", body=body).execute()
            p_id = res["id"]
            data["playlists"][tier]["playlist_id"] = p_id
            data["playlists"][tier]["title"] = meta["title"]
            data["playlists"][tier]["url"] = f"https://www.youtube.com/playlist?list={p_id}"
            self.save_playlists(data)
            logger.info(f"Successfully created new YouTube playlist for '{tier}': {p_id}")
            return p_id
        except Exception as e:
            logger.warning(
                f"Cannot automatically create playlist '{meta['title']}' via API ({e}). "
                f"You can create it in YouTube Studio and add ID to data/playlists.json."
            )
            return None

    def add_video_to_playlist(self, youtube_client, video_id: str, tier: str) -> bool:
        """Adds a video to its age-tiered playlist."""
        if not video_id:
            return False

        tier_key = tier if tier in self.DEFAULT_PLAYLIST_METADATA else "younger"
        playlist_id = self.get_or_create_playlist(youtube_client, tier_key)

        data = self.load_playlists()
        tier_data = data.setdefault("playlists", {}).setdefault(tier_key, {"videos": []})
        if video_id not in tier_data.setdefault("videos", []):
            tier_data["videos"].append(video_id)
        self.save_playlists(data)

        if not playlist_id or not youtube_client:
            logger.info(f"Recorded video {video_id} in local playlist manifest for [{tier_key}].")
            return False

        try:
            body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
            youtube_client.playlistItems().insert(part="snippet", body=body).execute()
            logger.info(f"✅ Added video {video_id} to YouTube Playlist '{tier_key}' ({playlist_id})")
            return True
        except Exception as e:
            logger.warning(f"Could not insert video {video_id} into playlist {playlist_id} via API: {e}")
            return False
