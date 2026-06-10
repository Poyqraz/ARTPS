from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import time
import logging
import requests
from pathlib import Path


_LOGGER = logging.getLogger(__name__)


@dataclass
class NASAAPIConfig:
    api_key: str = os.environ.get("NASA_API_KEY", "DEMO_KEY")
    base_url: str = "https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
    timeout: float = 30.0
    user_agent: str = "ARTPS/1.0 (Mars Rover Photos)"


class NASAPhotosClient:
    def __init__(self, config: Optional[NASAAPIConfig] = None) -> None:
        self.cfg = config or NASAAPIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.cfg.user_agent,
            "Accept": "application/json",
        })

    def list_photos(self, rover: str, sol: int, camera: Optional[str] = None) -> List[Dict]:
        url = self.cfg.base_url.format(rover=rover.lower())
        params = {"sol": int(sol), "api_key": self.cfg.api_key}
        if camera:
            params["camera"] = camera
        r = self.session.get(url, params=params, timeout=self.cfg.timeout)
        r.raise_for_status()
        data = r.json()
        return list(data.get("photos", []))

    def download_photo(self, img_url: str, dest_dir: Path) -> Optional[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = img_url.split("/")[-1]
        out = dest_dir / name
        if out.exists():
            return out
        with self.session.get(img_url, stream=True, timeout=self.cfg.timeout) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return out

    def fetch_range(self, rover: str, sol_start: int, sol_end: int, out_dir: Path, camera: Optional[str] = None,
                    max_per_sol: int = 10, backoff: float = 1.2) -> int:
        total = 0
        for sol in range(int(sol_start), int(sol_end) + 1):
            tries = 0
            while tries < 2:
                tries += 1
                try:
                    photos = self.list_photos(rover, sol, camera)
                    if not photos:
                        break
                    count = 0
                    for ph in photos:
                        img = ph.get("img_src") or ph.get("img_url")
                        if not img:
                            continue
                        if self.download_photo(img, Path(out_dir) / f"sol_{sol:05d}"):
                            total += 1
                            count += 1
                            if count >= max_per_sol:
                                break
                    break
                except Exception as e:
                    _LOGGER.warning("NASAPhotos fetch hata (sol=%d, deneme=%d): %s", sol, tries, e)
                    time.sleep(backoff ** tries)
        return total


