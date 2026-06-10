from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import logging
import requests
from pathlib import Path


_LOGGER = logging.getLogger(__name__)


@dataclass
class PDSAtlasQuery:
    instrument: Optional[str] = None
    instrument_host: Optional[str] = None  # rover/uzay aracı
    investigation: Optional[str] = None   # mission
    target: Optional[str] = None
    product_type: Optional[str] = None
    planet_day_range: Optional[Tuple[int, int]] = None  # (sol_start, sol_end)
    start_time: Optional[str] = None  # ISO 8601
    stop_time: Optional[str] = None   # ISO 8601
    title: Optional[str] = None
    image_content: Optional[str] = None
    rows: int = 50
    start: int = 0


class PDSAtlasClient:
    """PDS Imaging Atlas (Solr) istemcisi.

    Referans: https://pds-imaging.jpl.nasa.gov/beta/atlas/documentation/docs/category/api
    Tipik uç nokta: https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search
    """

    def __init__(self, base_url: str = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/select", timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = float(timeout)
        # Alternatif Solr uç noktaları (bazı ortamlarda /search yerine /select çalışır)
        self._alt_endpoints = [
            self.base_url,
            "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search",
            "https://pds-imaging.jpl.nasa.gov/solr/pds/select",
            "https://pds-imaging.jpl.nasa.gov/solr/pds/search",
            "https://pds-imaging.jpl.nasa.gov/solr/atlas/select",
            "https://pds-imaging.jpl.nasa.gov/solr/atlas/search",
        ]

    def _build_params(self, q: PDSAtlasQuery) -> List[Tuple[str, str]]:
        # Parametreleri list-of-tuples olarak döndürerek birden fazla 'fq' anahtarı gönderebiliriz
        params: List[Tuple[str, str]] = [
            ('q', '*:*'),
            ('wt', 'json'),
            ('rows', str(max(1, int(q.rows)))),
            ('start', str(max(0, int(q.start)))),
        ]
        # Solr filter queries
        fqs: List[str] = []
        if q.instrument:
            fqs.append(f'instrument:"{q.instrument}"')
        if q.instrument_host:
            fqs.append(f'instrument_host:"{q.instrument_host}"')
        if q.investigation:
            fqs.append(f'investigation:"{q.investigation}"')
        if q.target:
            fqs.append(f'target:"{q.target}"')
        if q.product_type:
            fqs.append(f'product_type:"{q.product_type}"')
        if q.title:
            fqs.append(f'title:"{q.title}"')
        if q.image_content:
            fqs.append(f'image_content:"{q.image_content}"')
        if q.planet_day_range:
            a, b = q.planet_day_range
            fqs.append(f'planet_day_number:[{int(a)} TO {int(b)}]')
        if q.start_time or q.stop_time:
            lo = q.start_time or '*'
            hi = q.stop_time or '*'
            fqs.append(f'start_time:[{lo} TO {hi}]')

        for cond in fqs:
            params.append(('fq', cond))
        return params

    def search(self, query: PDSAtlasQuery, *, max_retries: int = 2, backoff: float = 1.5) -> Dict:
        params = self._build_params(query)
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(max_retries)) + 1):
            for endpoint in self._alt_endpoints:
                try:
                    resp = requests.get(endpoint, params=params, timeout=self.timeout, headers={"Accept": "application/json"})
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    last_err = e
                    _LOGGER.warning("PDSAtlas search hata (deneme %d, %s): %s", attempt + 1, endpoint, e)
                    continue
            time.sleep((backoff ** attempt))
        raise RuntimeError(f"PDSAtlas search başarısız: {last_err}")

    @staticmethod
    def extract_docs(payload: Dict) -> List[Dict]:
        try:
            return list(payload.get('response', {}).get('docs', []))
        except Exception:
            return []

    @staticmethod
    def candidate_urls(doc: Dict) -> List[str]:
        keys = [
            'download_url', 'product_url', 'product_ref', 'file_ref', 'file_url', 'url', 'href'
        ]
        urls: List[str] = []
        for k in keys:
            v = doc.get(k)
            if isinstance(v, str):
                urls.append(v)
            elif isinstance(v, list):
                urls.extend([str(x) for x in v if isinstance(x, (str, bytes))])
        return urls

    def try_download(self, doc: Dict, dest_dir: Path, *, session: Optional[requests.Session] = None, timeout: Optional[float] = None) -> Optional[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        urls = self.candidate_urls(doc)
        if not urls:
            _LOGGER.info("İndirilebilir URL alanı bulunamadı (doc id: %s)", str(doc.get('identifier', 'unknown')))
            return None
        sess = session or requests.Session()
        t = timeout or self.timeout
        for u in urls:
            try:
                name = u.split('/')[-1] or f"pds_file_{int(time.time())}"
                local = dest_dir / name
                if local.exists():
                    return local
                with sess.get(u, stream=True, timeout=t) as r:
                    r.raise_for_status()
                    with open(local, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return local
            except Exception:
                continue
        return None


