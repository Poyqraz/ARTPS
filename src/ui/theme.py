"""
ARTPS — Arayüz tema ve bileşen yardımcıları.

Tüm görsel kimlik (CSS, hero bandı, telemetri, boş-durum) bu modülden gelir.
Davranış/algoritma kodu app.py içinde kalır; burada yalnızca sunum üretilir.
"""

import base64
from pathlib import Path

import streamlit as st

from .i18n import t

_ROOT = Path(__file__).resolve().parents[2]
_CSS_PATH = _ROOT / "assets" / "style.css"
_HERO_IMG_CANDIDATES = [
    _ROOT / "assets" / "img" / "hero_mars.jpg",
    _ROOT / "assets" / "img" / "hero_mars.png",
]


def _read_css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def _img_data_uri(path: Path) -> str | None:
    """Görseli base64 data-URI olarak döndürür (çevrimdışı/gömülü kullanım)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def inject_theme() -> None:
    """Global CSS'i tek noktadan enjekte eder. main() başında çağrılmalı."""
    css = _read_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _hero_background_css() -> str:
    """Hero arka plan görselini <style> bloğu olarak döndürür.

    Not: Streamlit HTML sanitizer'ı `url(data:...)` içeren inline `style`
    attribute'unu siler; bu yüzden değişken bir <style> kuralıyla verilir.
    """
    for cand in _HERO_IMG_CANDIDATES:
        uri = _img_data_uri(cand)
        if uri:
            return f"<style>.artps-hero{{--hero-img:url('{uri}');}}</style>"
    return ""


def render_hero(
    *,
    version: str = "v1.3.0",
    doi: str = "10.13140/RG.2.2.12215.18088",
    github_url: str = "https://github.com/Poyqraz/ARTPS",
    telemetry: list[dict] | None = None,
) -> None:
    """Referanstan daha zengin hero/landing bandını çizer.

    telemetry: [{"label": str, "value": str, "state": "ok"|"warn"|""}]
    """
    bg_css = _hero_background_css()
    chips = ""
    for item in telemetry or []:
        state = item.get("state", "")
        cls = f"artps-chip {state}".strip()
        chips += (
            f'<span class="{cls}">{item.get("label", "")} '
            f'<b>{item.get("value", "")}</b></span>'
        )

    html = f"""
    {bg_css}
    <div class="artps-hero">
      <div class="artps-nav">
        <div class="artps-brand">
          <div class="logo">A</div>
          <div>
            <div class="name">ARTPS</div>
            <div class="tag">{t("hero.brand_tag")}</div>
          </div>
        </div>
        <a class="artps-chip" href="{github_url}" target="_blank"
           style="text-decoration:none;">⧉ GitHub</a>
      </div>

      <span class="artps-badge">
        <span class="dot"></span>
        {t("hero.badge", version=version, doi=doi)}
      </span>

      <h1>{t("hero.title_html")}</h1>
      <div class="artps-tr">{t("hero.subtitle")}</div>

      <p class="artps-sub">{t("hero.body_html")}</p>

      <div class="artps-telemetry">{chips}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str) -> None:
    """Mars vurgulu bölüm başlığı."""
    st.markdown(f'<div class="artps-section">{title}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str, icon: str = "🛰️") -> None:
    """Model/görsel yokken gösterilecek zarif boş-durum kartı."""
    st.markdown(
        f"""
        <div class="artps-empty">
          <div style="font-size:2.4rem;margin-bottom:6px">{icon}</div>
          <h3>{title}</h3>
          <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
