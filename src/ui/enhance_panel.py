"""Mars-safe görüntü iyileştirme paneli (Streamlit)."""
from __future__ import annotations

import streamlit as st
from PIL import Image

from src.ui.i18n import t
from src.utils.image_enhancement import enhance_image_auto


def render_enhance_panel(image: Image.Image) -> Image.Image:
    """Kontrolleri çizer; enhance tıklanınca session + dönen görüntüyü günceller."""
    st.subheader(t("analysis.enhance_header"))
    st.caption(t("analysis.enhance_mars_note"))
    cols = st.columns(6)
    with cols[0]:
        opt_upscale = st.checkbox(t("analysis.opt_upscale"), value=True, help=t("analysis.opt_upscale_help"))
    with cols[1]:
        opt_denoise = st.checkbox(t("analysis.opt_denoise"), value=True, help=t("analysis.opt_denoise_help"))
    with cols[2]:
        opt_clahe = st.checkbox(t("analysis.opt_clahe"), value=True, help=t("analysis.opt_clahe_help"))
    with cols[3]:
        opt_gamma = st.checkbox(t("analysis.opt_gamma"), value=True, help=t("analysis.opt_gamma_help"))
    with cols[4]:
        opt_sharp = st.checkbox(t("analysis.opt_sharp"), value=True, help=t("analysis.opt_sharp_help"))
    with cols[5]:
        opt_realesrgan = st.checkbox(
            t("analysis.opt_realesrgan"),
            value=False,
            help=t("analysis.opt_realesrgan_help"),
        )
    if opt_realesrgan:
        st.caption(t("analysis.opt_realesrgan_risk"))

    if not st.button(t("analysis.enhance_btn")):
        return image

    # Yalnızca UI override'ları; Mars sabitleri ENHANCE_PROFILES["mars"]
    result = enhance_image_auto(
        image,
        {
            "enable_upscale": opt_upscale,
            "enable_denoise": opt_denoise,
            "enable_clahe": opt_clahe,
            "enable_gamma": opt_gamma,
            "enable_sharpen": opt_sharp,
            "enable_realesrgan": opt_realesrgan,
        },
        profile="mars",
    )
    st.success(t("analysis.steps_applied", steps=", ".join(result.steps)))
    if result.realesrgan_fallback:
        st.caption(t("analysis.realesrgan_fallback"))
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption=t("analysis.before"), use_container_width=True)
        st.json({t("analysis.before"): result.metrics_before})
    with c2:
        st.image(result.image, caption=t("analysis.after"), use_container_width=True)
        st.json({t("analysis.after"): result.metrics_after})
    st.session_state["enhanced_image_for_analysis"] = result.image
    return result.image
