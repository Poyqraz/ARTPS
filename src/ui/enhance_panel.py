"""Mars-safe görüntü iyileştirme paneli (Streamlit)."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.ui.i18n import t
from src.utils.image_enhancement import enhance_image_auto
from src.utils.shadow_visibility import lift_shadow_visibility


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

    opt_shadow_preview = st.checkbox(
        t("analysis.opt_shadow_preview"),
        value=True,
        help=t("analysis.opt_shadow_preview_help"),
    )
    opt_depth_gate = st.checkbox(
        t("analysis.opt_shadow_depth_gate"),
        value=False,
        help=t("analysis.opt_shadow_depth_gate_help"),
    )

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

    # Track A: preview only — never overwrite analysis RGB
    if opt_shadow_preview:
        shared_u8 = np.asarray(result.image.convert("RGB"), dtype=np.uint8)
        depth = st.session_state.get("last_depth_map")
        if opt_depth_gate and depth is None:
            st.caption(t("analysis.shadow_depth_gate_skip"))
        depth_arr = None
        if opt_depth_gate and depth is not None:
            depth_arr = np.asarray(depth, dtype=np.float32)
            if depth_arr.shape[:2] != shared_u8.shape[:2]:
                depth_arr = cv2.resize(
                    depth_arr,
                    (shared_u8.shape[1], shared_u8.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        lifted, mask = lift_shadow_visibility(
            shared_u8,
            depth_arr,
            use_depth_gate=bool(opt_depth_gate and depth_arr is not None),
        )
        mask_vis = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.image(result.image, caption=t("analysis.shadow_shared"), use_container_width=True)
        with p2:
            st.image(lifted, caption=t("analysis.shadow_lifted"), use_container_width=True)
        with p3:
            st.image(mask_vis, caption=t("analysis.shadow_mask"), use_container_width=True)

    st.session_state["enhanced_image_for_analysis"] = result.image
    return result.image
