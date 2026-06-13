"""ARTPS TR/EN arayuz cevirileri."""

from __future__ import annotations

import re
from typing import Any, Literal

import streamlit as st

from .en import MESSAGES as EN_MESSAGES
from .tr import MESSAGES as TR_MESSAGES

Locale = Literal["tr", "en"]

_LOCALES: dict[Locale, dict[str, str]] = {
    "tr": TR_MESSAGES,
    "en": EN_MESSAGES,
}

_CATEGORY_KEYS = {
    "rocky": "category.rocky",
    "boulder": "category.boulder",
    "flat_terrain": "category.flat_terrain",
    "hills_or_ridge": "category.hills_or_ridge",
    "dusty": "category.dusty",
    "rover": "category.rover",
}

_CLASS_KEYS = {
    0: "analysis.class.negligible",
    1: "analysis.class.low",
    2: "analysis.class.medium",
    3: "analysis.class.medium_high",
    4: "analysis.class.high",
}


def get_locale() -> Locale:
    if "locale" not in st.session_state:
        st.session_state.locale = "tr"
    loc = st.session_state.locale
    return loc if loc in _LOCALES else "tr"


def set_locale(locale: Locale) -> None:
    st.session_state.locale = locale


def t(key: str, /, **kwargs: Any) -> str:
    loc = get_locale()
    pack = _LOCALES[loc]
    if key not in pack:
        raise KeyError(f"Missing i18n key '{key}' for locale '{loc}'")
    text = pack[key]
    if not text:
        raise KeyError(f"Empty i18n value for key '{key}' ({loc})")
    return text.format(**kwargs) if kwargs else text


def category_label(name: str) -> str:
    key = _CATEGORY_KEYS.get(name)
    return t(key) if key else name.replace("_", " ").title()


def class_label(class_id: int) -> str:
    key = _CLASS_KEYS.get(class_id)
    return t(key) if key else t("analysis.class.unknown")


def lang_selector() -> None:
    labels = {"tr": "Türkçe", "en": "English"}
    current = get_locale()
    choice = st.sidebar.radio(
        t("sidebar.language"),
        options=["tr", "en"],
        index=0 if current == "tr" else 1,
        format_func=lambda code: labels[code],
        horizontal=True,
        key="locale_radio",
    )
    if choice != current:
        set_locale(choice)  # type: ignore[arg-type]
        st.rerun()


def placeholder_names(text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", text))


def all_keys() -> set[str]:
    return set(TR_MESSAGES.keys()) | set(EN_MESSAGES.keys())
