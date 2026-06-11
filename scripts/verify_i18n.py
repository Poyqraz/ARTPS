#!/usr/bin/env python3
"""TR/EN i18n sozluk butunlugu dogrulamasi."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.i18n import placeholder_names  # noqa: E402
from src.ui.i18n.en import MESSAGES as EN  # noqa: E402
from src.ui.i18n.tr import MESSAGES as TR  # noqa: E402


def main() -> int:
    errors: list[str] = []
    tr_keys = set(TR.keys())
    en_keys = set(EN.keys())

    missing_en = sorted(tr_keys - en_keys)
    missing_tr = sorted(en_keys - tr_keys)
    if missing_en:
        errors.append(f"EN eksik anahtarlar ({len(missing_en)}): {missing_en[:5]}...")
    if missing_tr:
        errors.append(f"TR eksik anahtarlar ({len(missing_tr)}): {missing_tr[:5]}...")

    common = tr_keys & en_keys
    for key in sorted(common):
        if not TR[key].strip():
            errors.append(f"TR bos deger: {key}")
        if not EN[key].strip():
            errors.append(f"EN bos deger: {key}")
        tr_ph = placeholder_names(TR[key])
        en_ph = placeholder_names(EN[key])
        if tr_ph != en_ph:
            errors.append(
                f"Placeholder uyumsuz '{key}': TR={sorted(tr_ph)} EN={sorted(en_ph)}"
            )

    print(f"TR keys: {len(tr_keys)}")
    print(f"EN keys: {len(en_keys)}")
    print(f"Common:  {len(common)}")

    if errors:
        print("\nHATALAR:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK: Tum anahtarlar eslesiyor, bos deger yok, placeholder'lar uyumlu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
