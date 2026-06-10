import argparse
import json
import numpy as np
from pathlib import Path


def robust_norm(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [2, 98])
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description="Curiosity proxy label üretici")
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--out", dest="out", type=str, required=True)
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    rows = []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    # Alanları topla
    def col(name):
        vals = []
        for r in rows:
            v = r.get(name, None)
            if v is None:
                continue
            vals.append(float(v))
        return vals

    anom = col("anomaly_mse")
    rough = col("roughness")
    dvar = col("depth_variance")

    # Normalize
    anom_n = robust_norm(anom)
    rough_n = robust_norm(rough)
    dvar_n = robust_norm(dvar)

    # Sözlük: orijinale map
    def assign_norm(name, norm_vals, original_vals):
        it = iter(norm_vals)
        for r in rows:
            v = r.get(original_vals, None)
            if v is None:
                r[name] = 0.0
            else:
                r[name] = float(next(it))

    assign_norm("anomaly_norm", anom_n, "anomaly_mse")
    assign_norm("roughness_norm", rough_n, "roughness")
    assign_norm("depth_variance_norm", dvar_n, "depth_variance")

    # Proxy label: keşif odaklı (anomali + pürüzlülük + derinlik çeşitliliği)
    for r in rows:
        a = float(r.get("anomaly_norm", 0.0))
        t = float(r.get("roughness_norm", 0.0))
        dv = float(r.get("depth_variance_norm", 0.0))
        kv = float(r.get("known_value_score", 0.5) or 0.5)
        # keşif ağırlıklı: 0.5 anomali + 0.3 rough + 0.2 dvar; bilinen değeri hafifçe azalt
        label = 0.5 * a + 0.3 * t + 0.2 * dv
        label = float(np.clip(label, 0.0, 1.0))
        r["label"] = label

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Yazıldı: {out} ({len(rows)} satır)")


if __name__ == "__main__":
    main()


