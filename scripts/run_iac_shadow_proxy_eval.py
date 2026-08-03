"""Run IAC shadow FP proxy ablation (fp_mode off vs on) on a real image set."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_path(rows: list[dict]) -> dict[str, dict]:
    return {str(r["image_path"]): r for r in rows}


def _prepare_images(root: Path, images_dir: Path, rover_extra: int) -> Path:
    """Copy round set + up to N rover images into a staging folder."""
    staging = root / "results" / "iac_shadow_proxy" / "images_stage"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    src = root / images_dir
    for p in src.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            shutil.copy2(p, staging / p.name)
    rover_dirs = [
        root / "mars_images" / "valid" / "rover",
        root / "mars_images" / "train" / "rover",
    ]
    n = 0
    for rd in rover_dirs:
        if not rd.is_dir() or n >= rover_extra:
            continue
        for p in sorted(rd.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(p, staging / f"rover__{p.name}")
            n += 1
            if n >= rover_extra:
                break
    return staging


def _export(root: Path, images: Path, jsonl: Path, overlays: Path, fp_mode: str, device: str) -> None:
    cmd = [
        sys.executable,
        str(root / "scripts" / "export_app_detections.py"),
        "--images_dir",
        str(images),
        "--out_dir",
        str(overlays),
        "--jsonl",
        str(jsonl),
        "--fp_mode",
        fp_mode,
        "--size_distance_policy",
        "on",
        "--device",
        device,
        "--backend",
        "heuristic",
    ]
    subprocess.run(cmd, check=True, cwd=root)


def _write_table(summary: dict, path: Path) -> None:
    lines = [
        "# IAC shadow / FP proxy ablation",
        "",
        summary.get("disclaimer", ""),
        "",
        "| Metric | OFF | ON | Delta |",
        "|--------|-----|----|-------|",
        f"| shadow-dense FPR | {summary.get('shadow_dense_fpr_off')} | {summary.get('shadow_dense_fpr_on')} | {summary.get('shadow_dense_fpr_delta')} |",
        f"| rover-body FP count | {summary.get('rover_fp_count_off')} | {summary.get('rover_fp_count_on')} | — |",
        f"| target recall proxy | — | {summary.get('target_recall_proxy_mean')} | (ON vs OFF pseudo-GT) |",
        f"| shadow-rock loss | — | {summary.get('shadow_rock_loss_mean')} | (lost OFF shadow-rocks) |",
        f"| avg detections | {summary.get('avg_detections_off')} | {summary.get('avg_detections_on')} | — |",
        "",
        f"n_images={summary.get('n_images')}, n_shadow_dense={summary.get('n_shadow_dense')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.eval.iac_shadow_proxy import aggregate_shadow_summaries, summarize_image_pair

    p = argparse.ArgumentParser(description="IAC shadow proxy eval")
    p.add_argument("--images_dir", type=str, default="results/benchmark_round3_set")
    p.add_argument("--rover_extra", type=int, default=10)
    p.add_argument("--tag", type=str, default="iac_v1")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--skip_export", action="store_true")
    args = p.parse_args()

    out = root / "results" / "iac_shadow_proxy"
    out.mkdir(parents=True, exist_ok=True)
    figs = root / "results" / "paper_figs"
    figs.mkdir(parents=True, exist_ok=True)

    staging = _prepare_images(root, Path(args.images_dir), int(args.rover_extra))
    off_jsonl = out / f"{args.tag}_fp_off.jsonl"
    on_jsonl = out / f"{args.tag}_fp_on.jsonl"

    if not args.skip_export:
        _export(root, staging, off_jsonl, out / f"overlays_{args.tag}_off", "off", args.device)
        _export(root, staging, on_jsonl, out / f"overlays_{args.tag}_on", "on", args.device)

    off_map = _index_by_path(_load_jsonl(off_jsonl))
    on_map = _index_by_path(_load_jsonl(on_jsonl))
    rows = []
    for path, off_row in off_map.items():
        on_row = on_map.get(path)
        if on_row is None:
            continue
        rows.append(
            summarize_image_pair(
                class_label=str(off_row.get("class_label", "unknown")),
                dets_off=list(off_row.get("detections") or []),
                dets_on=list(on_row.get("detections") or []),
                scene_shadow_density_value=float(off_row.get("scene_shadow_density", 0.0)),
                shadow_dense_flag=bool(off_row.get("shadow_dense", False)),
            )
        )
    summary = aggregate_shadow_summaries(rows)
    summary_path = figs / "iac_shadow_proxy_summary.json"
    table_path = figs / "iac_shadow_proxy_table.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_table(summary, table_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
