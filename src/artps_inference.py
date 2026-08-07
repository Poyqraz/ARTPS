"""Streamlit-free frozen ARTPS inference for batch / independent_eval_v1."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
from PIL import Image

from src.artps_detection_core import (
    _compute_protrusion_map,
    _score_object_detections,
    compute_combined_anomaly_map,
    set_runtime_params,
)
from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
from src.models.depth_estimation import MiDaSDepthEstimator
from src.models.optimized_autoencoder import OptimizedAutoencoder
from src.utils.image_enhancement import enhance_image_auto

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_load_state_dict(
    model: torch.nn.Module,
    path: Path,
    *,
    state_dict_key: str = "model_state_dict",
    label: str,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or state_dict_key not in checkpoint:
        raise RuntimeError(f"{label}: expected top-level key {state_dict_key!r} in {path}")
    missing, unexpected = model.load_state_dict(checkpoint[state_dict_key], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} strict load failed: missing={missing!r} unexpected={unexpected!r}"
        )
    return {"path": str(path), "sha256": _sha256_file(path)}


@dataclass(frozen=True)
class FrozenARTPSConfig:
    """Pinned production defaults from root app.py sidebar."""

    config_id: str = "artps_full_frozen_v1"
    protocol_id: str = "independent_eval_v1"
    protocol_lock_path: str = "reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml"
    protocol_lock_sha256: str | None = None

    ae_path: str = "results/optimized_autoencoder_curiosity_extended.pth"
    classifier_path: str = "results/depth_enhanced_classifier.pth"
    enable_classifier: bool = True

    device: str = "auto"
    use_amp: bool = False
    ae_resize: int = 128
    depth_fusion_size: int = 768

    hyst_high: int = 96
    hyst_low: int = 90
    nms_iou: float = 0.25
    top_k: int = 25
    w_recon: float = 0.50
    w_depth: float = 0.30
    w_texture: float = 0.20
    w_lap: float = 0.08
    w_detail: float = 0.12
    edge_reinf: float = 0.40
    merge_iou: float = 0.15
    merge_tol: float = 0.5
    min_area_pct: float = 0.10

    alpha_shad: float = 0.65
    beta_shadow_obj: float = 0.5
    beta_illum: float = 0.25
    spec_gamma: float = 0.35
    spec_lowvar_gamma: float = 0.35
    spec_var_thresh: float = 0.005
    shadow_cut: float = 0.45
    img_edge_min: float = 0.10
    depth_edge_min: float = 0.08
    spec_cut: float = 0.50

    fp_suppression_enabled: bool = True
    size_distance_policy: bool = True
    detector_backend: str = "heuristic"
    policy_enable: bool = False
    recall_ablation: str = "slim"

    preprocessing_profile: str = "raw_rgb_v1"  # raw_rgb_v1 | mars_enhancement_v1

    checkpoint_sha256: Mapping[str, str] = field(default_factory=dict)

    def detection_params(self) -> dict[str, Any]:
        return {
            "hyst_high": self.hyst_high,
            "hyst_low": self.hyst_low,
            "nms_iou": self.nms_iou,
            "top_k": self.top_k,
            "w_recon": self.w_recon,
            "w_depth": self.w_depth,
            "w_texture": self.w_texture,
            "w_lap": self.w_lap,
            "w_detail": self.w_detail,
            "edge_reinf": self.edge_reinf,
            "merge_iou": self.merge_iou,
            "merge_tol": self.merge_tol,
            "min_area_pct": self.min_area_pct,
            "alpha_shad": self.alpha_shad,
            "beta_shadow_obj": self.beta_shadow_obj,
            "beta_illum": self.beta_illum,
            "spec_gamma": self.spec_gamma,
            "spec_lowvar_gamma": self.spec_lowvar_gamma,
            "spec_var_thresh": self.spec_var_thresh,
            "shadow_cut": self.shadow_cut,
            "img_edge_min": self.img_edge_min,
            "depth_edge_min": self.depth_edge_min,
            "spec_cut": self.spec_cut,
            "fp_suppression_enabled": self.fp_suppression_enabled,
            "size_distance_policy": self.size_distance_policy,
            "recall_ablation": self.recall_ablation,
            "policy_crop_margin": 0.10,
        }

    def verify_checkpoint_hash(self, key: str, path: Path) -> None:
        expected = (self.checkpoint_sha256 or {}).get(key)
        if not expected:
            return
        actual = _sha256_file(path).lower()
        if actual != str(expected).strip().lower():
            raise RuntimeError(
                f"checkpoint sha256 mismatch for {key}: expected={expected} actual={actual}"
            )


@dataclass
class FrozenARTPS:
    config: FrozenARTPSConfig
    device: torch.device
    autoencoder: OptimizedAutoencoder
    depth_estimator: MiDaSDepthEstimator
    classifier: DepthEnhancedClassifier | None
    checkpoint_hashes: dict[str, str]
    model_name: str = "ARTPS_frozen"
    model_version: str = "iac2026_v1"


PredictionRecord = dict[str, Any]


def load_frozen_artps_profile(config: FrozenARTPSConfig | None = None) -> FrozenARTPS:
    cfg = config or FrozenARTPSConfig()
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU available")
        device = torch.device("cuda")
    elif cfg.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_path = _REPO_ROOT / cfg.ae_path
    cfg.verify_checkpoint_hash("ae", ae_path)
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    hashes: dict[str, str] = {}
    hashes["ae"] = _strict_load_state_dict(autoencoder, ae_path, label="AE")["sha256"]
    autoencoder.to(device)
    autoencoder.eval()

    depth_estimator = MiDaSDepthEstimator(
        model_type="DPT_Large",
        device=str(device),
        strict_local_only=True,
    )
    if not depth_estimator.is_real_dpt or depth_estimator.load_source != "local_state_dict":
        raise RuntimeError(
            "Primary eval requires DPT_Large local_state_dict; "
            f"got load_source={depth_estimator.load_source!r} is_real_dpt={depth_estimator.is_real_dpt}"
        )
    dpt_path = _REPO_ROOT / "raw_models" / "dpt_large_384.pt"
    cfg.verify_checkpoint_hash("dpt", dpt_path)
    if dpt_path.is_file():
        hashes["dpt"] = _sha256_file(dpt_path)

    classifier: DepthEnhancedClassifier | None = None
    if cfg.enable_classifier:
        clf_path = _REPO_ROOT / cfg.classifier_path
        cfg.verify_checkpoint_hash("classifier", clf_path)
        classifier = DepthEnhancedClassifier(num_classes=5, rgb_features=1024, depth_features=14)
        hashes["classifier"] = _strict_load_state_dict(classifier, clf_path, label="classifier")["sha256"]
        classifier.to(device)
        classifier.eval()

    return FrozenARTPS(
        config=cfg,
        device=device,
        autoencoder=autoencoder,
        depth_estimator=depth_estimator,
        classifier=classifier,
        checkpoint_hashes=hashes,
    )


def _preprocess_image(image_path: Path, profile: str) -> Image.Image:
    pil = Image.open(image_path).convert("RGB")
    if profile == "raw_rgb_v1":
        return pil
    if profile == "mars_enhancement_v1":
        result = enhance_image_auto(pil, profile="mars", config={"enable_realesrgan": False})
        return result.image
    raise ValueError(f"unknown preprocessing_profile: {profile!r}")


def _ae_forward(
    autoencoder: OptimizedAutoencoder,
    pil_image: Image.Image,
    device: torch.device,
    *,
    ae_resize: int,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    image = pil_image.resize((ae_resize, ae_resize), Image.LANCZOS)
    image_array = np.array(image, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                reconstructed, latent = autoencoder(input_tensor)
        else:
            reconstructed, latent = autoencoder(input_tensor)
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    latent_np = latent.squeeze().cpu().numpy()
    mse = float(np.mean((image_array - reconstructed_np) ** 2))
    return mse, image_array, reconstructed_np, latent_np


def _known_value_score(
    bundle: FrozenARTPS,
    image_array: np.ndarray,
    latent: np.ndarray,
    *,
    diagnostics_out: MutableMapping[str, Any] | None = None,
) -> float:
    if bundle.classifier is None:
        raise RuntimeError("classifier required for known_value_score but not loaded")
    depth_map, _ = bundle.depth_estimator.estimate_depth(image_array)
    depth_features = bundle.depth_estimator.extract_depth_features(depth_map)
    depth_vec = MiDaSDepthEstimator.vectorize_depth_features(depth_features)
    depth_t = torch.tensor(depth_vec, dtype=torch.float32).unsqueeze(0).to(bundle.device)
    rgb_t = torch.tensor(latent, dtype=torch.float32).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        if bundle.config.use_amp and bundle.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                predictions = bundle.classifier(rgb_t, depth_t)
        else:
            predictions = bundle.classifier(rgb_t, depth_t)
        predicted_class = int(torch.argmax(predictions, dim=1).item())
        if diagnostics_out is not None:
            probs = torch.softmax(predictions.float(), dim=1).squeeze(0).detach().cpu().tolist()
            diagnostics_out["classifier_argmax"] = predicted_class
            diagnostics_out["classifier_logits_or_probabilities"] = "|".join(
                f"{float(p):.6f}" for p in probs
            )
    return float(predicted_class / 4.0)


def _depth_for_fusion(bundle: FrozenARTPS, pil_image: Image.Image) -> np.ndarray:
    size = int(bundle.config.depth_fusion_size)
    image_for_depth = np.array(pil_image.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    depth_map, _ = bundle.depth_estimator.estimate_depth(
        image_for_depth,
        apply_enhancement=True,
        high_detail=True,
        tta_flips=True,
        use_fgs=True,
        use_wmf=True,
    )
    if depth_map is None:
        raise RuntimeError("DPT depth estimation returned None (primary eval refuses fallback)")
    return depth_map


def predict_image(
    image_path: str | Path,
    model_bundle: FrozenARTPS,
    config: FrozenARTPSConfig | None = None,
    *,
    sample_id: str | None = None,
    split: str | None = None,
    diagnostics_out: MutableMapping[str, Any] | None = None,
) -> PredictionRecord:
    cfg = config or model_bundle.config
    path = Path(image_path)
    warnings: list[str] = []
    status = "ok"
    cand_diags: list[dict[str, Any]] = []

    try:
        pil = _preprocess_image(path, cfg.preprocessing_profile)
        set_runtime_params(cfg.detection_params())

        mse, original, reconstructed, latent = _ae_forward(
            model_bundle.autoencoder,
            pil,
            model_bundle.device,
            ae_resize=cfg.ae_resize,
            use_amp=cfg.use_amp,
        )
        clf_diag: dict[str, Any] = {}
        if cfg.enable_classifier and model_bundle.classifier is not None:
            known_value = _known_value_score(
                model_bundle, original, latent, diagnostics_out=clf_diag if diagnostics_out is not None else None
            )
        else:
            known_value = 0.5
            warnings.append("classifier_disabled_known_value_fallback_0.5")
            if diagnostics_out is not None:
                clf_diag["classifier_argmax"] = ""
                clf_diag["classifier_logits_or_probabilities"] = ""
        depth_map = _depth_for_fusion(model_bundle, pil)
        protrusion_map = _compute_protrusion_map(depth_map)

        with torch.inference_mode():
            combined_map, detections, _diag = compute_combined_anomaly_map(
                original,
                reconstructed,
                depth_map,
                hyst_high_pct=cfg.hyst_high,
                hyst_low_pct=cfg.hyst_low,
                nms_iou=cfg.nms_iou,
                top_k=cfg.top_k,
                w_recon=cfg.w_recon,
                w_depth=cfg.w_depth,
                w_texture=cfg.w_texture,
                edge_reinforce=cfg.edge_reinf,
            )

        scored = _score_object_detections(
            detections,
            original_rgb_float=original,
            autoencoder=model_bundle.autoencoder,
            device=model_bundle.device,
            combined_map=combined_map,
            depth_map=depth_map,
            protrusion_map=protrusion_map,
            padim_map=None,
            patchcore_map=None,
            global_known_value=known_value,
            diagnostics_candidates=cand_diags if diagnostics_out is not None else None,
        )

        if scored:
            image_score = float(max(float(d.get("score", 0.0)) for d in scored))
            top_candidate_score = image_score
        else:
            image_score = 0.0
            top_candidate_score = 0.0

        candidates = [
            {
                "x": d["x"],
                "y": d["y"],
                "w": d["w"],
                "h": d["h"],
                "score": float(d.get("score", 0.0)),
            }
            for d in scored
        ]

        if diagnostics_out is not None:
            kept_n = len(scored)
            raw_n = len(detections)
            suppressed_n = max(0, len(cand_diags) - kept_n) if cand_diags else max(0, raw_n - kept_n)
            top = scored[0] if scored else (cand_diags[0] if cand_diags else None)
            drop_reasons = [c["drop_reason"] for c in cand_diags if c.get("drop_reason")]
            if status == "ok" and kept_n == 0:
                if raw_n == 0:
                    no_cand = "no_raw_proposal"
                elif drop_reasons:
                    no_cand = Counter(drop_reasons).most_common(1)[0][0]
                else:
                    no_cand = "no_valid_candidate"
            else:
                no_cand = ""
            diagnostics_out.clear()
            diagnostics_out.update(
                {
                    "raw_proposal_count": raw_n,
                    "scored_candidate_count": len(cand_diags) if cand_diags else raw_n,
                    "kept_candidate_count": kept_n,
                    "suppressed_candidate_count": suppressed_n,
                    "top_candidate_box": (
                        f"{top.get('x')},{top.get('y')},{top.get('w')},{top.get('h')}"
                        if top
                        else ""
                    ),
                    "combined_pool": float(top["combined_pool"]) if top and "combined_pool" in top else "",
                    "depth_pool": float(top["depth_pool"]) if top and "depth_pool" in top else "",
                    "detector_confidence": (
                        float(top.get("detector_confidence", top.get("detector_conf", "")))
                        if top
                        else ""
                    ),
                    "classifier_argmax": clf_diag.get("classifier_argmax", ""),
                    "classifier_logits_or_probabilities": clf_diag.get(
                        "classifier_logits_or_probabilities", ""
                    ),
                    "classifier_known_value": known_value,
                    "padim_pool": float(top["padim_pool"]) if top and "padim_pool" in top else 0.0,
                    "patchcore_pool": (
                        float(top["patchcore_pool"]) if top and "patchcore_pool" in top else 0.0
                    ),
                    "local_value": (
                        float(top.get("local_value", top.get("object_value_score", "")))
                        if top
                        else ""
                    ),
                    "anomaly_score_before_gate": (
                        float(
                            top.get(
                                "anomaly_score_before_gate",
                                top.get("object_anomaly_score", ""),
                            )
                        )
                        if top
                        else ""
                    ),
                    "final_candidate_score": top_candidate_score,
                    "keep_or_drop": "keep" if kept_n else "drop",
                    "drop_reason": no_cand if kept_n == 0 else "",
                    "mask_reason": no_cand if kept_n == 0 else "",
                    "no_valid_candidate_reason": no_cand,
                    "execution_path": "instrumented_validation_rerun",
                    "warning_flags": "|".join(warnings),
                    "candidates_detail": cand_diags,
                }
            )

    except Exception as exc:
        status = "error"
        warnings.append(str(exc))
        image_score = 0.0
        top_candidate_score = 0.0
        scored = []
        candidates = []
        mse = None
        if diagnostics_out is not None:
            diagnostics_out.clear()
            diagnostics_out.update(
                {
                    "raw_proposal_count": 0,
                    "scored_candidate_count": 0,
                    "kept_candidate_count": 0,
                    "suppressed_candidate_count": 0,
                    "top_candidate_box": "",
                    "combined_pool": "",
                    "depth_pool": "",
                    "detector_confidence": "",
                    "classifier_argmax": "",
                    "classifier_logits_or_probabilities": "",
                    "classifier_known_value": "",
                    "padim_pool": "",
                    "patchcore_pool": "",
                    "local_value": "",
                    "anomaly_score_before_gate": "",
                    "final_candidate_score": 0.0,
                    "keep_or_drop": "drop",
                    "drop_reason": "processing_error",
                    "mask_reason": "processing_error",
                    "no_valid_candidate_reason": "processing_error",
                    "execution_path": "instrumented_validation_rerun",
                    "warning_flags": "|".join(warnings),
                    "candidates_detail": [],
                }
            )

    record: MutableMapping[str, Any] = {
        "sample_id": sample_id or path.stem,
        "split": split,
        "image_score": image_score,
        "candidate_count": len(scored) if status == "ok" else 0,
        "valid_candidate_count": len(scored) if status == "ok" else 0,
        "top_candidate_score": top_candidate_score,
        "anomaly_mse": mse,
        "model_name": model_bundle.model_name,
        "model_version": model_bundle.model_version,
        "config_id": cfg.config_id,
        "checkpoint_hashes": dict(model_bundle.checkpoint_hashes),
        "protocol_id": cfg.protocol_id,
        "protocol_lock_sha256": cfg.protocol_lock_sha256,
        "processing_status": status,
        "warning_flags": warnings,
        "preprocessing_profile": cfg.preprocessing_profile,
        "detector_backend": cfg.detector_backend,
        "policy_enable": cfg.policy_enable,
    }
    if status == "ok" and candidates:
        record["candidates"] = candidates
    return dict(record)
