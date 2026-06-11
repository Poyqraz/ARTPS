"""English UI strings."""

MESSAGES: dict[str, str] = {
    # Core
    "sidebar.language": "Language",
    "sidebar.credits": "🛰️ Built by [Poyraz BAYDEMİR](https://github.com/Poyqraz) · [ResearchGate DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)",
    "sidebar.license": "📄 License: [MIT License](https://github.com/Poyqraz/ARTPS/blob/main/LICENSE)",
    "sidebar.control_panel": "🎛️ Control Panel",
    "sidebar.models_loading": "🤖 Loading hybrid models...",
    "sidebar.models_loaded_prefix": "Models loaded:\n",
    "sidebar.model.autoencoder": "✅ Autoencoder",
    "sidebar.model.classifier": "✅ Hybrid Classifier",
    "sidebar.model.depth": "✅ Depth Estimation ({model_type}) - {quality}",
    "sidebar.model.padim": "✅ PaDiM (Anomaly Fusion)",
    "sidebar.model.patchcore": "✅ PatchCore (Anomaly Fusion)",
    "sidebar.depth_active": "Active Depth Model: {model_type} — Parameters: {param_count:,} — {quality}",
    "sidebar.params_settings": "📊 Parameter Settings",

    # load_models messages
    "models.device": "🖥️ Device in use: {device}",
    "models.autoencoder_missing": "❌ Autoencoder model not found: {path}",
    "models.classifier_missing": "⚠️ Classifier model not found; only anomaly detection will be used",
    "models.padim_stats_missing": "⚠️ PaDiM statistics not found: results/padim_stats.pth. Only AE-based anomaly will be used",
    "models.padim_load_failed": "⚠️ PaDiM failed to load: {error}",
    "models.patchcore_missing": "ℹ️ PatchCore memory bank not found (generate with tools/prepare_patchcore_bank.py)",
    "models.patchcore_load_failed": "⚠️ PatchCore failed to load: {error}",
    "models.dpt_success": "✅ DPT_Large model loaded successfully (high accuracy) - {params:,} parameters",
    "models.dpt_fallback": "⚠️ DPT_Large model unavailable; using simple model - {params:,} parameters",
    "models.dpt_hub_fallback": "ℹ️ Fallback model active due to PyTorch Hub connection issue",
    "models.depth_load_failed": "❌ Depth estimation module failed to load: {error}",
    "models.curiosity_loaded": "🧭 Curiosity weights auto-loaded (results/curiosity_weights.json)",
    "models.curiosity_load_failed": "Curiosity weights failed to load: {error}",
    "models.quality.high": "High Accuracy",
    "models.quality.simple": "Simple Model",

    # Hero
    "hero.brand_tag": "Target Prioritization",
    "hero.badge": "ARTPS {version} · PUBLISHED · DOI {doi}",
    "hero.title_html": "Autonomous target prioritization for <span class=\"accent\">planetary rovers</span>.",
    "hero.subtitle": "Autonomously decide which scientific target a rover investigates next on Mars",
    "hero.body_html": (
        "ARTPS combines an autoencoder, two anomaly detectors (PaDiM + PatchCore), "
        "a Vision Transformer depth model, and a learnable "
        "<b style=\"color:#E2725B\">Curiosity Score</b> head to decide "
        "<i>which target</i> a rover should inspect next on the Martian surface."
    ),
    "telemetry.device": "DEVICE",
    "telemetry.active_models": "ACTIVE MODELS",
    "telemetry.depth": "DEPTH",

    # Demo mode
    "demo.title": "Demo Mode — Models could not be loaded",
    "demo.message": (
        "Trained model files (<code>results/*.pth</code>) were not found. "
        "The UI and design are browsable; add model files to the "
        "<code>results/</code> folder to run analysis."
    ),

    # Params sliders
    "params.alpha.label": "α (Alpha) - Known Value Weight",
    "params.alpha.help": "Contribution of the classifier's predicted 'known value' to the Curiosity score. Higher values favor images similar to scientifically valuable known classes.",
    "params.beta.label": "β (Beta) - Anomaly Weight",
    "params.beta.help": "Contribution of AE-based anomaly MSE to the Curiosity score. Higher values favor unexpected or irregular structures.",
    "params.w_combined.label": "w_combined (Combined Anomaly)",
    "params.w_combined.help": "Contribution of the mean density of the combined anomaly map to the Curiosity score. Built from AE difference, depth edges, and texture components.",
    "params.w_dvar.label": "w_depth_variance",
    "params.w_dvar.help": "Contribution of depth variance (3D structural diversity) to the Curiosity score. High variance may indicate more complex geomorphology.",
    "params.w_rough.label": "w_roughness",
    "params.w_rough.help": "Contribution of roughness (gradient and Laplacian variability). Can highlight fine details such as small rocks or sand ripples.",
    "params.anomaly_threshold.label": "Anomaly Threshold",
    "params.anomaly_threshold.help": "Decision threshold for AE MSE. Values above this threshold may be considered anomalous on their own.",
    "params.ref_mse.label": "Curiosity Reference MSE",
    "params.ref_mse.help": "AE MSE reference for Curiosity normalization. Roughly 2×ref MSE maps to a score of 1.0.",

    # Policy expander
    "params.policy.expander": "🛡️ Operational Selection Policy (Clustering + Buffer)",
    "params.policy.enable": "Enable (generate recommended target set)",
    "params.policy.enable_help": "Selects targets from diverse shape types via latent-space clustering and places high-value targets suppressed by similarity into the Priority Buffer.",
    "params.policy.budget": "Target budget (B)",
    "params.policy.method": "Clustering method",
    "params.policy.k": "K (KMeans)",
    "params.policy.eps": "eps (DBSCAN)",
    "params.policy.min_samples": "min_samples (DBSCAN)",
    "params.policy.lambda_penalty": "λ (Soft Penalty)",
    "params.policy.tau_high": "Buffer τ_high (raw score)",
    "params.policy.tau_delta": "Buffer τ_Δ (drop)",
    "params.policy.history_m": "History length (m)",
    "params.policy.history_m_help": "If 0, historical diversity pressure is disabled.",
    "params.policy.crop_margin": "Crop margin",
    "params.policy.crop_margin_help": "Context padding added to the box for latent extraction.",

    # Detection expander
    "params.detection.expander": "🔧 Detection Settings (Advanced)",
    "params.detection.unified_threshold": "Combined Anomaly Threshold",
    "params.detection.hyst_high": "Hysteresis High (%)",
    "params.detection.hyst_low": "Hysteresis Low (%)",
    "params.detection.nms_iou": "NMS IoU",
    "params.detection.top_k": "Top-K Boxes",
    "params.detection.min_area": "Min Box Area (%)",
    "params.detection.min_area_help": "Relative to image area",
    "params.detection.weights_header": "⚖️ Weights",
    "params.detection.w_recon": "w_recon (difference)",
    "params.detection.w_depth": "w_depthEdge (∇depth)",
    "params.detection.w_texture": "w_texture (shadow+edge)",
    "params.detection.w_lap": "w_lap (Δ depth)",
    "params.detection.edge_reinf": "edge reinforce",
    "params.detection.w_detail": "w_detail (fine detail)",
    "params.detection.w_detail_help": "Multi-scale detail component highlighting small rocks and sand ripples",
    "params.detection.w_padim": "w_padim (PaDiM fusion)",
    "params.detection.w_padim_help": "Contribution of the PaDiM anomaly map to the combined map",
    "params.detection.w_patchcore": "w_patchcore (PatchCore fusion)",
    "params.detection.w_patchcore_help": "Contribution of the PatchCore anomaly map to the combined map",
    "params.detection.merge_header": "🔗 Box Merging",
    "params.detection.merge_iou": "Merge IoU",
    "params.detection.merge_tol": "Center Proximity (diagonal ratio)",
    "params.detection.merge_caption": "Merges nearby small boxes into a unified target; lower IoU preserves small details in distant areas.",
    "params.detection.shadow_header": "🌑 Shadow Suppression (Field Tuning)",
    "params.detection.alpha_shad": "Shadow Suppression Strength",
    "params.detection.alpha_shad_help": "Suppress dark regions with low edge response",
    "params.detection.beta_illum": "Illumination-Edge Reduction",
    "params.detection.beta_illum_help": "Reduces influence when image edges are high but depth edges are low",
    "params.detection.shadow_cut": "Shadow Rejection Threshold",
    "params.detection.shadow_cut_help": "Lower bound for rejecting pure shadow regions",
    "params.detection.img_edge_min": "Min Image Edge",
    "params.detection.depth_edge_min": "Min Depth Edge",
    "params.detection.spec_gamma": "Specular Suppression Strength",
    "params.detection.spec_gamma_help": "Suppress regions with high brightness and low saturation",
    "params.detection.spec_cut": "Specular Rejection Threshold",
    "params.detection.spec_lowvar_gamma": "Low Variance Reduction",
    "params.detection.spec_lowvar_help": "Applies extra reduction to low-texture (low variance) specular points",
    "params.detection.spec_var_thresh": "Low Variance Threshold",
    "params.detection.focus_header": "🎯 Focus Tiles",
    "params.detection.focus_h": "Focus Tile Height",
    "params.detection.focus_overlay": "Show heat + original blend (overlay)",
    "params.detection.focus_sharpen": "Focus Sharpening (unsharp)",
    "params.detection.focus_hide_empty_depth": "Hide depth tile when empty",
    "params.detection.focus_interp": "Resampling",
    "params.detection.focus_caption": "Focus tiles are pre-generated right after analysis for speed.",

    # Curiosity weights expander
    "params.curiosity.expander": "🧭 Curiosity Weights (Optional)",
    "params.curiosity.use_loaded": "Use weights loaded from file",
    "params.curiosity.weights_path": "Weights file (JSON)",
    "params.curiosity.load_btn": "Load",
    "params.curiosity.reset_btn": "Reset to defaults",
    "params.curiosity.loaded_ok": "Weights loaded",
    "params.curiosity.load_error": "Load error: {error}",
    "params.curiosity.defaults_active": "Default weights active",
    "params.curiosity.active_caption": "Active: known={known:.3f}, anomaly={anomaly:.3f}, combined={combined:.3f}, dvar={dvar:.3f}, rough={rough:.3f}",

    # Tabs and sections
    "tabs.image_analysis": "📸 Image Analysis",
    "tabs.depth": "🔍 Depth Analysis",
    "tabs.system": "📊 System Status",
    "tabs.demo": "🎯 Demo Data",
    "tabs.about": "ℹ️ About",
    "section.image_analysis": "📸 Mars Image Hybrid Analysis",
    "section.depth": "🔍 Depth Analysis",
    "section.system": "📊 System Status",
    "section.demo": "🎯 Demo Data",
    "section.about": "ℹ️ About the ARTPS Hybrid System",

    # Tab1 image analysis
    "analysis.upload_label": "Upload a Mars image (JPG, PNG)",
    "analysis.enhance_header": "🧹 Automatic Image Enhancement",
    "analysis.opt_upscale": "Upscale",
    "analysis.opt_upscale_help": "Smart upscaling for low-resolution images",
    "analysis.opt_denoise": "Denoise",
    "analysis.opt_denoise_help": "Color noise reduction for high-noise images",
    "analysis.opt_clahe": "Contrast (CLAHE)",
    "analysis.opt_gamma": "Exposure (Gamma)",
    "analysis.opt_sharp": "Sharpening",
    "analysis.enhance_btn": "✨ Auto-Enhance Image",
    "analysis.steps_applied": "Steps applied: {steps}",
    "analysis.before": "Before",
    "analysis.after": "After",
    "analysis.original_header": "📷 Original Image",
    "analysis.original_caption": "Uploaded Mars image",
    "analysis.analyze_btn": "🔍 Run Hybrid Analysis",
    "analysis.spinner": "Running hybrid analysis (Anomaly + Depth + Dynamic Value)...",
    "analysis.reconstructed_header": "🔄 Reconstructed Image",
    "analysis.anomaly_caption": "Anomaly Score: {score:.6f}",
    "analysis.results_header": "📊 Hybrid Analysis Results",
    "analysis.metric.anomaly_mse": "Anomaly Score (MSE)",
    "analysis.metric.combined": "Combined Anomaly",
    "analysis.metric.combined_na": "N/A",
    "analysis.metric.anomaly_status": "Anomaly Status",
    "analysis.status.anomaly": "🚨 Anomalous",
    "analysis.status.normal": "✅ Normal",
    "analysis.metric.known_value": "Known Value",
    "analysis.metric.curiosity": "Curiosity Score",
    "analysis.metric.predicted_class": "Predicted Class",
    "analysis.diff_header": "🔍 Difference and Combined Anomaly Map",
    "plot.original": "Original",
    "plot.reconstructed": "Reconstructed",
    "plot.difference": "Difference",
    "plot.combined_overlay": "Combined Anomaly (overlay)",
    "plot.difference_anomaly": "Difference (Anomaly)",
    "analysis.diag_header": "🔎 Detection Diagnostics Panel",
    "analysis.diag_help": (
        "- **sc**: Combined anomaly score\n"
        "- **e**: Edge density indicator\n"
        "- **s**: Shadow/darkness effect (reduction)\n"
        "- **sp**: Specular glare effect (reduction)\n"
        "- **lv**: Low texture/variance effect (reduction)"
    ),
    "analysis.quick_select": "Quick Select",
    "analysis.quick_all": "All",
    "analysis.detections_caption": "Combined Anomaly Detections",
    "analysis.detections_none_suffix": " — no detections found",
    "analysis.detections_small_objects": " (small objects included)",
    "analysis.no_detections": "No detections found or no diagnostic data available.",
    "analysis.focus_tile": "Focus: #{idx}",
    "analysis.recommendations_header": "💡 Hybrid Recommendations",
    "analysis.reco.high": "🎯 **HIGH PRIORITY**: This target is both anomalous and has high scientific value!",
    "analysis.reco.medium": "🔍 **MEDIUM PRIORITY**: This target is anomalous but has moderate scientific value.",
    "analysis.reco.low_known": "📋 **LOW PRIORITY**: This target appears normal but resembles known valuable targets.",
    "analysis.reco.low_normal": "📋 **LOW PRIORITY**: This target appears to be normal Martian terrain.",
    "analysis.anomaly_calc_error": "❌ Anomaly calculation error: {error}",
    "analysis.known_value_error": "⚠️ Known value calculation error: {error}",

    # analysis classes
    "analysis.class.negligible": "Negligible",
    "analysis.class.low": "Low",
    "analysis.class.medium": "Medium",
    "analysis.class.medium_high": "Medium-High",
    "analysis.class.high": "High",
    "analysis.class.unknown": "Unknown",

    # categories
    "category.rocky": "Rocky",
    "category.boulder": "Boulder",
    "category.flat_terrain": "Flat Terrain",
    "category.hills_or_ridge": "Hills/Ridge",
    "category.dusty": "Dusty",
    "category.rover": "Rover",

    # Tab2 depth
    "depth.map_header": "🌊 Depth Map ({model_type}) - {quality}",
    "depth.resolution": "Resolution",
    "depth.resolution_help": "Input image resolution used for analysis",
    "depth.apply_enhancement": "Apply Enhancement (contrast + sharpening)",
    "depth.raw_compare": "Compare with raw output",
    "depth.raw_compare_help": "Show raw (enhancement off) and enhanced outputs side by side",
    "depth.original_caption": "Original Image",
    "depth.map_title": "Enhanced Depth Map",
    "depth.colorbar_label": "Depth (0=Near, 1=Far)",
    "depth.raw_output": "Raw DPT Output",
    "depth.enhanced_output": "Enhancement Applied",
    "depth.enhancement_off": "Enhancement Off",
    "depth.summary": "📊 **Depth Analysis ({model_type})**: {width}x{height} resolution, Contrast: {contrast:.3f}, Mean Depth: {mean:.3f}, Time: {ms:.1f} ms",
    "depth.tuning_expander": "🔧 Depth Fine Tuning (Advanced)",
    "depth.apply_tuning_btn": "Apply (Update Depth Refinement)",
    "depth.tuning_applied": "Fine-tuning parameters updated. Re-run the 'Depth Analysis' section.",
    "depth.viz_options": "🎨 Depth Visualization Options",
    "depth.colormap": "Select Colormap:",
    "depth.features_header": "📊 Enhanced Depth Features",
    "depth.metric.mean": "🌊 Mean Depth",
    "depth.metric.std": "📏 Depth Std",
    "depth.metric.variance": "📊 Depth Variance",
    "depth.metric.min": "⬇️ Min Depth",
    "depth.metric.max": "⬆️ Max Depth",
    "depth.metric.median": "📈 Depth Median",
    "depth.metric.complexity": "🏔️ Surface Complexity",
    "depth.metric.grad_mean": "🌊 Gradient Mean",
    "depth.metric.grad_std": "📐 Gradient Std",
    "depth.metric.skewness": "📊 Skewness",
    "depth.metric.kurtosis": "📈 Kurtosis",
    "depth.metric.p75_p25": "🎯 P75-P25",
    "depth.metadata_header": "📋 Depth Metadata",
    "depth.distribution_header": "📊 Depth Distribution",
    "depth.quality_header": "🎯 Depth Quality Assessment",
    "depth.contrast.high": "✅ **High Contrast**: {value:.3f}",
    "depth.contrast.medium": "⚠️ **Medium Contrast**: {value:.3f}",
    "depth.contrast.low": "❌ **Low Contrast**: {value:.3f}",
    "depth.range.wide": "✅ **Wide Depth Range**: {value:.3f}",
    "depth.range.medium": "⚠️ **Medium Depth Range**: {value:.3f}",
    "depth.range.narrow": "❌ **Narrow Depth Range**: {value:.3f}",
    "depth.surface.smooth": "✅ **Smooth Surface**: {value:.3f}",
    "depth.surface.medium": "⚠️ **Medium Surface**: {value:.3f}",
    "depth.surface.rough": "❌ **Complex Surface**: {value:.3f}",
    "depth.analysis_error": "❌ Depth analysis error: {error}",
    "depth.upload_first": "📸 Upload an image first to run depth analysis.",
    "depth.load_failed": "Depth estimation module is unavailable.",
    "depth.plot.histogram_title": "Depth Histogram",
    "depth.plot.depth_value": "Depth Value",
    "depth.plot.frequency": "Frequency",
    "depth.plot.surface_3d": "3D Depth Surface (Sample)",
    "depth.plot.contour_2d": "2D Depth Contour (Fallback)",
    "depth.plot.axis_x": "X",
    "depth.plot.axis_y": "Y",
    "depth.plot.axis_z": "Depth",

    # Tab3 system
    "system.model_info": "🤖 Hybrid Model Information",
    "system.ae_params": "Autoencoder Parameters",
    "system.ae_size": "Autoencoder Size",
    "system.latent_size": "Latent Size",
    "system.clf_params": "Classifier Parameters",
    "system.clf_size": "Classifier Size",
    "system.class_count": "Class Count",
    "system.training_data": "📈 Training Data",
    "system.total_images": "Total Images",
    "system.category_count": "Category Count",
    "system.pie_title": "Category Distribution",

    # Tab4 demo
    "system.test_images": "📸 Test Images",
    "demo.analyze_btn": "🔍 {category} Hybrid Analysis",
    "demo.spinner": "Running hybrid analysis on {category}...",
    "demo.anomaly_result": "Anomaly: {score:.6f}",
    "demo.known_result": "Known Value: {score:.3f}",
    "demo.curiosity_metric": "Curiosity Score",

    # about.markdown
    "about.markdown": """
## 🚀 ARTPS - Autonomous Scientific Discovery System (Hybrid)

**ARTPS (Autonomous Rover Target Prioritization System)** is a **hybrid AI system**
that enables Mars rovers to detect scientifically interesting targets autonomously
without waiting for commands from Earth.

**🛰️ Built by:** [Poyraz BAYDEMİR](https://github.com/Poyqraz) · [ResearchGate DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
**📄 License:** [MIT License](https://github.com/Poyqraz/ARTPS/blob/main/LICENSE)

### 🎯 System Purpose
- Autonomously detect scientifically valuable targets on the Martian surface
- Perform 3D analysis with **depth perception**
- Compute a dynamic **Known Value** score
- Rank targets by priority
- Improve rover efficiency

### 🔬 Hybrid Technical Features (Current)
- **Convolutional Autoencoder**: Anomaly detection (optimized 17M params)
- **Depth-Enhanced Classifier**: Dynamic value (RGB latent + 14 depth features)
- **DPT_Large Depth Estimation**: High accuracy (CUDA accelerated)
- **PaDiM (Patch Distribution Modeling)**: Fuses image-based anomaly with AE+Depth
- **Multi-Scale Fine Detail**: Laplacian(3,5) + DoG highlighting small rocks and sand ripples
- **Far-Field Sensitivity**: Proximity blending and depth-conditioned area threshold
- **Curiosity Data**: ~2,575 images (train/valid)
- **Soft Focus Mask**: Gaussian-blended emphasis around the selected target

### 📊 Advanced Curiosity Score
```
Curiosity Score = α × Dynamic Known Value + β × Anomaly Score
```

- **α (Alpha)**: Dynamic known value weight (0-1)
- **β (Beta)**: Anomaly/exploration weight (0-1)
- **Dynamic Known Value**: Category-based automatic labeling (0-1)

### 🌊 Depth Analysis (Current)
- **DPT_Large**: High-accuracy monocular depth with guided refinement and filtering
- **14 Depth Features**: Mean, std, min, max, surface complexity, gradient, etc.
- **Far/Near Balance**: Adaptive thresholding to preserve small details in distant areas
- **3D/2D Visualization**: Turbo colormap, 3D surface, histogram, and statistics

### 🎮 Hybrid Usage
1. Upload a Mars image
2. Adjust parameters (α, β)
3. Click "Run Hybrid Analysis"
4. Review Anomaly + Depth + Dynamic Value results

### 🔍 Advanced Anomaly Detection
- **Low MSE**: Normal Martian terrain
- **High MSE**: Anomalous/interesting target
- **Depth Integration**: 3D anomaly detection
- **Dynamic Classification**: Automatic category assignment

### 📈 Hybrid Model Performance
- **Anomaly Detection**: 95%+ accuracy
- **Classification**: 74% accuracy
- **Depth Estimation**: DPT_Large (High Accuracy) + Fallback
- **Real-Time**: <1 second analysis time

### 🚀 Future Improvements
- Perseverance data integration
- Advanced segmentation algorithms
- Stereo vision integration
- Real-time rover integration
- Multi-rover support
- Space station integration
""",
}
