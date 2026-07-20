# AGENTS.md

## Learned User Preferences

## Learned Workspace Facts

- This repository uses git; active app is root `app.py` (not `ARTPS/app.py`).
- The active Streamlit app is the ROOT `app.py` (3000+ lines); `ARTPS/app.py` is an older duplicate copy and not the one in use.
- Optional Real-ESRGAN x2: place `raw_models/RealESRGAN_x2plus.pth` and `pip install realesrgan` (defaults off in Mars enhance UI).
- Scraped/NASA JPEG authenticity checks: use external [Sherloq](https://github.com/GuidoBartoli/sherloq) (GPL-3 Qt toolset) separately — do not port into ARTPS (MIT); it is not part of the hybrid anomaly/shape scoring pipeline.
- The Windows console (cp1254) crashes on emoji `print()` calls; `app.py` reconfigures `sys.stdout`/`sys.stderr` to UTF-8 at the top to avoid this. `depth_estimation.py` contains many emoji prints.
- Streamlit 1.41 forbids nested expanders: an `st.expander` inside an `st.sidebar.expander` crashes with "Expanders may not be nested"; the fix used here replaces inner expanders with an `st.markdown` label plus `st.container(border=True)`.
- DPT_Large depth weights live at `raw_models/dpt_large_384.pt` (~1.3GB, state_dict format) and are gitignored (`raw_models/`, `*.pt`). They ARE present locally on this machine; `timm` is required to load the Hub architecture offline. Without the file or `timm`, depth falls back to a 424K-param simple model. `MiDaSDepthEstimator` exposes `is_real_dpt` and `load_source` (`local_state_dict`, `hub`, `fallback`).
- UI uses a Mars/space dark theme defined in `.streamlit/config.toml`, `assets/style.css`, and `src/ui/theme.py` (`inject_theme`/`render_hero`/`empty_state`). Streamlit's HTML sanitizer strips inline `style` attributes containing `url(data:...)`, so the hero background image must be injected via a `<style>` block, not an inline style attribute.
- Model files (`results/*.pth`) and `mars_images/` ARE present locally on this machine.
- The Curiosity Score formula is verified consistent with the paper: `C = alpha*known + beta*anomaly` (plus optional combined / depth_variance / roughness terms).
