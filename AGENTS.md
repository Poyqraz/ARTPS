# AGENTS.md

## Learned User Preferences

## Learned Workspace Facts

- This repository has NO git initialized, so `git diff` and other before/after git validation are unavailable.
- The active Streamlit app is the ROOT `app.py` (2242+ lines); `ARTPS/app.py` is an older duplicate copy and not the one in use.
- The Windows console (cp1254) crashes on emoji `print()` calls; `app.py` reconfigures `sys.stdout`/`sys.stderr` to UTF-8 at the top to avoid this. `depth_estimation.py` contains many emoji prints.
- Streamlit 1.41 forbids nested expanders: an `st.expander` inside an `st.sidebar.expander` crashes with "Expanders may not be nested"; the fix used here replaces inner expanders with an `st.markdown` label plus `st.container(border=True)`.
- DPT_Large depth weights are NOT present locally (expected at `raw_models/dpt_large_384.pt`, ~1.3GB); without them and without internet, depth estimation falls back to a 424K-param simple model instead of the real ~345M DPT_Large, which diverges from the paper.
- UI uses a Mars/space dark theme defined in `.streamlit/config.toml`, `assets/style.css`, and `src/ui/theme.py` (`inject_theme`/`render_hero`/`empty_state`). Streamlit's HTML sanitizer strips inline `style` attributes containing `url(data:...)`, so the hero background image must be injected via a `<style>` block, not an inline style attribute.
- Model files (`results/*.pth`) and `mars_images/` ARE present locally on this machine.
- The Curiosity Score formula is verified consistent with the paper: `C = alpha*known + beta*anomaly` (plus optional combined / depth_variance / roughness terms).
