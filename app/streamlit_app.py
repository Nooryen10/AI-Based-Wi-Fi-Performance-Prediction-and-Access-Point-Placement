"""
STEP 6: STREAMLIT WEB APP
================================================================================
A polished, interactive alternative to app/flask_app.py, built with Streamlit +
Plotly. Reuses the exact same underlying functions as the CLI (app/interface.py)
and the Flask app, so predictions and placement recommendations are identical
across every interface -- only the presentation differs.

Why Streamlit instead of hand-rolled HTML/CSS:
  - Native widgets (sliders, selectboxes, metrics, tabs) look polished by default,
    with zero custom CSS required.
  - The coverage map is an interactive Plotly heatmap instead of a static image:
    you can hover over any point to see its exact dBm/SINR value, zoom, and pan.
  - Re-runs live in the same page (no full page reload on every submit).

Run with:
    cd wifi_ai_project (or the project folder)
    streamlit run app/streamlit_app.py
Then it opens automatically in your browser (usually http://localhost:8501).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.interface import (load_artifacts, predict_performance, recommend_placement,
                            recommend_multi_ap_placement, interpret_results,
                            _apply_scaled_floorplan)
import placement.ap_placement as ap

THRESHOLD = -65.0
AP_COLORS = ["#111827", "#7c3aed", "#ea580c", "#db2777", "#0891b2"]

st.set_page_config(page_title="AI Wi-Fi Predictor & AP Placement", page_icon="📡", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources / helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_models():
    return load_artifacts()


ARTIFACTS, MODELS = get_models()


def signal_badge(sig):
    if sig >= -50:
        return "Excellent", "🟢"
    elif sig >= -65:
        return "Good", "🟢"
    elif sig >= -75:
        return "Fair", "🟠"
    return "Weak", "🔴"


def throughput_badge(thr):
    if thr >= 100:
        return "High", "🟢"
    elif thr >= 25:
        return "Moderate", "🟠"
    return "Low", "🔴"


def latency_badge(lat):
    if lat <= 30:
        return "Low", "🟢"
    elif lat <= 80:
        return "Moderate", "🟠"
    return "High", "🔴"


def build_coverage_figure(router_positions, band, room_w, room_h, threshold=THRESHOLD,
                           interference_aware=False, sinr_threshold_db=9.0):
    """Builds an interactive Plotly heatmap of the coverage on the user's own (rescaled)
    floor plan -- hoverable, zoomable, with walls, AP markers, and threshold contours."""
    _apply_scaled_floorplan(room_w, room_h)

    sinr_2d = None
    if len(router_positions) == 1:
        X, Y, points = ap.build_grid(0.5)
        dist = np.linalg.norm(points - np.asarray(router_positions[0]), axis=1)
        walls_crossed = ap.count_wall_crossings(router_positions[0], points)
        signal = ap.path_loss_signal(dist, walls_crossed, band=band)
        signal_2d = signal.reshape(X.shape)
        coverage_pct = 100.0 * np.mean(signal >= threshold)
    else:
        X, Y, signal_2d, _, coverage_pct, sinr_2d = ap.simulate_coverage_multi(
            router_positions, band=band, threshold=threshold,
            interference_aware=interference_aware, sinr_threshold_db=sinr_threshold_db)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=X[0, :], y=Y[:, 0], z=signal_2d,
        colorscale="RdYlGn", zmin=-90, zmax=-30,
        colorbar=dict(title="dBm"),
        hovertemplate="x=%{x:.1f}m, y=%{y:.1f}m<br>Signal: %{z:.1f} dBm<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        x=X[0, :], y=Y[:, 0], z=signal_2d,
        contours=dict(start=threshold, end=threshold, size=1, coloring="lines"),
        line=dict(color="#1d4ed8", width=2.5), showscale=False, showlegend=False,
        hoverinfo="skip", name=f"{threshold:.0f} dBm boundary",
    ))
    if interference_aware and sinr_2d is not None:
        fig.add_trace(go.Contour(
            x=X[0, :], y=Y[:, 0], z=sinr_2d,
            contours=dict(start=sinr_threshold_db, end=sinr_threshold_db, size=1, coloring="lines"),
            line=dict(color="#dc2626", width=2.5, dash="dash"), showscale=False, showlegend=False,
            hoverinfo="skip", name=f"{sinr_threshold_db:.0f} dB SINR boundary",
        ))
    for (x1, y1), (x2, y2) in ap.WALLS:
        fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                      line=dict(color="black", width=3))
    for i, pos in enumerate(router_positions):
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]], mode="markers+text",
            marker=dict(symbol="star", size=22, color=AP_COLORS[i % len(AP_COLORS)],
                       line=dict(color="white", width=1.5)),
            text=[f"AP {i + 1}"], textposition="top center", textfont=dict(size=12, color="black"),
            name=f"AP {i + 1}", hovertemplate=f"AP {i + 1}: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
        ))

    mode_note = " (SINR-checked)" if interference_aware else ""
    fig.update_layout(
        title=f"Coverage{mode_note} @ {threshold:.0f} dBm — {coverage_pct:.1f}%",
        xaxis_title="Room Width (m)", yaxis_title="Room Height (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=560, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    return fig, round(coverage_pct, 1)


# ---------------------------------------------------------------------------
# Custom styling (light touch on top of Streamlit's own clean defaults)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  .main .block-container { padding-top: 1.6rem; max-width: 1200px; }
  div[data-testid="stMetric"] {
      background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 14px 16px 10px;
  }
  div[data-testid="stMetricLabel"] { font-weight: 600; color: #475569; }
  .hero {
      background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
      padding: 28px 30px; border-radius: 16px; color: white; margin-bottom: 22px;
  }
  .hero h1 { margin: 0 0 6px; font-size: 26px; }
  .hero p { margin: 0; opacity: 0.92; font-size: 14.5px; }
  .gap-note {
      background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
      padding: 10px 14px; font-size: 13.5px; color: #1e3a8a; margin-top: 10px;
  }
  .gap-note-warn {
      background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px;
      padding: 10px 14px; font-size: 13.5px; color: #9a3412; margin-top: 10px;
  }
</style>
<div class="hero">
  <h1>📡 AI-Based Wi-Fi Performance Prediction &amp; Access Point Placement</h1>
  <p>Predict signal / throughput / latency with ML, then get an AI-optimized router placement for your room</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar -- inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Room & Network Setup")
    with st.form("inputs"):
        col1, col2 = st.columns(2)
        room_w = col1.number_input("Room width (m)", min_value=0.1, max_value=200.0, value=30.0, step=0.5)
        room_h = col2.number_input("Room height (m)", min_value=0.1, max_value=200.0, value=20.0, step=0.5)

        walls = st.number_input("Walls between router & device", min_value=0, max_value=20, value=2)
        users = st.number_input("Concurrent users on the network", min_value=1, max_value=500, value=15)

        col3, col4 = st.columns(2)
        interference = col3.selectbox("Interference level", ["low", "medium", "high"], index=1)
        band = col4.selectbox("Frequency band", ["2.4GHz", "5GHz"], index=0)

        num_aps = st.slider("Number of access points", 1, 4, 1,
                            help="2+ switches on the multi-AP greedy joint-placement optimizer "
                                 "(research-gap extension #1) instead of the single-router search.")

        interference_aware = st.checkbox(
            "Account for AP-to-AP interference (advanced)",
            help="Research-gap extension #2: also penalizes placing APs close enough to "
                 "co-channel-interfere with each other, instead of only chasing raw coverage. "
                 "Only applies with 2+ APs.")
        sinr_threshold_db = st.slider(
            "Minimum acceptable SINR (dB)", 0.0, 30.0, 3.0, step=0.5,
            help="Only used when 'Account for AP-to-AP interference' above is checked. "
                 "Lower = more tolerant of interference (APs can sit closer together); "
                 "higher = stricter, may spread APs further apart or recommend fewer of them. "
                 "(Note: this slider stays enabled regardless of the checkbox above, since "
                 "widgets inside a form only update together on submit -- it's simply ignored "
                 "unless the checkbox is on.)")

        submitted = st.form_submit_button("Predict & Recommend Placement", width="stretch", type="primary")


# ---------------------------------------------------------------------------
# Main area -- results
# ---------------------------------------------------------------------------

if not submitted and "last_result" not in st.session_state:
    st.info("Fill in the form on the left and click **Predict & Recommend Placement** to see results here.")
    st.stop()

if submitted:
    distance = float(np.hypot(room_w, room_h) / 2)
    preds = predict_performance(distance, int(walls), int(users), interference, band, ARTIFACTS, MODELS)
    interpretation = interpret_results(preds)

    eff_interference_aware = interference_aware and num_aps > 1
    if num_aps == 1:
        pos, coverage = recommend_placement(room_w, room_h, band=band)
        positions, gains = [pos], None
    else:
        positions, coverage, gains = recommend_multi_ap_placement(
            room_w, room_h, num_aps=num_aps, band=band,
            interference_aware=eff_interference_aware, sinr_threshold_db=sinr_threshold_db)

    fig, _ = build_coverage_figure(positions, band, room_w, room_h,
                                    interference_aware=eff_interference_aware,
                                    sinr_threshold_db=sinr_threshold_db)

    st.session_state["last_result"] = dict(
        preds=preds, interpretation=interpretation, positions=positions, gains=gains,
        coverage=coverage, fig=fig, num_aps_requested=num_aps, num_aps_placed=len(positions),
        interference_aware=eff_interference_aware,
    )

r = st.session_state["last_result"]

st.subheader("📊 Predicted Performance")
c1, c2, c3 = st.columns(3)
sig_label, sig_icon = signal_badge(r["preds"]["signal_strength_dbm"])
thr_label, thr_icon = throughput_badge(r["preds"]["throughput_mbps"])
lat_label, lat_icon = latency_badge(r["preds"]["latency_ms"])
c1.metric("Signal Strength", f"{r['preds']['signal_strength_dbm']:.1f} dBm", f"{sig_icon} {sig_label}")
c2.metric("Throughput", f"{r['preds']['throughput_mbps']:.1f} Mbps", f"{thr_icon} {thr_label}")
c3.metric("Latency", f"{r['preds']['latency_ms']:.1f} ms", f"{lat_icon} {lat_label}")

with st.expander("Interpretation", expanded=True):
    for line in r["interpretation"]:
        st.markdown(f"- {line}")

st.subheader("📍 Recommended Access Point Placement")
st.markdown(f"**Coverage: {r['coverage']}% @ -65 dBm threshold**")

for i, pos in enumerate(r["positions"]):
    gain_note = f"  (+{r['gains'][i]}% coverage gain)" if r["gains"] and i < len(r["gains"]) else ""
    st.markdown(f"- **AP {i + 1}:** x = {pos[0]} m, y = {pos[1]} m (from bottom-left corner){gain_note}")

if r["num_aps_placed"] < r["num_aps_requested"]:
    st.warning(
        f"You asked for {r['num_aps_requested']} APs, but the search stopped after "
        f"**{r['num_aps_placed']}** because coverage already reached {r['coverage']}% — adding "
        "more APs wouldn't gain any additional area (and, if interference-awareness is on, "
        "could actually make real-world coverage worse by adding unnecessary interference). "
        "This is intentional early-stopping, not an error."
    )

if r["num_aps_requested"] > 1:
    st.markdown(
        '<div class="gap-note"><b>Multi-AP mode (research-gap extension #1):</b> a single AP is '
        'often insufficient once a floor plan gets large or walls attenuate the signal heavily '
        '(especially true on 5GHz). Multiple access points were jointly placed with a greedy '
        'coverage-maximization search so each additional AP targets the biggest remaining dead '
        'zone left by the previous ones.</div>', unsafe_allow_html=True)

if r["interference_aware"]:
    st.markdown(
        '<div class="gap-note-warn"><b>Interference-aware mode (research-gap extension #2):</b> '
        "APs were also kept far enough apart to avoid co-channel interference (assuming, "
        "conservatively, that they'd share one Wi-Fi channel) — not just placed for raw coverage. "
        "The red dashed line on the map marks where the signal-to-interference ratio drops below "
        'a usable level.</div>', unsafe_allow_html=True)

st.plotly_chart(r["fig"], width="stretch")

st.caption("AI-Based Wi-Fi Performance Prediction & AP Placement — CN academic project")
