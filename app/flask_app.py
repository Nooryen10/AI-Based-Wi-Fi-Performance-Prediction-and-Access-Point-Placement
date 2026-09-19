"""
STEP 6: FLASK WEB APP  (v2 -- redesigned UI + multi-AP research-gap extension)
================================================================================
A single-file Flask app wrapping the same predict_performance() / recommend_placement() /
recommend_multi_ap_placement() functions used by the CLI (app/interface.py), so every
interface in the project stays consistent with one source of truth.

What's new in this version:
  - Complete visual redesign: responsive card-based layout, gradient header, color-coded
    result badges, mobile-friendly (single column below 800px), loading state on submit.
  - The AP placement recommendation is no longer just text -- the actual coverage heatmap
    (floor plan + walls + AP position(s) + signal color-map) is rendered as an image
    directly in the page, generated on the fly with matplotlib for the user's own room size.
  - RESEARCH-GAP EXTENSION exposed in the UI: a "Number of Access Points" selector (1-4).
    Choosing >1 switches to the greedy multi-AP joint-coverage optimizer implemented in
    placement/ap_placement.py::find_optimal_multi_ap_placement, and the page explains *why*
    (single AP cannot always reach full coverage, especially on 5GHz / larger floor plans).
  - Server-side input validation with friendly inline error messages instead of raw
    tracebacks/500s.

Run with:
    cd wifi_ai_project
    python3 app/flask_app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import base64
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template_string, request

from app.interface import (load_artifacts, predict_performance, recommend_placement,
                            recommend_multi_ap_placement, interpret_results,
                            _apply_scaled_floorplan)
import placement.ap_placement as ap

app = Flask(__name__)
ARTIFACTS, MODELS = load_artifacts()

THRESHOLD = -65.0
AP_COLORS = ["#111827", "#7c3aed", "#ea580c", "#db2777", "#0891b2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_coverage_figure(router_positions, band, room_w, room_h, threshold=THRESHOLD,
                            interference_aware=False, sinr_threshold_db=9.0):
    """Renders the coverage heatmap for one or more AP positions on the user's own
    (rescaled) floor plan and returns it as a base64 PNG data-URI string, plus the
    resulting coverage percentage. Kept in-memory -- nothing is written to disk here."""
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

    fig, axis = plt.subplots(figsize=(7.5, 5.6))
    hm = axis.pcolormesh(X, Y, signal_2d, cmap="RdYlGn", vmin=-90, vmax=-30, shading="auto")
    plt.colorbar(hm, ax=axis, label="Signal Strength (dBm)")
    axis.contour(X, Y, signal_2d, levels=[threshold], colors="#1d4ed8", linewidths=1.8)
    if interference_aware and sinr_2d is not None:
        axis.contour(X, Y, sinr_2d, levels=[sinr_threshold_db], colors="#dc2626",
                     linewidths=1.6, linestyles="dashed")
    for (x1, y1), (x2, y2) in ap.WALLS:
        axis.plot([x1, x2], [y1, y2], color="black", linewidth=2.6, solid_capstyle="round")
    for i, pos in enumerate(router_positions):
        axis.scatter(*pos, marker="*", s=420, c=AP_COLORS[i % len(AP_COLORS)],
                    edgecolors="white", linewidths=1.2, zorder=5, label=f"AP {i + 1}")
    axis.set_xlabel("Room Width (m)")
    axis.set_ylabel("Room Height (m)")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    mode_note = " (SINR-checked)" if interference_aware else ""
    axis.set_title(f"Coverage{mode_note} @ {threshold:.0f} dBm threshold  —  {coverage_pct:.1f}%", fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}", round(coverage_pct, 1)


def signal_badge(sig):
    if sig >= -50:
        return "excellent", "#059669"
    elif sig >= -65:
        return "good", "#16a34a"
    elif sig >= -75:
        return "fair", "#d97706"
    return "weak", "#dc2626"


def throughput_badge(thr):
    if thr >= 100:
        return "high", "#059669"
    elif thr >= 25:
        return "moderate", "#d97706"
    return "low", "#dc2626"


def latency_badge(lat):
    if lat <= 30:
        return "low", "#059669"
    elif lat <= 80:
        return "moderate", "#d97706"
    return "high", "#dc2626"


def validate_form(form):
    """Returns a list of human-readable error strings (empty list = valid)."""
    errors = []
    if form["room_w"] <= 0 or form["room_h"] <= 0:
        errors.append("Room width and height must both be greater than 0.")
    if form["room_w"] > 200 or form["room_h"] > 200:
        errors.append("Room dimensions above 200 m aren't realistic for a single floor plan -- "
                      "please enter a smaller value.")
    if form["walls"] < 0:
        errors.append("Number of walls can't be negative.")
    if form["walls"] > 20:
        errors.append("Please enter 20 or fewer walls between router and device.")
    if form["users"] < 1:
        errors.append("Number of concurrent users must be at least 1.")
    if form["users"] > 500:
        errors.append("Please enter 500 or fewer concurrent users.")
    if form["num_aps"] < 1 or form["num_aps"] > 4:
        errors.append("Number of access points must be between 1 and 4.")
    if form["sinr_threshold_db"] < 0 or form["sinr_threshold_db"] > 30:
        errors.append("Minimum SINR must be between 0 and 30 dB.")
    return errors


# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Wi-Fi Performance Predictor &amp; AP Placement</title>
<style>
  :root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --bg: #f1f5f9;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    color: white;
    padding: 34px 20px 46px;
    text-align: center;
  }
  header h1 { margin: 0 0 6px; font-size: 26px; font-weight: 700; }
  header p { margin: 0; opacity: 0.9; font-size: 14.5px; }
  .wrap {
    max-width: 1080px;
    margin: -26px auto 40px;
    padding: 0 18px;
  }
  .grid {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 22px;
    align-items: start;
  }
  @media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid var(--border);
    padding: 22px 22px 24px;
  }
  .card h2 {
    margin: 0 0 16px;
    font-size: 16px;
    letter-spacing: 0.02em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 5px;
  }
  input, select {
    width: 100%;
    padding: 10px 11px;
    border-radius: 9px;
    border: 1.5px solid var(--border);
    font-size: 14px;
    background: #f8fafc;
    transition: border-color 0.15s;
  }
  input:focus, select:focus {
    outline: none;
    border-color: var(--primary);
    background: white;
  }
  .hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  button.submit {
    width: 100%;
    margin-top: 22px;
    padding: 13px;
    background: var(--primary);
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.05s, box-shadow 0.15s;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
  }
  button.submit:hover { box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45); }
  button.submit:active { transform: translateY(1px); }
  button.submit:disabled { opacity: 0.6; cursor: wait; }

  .errors {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 18px;
    font-size: 14px;
  }
  .errors ul { margin: 6px 0 0; padding-left: 20px; }

  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 6px; }
  @media (max-width: 620px) { .metrics { grid-template-columns: 1fr; } }
  .metric-card {
    border-radius: 12px;
    padding: 14px 14px 12px;
    background: #f8fafc;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }
  .metric-card .label { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .metric-card .value { font-size: 22px; font-weight: 700; margin: 4px 0 6px; }
  .badge {
    display: inline-block;
    font-size: 11.5px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .note-list { margin: 18px 0 0; padding-left: 20px; font-size: 14px; color: #334155; }
  .note-list li { margin-bottom: 6px; }

  .placement-header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }
  .coverage-pill {
    font-size: 13px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
  }
  .ap-list { list-style: none; margin: 14px 0 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .ap-list li {
    display: flex; align-items: center; gap: 10px;
    font-size: 14px; background: #f8fafc; border: 1px solid var(--border);
    border-radius: 9px; padding: 8px 12px;
  }
  .dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; border: 2px solid white; box-shadow: 0 0 0 1px #cbd5e1; }
  .coverage-img { width: 100%; border-radius: 10px; border: 1px solid var(--border); margin-top: 14px; }

  .gap-callout {
    margin-top: 16px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13px;
    color: #1e3a8a;
  }
  .gap-callout b { color: #1e40af; }

  .placeholder {
    color: var(--muted);
    font-size: 14px;
    padding: 30px 10px;
    text-align: center;
    border: 1.5px dashed var(--border);
    border-radius: 12px;
  }
  footer { text-align: center; color: var(--muted); font-size: 12.5px; padding: 10px 0 30px; }
</style>
</head>
<body>

<header>
  <h1>📡 AI-Based Wi-Fi Performance Prediction &amp; Access Point Placement</h1>
  <p>Predict signal / throughput / latency with ML, then get an AI-optimized router placement for your room</p>
</header>

<div class="wrap">
  {% if errors %}
  <div class="errors">
    <b>Please fix the following before continuing:</b>
    <ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul>
  </div>
  {% endif %}

  <div class="grid">
    <!-- INPUT CARD -->
    <div class="card">
      <h2>⚙️ Room &amp; Network Setup</h2>
      <form method="POST" id="predict-form">
        <div class="row2">
          <div>
            <label>Room Width (m)</label>
            <input type="number" step="0.1" min="0.1" name="room_w" value="{{ form.room_w }}" required>
          </div>
          <div>
            <label>Room Height (m)</label>
            <input type="number" step="0.1" min="0.1" name="room_h" value="{{ form.room_h }}" required>
          </div>
        </div>

        <label>Walls Between Router &amp; Device</label>
        <input type="number" min="0" name="walls" value="{{ form.walls }}" required>
        <div class="hint">Used for the ML performance prediction at your specific point.</div>

        <label>Concurrent Users on the Network</label>
        <input type="number" min="1" name="users" value="{{ form.users }}" required>

        <div class="row2">
          <div>
            <label>Interference Level</label>
            <select name="interference">
              <option value="low" {{ 'selected' if form.interference=='low' }}>Low</option>
              <option value="medium" {{ 'selected' if form.interference=='medium' }}>Medium</option>
              <option value="high" {{ 'selected' if form.interference=='high' }}>High</option>
            </select>
          </div>
          <div>
            <label>Frequency Band</label>
            <select name="band">
              <option value="2.4GHz" {{ 'selected' if form.band=='2.4GHz' }}>2.4 GHz</option>
              <option value="5GHz" {{ 'selected' if form.band=='5GHz' }}>5 GHz</option>
            </select>
          </div>
        </div>

        <label>Number of Access Points</label>
        <select name="num_aps">
          {% for n in [1,2,3,4] %}
          <option value="{{ n }}" {{ 'selected' if form.num_aps==n }}>{{ n }} {{ 'AP (single router)' if n==1 else 'APs (multi-AP)' }}</option>
          {% endfor %}
        </select>
        <div class="hint">Choosing 2+ switches on the multi-AP greedy joint-placement optimizer (research-gap extension) instead of the single-router search.</div>

        <label style="display:flex;align-items:center;gap:8px;margin-top:16px;">
          <input type="checkbox" name="interference_aware" value="1" style="width:auto;"
                 {{ 'checked' if form.interference_aware }}>
          Account for AP-to-AP interference (advanced)
        </label>
        <div class="hint">2nd research-gap extension: also penalizes placing multiple APs close enough to co-channel-interfere with each other, instead of only chasing raw coverage. Only applies with 2+ APs. A stricter (higher) SINR requirement below may lead to FEWER APs being recommended than you asked for — that's intentional, it means extra APs wouldn't actually help once real interference is accounted for.</div>

        <label>Minimum acceptable SINR (dB)</label>
        <input type="number" step="0.5" min="0" max="30" name="sinr_threshold_db" value="{{ form.sinr_threshold_db }}">
        <div class="hint">Only used when the checkbox above is on. Lower = more tolerant of interference (APs can sit closer together); higher = stricter, may spread APs further apart or recommend fewer of them.</div>

        <button class="submit" type="submit" id="submit-btn">Predict &amp; Recommend Placement</button>
      </form>
    </div>

    <!-- RESULTS -->
    <div class="card">
      <h2>📊 Predicted Performance</h2>
      {% if result %}
        <div class="metrics">
          <div class="metric-card">
            <div class="label">Signal Strength</div>
            <div class="value">{{ result.preds.signal_strength_dbm | round(1) }} dBm</div>
            <span class="badge" style="background:{{ result.sig_color }}">{{ result.sig_label }}</span>
          </div>
          <div class="metric-card">
            <div class="label">Throughput</div>
            <div class="value">{{ result.preds.throughput_mbps | round(1) }} Mbps</div>
            <span class="badge" style="background:{{ result.thr_color }}">{{ result.thr_label }}</span>
          </div>
          <div class="metric-card">
            <div class="label">Latency</div>
            <div class="value">{{ result.preds.latency_ms | round(1) }} ms</div>
            <span class="badge" style="background:{{ result.lat_color }}">{{ result.lat_label }}</span>
          </div>
        </div>
        <ul class="note-list">
          {% for line in result.interpretation %}<li>{{ line }}</li>{% endfor %}
        </ul>
      {% else %}
        <div class="placeholder">Fill in the form and click <b>Predict &amp; Recommend Placement</b> to see results here.</div>
      {% endif %}
    </div>
  </div>

  {% if result %}
  <div class="card" style="margin-top:22px;">
    <div class="placement-header">
      <h2 style="margin:0;">📍 Recommended Access Point Placement</h2>
      <span class="coverage-pill">{{ result.coverage }}% coverage @ -65 dBm</span>
    </div>

    <ul class="ap-list">
      {% for pos in result.positions %}
      <li><span class="dot" style="background:{{ result.ap_colors[loop.index0] }}"></span>
          AP {{ loop.index }}: x = {{ pos[0] }} m, y = {{ pos[1] }} m (from bottom-left corner)
          {% if result.gains and loop.index0 < result.gains|length %}
            <span style="margin-left:auto;color:#64748b;font-size:12.5px;">+{{ result.gains[loop.index0] }}% coverage gain</span>
          {% endif %}
      </li>
      {% endfor %}
    </ul>

    {% if result.num_aps_placed < result.num_aps %}
    <div class="gap-callout" style="background:#fef2f2;border-color:#fecaca;color:#991b1b;">
      <b>Note:</b> you asked for {{ result.num_aps }} APs, but the search stopped after
      {{ result.num_aps_placed }} because coverage already reached {{ result.coverage }}% —
      adding more APs wouldn't gain any additional area (and, if interference-awareness is on,
      could actually make real-world coverage worse by adding unnecessary interference).
      This is intentional early-stopping, not an error.
    </div>
    {% endif %}

    {% if result.num_aps > 1 %}
    <div class="gap-callout">
      <b>Multi-AP mode (research-gap extension):</b> a single AP is often insufficient once a floor
      plan gets large or walls attenuate the signal heavily (this is especially true on 5GHz).
      Instead of one router, {{ result.num_aps }} access points were jointly placed with a
      greedy coverage-maximization search so each additional AP targets the biggest remaining
      dead zone left by the previous ones.
    </div>
    {% endif %}
    {% if result.interference_aware %}
    <div class="gap-callout" style="background:#fff7ed;border-color:#fed7aa;color:#9a3412;">
      <b>Interference-aware mode (2nd research-gap extension):</b> APs were also kept far enough
      apart to avoid co-channel interference (assuming, conservatively, that they'd share one
      Wi-Fi channel) — not just placed for raw coverage. The red dashed line on the map below
      marks where the signal-to-interference ratio drops below a usable level.
    </div>
    {% endif %}

    <img class="coverage-img" src="{{ result.coverage_img }}" alt="Coverage heatmap">
  </div>
  {% endif %}
</div>

<footer>AI-Based Wi-Fi Performance Prediction &amp; AP Placement — CN academic project by Nooryen, Tanvi & Vedanti</footer>

<script>
  document.getElementById('predict-form').addEventListener('submit', function () {
    var btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.textContent = 'Computing…';
  });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    form = {"room_w": 30, "room_h": 20, "walls": 2, "users": 15,
            "interference": "medium", "band": "2.4GHz", "num_aps": 1, "interference_aware": False,
            "sinr_threshold_db": 3.0}
    result = None
    errors = []

    if request.method == "POST":
        try:
            form["room_w"] = float(request.form["room_w"])
            form["room_h"] = float(request.form["room_h"])
            form["walls"] = int(request.form["walls"])
            form["users"] = int(request.form["users"])
            form["interference"] = request.form["interference"]
            form["band"] = request.form["band"]
            form["num_aps"] = int(request.form["num_aps"])
            form["interference_aware"] = "interference_aware" in request.form
            form["sinr_threshold_db"] = float(request.form.get("sinr_threshold_db", 3.0))
        except (ValueError, KeyError):
            errors = ["Please make sure every field is filled in with a valid number."]

        if not errors:
            errors = validate_form(form)

        if not errors:
            distance = float(np.hypot(form["room_w"], form["room_h"]) / 2)
            preds = predict_performance(distance, form["walls"], form["users"],
                                         form["interference"], form["band"], ARTIFACTS, MODELS)
            interpretation = interpret_results(preds)

            interference_aware = form["interference_aware"] and form["num_aps"] > 1
            if form["num_aps"] == 1:
                pos, coverage = recommend_placement(form["room_w"], form["room_h"], band=form["band"])
                positions, gains = [pos], None
            else:
                positions, coverage, gains = recommend_multi_ap_placement(
                    form["room_w"], form["room_h"], num_aps=form["num_aps"], band=form["band"],
                    interference_aware=interference_aware, sinr_threshold_db=form["sinr_threshold_db"])

            coverage_img, _ = render_coverage_figure(positions, form["band"],
                                                       form["room_w"], form["room_h"],
                                                       interference_aware=interference_aware,
                                                       sinr_threshold_db=form["sinr_threshold_db"])

            sig_label, sig_color = signal_badge(preds["signal_strength_dbm"])
            thr_label, thr_color = throughput_badge(preds["throughput_mbps"])
            lat_label, lat_color = latency_badge(preds["latency_ms"])

            result = {
                "preds": preds,
                "interpretation": interpretation,
                "positions": positions,
                "gains": gains,
                "coverage": coverage,
                "coverage_img": coverage_img,
                "num_aps": form["num_aps"],
                "num_aps_placed": len(positions),
                "interference_aware": interference_aware,
                "ap_colors": AP_COLORS,
                "sig_label": sig_label, "sig_color": sig_color,
                "thr_label": thr_label, "thr_color": thr_color,
                "lat_label": lat_label, "lat_color": lat_color,
            }

    return render_template_string(PAGE, form=form, result=result, errors=errors)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
