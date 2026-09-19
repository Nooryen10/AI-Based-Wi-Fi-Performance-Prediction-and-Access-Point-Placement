# AI-Based Wi-Fi Performance Prediction and Access Point Placement

A complete, runnable academic project: synthetic-but-physics-grounded Wi-Fi dataset →
ML performance prediction (signal / throughput / latency) → geometry-based AP placement
optimization on a real floor plan → CLI and web interfaces.

## Project Structure

```
wifi_ai_project/
├── data/
│   ├── generate_dataset.py       # Step 1: physics-informed synthetic dataset generator
│   └── wifi_dataset.csv          # 25,000-row generated dataset
├── preprocessing/
│   ├── preprocess.py             # Step 2: missing values, encoding, scaling, train/test split
│   └── *.pkl / *.npy / *.csv     # saved preprocessing artifacts + splits
├── models/
│   ├── train_models.py           # Step 3: trains + compares Linear Reg / RF / XGBoost
│   ├── best_model_*.pkl          # best model per target, auto-selected by R2
│   └── model_comparison_results.csv
├── visualization/
│   ├── visualize.py              # Step 4: signal-vs-distance, throughput-vs-users, correlation heatmap
│   └── bonus_analysis.py         # Bonus: 2.4GHz vs 5GHz aggregate comparison
├── placement/
│   └── ap_placement.py           # Step 5 (CORE): real floor plan, ray-based wall counting,
│                                  # exhaustive-grid-search coverage optimizer, before/after comparison
├── app/
│   ├── interface.py               # Step 6: CLI -- predict performance + recommend AP position
│   └── flask_app.py               # Step 6 (optional): web UI wrapping the same functions
├── outputs/                      # all generated PNGs + summary CSV/TXT land here
├── report/                       # generated academic report (docx)
└── requirements.txt
```

## How to run everything, in order

```bash
pip install -r requirements.txt

python3 data/generate_dataset.py          # Step 1
python3 preprocessing/preprocess.py       # Step 2
python3 models/train_models.py            # Step 3
python3 visualization/visualize.py        # Step 4
python3 visualization/bonus_analysis.py   # Bonus
python3 placement/ap_placement.py         # Step 5 (also runs both research-gap demos)

# Step 6 -- interactive CLI:
python3 app/interface.py
# or non-interactive:
python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 --interference medium --band 2.4GHz
# or with the multi-AP research-gap extension:
python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 --interference medium --band 5GHz --num_aps 2
# or with the interference-aware research-gap extension on top:
python3 app/interface.py --room_w 35 --room_h 25 --walls 2 --users 15 --interference medium --band 5GHz --num_aps 4 --interference_aware --sinr_threshold_db 3.0

# Step 6 -- web UI (redesigned):
python3 app/flask_app.py     # then open http://127.0.0.1:5000

# Step 6 -- web UI (Streamlit alternative -- interactive Plotly map, no page reloads):
streamlit run app/streamlit_app.py    # opens automatically in your browser
```

## What's new in this version

**1. Redesigned web UI (`app/flask_app.py`) + new Streamlit alternative (`app/streamlit_app.py`)**
- Full visual redesign of the Flask UI: gradient header, responsive two-column card layout
  (collapses to one column on mobile), color-coded result badges (excellent/good/fair/weak).
- A brand-new **Streamlit + Plotly** interface as a second, more interactive option: native
  polished widgets (sliders, selectboxes, metrics) with zero custom CSS needed, and an
  interactive coverage map you can hover/zoom (exact dBm and SINR values on hover) instead
  of a static image. Both UIs call the exact same underlying functions in `app/interface.py`,
  so predictions and placement recommendations are identical -- only presentation differs.
  Run with `streamlit run app/streamlit_app.py`.
- The placement recommendation is no longer just text -- the actual coverage heatmap
  (your floor plan, your walls, your AP position(s), color-mapped signal strength) is
  rendered live and embedded directly in the page in both UIs.
- Server-side/widget-level input validation with friendly inline error messages (no more raw
  tracebacks if you type a negative room size or a garbage value).
- A "Number of Access Points" control that turns on the multi-AP research-gap
  extension described below, in both UIs.

**2. Research-gap extension #1: multi-AP joint placement (`placement/ap_placement.py`,
   `find_optimal_multi_ap_placement`)**
- The original Step 5 optimizer only ever placed ONE access point. That's a real,
  well-documented limitation of single-AP grid-search approaches used in most course
  projects: on this project's own floor plan, a single 5GHz AP tops out at **93.6%**
  coverage no matter where you put it -- some rooms are just too far / too many walls away.
- This version adds a **greedy sequential multi-AP placement** algorithm: place AP #1 to
  maximize coverage alone, then place AP #2 to maximize the *marginal* coverage gain given
  AP #1 is already there (each client is served by whichever AP gives it the strongest
  signal), and so on for as many APs as you ask for. This is the standard, well-studied
  approximation for the "maximal coverage location problem" (a form of the classic
  NP-hard facility-location problem) and comes with a theoretical guarantee: greedy
  achieves at least (1 - 1/e) ≈ 63% of the true joint optimum, while staying fast enough
  to run interactively (unlike full joint brute-force search over all AP combinations).
- Result on this project's floor plan: 2 APs on 5GHz reach **100%** coverage
  (+6.4 points over the best possible single AP). Try it yourself with
  `python3 app/interface.py --band 5GHz --num_aps 2` or via the web UI.
- Available everywhere: `placement/ap_placement.py` (core algorithm + demo block that
  produces `outputs/10_multi_ap_placement_5ghz.png`), `app/interface.py`
  (`--num_aps` CLI flag), and `app/flask_app.py` (the "Number of Access Points" dropdown).

**3. Research-gap extension #2: interference-aware multi-AP placement (`placement/ap_placement.py`,
   `simulate_coverage_multi(interference_aware=True)`)**
- Extension #1 above only optimizes *coverage* -- a cell counts as "covered" as soon as
  SOME AP is loud enough there. Nothing stops the greedy search from placing two APs right
  next to each other, because from a coverage-only view that looks fine. In a real
  deployment, though, nearby APs very likely share the same Wi-Fi channel (2.4GHz has only
  3 non-overlapping channels; dense deployments run out of clean channels fast), so they'd
  **co-channel interfere** with each other and degrade real-world throughput exactly where
  their coverage overlaps -- a gap coverage-only placement tools have no way to notice.
- This extension computes a real **SINR (signal-to-interference ratio)** at every point:
  every placed AP is conservatively treated as being on the same channel as every other one,
  and a cell only counts as covered if the serving AP's signal also clears a minimum SINR
  bar over the combined interference from every other AP (plus the receiver noise floor),
  not just the usual absolute RSSI threshold. This makes the greedy search naturally spread
  APs apart instead of clustering them, since clustering now measurably lowers SINR without
  buying any extra coverage.
- Result on this project's floor plan (5GHz, 4 APs requested, a relaxed 3 dB minimum SINR):
  a coverage-only placement clusters 2 APs just 5m apart, claiming **100%** coverage on
  paper -- but under a real interference check that same placement only actually achieves
  **79.2%**. The interference-aware search instead spreads the 2 APs 20m apart and reaches
  **95%** of *real*, interference-limited coverage. See
  `outputs/12_coverage_only_vs_interference_aware_A_coverage_only.png` vs
  `..._B_interference_aware.png` for the visual before/after.
- A stricter SINR requirement can legitimately recommend *fewer* APs than requested --
  that's not a bug, it means adding another AP wouldn't actually help once interference is
  accounted for realistically. This is itself a useful, sometimes counterintuitive finding:
  more APs isn't automatically better without also doing channel planning (noted as a
  remaining limitation/future-work item -- this project doesn't model channel assignment).
- Available everywhere: the demo block in `ap_placement.py`, `app/interface.py`
  (`--interference_aware` and `--sinr_threshold_db` CLI flags), and `app/flask_app.py`
  (the "Account for AP-to-AP interference" checkbox + adjustable SINR field, with a red
  dashed contour on the coverage map showing the SINR boundary).

**4. Bug fix: room-rescaling now correctly affects wall geometry**
- Previously, when a user entered custom room dimensions in the CLI/web UI, the
  optimizer's grid extent was rescaled but the wall-crossing calculation silently kept
  using the *original* 35m x 25m wall coordinates (a stale-default-argument bug in
  `count_wall_crossings`), so line-of-sight results were wrong for any room size other
  than the hardcoded default. This has been fixed so any room size you enter uses the
  correctly rescaled floor plan geometry throughout.

**5. Pinned dependency versions in `requirements.txt`** to the exact versions this project
   was built and tested against, so re-running the pipeline months from now reproduces the
   same environment instead of silently picking up newer, untested library versions.

**6. Hardened CLI input validation (`app/interface.py`)** to match the web UI: room size,
   wall count, user count, and `--num_aps` are now range-checked the same way in both
   interfaces, with a fixed falsy-zero bug (`--walls 0` used to be silently ignored and
   re-prompted for; it's now respected).

**7. Models retrained against the exact library versions in `requirements.txt`** to
   remove `InconsistentVersionWarning` noise from pickled scikit-learn objects.

**8. Git hygiene:** added a `.gitignore` and untracked stray committed `__pycache__` files.

## Key design decisions (good viva talking points)

1. **Why synthetic data, and why it's not "fake":** the dataset generator implements the
   log-distance path loss model + wall attenuation factor (WAF) + Shannon-capacity-style
   throughput derivation + M/M/1-style queuing delay for latency -- the same models used in
   real RF planning tools. This gives ground-truth control over every input variable while
   keeping the input→output relationships physically realistic and non-linear.

2. **Why tree ensembles beat Linear Regression so much on latency (R²: 0.49 → 0.9996):**
   latency has a queuing-delay term that blows up non-linearly as user count approaches
   channel saturation. A linear model structurally cannot represent that; Random Forest /
   XGBoost can.

3. **Why the AP placement uses real geometry, not a distance-only proxy:** Step 5 defines an
   actual floor plan (wall line segments) and uses classic 2D segment-intersection tests to
   count how many walls the signal must cross to reach each point -- this is what real
   placement decisions actually depend on (a room 3m away through 2 walls is worse than a
   room 10m away with a clear line of sight).

4. **Why exhaustive grid search for optimization, not just greedy:** the search space (room
   positions on a coarse grid) is small enough that brute-force search is fast (~2-4 seconds)
   and guarantees the best position on the evaluated grid, unlike a greedy hill-climb which
   can converge to a local optimum behind a wall.
