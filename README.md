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
python3 placement/ap_placement.py         # Step 5 (also runs the multi-AP research-gap demo)

# Step 6 -- interactive CLI:
python3 app/interface.py
# or non-interactive:
python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 --interference medium --band 2.4GHz
# or with the multi-AP research-gap extension:
python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 --interference medium --band 5GHz --num_aps 2

# Step 6 -- web UI (redesigned):
python3 app/flask_app.py     # then open http://127.0.0.1:5000
```

## What's new in this version

**1. Redesigned web UI (`app/flask_app.py`)**
- Full visual redesign: gradient header, responsive two-column card layout (collapses to
  one column on mobile), color-coded result badges (excellent/good/fair/weak, etc.).
- The placement recommendation is no longer just text -- the actual coverage heatmap
  (your floor plan, your walls, your AP position(s), color-mapped signal strength) is
  rendered live with matplotlib and embedded directly in the page.
- Server-side input validation with friendly inline error messages (no more raw
  tracebacks if you type a negative room size or a garbage value).
- A "Number of Access Points" selector that turns on the multi-AP research-gap
  extension described below.

**2. Research-gap extension: multi-AP joint placement (`placement/ap_placement.py`,
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

**3. Bug fix: room-rescaling now correctly affects wall geometry**
- Previously, when a user entered custom room dimensions in the CLI/web UI, the
  optimizer's grid extent was rescaled but the wall-crossing calculation silently kept
  using the *original* 35m x 25m wall coordinates (a stale-default-argument bug in
  `count_wall_crossings`), so line-of-sight results were wrong for any room size other
  than the hardcoded default. This has been fixed so any room size you enter uses the
  correctly rescaled floor plan geometry throughout.

**4. Models retrained against the exact library versions in `requirements.txt`** to
   remove `InconsistentVersionWarning` noise from pickled scikit-learn objects.

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
