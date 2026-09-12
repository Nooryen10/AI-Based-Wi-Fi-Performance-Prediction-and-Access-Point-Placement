const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak, LevelFormat,
} = require("docx");

const path = require("path");
const PROJECT_ROOT = path.resolve(__dirname, "..");
const OUT = path.join(PROJECT_ROOT, "outputs");

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 150 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}
function figure(path, widthPx, heightPx, captionText) {
  return [
    new Paragraph({
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: widthPx, height: heightPx } })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 150, after: 80 },
    }),
    caption(captionText),
  ];
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E79" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}

function resultsTable() {
  const header = new TableRow({
    children: ["Target Metric", "Model", "R²", "RMSE", "MAE"].map((t) => cell(t, { header: true, width: 1800 })),
  });
  const rows = [
    ["Signal Strength (dBm)", "Linear Regression", "0.8587", "5.5854", "4.1268"],
    ["Signal Strength (dBm)", "Random Forest", "0.9758", "2.3105", "1.7461"],
    ["Signal Strength (dBm)", "XGBoost (Best)", "0.9785", "2.1777", "1.6589"],
    ["Throughput (Mbps)", "Linear Regression", "0.5000", "28.7668", "16.6576"],
    ["Throughput (Mbps)", "Random Forest", "0.9606", "8.0768", "4.6208"],
    ["Throughput (Mbps)", "XGBoost (Best)", "0.9758", "6.3241", "4.0449"],
    ["Latency (ms)", "Linear Regression", "0.4939", "87.3050", "62.3736"],
    ["Latency (ms)", "Random Forest", "0.9996", "2.5321", "1.9701"],
    ["Latency (ms)", "XGBoost (Best)", "0.9996", "2.3470", "1.8466"],
  ].map((r) => new TableRow({ children: r.map((t) => cell(t, { width: 1800 })) }));

  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: [1800, 1800, 1800, 1800, 1800], rows: [header, ...rows] });
}

function placementTable() {
  const header = new TableRow({
    children: ["Scenario", "Router Position (x, y)", "Coverage @ -65 dBm"].map((t) => cell(t, { header: true, width: 3000 })),
  });
  const rows = [
    ["BEFORE — Naive corner placement", "(2.0 m, 2.0 m)", "70.9%"],
    ["AFTER — Optimized placement (2.4GHz)", "(16.0 m, 8.0 m)", "100.0%"],
    ["Optimized placement (5GHz)", "(18.0 m, 9.0 m)", "93.6%"],
  ].map((r) => new TableRow({ children: r.map((t) => cell(t, { width: 3000 })) }));
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: [3000, 3000, 3000], rows: [header, ...rows] });
}

function bandTable() {
  const header = new TableRow({
    children: ["Frequency Band", "Avg. Signal (dBm)", "Avg. Throughput (Mbps)", "Avg. Latency (ms)"].map((t) => cell(t, { header: true, width: 2250 })),
  });
  const rows = [
    ["2.4GHz", "-66.88", "23.20", "81.49"],
    ["5GHz", "-75.50", "45.14", "87.29"],
  ].map((r) => new TableRow({ children: r.map((t) => cell(t, { width: 2250 })) }));
  return new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: [2250, 2250, 2250, 2250], rows: [header, ...rows] });
}

const doc = new Document({
  numbering: {
    config: [{ reference: "bullet-points", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT }] }],
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: [
        new Paragraph({
          children: [new TextRun({ text: "AI-Based Wi-Fi Performance Prediction and", bold: true, size: 40 })],
          alignment: AlignmentType.CENTER, spacing: { before: 2000, after: 50 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Access Point Placement", bold: true, size: 40 })],
          alignment: AlignmentType.CENTER, spacing: { after: 400 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "A Machine Learning and Computer Networks Approach to Indoor Wi-Fi Optimization", italics: true, size: 24, color: "444444" })],
          alignment: AlignmentType.CENTER, spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "B.Tech Project Report — Artificial Intelligence & Machine Learning", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 100 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Third Year", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 2000 },
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- ABSTRACT ----------------
        h1("Abstract"),
        p(
          "Indoor Wi-Fi performance is governed by a complex interaction of distance, physical " +
          "obstacles, user load, and interference, making manual router placement largely a matter " +
          "of guesswork. This project presents an end-to-end system that (1) predicts Wi-Fi signal " +
          "strength, throughput, and latency using supervised machine learning trained on a physics-" +
          "informed synthetic dataset of 25,000 samples, and (2) recommends optimal Access Point (AP) " +
          "placement in a given indoor floor plan using a geometry-aware coverage simulation and " +
          "exhaustive search optimization. Three regression models — Linear Regression, Random Forest, " +
          "and XGBoost — were trained and compared; XGBoost achieved the best performance across all " +
          "three targets (R² of 0.9785, 0.9758, and 0.9996 for signal strength, throughput, and latency " +
          "respectively). The placement module models real wall geometry using 2D ray-based wall-" +
          "crossing detection and demonstrates a coverage improvement from 70.9% (naive corner " +
          "placement) to 100% (optimized placement) on a representative floor plan. The system is " +
          "exposed through both a command-line interface and a Flask web application for practical use."
        ),

        // ---------------- PROBLEM STATEMENT ----------------
        h1("1. Problem Statement"),
        p(
          "Home and office Wi-Fi routers are typically placed based on convenience (e.g., near the " +
          "point where the ISP cable enters the building) rather than on any analysis of coverage. " +
          "This frequently results in dead zones, poor throughput in far rooms, and inconsistent " +
          "latency for real-time applications such as video calls and gaming. Two distinct but related " +
          "problems are addressed in this project:"
        ),
        bullet("Prediction: Given environmental conditions (distance, obstacles, user load, interference, frequency band), estimate the resulting signal strength, throughput, and latency."),
        bullet("Placement: Given a floor plan with walls and rooms, determine the AP position that maximizes usable coverage area."),
        p("Both problems are tackled computationally so that the reasoning is reproducible, data-driven, and independent of guesswork."),

        // ---------------- OBJECTIVES ----------------
        h1("2. Objectives"),
        bullet("Build a physics-informed synthetic dataset simulating realistic indoor Wi-Fi behaviour."),
        bullet("Train and compare multiple regression models to predict signal strength, throughput, and latency."),
        bullet("Design a geometry-aware AP placement optimization algorithm using real wall layouts."),
        bullet("Visualize Wi-Fi coverage using heatmaps, correlation plots, and comparative graphs."),
        bullet("Provide a usable CLI and web interface for practical demonstration."),

        // ---------------- METHODOLOGY ----------------
        h1("3. Methodology"),
        h2("3.1 Dataset Generation"),
        p(
          "A real-world dataset controlling for distance, wall count, user load, interference, and " +
          "frequency band simultaneously does not exist in the public domain (public indoor datasets " +
          "such as UJIIndoorLoc are built for localization, not controlled performance study). A " +
          "synthetic dataset of 25,000 samples was therefore generated using three well-established " +
          "wireless propagation models:"
        ),
        bullet("Log-distance Path Loss Model: PL(d) = PL₀ + 10·n·log₁₀(d/d₀), with path loss exponent n = 2.7 (2.4GHz) / 3.2 (5GHz)."),
        bullet("Wall Attenuation Factor (WAF): an additional dB penalty per wall crossed (3.5 dB for 2.4GHz, 5.5 dB for 5GHz, reflecting 5GHz's shorter wavelength and higher material absorption)."),
        bullet("Shannon-capacity-style throughput derivation, de-rated for MAC-layer contention (∝ 1/√users) and interference."),
        bullet("M/M/1-queue-inspired latency model, where queuing delay grows non-linearly as channel utilization approaches saturation."),
        p("Gaussian noise was added to each target to simulate real-world measurement variability, and ~1% of feature values were deliberately set to missing to make the preprocessing stage meaningful."),

        h2("3.2 Data Preprocessing"),
        bullet("Missing numeric features were imputed using median imputation; rows with missing target values were dropped (never imputed, to preserve label integrity)."),
        bullet("Categorical features (interference_level, frequency_band) were one-hot encoded."),
        bullet("Numeric features were standardized (StandardScaler), fit only on the training split to prevent data leakage."),
        bullet("An 80/20 train-test split was used throughout."),

        h2("3.3 Model Training and Comparison"),
        p(
          "A separate regressor was trained per target rather than a single multi-output model, since " +
          "each target has a distinct noise structure and non-linearity profile. Three models were " +
          "compared per target: Linear Regression (baseline), Random Forest Regressor, and XGBoost " +
          "Regressor, evaluated on R², RMSE, and MAE."
        ),

        h2("3.4 Access Point Placement Algorithm"),
        p(
          "The placement module represents an indoor floor plan as a set of 2D line segments (walls). " +
          "For any candidate router position, the number of walls crossed en route to every point on a " +
          "fine grid is computed using the classical orientation-based segment-intersection test from " +
          "computational geometry — the same technique used in basic ray-tracing propagation tools. " +
          "This wall count feeds into the same path-loss model used for dataset generation, giving a " +
          "geometrically consistent signal-strength heatmap. Coverage is defined as the percentage of " +
          "grid cells at or above a usable signal threshold (-65 dBm). An exhaustive grid search over " +
          "candidate router positions (spaced 1m apart) is used to find the position that maximizes " +
          "coverage — chosen over a greedy hill-climb because the search space is small enough for " +
          "brute force to be fast (2-4 seconds) while guaranteeing no local-optimum traps behind walls."
        ),

        // ---------------- IMPLEMENTATION ----------------
        h1("4. Implementation"),
        p("The system was implemented entirely in Python, organized into independent, testable modules:"),
        bullet("data/generate_dataset.py — synthetic dataset generation (NumPy, Pandas)"),
        bullet("preprocessing/preprocess.py — cleaning, encoding, scaling, splitting (scikit-learn)"),
        bullet("models/train_models.py — model training, comparison, persistence (scikit-learn, XGBoost, joblib)"),
        bullet("visualization/visualize.py, bonus_analysis.py — all graphs and heatmaps (Matplotlib, Seaborn)"),
        bullet("placement/ap_placement.py — floor plan, geometry-based coverage simulation, optimizer"),
        bullet("app/interface.py — command-line interface tying prediction + placement together"),
        bullet("app/flask_app.py — web interface (Flask) reusing the same underlying functions"),

        // ---------------- RESULTS ----------------
        h1("5. Results"),
        h2("5.1 Model Comparison"),
        p("The table below summarizes test-set performance for all three models across all three prediction targets."),
        resultsTable(),
        p(""),
        p(
          "XGBoost was selected as the best model for all three targets. The most notable result is on " +
          "latency: Linear Regression achieves only R² = 0.49, because latency includes a queuing-delay " +
          "term that grows non-linearly as user load approaches channel saturation — a relationship a " +
          "linear model cannot represent. Both tree-based ensembles capture this non-linearity, with " +
          "XGBoost reaching R² = 0.9996."
        ),

        h2("5.2 Visualizations"),
        ...figure(`${OUT}/01_signal_vs_distance.png`, 550, 420, "Figure 1: Signal strength decays with distance; 5GHz attenuates faster than 2.4GHz."),
        ...figure(`${OUT}/02_throughput_vs_users.png`, 550, 420, "Figure 2: Throughput drops as concurrent users increase, worsened by higher interference."),
        ...figure(`${OUT}/03_correlation_heatmap.png`, 500, 420, "Figure 3: Correlation heatmap — num_users correlates strongly with latency (0.70), consistent with the queuing model."),
        ...figure(`${OUT}/09_band_comparison_boxplots.png`, 580, 200, "Figure 4: 2.4GHz vs 5GHz — 5GHz shows weaker average signal but a higher throughput ceiling."),
        new Paragraph({ children: [new PageBreak()] }),

        h2("5.3 Access Point Placement Results"),
        p(
          "A representative 35m × 25m floor plan (5 rooms + a central hallway) was used to demonstrate " +
          "the placement optimizer. The table and figures below compare a naive corner placement " +
          "(a common real-world mistake — placing the router wherever the ISP cable enters the home) " +
          "against the algorithm-recommended optimal placement."
        ),
        placementTable(),
        p(""),
        ...figure(`${OUT}/05_before_naive_placement.png`, 480, 380, "Figure 5: BEFORE — naive corner placement, 70.9% coverage. Far rooms fall below the usable-signal threshold."),
        ...figure(`${OUT}/06_after_optimized_placement.png`, 480, 380, "Figure 6: AFTER — optimized central placement, 100% coverage across the floor plan."),
        ...figure(`${OUT}/07_placement_search_surface.png`, 480, 380, "Figure 7: Coverage achievable from every candidate AP position. Isolated corner rooms (e.g., the bathroom) reduce achievable coverage regardless of AP position, informing realistic expectations."),

        h2("5.4 Frequency Band Comparison (Aggregate)"),
        bandTable(),
        p(""),
        p(
          "5GHz consistently shows weaker average signal strength but higher average throughput — the " +
          "well-known range-vs-speed trade-off of the two bands. This is reflected in the placement " +
          "results too: the optimal 5GHz position only achieves 93.6% coverage on the same floor plan " +
          "where 2.4GHz reaches 100%, implying that 5GHz deployments in larger or wall-dense homes " +
          "often need a second AP or a mesh system to match 2.4GHz's footprint."
        ),

        // ---------------- CONCLUSION ----------------
        h1("6. Conclusion and Future Scope"),
        p(
          "This project demonstrates that combining physics-grounded simulation with machine learning " +
          "produces a Wi-Fi performance predictor that is both accurate (R² > 0.97 for signal and " +
          "throughput, R² > 0.999 for latency using XGBoost) and interpretable, since every input " +
          "feature maps to a real, explainable RF phenomenon. Pairing this with a geometry-aware " +
          "placement optimizer turns an ad-hoc decision (\"where do I put the router?\") into a " +
          "measurable, optimizable one — improving coverage from 70.9% to 100% in the demonstrated " +
          "scenario."
        ),
        h2("Future Scope"),
        bullet("Multi-AP placement optimization (mesh network design) using set-cover or genetic algorithms."),
        bullet("Incorporating real sensor-collected RSSI data (e.g., via a mobile app) to validate/calibrate the synthetic model."),
        bullet("3D propagation modeling to account for multi-floor buildings."),
        bullet("Real-time adaptive channel/band selection based on live interference measurements."),

        h1("References"),
        p("1. Rappaport, T. S. \"Wireless Communications: Principles and Practice.\" Prentice Hall."),
        p("2. IEEE 802.11 Standards Documentation."),
        p("3. scikit-learn, XGBoost, Matplotlib, Seaborn, Flask — official documentation."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(__dirname, "Wifi_AI_Project_Report.docx"), buffer);
  console.log("Report written.");
});
