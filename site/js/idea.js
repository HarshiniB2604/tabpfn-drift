/* Sections 3 ("The idea") and 4 ("How we tested").
   Every setting in the text (context size, window, half-life, delta, candidates, block size, seeds,
   feature counts, ...) comes from window.RESULTS, which make_site.py reads from the code.
   The block strips on the policy cards are ILLUSTRATIONS (labelled so), not results. */
(function () {
  "use strict";
  var R = window.RESULTS, F = window.FMT, el = F.el;
  var S = R.settings, PS = R.policy_settings;

  /* ------------------------------------------------------------ small drawing helpers */
  function lineSample(method) {
    // a short line in the method's colour and dash style: the same key the result charts use
    var s = F.svg("svg", { class: "line-sample", viewBox: "0 0 44 12", width: 44, height: 12, "aria-hidden": "true" });
    s.appendChild(F.svg("line", { x1: 2, x2: 42, y1: 6, y2: 6, "stroke-width": 2.5,
      stroke: F.cssVar(method.kind === "baseline" ? "--accent-baseline" : "--accent-tabpfn"),
      "stroke-dasharray": method.dash || null }));
    return s;
  }

  var N_PAST = 12;   // blocks drawn in each illustration strip (a drawing choice, not a setting)

  /* weights 0..1 per past block (oldest first) for each policy's illustration */
  function illustrationWeights(key) {
    var w = [], i;
    if (key === "P0_vanilla") { for (i = 0; i < N_PAST; i++) w.push(0.5); return { w: w }; }
    if (key === "P1_window") {
      for (i = 0; i < N_PAST; i++) w.push(i >= N_PAST - PS.P1_window.window ? 1 : 0);
      return { w: w };
    }
    if (key === "P2_recency") {
      for (i = 0; i < N_PAST; i++) w.push(Math.pow(0.5, (N_PAST - 1 - i) / PS.P2_recency.half_life));
      return { w: w };
    }
    if (key === "P3_drift") {
      var cut = Math.round(N_PAST * 0.6);   // where the (illustrative) drift is flagged
      for (i = 0; i < N_PAST; i++) w.push(i >= cut ? 0.5 : 0);
      return { w: w, marker: cut, markerLabel: "drift flagged" };
    }
    if (key === "P4_valguided") {
      // show one candidate rule being used: the shortest window among the candidates
      var wins = PS.P4_valguided.candidates.map(function (c) { var m = /^last (\d+) blocks$/.exec(c); return m ? +m[1] : null; })
                   .filter(function (x) { return x; });
      var k = Math.min.apply(null, wins);
      for (i = 0; i < N_PAST; i++) w.push(i >= N_PAST - k ? 1 : 0);
      return { w: w, scored: PS.P4_valguided.n_val_blocks, note: "currently picked: last " + k + " blocks" };
    }
    if (key === "kNN_context") {
      var similar = [1, 4, 5, 8, 10];        // illustrative: blocks whose inputs look like the new block
      for (i = 0; i < N_PAST; i++) w.push(similar.indexOf(i) >= 0 ? 1 : 0);
      return { w: w };
    }
    return { w: w };
  }

  function strip(key, ariaText) {
    var ill = illustrationWeights(key);
    var cell = 18, gap = 4, newGap = 12, W = N_PAST * (cell + gap) + newGap + cell + 2, H = 46;
    var root = F.svg("svg", { class: "strip", viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": ariaText });
    var blue = F.cssVar("--accent-tabpfn"), line = F.cssVar("--line-strong"), ink2 = F.cssVar("--ink-2");
    var top = 14;
    ill.w.forEach(function (v, i) {
      var x = 1 + i * (cell + gap);
      root.appendChild(F.svg("rect", { x: x, y: top, width: cell, height: cell, rx: 3, fill: blue, "fill-opacity": v * 0.9,
                                       stroke: v > 0 ? blue : line, "stroke-width": 1 }));
      if (ill.scored && i >= N_PAST - ill.scored) {
        root.appendChild(F.svg("rect", { x: x - 2, y: top - 2, width: cell + 4, height: cell + 4, rx: 4, fill: "none",
                                         stroke: F.cssVar("--ink"), "stroke-width": 1.5, "stroke-dasharray": "3 2" }));
      }
    });
    var nx = 1 + N_PAST * (cell + gap) + newGap - gap;
    root.appendChild(F.svg("rect", { x: nx, y: top, width: cell, height: cell, rx: 3, fill: "none", stroke: F.cssVar("--ink"),
                                     "stroke-width": 1.5 }));
    var t = F.svg("text", { x: nx + cell / 2, y: top + cell + 13, "text-anchor": "middle", fill: ink2, "font-size": 10 });
    t.textContent = "new";
    root.appendChild(t);
    var o = F.svg("text", { x: 1, y: top + cell + 13, fill: ink2, "font-size": 10 });
    o.textContent = "oldest";
    root.appendChild(o);
    if (ill.marker !== undefined) {
      var mx = 1 + ill.marker * (cell + gap) - gap / 2;
      root.appendChild(F.svg("line", { x1: mx, x2: mx, y1: top - 4, y2: top + cell + 4, stroke: F.cssVar("--ink"), "stroke-width": 1.5 }));
      var mt = F.svg("text", { x: mx, y: 9, "text-anchor": "middle", fill: ink2, "font-size": 10 });
      mt.textContent = ill.markerLabel;
      root.appendChild(mt);
    }
    if (ill.scored) {
      var sx = 1 + (N_PAST - ill.scored) * (cell + gap);
      var st = F.svg("text", { x: sx + (ill.scored * (cell + gap) - gap) / 2, y: 9, "text-anchor": "middle", fill: ink2, "font-size": 10 });
      st.textContent = "scored on";
      root.appendChild(st);
    }
    return { svg: root, note: ill.note };
  }

  /* ------------------------------------------------------------ section 3: the idea */
  function policyText(key) {
    var p = PS[key] || {};
    switch (key) {
      case "P0_vanilla": return "A random sample of up to " + S.c_max + " rows from all of history. The starting point every other policy is compared with.";
      case "P1_window": return "Only the most recent " + p.window + " blocks. Everything older is forgotten.";
      case "P2_recency": return "A random sample from all of history, but newer rows are more likely: a row's chance halves every " +
        p.half_life + " blocks of age.";
      case "P3_drift": return "Like vanilla, but it watches its own prediction errors with the ADWIN detector (significance δ = " +
        p.delta + "). When the error rises, it forgets the history from before the change.";
      case "P4_valguided": return "Tries " + p.candidates.length + " candidate rules (" + p.candidates.join("; ") + ") on the latest " +
        p.n_val_blocks + " labelled blocks and switches to the best one only if it is more than " + F.num(p.min_gain_pct, 0) +
        "% better. Starts with " + p.start + ".";
      case "kNN_context": return "Picks the " + S.c_max + " past rows whose inputs are closest to the new block's inputs, wherever in history they are.";
    }
    return "";
  }

  function buildIdea() {
    document.getElementById("idea-lede").textContent =
      "TabPFN itself never changes and is never retrained. Before each new block, a context policy decides which " +
      "labelled rows from the past go into the model's context. That choice is the only difference between the " +
      R.counts.tabpfn_policies + " policies we compare.";

    var steps = document.getElementById("idea-steps");
    [["Collect", "Labelled rows arrive in blocks of " + S.block_size + " and are kept as history."],
     ["Choose", "A context policy picks up to " + S.c_max + " of those past rows."],
     ["Predict", "TabPFN reads the chosen rows plus the new block's inputs and predicts its answers in one pass."]
    ].forEach(function (s, i) {
      steps.appendChild(el("li", { class: "card step" },
        el("span", { class: "step-num", "aria-hidden": "true", text: String(i + 1) }),
        el("h3", { text: s[0] }), el("p", { text: s[1] })));
    });

    var grid = document.getElementById("policy-cards");
    R.methods.filter(function (m) { return m.kind === "tabpfn"; }).forEach(function (m) {
      var text = policyText(m.key);
      var st = strip(m.key, "Illustration, not data: which past blocks " + m.label + " can use. " + text);
      grid.appendChild(el("li", { class: "card policy", "data-key": m.key },
        el("div", { class: "policy-head" }, lineSample(m), el("h4", { text: m.label })),
        el("p", { text: text }),
        el("div", { class: "strip-wrap" }, st.svg),
        el("p", { class: "strip-note", text: st.note || "", "aria-hidden": st.note ? null : "true" })));
    });
  }

  /* ------------------------------------------------------------ section 4: how we tested */
  function buildHow() {
    var ttt = document.getElementById("ttt");
    ttt.appendChild(el("h3", { text: "Test-then-train" }));
    ttt.appendChild(el("p", { text: "Every method sees the stream in the same order. For each block:" }));
    var ol = el("ol", { class: "ttt-steps" });
    [["Predict", "the block's answers, using only blocks that came before it."],
     ["Score", "the predictions: RMSE on its " + S.block_size + " rows (one point on the error-over-time charts)."],
     ["Learn", "the true answers are revealed and the block joins the history."]
    ].forEach(function (s) { ol.appendChild(el("li", null, el("b", { text: s[0] + " " }), s[1])); });
    ttt.appendChild(ol);
    ttt.appendChild(el("p", { class: "footnote", text: "The first " + (R.warmup_blocks === 1 ? "block is" : R.warmup_blocks + " blocks are") +
      " warm-up only (nothing to learn from yet), so it is not scored. No method ever sees an answer before predicting it." }));

    var met = document.getElementById("metrics");
    met.appendChild(el("h3", { text: "What we measure" }));
    var dl = el("dl", { class: "metric-list" });
    [["RMSE (lower is better)", "Root mean squared error: the typical size of a prediction mistake, in the target's own units."],
     ["Recovery time", "After a known drift: blocks until the error is back within " + S.tolerance_pct + "% of the pre-drift level " +
       "(the mean of the " + S.pre_window + " blocks before the drift). If it never gets back, the time is capped and marked “not recovered”. Synthetic streams only."],
     ["No-drift cost", "On the no-drift control: how much worse (in %) a policy is than vanilla, paired by seed."],
     ["Uncertainty", "Synthetic streams are run with " + F.seeds(S.seeds.length) + "; results show the mean ± a 95% interval across seeds. " +
       "With so few seeds the interval is rough. A method run once has no interval, and we say so."]
    ].forEach(function (d) { dl.appendChild(el("dt", { text: d[0] })); dl.appendChild(el("dd", { text: d[1] })); });
    met.appendChild(dl);

    var syn = R.datasets.filter(function (d) { return d.kind === "synthetic"; });
    var real = R.datasets.filter(function (d) { return d.kind === "real"; });
    document.getElementById("syn-label").textContent = "Synthetic streams: we create the drift, so we know exactly when it happens";
    document.getElementById("real-label").textContent = "Real datasets: kept in time order; the true drift points are unknown";
    function dataCard(d) {
      var facts = d.scored_blocks + " scored blocks of " + S.block_size + " rows · " +
        (d.n_features === null ? "" : d.n_features + " input features");
      return el("li", { class: "card data-item" },
        el("div", { class: "data-head" }, el("h4", { text: d.label }),
          el("span", { class: "pill " + (d.kind === "real" ? "pill-real" : "pill-syn"),
                       text: d.kind === "real" ? "real · drift unknown" : (d.drift_blocks.length ? "synthetic · drift known" : "synthetic · no drift") })),
        el("p", { text: d.story }), el("p", { class: "meta tabular", text: facts }));
    }
    syn.forEach(function (d) { document.getElementById("data-synthetic").appendChild(dataCard(d)); });
    real.forEach(function (d) { document.getElementById("data-real").appendChild(dataCard(d)); });

    var BL = R.baseline_settings.refit_blocks;
    var base = { XGBoost: "Gradient-boosted decision trees, trained from scratch before every block on the last " + BL + " blocks.",
                 MLP: "A small neural network, trained from scratch before every block on the last " + BL + " blocks.",
                 AdaptiveForest: "An adaptive random forest (River) that learns row by row and uses its own drift detectors." };
    var bg = document.getElementById("baseline-cards");
    R.methods.filter(function (m) { return m.kind === "baseline"; }).forEach(function (m) {
      bg.appendChild(el("li", { class: "card baseline" },
        el("div", { class: "policy-head" }, lineSample(m), el("h4", { text: m.label })), el("p", { text: base[m.key] || "" })));
    });
  }

  /* redraw the theme-coloured SVGs (line samples, strips) when the theme changes */
  function rebuild() {
    ["policy-cards", "idea-steps", "ttt", "metrics", "data-synthetic", "data-real", "baseline-cards"].forEach(function (id) {
      document.getElementById(id).textContent = "";
    });
    buildIdea();
    buildHow();
  }

  window.SECTIONS = { buildIdea: buildIdea, buildHow: buildHow };
  window.addEventListener("themechange", rebuild);
})();
