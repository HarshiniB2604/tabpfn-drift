/* Sections 5 (Results), 6 (What we found) and 7 (Limitations).
   Every number comes from window.RESULTS (site/data.js); this file only picks words and formats.
   The chosen dataset is kept in the URL hash (#results-bike etc.) so a view can be shared. */
(function () {
  "use strict";
  var R = window.RESULTS, F = window.FMT, el = F.el;
  var S = R.settings;
  var DEFAULT_ON = ["P0_vanilla", "P1_window", "P4_valguided", "P3_drift", "XGBoost"];   // chips on at start
  var state = { ds: null, on: {}, flags: true };
  DEFAULT_ON.forEach(function (k) { state.on[k] = true; });
  var lineChart = null, barChart = null;

  function replay() { return el("span", { class: "replay", text: "Replay of saved results" }); }
  function rows(ds) { return R.summary.filter(function (r) { return r.stream === ds; }); }
  function row(ds, key) { return F.summaryRow(ds, key); }
  function sample(m) { return window.SECTIONS.lineSample(m); }

  /* "TabPFN policies: 3 seeds (Validation: 1 seed); baselines: 3 seeds" */
  function seedsLine(ds) {
    var out = [], warn = false;
    [["tabpfn", "TabPFN policies"], ["baseline", "baselines"]].forEach(function (k) {
      var rs = R.methods.filter(function (m) { return m.kind === k[0]; }).map(function (m) { return row(ds, m.key); }).filter(Boolean);
      if (!rs.length) { out.push(k[1] + ": no results yet"); return; }
      var counts = {}; rs.forEach(function (r) { counts[r.n_seeds] = (counts[r.n_seeds] || 0) + 1; });
      var most = +Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; })[0];
      var odd = rs.filter(function (r) { return r.n_seeds !== most; })
                  .map(function (r) { return F.method(r.method).label + ": " + F.seeds(r.n_seeds); });
      if (rs.some(function (r) { return r.n_seeds < 2; })) warn = true;
      out.push(k[1] + ": " + F.seeds(most) + (odd.length ? " (" + odd.join(", ") + ")" : ""));
    });
    return out.join("; ") + "." + (warn ? " A method with 1 seed is a single run: no confidence interval, so no claim that it is better or worse." : "");
  }

  /* ------------------------------------------------------------------ results: switcher */
  function buildSwitch() {
    var box = document.getElementById("ds-switch");
    R.datasets.forEach(function (d) {
      var id = "ds-" + d.key;
      var input = el("input", { type: "radio", name: "dataset", id: id, value: d.key, class: "visually-hidden" });
      input.addEventListener("change", function () { if (input.checked) select(d.key, true); });
      box.appendChild(el("span", { class: "seg" }, input,
        el("label", { for: id }, d.label, el("span", { class: "seg-kind", text: d.kind === "real" ? "real" : "synthetic" }))));
    });
  }

  function fromHash() {
    var m = /^#results-([a-z0-9]+)$/.exec(location.hash);
    return m && F.dataset(m[1]) ? m[1] : null;
  }

  function select(key, fromUser) {
    state.ds = key;
    document.getElementById("ds-" + key).checked = true;
    if (fromUser && history.replaceState) history.replaceState(null, "", "#results-" + key);
    render();
  }

  /* ------------------------------------------------------------------ results: one dataset */
  function render() {
    var d = F.dataset(state.ds), real = d.kind === "real";
    var avail = R.methods.filter(function (m) { return R.curves[d.key] && R.curves[d.key][m.key] && row(d.key, m.key); });

    // intro: story, what kind of data, replay + seeds
    var intro = document.getElementById("results-intro");
    intro.textContent = "";
    intro.appendChild(el("p", { class: "ds-story" }, el("b", { text: d.label + ". " }), d.story));
    intro.appendChild(el("p", { class: "meta" }, replay(),
      (real ? "Real data: the true drift points are unknown, so there is no drift line and no recovery time. "
            : (d.drift_blocks.length ? "Synthetic data: we made the drift, so the true drift points are marked. "
                                     : "Synthetic control: nothing drifts, so this shows what each policy costs when it is not needed. ")) +
      seedsLine(d.key)));

    // chips
    var chips = document.getElementById("method-chips");
    chips.textContent = "";
    avail.forEach(function (m) {
      var r = row(d.key, m.key);
      var b = el("button", { type: "button", class: "chip chip-" + m.kind, "aria-pressed": state.on[m.key] ? "true" : "false" },
        sample(m), el("span", { text: m.label }), r.n_seeds < 2 ? el("span", { class: "chip-note", text: "1 seed" }) : null);
      b.addEventListener("click", function () { state.on[m.key] = !state.on[m.key]; render(); });
      chips.appendChild(b);
    });
    var adwOn = !!state.on.P3_drift && avail.some(function (m) { return m.key === "P3_drift"; });
    document.getElementById("flag-toggle-wrap").hidden = !adwOn;

    // chart
    var shown = avail.filter(function (m) { return state.on[m.key]; });
    var series = shown.map(function (m) {
      var c = R.curves[d.key][m.key], markers = [];
      if (m.key === "P3_drift" && state.flags) {
        var set = {};
        Object.keys(c.flags || {}).forEach(function (s) { c.flags[s].forEach(function (b) { set[b] = true; }); });
        markers = Object.keys(set).map(Number);
      }
      return { key: m.key, label: m.label, short: m.short, kind: m.kind, dash: m.dash, x: c.blocks, y: c.mean,
               ci: c.ci95, n: c.n_seeds, markers: markers };
    });
    var vlines = [], vrects = [];
    if (!real && d.gradual) vrects.push({ x0: d.gradual[0], x1: d.gradual[1], label: "gradual change (true)" });
    else if (!real) d.drift_blocks.forEach(function (b) { vlines.push({ x: b, label: "true drift (block " + b + ")", short: "drift " + b }); });
    var unit = d.unit ? ", " + d.unit : "";
    var opts = {
      series: series, vlines: vlines, vrects: vrects, endLabels: true,
      xLabel: "block (" + S.block_size + " rows each)", xName: "Block",
      yLabel: real ? "RMSE over the last " + S.rolling_window + " blocks" + unit + " (lower is better)" : "RMSE per block (lower is better)",
      emptyText: "Pick at least one method above",
      ariaLabel: "Line chart of error over time on " + d.label + " for " + shown.map(function (m) { return m.label; }).join(", ") +
        ". The table below the chart lists every value.",
      formatY: function (s, i) { return F.withCI(s.y[i], s.ci[i], s.n); },
    };
    if (lineChart) lineChart.update(opts); else lineChart = window.CHARTS.lineChart(document.getElementById("results-chart"), opts);

    document.getElementById("results-chart-title").textContent = "Error over time: " + d.label + " · " + (real ? "real data" : "synthetic");
    var note = real
      ? "Each point is the RMSE over the last " + S.rolling_window + " blocks (" + S.rolling_window * S.block_size +
        " rows), so lines start at block " + (S.rolling_window) + ". Line = mean over seeds. No drift line: real drift points are unknown."
      : "Each point is the RMSE on one block of " + S.block_size + " new rows, mean over seeds. " +
        (d.gradual ? "Shaded: the true gradual change." : d.drift_blocks.length ? "Dashed: true drift." : "No drift in this stream.");
    if (adwOn && state.flags) note += " Triangles: blocks where ADWIN flagged a drift (in any seed).";
    document.getElementById("results-chart-note").textContent = note;

    curveTable(d, shown);
    resultsTable(d, avail);
    bars(d, avail);
  }

  function curveTable(d, shown) {
    var box = document.getElementById("results-curve-table");
    box.textContent = "";
    if (!shown.length) return;
    var c0 = R.curves[d.key][shown[0].key];
    var head = el("tr", null, el("th", { scope: "col", text: "Block" }));
    shown.forEach(function (m) { head.appendChild(el("th", { scope: "col", text: m.label })); });
    var body = el("tbody");
    c0.blocks.forEach(function (b, i) {
      var tr = el("tr", null, el("td", { text: String(b) }));
      shown.forEach(function (m) { var c = R.curves[d.key][m.key]; tr.appendChild(el("td", { text: c.mean[i] === null ? "–" : F.num(c.mean[i]) })); });
      body.appendChild(tr);
    });
    box.appendChild(el("table", null, el("caption", { class: "visually-hidden", text: "Error per block, " + d.label }), el("thead", null, head), body));
    document.getElementById("results-curve-details").open = false;
  }

  function resultsTable(d, avail) {
    var real = d.kind === "real", drift = !real && d.drift_blocks.length > 0, control = !real && !drift;
    var cols = ["Method", "Kind", "Seeds", "Mean RMSE (95% CI)"];
    if (drift) cols.push("Recovery time, blocks (95% CI)", "Drifts recovered");
    if (control) cols.push("Cost vs vanilla, % (95% CI)");
    cols.push("Drift flags per run");
    var rs = avail.map(function (m) { return { m: m, r: row(d.key, m.key) }; })
                  .sort(function (a, b) { return a.r.rmse_mean - b.r.rmse_mean; });
    var body = el("tbody");
    rs.forEach(function (x) {
      var r = x.r, m = x.m;
      var tr = el("tr", null,
        el("th", { scope: "row", class: "method-cell" }, sample(m), el("span", { text: m.label })),
        el("td", { class: "col-kind", text: m.kind === "baseline" ? "baseline" : "TabPFN policy" }),
        el("td", { text: String(r.n_seeds) }),
        el("td", { text: F.withCI(r.rmse_mean, r.rmse_ci95, r.n_seeds) }));
      if (drift) {
        var ev = String(r.recovered_events).split("/").map(Number);
        var rec = F.withCI(r.recovery_mean, r.recovery_ci95, r.n_seeds, 1);
        var cell = el("td", null, rec);
        if (ev[0] < ev[1]) cell.appendChild(el("span", { class: "flag-capped",
          text: ev[0] === 0 ? "not recovered (capped)" : "capped: not recovered in " + (ev[1] - ev[0]) + " of " + ev[1] }));
        tr.appendChild(cell);
        tr.appendChild(el("td", { text: r.recovered_events }));
      }
      if (control) tr.appendChild(el("td", { text: r.method === "P0_vanilla" ? "reference"
        : (r.regret_pct_mean > 0 ? "+" : "") + F.withCI(r.regret_pct_mean, r.regret_pct_ci95, r.n_seeds, 1) }));
      tr.appendChild(el("td", { text: r.method === "P3_drift" ? F.num(r.drift_flags_mean, 1) : "–" }));
      body.appendChild(tr);
    });
    var head = el("tr");
    cols.forEach(function (c) { head.appendChild(el("th", { scope: "col", class: c === "Kind" ? "col-kind" : null, text: c })); });
    var box = document.getElementById("results-table");
    box.textContent = "";
    box.appendChild(el("table", { class: "results" }, el("caption", { class: "visually-hidden", text: "Results for " + d.label }),
      el("thead", null, head), body));
    document.getElementById("results-table-note").textContent =
      "Best mean RMSE first. Mean ± 95% interval across seeds (rough with 3 seeds); “1 seed, no CI” where a method ran once." +
      (drift ? " Recovery: blocks until the error is back within " + S.tolerance_pct + "% of the pre-drift level; if it never gets back the value is the cap and is marked." : "") +
      (control ? " Cost vs vanilla: how much worse than vanilla with no drift, paired by seed." : "");
  }

  function bars(d, avail) {
    var list = avail.map(function (m) {
      var r = row(d.key, m.key);
      return { label: m.label, short: m.short, kind: m.kind, value: r.rmse_mean, ci: r.n_seeds < 2 ? null : r.rmse_ci95,
               text: F.withCI(r.rmse_mean, r.rmse_ci95, r.n_seeds) };
    });
    var opts = { bars: list, xLabel: "mean RMSE" + (d.unit ? ", " + d.unit : "") + " (lower is better)",
                 ariaLabel: "Bar chart of mean RMSE per method on " + d.label + "; values are in the results table." };
    if (barChart) barChart.update(opts); else barChart = window.CHARTS.barChart(document.getElementById("results-bars"), opts);
    document.getElementById("results-bars-note").textContent =
      "Blue: TabPFN context policies. Orange: baselines. Whiskers: 95% interval across seeds; none where a method ran once.";
  }

  /* ------------------------------------------------------------------ findings */
  function buildFindings() {
    ["finding-cards", "robust-table", "sens-body"].forEach(function (id) { document.getElementById(id).textContent = ""; });
    document.getElementById("findings-note").textContent =
      "Each finding is computed from the saved results when the site is built; its wording depends on the numbers, " +
      "so a claim only appears if the data supports it.";
    var list = document.getElementById("finding-cards");
    R.findings.forEach(function (f) {
      var li = el("li", { class: "card finding" },
        el("h3", { text: f.headline }), el("p", { text: f.body }),
        f.real ? el("p", { text: f.real }) : null,
        f.caveat ? el("p", { class: "caveat" }, el("b", { text: "Caveat: " }), f.caveat) : null);
      var ev = el("p", { class: "meta" }, replay());
      f.evidence.forEach(function (e, i) { ev.appendChild(document.createTextNode((i ? " " : "") + e.text)); });
      li.appendChild(ev);
      list.appendChild(li);
    });

    // robustness: % gap to the best TabPFN policy on each real dataset
    var sets = Object.keys(R.robustness);
    document.getElementById("robust-note").textContent =
      "For each real dataset: how much higher each TabPFN policy's mean RMSE is than the best TabPFN policy there " +
      "(0% = best). Computed from the shown means. This is a ranking, not a significance test.";
    var head = el("tr", null, el("th", { scope: "col", text: "Policy" }));
    sets.forEach(function (ds) { head.appendChild(el("th", { scope: "col", text: F.dataset(ds).label })); });
    var body = el("tbody");
    R.methods.filter(function (m) { return m.kind === "tabpfn"; }).forEach(function (m) {
      var tr = el("tr", null, el("th", { scope: "row", class: "method-cell" }, sample(m), el("span", { text: m.label })));
      sets.forEach(function (ds) {
        var r = R.robustness[ds].filter(function (x) { return x.method === m.key; })[0];
        if (!r) { tr.appendChild(el("td", { text: "–" })); return; }
        tr.appendChild(el("td", { class: r.best ? "best" : null },
          r.best ? "best" : "+" + F.num(r.gap_pct, 1) + "%", r.n_seeds < 2 ? el("span", { class: "cell-note", text: " (1 seed)" }) : null));
      });
      body.appendChild(tr);
    });
    document.getElementById("robust-table").appendChild(el("table", { class: "results" },
      el("caption", { class: "visually-hidden", text: "Gap to the best TabPFN policy per real dataset" }), el("thead", null, head), body));

    // ADWIN sensitivity: only if the ablation exists (data-driven hook)
    var A = R.adwin_sensitivity;
    if (!A) return;
    document.getElementById("sensitivity").hidden = false;
    document.getElementById("sens-title").textContent = A.title;
    document.getElementById("sens-note").textContent = A.intro;
    var b = document.getElementById("sens-body");
    b.appendChild(el("p", { class: "meta" }, replay(), "From results_ablation/ (ADWIN only, other δ values)."));
    if (A.summary) b.appendChild(el("p", { class: "sens-summary", text: A.summary }));
    var ul = el("ul", { class: "sens-points" });
    A.points.forEach(function (p) { ul.appendChild(el("li", { text: p })); });
    b.appendChild(ul);
    var th = el("tr");
    ["Dataset", "δ", "Seeds", "Mean RMSE (95% CI)", "Drift flags per run", "vs vanilla, same seeds"].forEach(function (c) { th.appendChild(el("th", { scope: "col", text: c })); });
    var tb = el("tbody");
    A.table.forEach(function (r) {
      tb.appendChild(el("tr", null, el("th", { scope: "row", text: F.dataset(r.dataset).label }), el("td", { text: r.delta }),
        el("td", { text: String(r.seeds) }), el("td", { text: F.withCI(r.rmse, r.rmse_ci95, r.seeds) }),
        el("td", { text: F.num(r.flags, 1) + " of " + r.blocks + " blocks" }),
        el("td", { text: (r.vs_vanilla_pct > 0 ? "+" : "") + F.num(r.vs_vanilla_pct, 1) + "%" +
          (r.seeds < 2 || r.vs_vanilla_ci95 === null ? " (1 seed, no CI)" : " ± " + F.num(r.vs_vanilla_ci95, 1)) })));
    });
    b.appendChild(el("div", { class: "table-scroll" }, el("table", { class: "results" },
      el("caption", { class: "visually-hidden", text: "ADWIN sensitivity table" }), el("thead", null, th), tb)));
    b.appendChild(el("p", { class: "caveat" }, el("b", { text: "Caveat: " }), A.caveat));
  }

  /* ------------------------------------------------------------------ limitations */
  function buildLimitations() {
    var ol = document.getElementById("limit-list");
    R.limitations.forEach(function (l) {
      ol.appendChild(el("li", { class: "card limit" }, el("h3", { text: l.title }), el("p", { text: l.text })));
    });
  }

  /* Tables that may scroll sideways must be reachable with the keyboard (arrow keys scroll them). */
  function keyboardScroll() {
    document.querySelectorAll(".table-scroll").forEach(function (box) {
      if (box.hasAttribute("tabindex")) return;
      var card = box.closest(".card, details"), h = card && card.querySelector("h3, summary");
      if (!h && card && card.previousElementSibling) h = card.previousElementSibling.querySelector("h3");   // e.g. robustness
      box.setAttribute("tabindex", "0");
      box.setAttribute("role", "region");
      box.setAttribute("aria-label", (h ? h.textContent : "Table") + " (scrollable)");
    });
  }

  /* ------------------------------------------------------------------ start */
  buildSwitch();
  document.getElementById("flag-toggle").addEventListener("change", function (e) { state.flags = e.target.checked; render(); });
  var start = fromHash();
  select(start || R.datasets[0].key, false);
  if (start) window.addEventListener("load", function () {
    document.documentElement.style.scrollBehavior = "auto";
    document.getElementById("results").scrollIntoView();
    document.documentElement.style.scrollBehavior = "";
  });
  window.addEventListener("hashchange", function () { var k = fromHash(); if (k && k !== state.ds) select(k, false); });
  window.addEventListener("themechange", function () { render(); buildFindings(); });   // line samples use theme colours
  buildFindings();
  buildLimitations();
  keyboardScroll();
  window.addEventListener("themechange", keyboardScroll);   // the sensitivity table is rebuilt
})();
