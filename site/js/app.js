/* Builds the page from window.RESULTS (site/data.js, made by python -m eval.make_site).
   Rule: every number shown comes from RESULTS; this file only chooses words and formats. */
(function () {
  "use strict";
  var R = window.RESULTS, F = window.FMT, el = F.el;
  if (!R) {
    document.getElementById("main").prepend(el("p", { class: "wrap", text:
      "site/data.js is missing: run python -m eval.make_site" }));
    return;
  }

  function replayBadge() { return el("span", { class: "replay", text: "Replay of saved results" }); }

  /* ------------------------------------------------------------------ nav */
  function buildNav() {
    var list = document.getElementById("navlinks");
    var links = [];
    document.querySelectorAll("section[data-nav]").forEach(function (s) {
      var a = el("a", { href: "#" + s.id, text: s.getAttribute("data-nav") });
      list.appendChild(el("li", null, a));
      links.push({ a: a, s: s });
    });
    if (!window.IntersectionObserver) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (l) { l.a.setAttribute("aria-current", l.s === e.target ? "true" : "false"); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach(function (l) { io.observe(l.s); });
    // buttons that point at sections not built yet are hidden, never dead links
    document.querySelectorAll('a[href^="#"][hidden]').forEach(function (a) {
      if (document.querySelector(a.getAttribute("href"))) a.hidden = false;
    });
  }

  /* ------------------------------------------------------------------ hero */
  function buildHero() {
    var c = R.counts, s = R.settings;
    document.getElementById("hero-lede").textContent =
      "TabPFN is a pretrained model for tables. It is never trained on your data: it reads a set of " +
      "labelled example rows (its context, up to " + s.c_max + " rows here) and predicts new rows from them. " +
      "We compare " + c.tabpfn_policies + " simple ways to choose those rows when the data drifts, " +
      "against " + c.baselines + " models that are retrained instead.";

    var badges = document.getElementById("hero-badges");
    [c.synthetic_drift + " synthetic streams + " + c.real + " real datasets",
     c.synthetic_control + " no-drift control stream",
     "no retraining of TabPFN"].forEach(function (t) { badges.appendChild(el("li", { class: "badge", text: t })); });

    // Headline: finding 1, only if the data supports it.
    var f = R.findings.filter(function (x) { return x.id === "vanilla"; })[0];
    if (!f || !f.supported) return;
    var v = f.values, ev = f.evidence[0];
    var box = document.getElementById("hero-headline");
    box.appendChild(el("p", { class: "eyebrow", text: "Headline result" }));
    box.appendChild(el("p", { class: "big", "aria-label": F.num(v.ratio_min, 1) + " to " + F.num(v.ratio_max, 1) + " times" },
      F.num(v.ratio_min, 1) + "–" + F.num(v.ratio_max, 1), el("span", { class: "unit", text: "×" })));
    box.appendChild(el("p", { text:
      "Vanilla TabPFN's average error on the drifting streams, as a multiple of its error on the no-drift " +
      "control. It got back to its pre-drift error after only " + v.recovered + " of " + v.drifts + " drifts." }));
    box.appendChild(el("p", { class: "meta" }, replayBadge(),
      "Synthetic streams · " + F.seeds(ev.seeds[0]) + " per method · " +
      (ev.has_ci ? "95% CIs do not overlap (rough with " + F.seeds(ev.seeds[0]) + ")" : "no CI")));
    box.hidden = false;
  }

  /* ------------------------------------------------------------------ problem */
  function driftIllustration() {
    // Two tiny panels: same inputs, different rule. Points are drawn from a formula: NOT data.
    // Drawn at the real width of its box (redrawn on resize), so the text never shrinks.
    var fig = document.getElementById("drift-illus"), F2 = F.svg;
    // Each panel: a y-axis label ("answer") on its left, then the plot box. Everything is placed
    // 1px inside the SVG edges, so no border is clipped.
    var W = Math.max(280, fig.clientWidth), gapX = 14, axisW = 18, inset = 1;
    var pw = (W - gapX - 2 * axisW - 2 * inset) / 2, ph = Math.round(Math.min(230, pw * 0.95)), top = 28, H = top + ph + 30;
    var root = F2("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img",
      "aria-label": "Illustration, not data: before the change the answer rises with the input; after the change the same inputs give falling answers." });
    function panel(x0, title, slope) {
      var g = F2("g", { transform: "translate(" + x0 + ",0)" });
      var bx = axisW;   // plot box starts after the y-axis label
      var yl = F2("text", { x: 0, y: 0, transform: "translate(12," + (top + ph / 2) + ") rotate(-90)", "text-anchor": "middle",
                            fill: F.cssVar("--ink-2"), "font-size": 12 });
      yl.textContent = "answer →";
      g.appendChild(yl);
      g.appendChild(F2("rect", { x: bx, y: top, width: pw, height: ph, rx: 10, fill: F.cssVar("--surface"), stroke: F.cssVar("--line") }));
      var t = F2("text", { x: bx, y: 16, fill: F.cssVar("--ink"), "font-size": 14, "font-weight": 600 });
      t.textContent = title;
      g.appendChild(t);
      for (var i = 0; i < 14; i++) {
        var u = (i + 0.5) / 14, wobble = Math.sin(i * 2.7) * 0.08;
        var v = slope > 0 ? 0.15 + 0.7 * u + wobble : 0.85 - 0.7 * u + wobble;
        g.appendChild(F2("circle", { cx: bx + 16 + u * (pw - 32), cy: top + ph - 14 - v * (ph - 28), r: 4.5,
                                     fill: F.cssVar("--ink-2"), "fill-opacity": 0.55 }));
      }
      var y1 = slope > 0 ? 0.15 : 0.85, y2 = slope > 0 ? 0.85 : 0.15;
      g.appendChild(F2("line", { x1: bx + 16, x2: bx + pw - 16, y1: top + ph - 14 - y1 * (ph - 28), y2: top + ph - 14 - y2 * (ph - 28),
                                 stroke: F.cssVar("--accent-tabpfn"), "stroke-width": 2.5 }));
      var ax = F2("text", { x: bx + pw / 2, y: top + ph + 18, "text-anchor": "middle", fill: F.cssVar("--ink-2"), "font-size": 12 });
      ax.textContent = "input →";
      g.appendChild(ax);
      return g;
    }
    root.appendChild(panel(inset, "Before the change", 1));
    root.appendChild(panel(inset + axisW + pw + gapX, "After the change", -1));
    var old = fig.querySelector("svg");
    if (old) { fig.replaceChild(root, old); return; }   // redraw after a resize: caption stays
    fig.appendChild(root);
    if (window.ResizeObserver) {
      var lastW = fig.clientWidth;
      new ResizeObserver(function () {
        if (Math.abs(fig.clientWidth - lastW) > 2) { lastW = fig.clientWidth; driftIllustration(); }
      }).observe(fig);
    }
    window.addEventListener("themechange", driftIllustration);   // colours come from the theme
    fig.appendChild(el("figcaption", null, el("span", { class: "tag-illus", text: "Illustration, not data" }),
      " Same inputs, different answers: a model that copies the old rule is now wrong."));
  }

  function buildProblem() {
    var P = R.problem, s = R.settings;
    var ds = F.dataset(P.stream), m = F.method(P.method);
    var c = R.curves[P.stream][P.method];
    var tol = P.tolerance_pct;

    var after = P.drift_block + 1, iAfter = c.blocks.indexOf(after);
    document.getElementById("problem-why").textContent =
      m.label + " TabPFN takes a random sample of up to " + s.c_max + " rows from all the history it has seen. " +
      "When it predicts block " + after + ", " + F.num(P.old_rule_share_pct[iAfter], 0) +
      "% of that history still follows the old rule (" + P.drift_block + " of " + after + " blocks).";
    document.getElementById("problem-recovery").textContent =
      "By the report's recovery rule (back within " + tol + "% of the pre-drift level), vanilla recovered in " +
      P.recovered_events + " runs on this stream.";
    document.getElementById("problem-chart-title").textContent =
      m.label + " TabPFN on the " + ds.label.toLowerCase() + " stream (synthetic)";
    var meta = document.getElementById("problem-chart-meta");
    meta.appendChild(el("span", { class: "define-line" }, el("b", { text: "RMSE" }),
      " (root mean squared error) = the typical size of a prediction mistake; lower is better. "));
    meta.appendChild(el("span", null, replayBadge(),
      "One point = RMSE on a block of " + s.block_size + " new rows. Line = mean of " + F.seeds(c.n_seeds) +
      ", band = 95% CI across seeds. Dashed: true drift (we made it). Dotted: pre-drift level (blocks " +
      P.pre_blocks[0] + "–" + P.pre_blocks[P.pre_blocks.length - 1] + ")."));

    var lo = c.mean.map(function (v, i) { return c.ci95[i] === null ? null : v - c.ci95[i]; });
    var hi = c.mean.map(function (v, i) { return c.ci95[i] === null ? null : v + c.ci95[i]; });
    var chart = window.CHARTS.lineChart(document.getElementById("problem-chart"), {
      series: [{ label: m.label, kind: m.kind, dash: m.dash, x: c.blocks, y: c.mean, lo: lo, hi: hi }],
      vlines: [{ x: P.drift_block, label: "true drift (block " + P.drift_block + ")" }],
      hlines: [{ y: P.pre_mean }],
      xLabel: "block (" + s.block_size + " rows each)", yLabel: "RMSE per block (lower is better)", xName: "Block",
      ariaLabel: "Line chart: error of " + m.label + " TabPFN per block on the " + ds.label.toLowerCase() +
        " stream. Around " + F.num(P.pre_mean) + " before block " + P.drift_block + ", peaking at " +
        F.num(P.peak) + " at block " + P.peak_block + ". The table below lists every value.",
      formatY: function (sr, i) { return F.withCI(sr.y[i], c.ci95[i], c.n_seeds); },
      onHover: function (x) { setBlock(x); },
      // two-column layout: make the chart card end level with the left column (no empty gap)
      height: function (current) {
        if (window.matchMedia("(max-width: 1100px)").matches) return null;
        var left = document.querySelector("#problem .problem-left"), card = document.querySelector("#problem .chart-card");
        var svgEl = card.querySelector(".chart svg");
        if (!svgEl) return null;
        return svgEl.getBoundingClientRect().height + (left.getBoundingClientRect().height - card.getBoundingClientRect().height);
      },
    });

    var input = document.getElementById("problem-scrub"), story = document.getElementById("problem-story");
    input.min = c.blocks[0];
    input.max = c.blocks[c.blocks.length - 1];

    function setBlock(b) {
      b = +b;
      input.value = b;
      chart.setCursor(b);
      var i = c.blocks.indexOf(b), v = c.mean[i], r = P.ratio_to_pre[i];
      story.textContent = "";
      var B = function (t) { return el("b", { text: t }); };
      story.appendChild(B("Block " + b + ": "));
      if (b < P.drift_block) {
        story.append("before the change. New rows follow the same rule as the context rows. Error ",
          B(F.num(v)), "; the pre-drift level averages ", B(F.num(P.pre_mean)), ".");
      } else if (b - P.drift_block <= 1) {
        story.append("the rule just changed. Error jumped from ", B(F.num(P.pre_mean)), " (pre-drift level) to ",
          B(F.num(v)), ": ", B(F.num(r, 1) + "×"), " the pre-drift level.");
      } else {
        var back = r <= 1 + tol / 100;
        story.append((b - P.drift_block) + " blocks after the change. Error ", B(F.num(v)), ", ",
          back ? "back within " + tol + "% of the pre-drift level."
               : el("span", null, B(F.num(r, 1) + "×"), " the pre-drift level (" + F.num(P.pre_mean) + "). ",
                    F.num(P.old_rule_share_pct[i], 0) + "% of the history it samples from still follows the old rule."));
      }
    }
    input.addEventListener("input", function () { setBlock(input.value); });
    setBlock(P.peak_block);
    // after the first layout (and on resize), redraw at the height that matches the left column
    var fitChart = function () { chart.redraw(); chart.setCursor(+input.value); };
    window.addEventListener("load", fitChart);
    window.addEventListener("resize", fitChart);

    // accessible table with every value of the chart
    var rows = c.blocks.map(function (b, i) {
      return el("tr", null, el("td", { text: String(b) }), el("td", { text: F.num(c.mean[i]) }),
        el("td", { text: c.ci95[i] === null ? "no CI" : "± " + F.num(c.ci95[i]) }),
        el("td", { text: F.num(P.ratio_to_pre[i], 2) + "×" }));
    });
    var table = el("table", null,
      el("caption", { class: "visually-hidden", text: m.label + " TabPFN error per block, " + ds.label.toLowerCase() + " stream" }),
      el("thead", null, el("tr", null, el("th", { scope: "col", text: "Block" }), el("th", { scope: "col", text: "Mean RMSE" }),
        el("th", { scope: "col", text: "95% CI" }), el("th", { scope: "col", text: "vs pre-drift" }))),
      el("tbody", null));
    rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
    document.getElementById("problem-table").appendChild(table);
    document.querySelector("#problem details.data-table").open = false;   // always starts collapsed
  }

  /* ------------------------------------------------------------------ theme toggle */
  function buildThemeToggle() {
    var btn = document.getElementById("theme-toggle"), label = btn.querySelector(".label");
    // The label says what a click will DO ("Switch to light" while dark is on), and the icon matches.
    var moon = btn.querySelector(".icon-moon"), sun = btn.querySelector(".icon-sun");
    function sync() {
      var dark = window.THEME.current() === "dark";
      label.textContent = dark ? "Switch to light" : "Switch to dark";
      btn.setAttribute("aria-label", label.textContent);
      sun.style.display = dark ? "" : "none";
      moon.style.display = dark ? "none" : "";
    }
    btn.addEventListener("click", function () {
      window.THEME.set(window.THEME.current() === "dark" ? "light" : "dark");
      sync();
    });
    sync();
  }

  document.getElementById("footer-built").textContent =
    "Page data built on " + R.generated + " from " + R.sources.summary_csv + " and " + R.sources.results_json +
    (R.sources.ablation_csv ? " and " + R.sources.ablation_csv : "") + ".";
  buildThemeToggle();
  buildHero();
  driftIllustration();
  buildProblem();
  window.SECTIONS.buildIdea();
  window.SECTIONS.buildHow();
  buildNav();
})();
