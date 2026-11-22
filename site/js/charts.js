/* Plain-SVG charts (no library, so the page works offline). Colours come from the theme (CSS
   variables) and every chart redraws when the theme or the container width changes.

   lineChart(container, opts): optional CI band, lines (colour by kind, dash by method), direct
     labels at line ends (spread apart, with a leader line when a label had to move), vertical
     drift lines and shaded drift periods, optional flag markers, hover tooltip.
     Returns { setCursor(x), redraw(), update(newOpts) }.
   barChart(container, opts): horizontal bars with 95% CI whiskers and a value label per bar. */
(function () {
  "use strict";
  var F = window.FMT;

  function niceTicks(max, count) {
    var raw = max / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).filter(function (s) { return s >= raw; })[0];
    var ticks = [];   // from 0 up to the first tick at or above max, so nothing is drawn past the axis
    for (var v = 0; ; v += step) { ticks.push(+v.toFixed(10)); if (v >= max - 1e-9) break; }
    return ticks;
  }

  function watch(container, draw) {
    // redraw on theme change and on width change; returns a function that stops watching
    window.addEventListener("themechange", draw);
    var lastW = container.clientWidth, ro = null;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(function () {
        if (Math.abs(container.clientWidth - lastW) > 2) { lastW = container.clientWidth; draw(); }
      });
      ro.observe(container);
    }
    return function () { window.removeEventListener("themechange", draw); if (ro) ro.disconnect(); };
  }

  /* ------------------------------------------------------------------ line chart */
  function lineChart(container, opts) {
    var state = { cursor: null };
    var tip = F.el("div", { class: "tooltip", role: "presentation" });
    container.appendChild(tip);

    function draw() {
      var old = container.querySelector("svg");
      if (old) old.remove();
      var W = Math.max(300, container.clientWidth);
      var small = W < 520;
      var H = Math.round(Math.min(420, Math.max(250, W * (small ? 0.72 : 0.52))));
      var fit = opts.height ? opts.height(H) : null;   // optional: caller asks for a given height
      if (fit) H = Math.round(Math.min(480, Math.max(230, fit)));
      var series = opts.series.filter(function (s) { return s.y.some(function (v) { return v !== null; }); });
      var labelled = series.length > 1 || opts.endLabels;
      var labelOf = function (s) { return small && s.short ? s.short : s.label; };
      var longest = series.reduce(function (n, s) { return Math.max(n, labelOf(s).length); }, 0);
      var labelW = labelled ? Math.min(small ? 70 : 130, 18 + longest * 7) : 16;
      var m = { top: 26, right: labelW, bottom: 44, left: small ? 40 : 52 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

      var root = F.svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img",
                                "aria-label": opts.ariaLabel });
      if (!series.length) {
        var t0 = F.svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" });
        t0.textContent = opts.emptyText || "Nothing selected";
        root.appendChild(t0);
        container.insertBefore(root, tip);
        return;
      }

      var xs = [], ymax = 0;
      series.forEach(function (s) {
        s.x.forEach(function (x, i) {
          xs.push(x);
          var top = s.hi && s.hi[i] !== null ? s.hi[i] : s.y[i];
          if (top !== null && top > ymax) ymax = top;
        });
      });
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ticksY = niceTicks(ymax * 1.06, small ? 4 : 5);
      var ytop = ticksY[ticksY.length - 1];
      var X = function (x) { return m.left + (x - xmin) / (xmax - xmin) * iw; };
      var Y = function (y) { return m.top + ih - y / ytop * ih; };
      var g = F.svg("g");
      root.appendChild(g);

      // shaded periods (gradual drift) go under everything else
      (opts.vrects || []).forEach(function (r) {
        g.appendChild(F.svg("rect", { x: X(r.x0), y: m.top, width: X(r.x1) - X(r.x0), height: ih, fill: F.cssVar("--line"),
                                      "fill-opacity": 0.55 }));
        var t = F.svg("text", { class: "drift-text", x: X(r.x0) + 6, y: m.top - 8 });
        t.textContent = r.label;
        g.appendChild(t);
      });

      // grid + y axis
      var grid = F.svg("g", { class: "grid" });
      ticksY.forEach(function (t) {
        grid.appendChild(F.svg("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t) }));
        var tx = F.svg("text", { x: m.left - 8, y: Y(t) + 4, "text-anchor": "end" });
        tx.textContent = t;
        grid.appendChild(tx);
      });
      g.appendChild(grid);

      // x axis: tick step chosen so labels are at least ~55px apart
      var ax = F.svg("g", { class: "axis" });
      ax.appendChild(F.svg("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }));
      var pxPerBlock = iw / (xmax - xmin);
      var step = [5, 10, 20, 25, 50, 100].filter(function (s) { return s * pxPerBlock >= 55; })[0] || 100;
      for (var t = Math.ceil(xmin / step) * step; t <= xmax; t += step) {
        ax.appendChild(F.svg("line", { x1: X(t), x2: X(t), y1: m.top + ih, y2: m.top + ih + 5 }));
        var lx = F.svg("text", { x: X(t), y: m.top + ih + 19, "text-anchor": "middle" });
        lx.textContent = t;
        ax.appendChild(lx);
      }
      var xl = F.svg("text", { x: m.left + iw / 2, y: H - 6, "text-anchor": "middle" });
      xl.textContent = opts.xLabel;
      ax.appendChild(xl);
      var yl = F.svg("text", { x: 0, y: 0, transform: "translate(12," + (m.top + ih / 2) + ") rotate(-90)",
                               "text-anchor": "middle" });
      yl.textContent = opts.yLabel;
      ax.appendChild(yl);
      g.appendChild(ax);

      // vertical drift lines; a label is shortened when the next line is too close for it
      var vl = opts.vlines || [];
      vl.forEach(function (v, k) {
        g.appendChild(F.svg("line", { class: "drift-line", x1: X(v.x), x2: X(v.x), y1: m.top, y2: m.top + ih }));
        var room = k + 1 < vl.length ? X(vl[k + 1].x) - X(v.x) : m.left + iw + m.right - X(v.x);
        var text = v.label.length * 6.6 + 12 <= room ? v.label : (v.short || "");
        if (!text) return;
        var tt = F.svg("text", { class: "drift-text", x: X(v.x) + 6, y: m.top - 8 });
        tt.textContent = text;
        g.appendChild(tt);
      });
      (opts.hlines || []).forEach(function (h) {
        g.appendChild(F.svg("line", { class: "pre-line", x1: m.left, x2: m.left + iw, y1: Y(h.y), y2: Y(h.y) }));
      });

      // series: band, then line
      var ends = [];
      series.forEach(function (s) {
        var colour = F.cssVar(s.kind === "baseline" ? "--accent-baseline" : "--accent-tabpfn");
        if (s.lo && s.hi) {
          var up = [], down = [];
          s.x.forEach(function (x, i) {
            if (s.hi[i] === null || s.lo[i] === null) return;
            up.push(X(x) + "," + Y(s.hi[i]));
            down.unshift(X(x) + "," + Y(Math.max(0, s.lo[i])));
          });
          if (up.length) g.appendChild(F.svg("polygon", { points: up.concat(down).join(" "), fill: F.cssVar("--band") }));
        }
        var d = "", pen = false;
        s.x.forEach(function (x, i) {
          if (s.y[i] === null) { pen = false; return; }
          d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(s.y[i]).toFixed(1);
          pen = true;
        });
        g.appendChild(F.svg("path", { d: d, fill: "none", stroke: colour, "stroke-width": 2.25,
                                       "stroke-dasharray": s.dash || null, "stroke-linejoin": "round" }));
        (s.markers || []).forEach(function (x) {   // e.g. ADWIN drift flags: small triangles on the line
          var i = s.x.indexOf(x);
          if (i < 0 || s.y[i] === null) return;
          var cx = X(x), cy = Y(s.y[i]) - 7;
          g.appendChild(F.svg("path", { d: "M" + (cx - 4.5) + "," + (cy - 5) + "L" + (cx + 4.5) + "," + (cy - 5) + "L" + cx + "," + (cy + 2) + "Z",
                                        fill: colour, stroke: F.cssVar("--surface"), "stroke-width": 1 }));
        });
        var last = s.y.length - 1;
        while (last > 0 && s.y[last] === null) last--;
        ends.push({ y: Y(s.y[last]), x: X(s.x[last]), label: labelOf(s) });
      });

      // direct labels at line ends, pushed apart so they never overlap (leader line if moved)
      if (labelled) {
        ends.sort(function (a, b) { return a.y - b.y; });
        var gap = 15;
        for (var i = 0; i < ends.length; i++) ends[i].ly = i ? Math.max(ends[i].y, ends[i - 1].ly + gap) : ends[i].y;
        var over = ends.length ? ends[ends.length - 1].ly - (m.top + ih) : 0;   // keep the stack inside the plot
        if (over > 0) ends.forEach(function (e) { e.ly -= over; });
        for (var j = 1; j < ends.length; j++) if (ends[j].ly < ends[j - 1].ly + gap) ends[j].ly = ends[j - 1].ly + gap;
        ends.forEach(function (e) {
          if (Math.abs(e.ly - e.y) > 3) {
            g.appendChild(F.svg("line", { x1: e.x + 2, y1: e.y, x2: e.x + 7, y2: e.ly, stroke: F.cssVar("--mark-muted"), "stroke-width": 1 }));
          }
          var t = F.svg("text", { class: "series-label", x: e.x + 9, y: e.ly + 4 });
          t.textContent = e.label;
          g.appendChild(t);
        });
      }

      // cursor (scrubber) + hover
      var cursor = F.svg("g", { "aria-hidden": "true" });
      g.appendChild(cursor);
      function placeCursor(x) {
        while (cursor.firstChild) cursor.removeChild(cursor.firstChild);
        if (x === null || x === undefined) return;
        cursor.appendChild(F.svg("line", { class: "scrub-line", x1: X(x), x2: X(x), y1: m.top, y2: m.top + ih }));
        series.forEach(function (s) {
          var i = s.x.indexOf(x);
          if (i >= 0 && s.y[i] !== null) cursor.appendChild(F.svg("circle", { class: "scrub-dot", cx: X(x), cy: Y(s.y[i]), r: 6 }));
        });
      }
      state.place = placeCursor;
      placeCursor(state.cursor);

      var hit = F.svg("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" });
      g.appendChild(hit);
      function nearest(evt) {
        var box = root.getBoundingClientRect();
        var px = (evt.clientX - box.left) * (W / box.width);
        var x = Math.round(xmin + (px - m.left) / iw * (xmax - xmin));
        return Math.max(xmin, Math.min(xmax, x));
      }
      hit.addEventListener("pointermove", function (evt) {
        var x = nearest(evt);
        var lines = series.map(function (s) {
          var i = s.x.indexOf(x);
          return i < 0 || s.y[i] === null ? null : s.label + ": " + opts.formatY(s, i);
        }).filter(Boolean);
        if (!lines.length) return;
        tip.textContent = "";
        tip.appendChild(F.el("strong", { text: opts.xName + " " + x }));
        lines.forEach(function (l) { tip.appendChild(F.el("div", { text: l })); });
        var box = root.getBoundingClientRect(), cb = container.getBoundingClientRect();
        var left = X(x) * box.width / W + (box.left - cb.left);
        tip.style.left = Math.min(Math.max(0, left + 12), cb.width - tip.offsetWidth) + "px";
        tip.style.top = (m.top * box.height / H) + "px";
        tip.classList.add("on");
        if (opts.onHover) opts.onHover(x);
      });
      hit.addEventListener("pointerleave", function () { tip.classList.remove("on"); });

      container.insertBefore(root, tip);
    }

    draw();
    watch(container, draw);
    return {
      setCursor: function (x) { state.cursor = x; if (state.place) state.place(x); },
      redraw: draw,
      update: function (newOpts) { opts = newOpts; tip.classList.remove("on"); draw(); }
    };
  }

  /* ------------------------------------------------------------------ bar chart */
  // opts.bars: [{label, short, kind, value, ci (or null), text}], opts.xLabel, opts.ariaLabel
  function barChart(container, opts) {
    function draw() {
      var old = container.querySelector("svg");
      if (old) old.remove();
      var W = Math.max(300, container.clientWidth), small = W < 520;
      var bars = opts.bars, rowH = small ? 34 : 30;
      var labelW = small ? 92 : 132, valueW = W < 560 ? 0 : 170;   // values also in the results table
      var m = { top: 8, right: valueW + 10, bottom: 40, left: labelW };
      var H = m.top + bars.length * rowH + m.bottom, iw = W - m.left - m.right;
      var max = 0;
      bars.forEach(function (b) { max = Math.max(max, b.value + (b.ci || 0)); });
      var ticks = niceTicks(max * 1.05, small ? 3 : 5), top = ticks[ticks.length - 1];
      var X = function (v) { return m.left + v / top * iw; };
      var root = F.svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": opts.ariaLabel });
      ticks.forEach(function (t) {
        root.appendChild(F.svg("line", { x1: X(t), x2: X(t), y1: m.top, y2: H - m.bottom, stroke: F.cssVar("--line") }));
        var tx = F.svg("text", { x: X(t), y: H - m.bottom + 16, "text-anchor": "middle", fill: F.cssVar("--ink-2"), "font-size": 12 });
        tx.textContent = t;
        root.appendChild(tx);
      });
      var xl = F.svg("text", { x: m.left + iw / 2, y: H - 6, "text-anchor": "middle", fill: F.cssVar("--ink-2"), "font-size": 12 });
      xl.textContent = opts.xLabel;
      root.appendChild(xl);
      bars.forEach(function (b, i) {
        var y = m.top + i * rowH, bh = rowH - 12, cy = y + rowH / 2;
        var colour = F.cssVar(b.kind === "baseline" ? "--accent-baseline" : "--accent-tabpfn");
        var name = F.svg("text", { x: m.left - 8, y: cy + 4, "text-anchor": "end", fill: F.cssVar("--ink"), "font-size": 12.5 });
        name.textContent = small && b.short ? b.short : b.label;
        root.appendChild(name);
        root.appendChild(F.svg("rect", { x: m.left, y: cy - bh / 2, width: Math.max(1, X(b.value) - m.left), height: bh, rx: 3, fill: colour }));
        if (b.ci !== null && b.ci !== undefined) {
          var x0 = X(Math.max(0, b.value - b.ci)), x1 = X(b.value + b.ci), ink = F.cssVar("--ink");
          root.appendChild(F.svg("line", { x1: x0, x2: x1, y1: cy, y2: cy, stroke: ink, "stroke-width": 1.5 }));
          [x0, x1].forEach(function (x) { root.appendChild(F.svg("line", { x1: x, x2: x, y1: cy - 5, y2: cy + 5, stroke: ink, "stroke-width": 1.5 })); });
        }
        if (valueW) {
          var vt = F.svg("text", { x: W - valueW + 6, y: cy + 4, fill: F.cssVar("--ink-2"), "font-size": 12 });
          vt.textContent = b.text;
          root.appendChild(vt);
        }
      });
      container.appendChild(root);
    }
    draw();
    var stop = watch(container, draw);
    return { update: function (o) { opts = o; draw(); }, destroy: stop };
  }

  window.CHARTS = { lineChart: lineChart, barChart: barChart };
})();
