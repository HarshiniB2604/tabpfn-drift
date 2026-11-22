/* Plain-SVG line chart (no library, so the page works offline).
   lineChart(container, opts) draws: optional CI band, lines (colour by kind, dash by method),
   direct labels at line ends (spread so they never overlap), vertical markers, hover tooltip.
   It redraws itself when the container changes size. Returns {setCursor(x)} for scrubbers. */
(function () {
  "use strict";
  var F = window.FMT;

  function niceTicks(max, count) {
    var raw = max / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).filter(function (s) { return s >= raw; })[0];
    var ticks = [];
    for (var v = 0; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

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
      var labelW = opts.series.length > 1 || opts.endLabels ? (small ? 74 : 96) : 16;
      var m = { top: 26, right: labelW, bottom: 44, left: small ? 40 : 52 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

      var xs = [], ymax = 0;
      opts.series.forEach(function (s) {
        s.x.forEach(function (x, i) {
          xs.push(x);
          var top = s.hi ? s.hi[i] : s.y[i];
          if (top !== null && top > ymax) ymax = top;
        });
      });
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ticksY = niceTicks(ymax * 1.06, small ? 4 : 5);
      var ytop = ticksY[ticksY.length - 1];
      var X = function (x) { return m.left + (x - xmin) / (xmax - xmin) * iw; };
      var Y = function (y) { return m.top + ih - y / ytop * ih; };

      var root = F.svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img",
                                "aria-label": opts.ariaLabel });
      var g = F.svg("g");
      root.appendChild(g);

      // grid + y axis
      var grid = F.svg("g", { class: "grid" });
      ticksY.forEach(function (t) {
        grid.appendChild(F.svg("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t) }));
        var tx = F.svg("text", { x: m.left - 8, y: Y(t) + 4, "text-anchor": "end" });
        tx.textContent = t;
        grid.appendChild(tx);
      });
      g.appendChild(grid);

      // x axis
      var ax = F.svg("g", { class: "axis" });
      ax.appendChild(F.svg("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }));
      var step = iw < 260 ? 20 : 10;
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

      // markers (true drift lines, reference levels)
      (opts.vlines || []).forEach(function (v) {
        g.appendChild(F.svg("line", { class: "drift-line", x1: X(v.x), x2: X(v.x), y1: m.top, y2: m.top + ih }));
        var t = F.svg("text", { class: "drift-text", x: X(v.x) + 6, y: m.top - 8 });
        t.textContent = v.label;
        g.appendChild(t);
      });
      (opts.hlines || []).forEach(function (h) {
        g.appendChild(F.svg("line", { class: "pre-line", x1: m.left, x2: m.left + iw, y1: Y(h.y), y2: Y(h.y) }));
      });

      // series: band, then line
      var ends = [];
      opts.series.forEach(function (s) {
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
        var last = s.y.length - 1;
        while (last > 0 && s.y[last] === null) last--;
        ends.push({ y: Y(s.y[last]), x: X(s.x[last]), label: s.label });
      });

      // direct labels at line ends, pushed apart so they never overlap
      if (opts.series.length > 1 || opts.endLabels) {
        ends.sort(function (a, b) { return a.y - b.y; });
        var gap = 15;
        for (var i = 1; i < ends.length; i++) ends[i].ly = Math.max(ends[i].y, (ends[i - 1].ly || ends[i - 1].y) + gap);
        ends.forEach(function (e) {
          var ly = e.ly || e.y;
          var t = F.svg("text", { class: "series-label", x: e.x + 8, y: ly + 4 });
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
        opts.series.forEach(function (s) {
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
        var lines = opts.series.map(function (s) {
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
    window.addEventListener("themechange", draw);   // colours come from the theme: redraw
    var lastW = container.clientWidth;
    if (window.ResizeObserver) {
      new ResizeObserver(function () {
        if (Math.abs(container.clientWidth - lastW) > 2) { lastW = container.clientWidth; draw(); }
      }).observe(container);
    }
    return {
      setCursor: function (x) { state.cursor = x; if (state.place) state.place(x); },
      redraw: draw
    };
  }

  window.CHARTS = { lineChart: lineChart };
})();
