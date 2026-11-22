/* Small helpers shared by the page scripts. Only formatting: no statistics are computed here
   (all numbers are prepared by python -m eval.make_site and arrive in window.RESULTS). */
(function () {
  "use strict";
  var R = window.RESULTS;

  function num(x, digits) {
    if (x === null || x === undefined || isNaN(x)) return "n/a";
    return Number(x).toFixed(digits === undefined ? 2 : digits);
  }

  /* "50.22 ± 1.30", or "64.84 (1 seed, no CI)" - same wording as the app and the report */
  function withCI(mean, ci, nSeeds, digits) {
    if (mean === null || mean === undefined) return "n/a";
    if (nSeeds < 2 || ci === null || ci === undefined) return num(mean, digits) + " (1 seed, no CI)";
    return num(mean, digits) + " ± " + num(ci, digits);
  }

  function seeds(n) { return n + (n === 1 ? " seed" : " seeds"); }

  /* el("p", {class: "x"}, "text", child...) - builds DOM nodes; strings become text (never HTML) */
  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (attrs[k] === null || attrs[k] === undefined || attrs[k] === false) return;
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function svg(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    return node;
  }

  function method(key) { return R.methods.filter(function (m) { return m.key === key; })[0]; }
  function dataset(key) { return R.datasets.filter(function (d) { return d.key === key; })[0]; }
  function summaryRow(stream, methodKey) {
    return R.summary.filter(function (r) { return r.stream === stream && r.method === methodKey; })[0];
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  window.FMT = { num: num, withCI: withCI, seeds: seeds, el: el, svg: svg, method: method,
                 dataset: dataset, summaryRow: summaryRow, cssVar: cssVar };
})();
