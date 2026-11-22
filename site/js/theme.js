/* Light/dark theme. Loaded in <head> so the right theme is set before the page paints.
   Default is DARK on a first visit; the OS setting is ignored on purpose. Once the user
   uses the toggle, that choice is remembered in localStorage and wins from then on. */
(function () {
  "use strict";
  var KEY = "tabpfn-drift-theme";

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }   // storage can be blocked
  }
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme === "dark" ? "dark" : "light");
  }
  apply(stored() === "light" ? "light" : "dark");

  // Printing: always use the light colours (charts read them when they draw), then restore.
  var beforePrint = null;
  function printing(on) {
    if (on && beforePrint === null && document.documentElement.getAttribute("data-theme") !== "light") {
      beforePrint = document.documentElement.getAttribute("data-theme");
      apply("light");
      window.dispatchEvent(new Event("themechange"));
    } else if (!on && beforePrint !== null) {
      apply(beforePrint);
      beforePrint = null;
      window.dispatchEvent(new Event("themechange"));
    }
  }
  window.addEventListener("beforeprint", function () { printing(true); });
  window.addEventListener("afterprint", function () { printing(false); });
  if (window.matchMedia) {
    var mq = window.matchMedia("print");
    var onChange = function (e) { printing(e.matches); };
    if (mq.addEventListener) mq.addEventListener("change", onChange); else if (mq.addListener) mq.addListener(onChange);
  }

  window.THEME = {
    current: function () { return document.documentElement.getAttribute("data-theme"); },
    set: function (theme) {
      apply(theme);
      try { localStorage.setItem(KEY, theme); } catch (e) { /* not saved, still applied */ }
      window.dispatchEvent(new Event("themechange"));   // charts redraw with the new colours
    },
  };
})();
