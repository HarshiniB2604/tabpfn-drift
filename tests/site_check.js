/*
Browser quality gate for site/index.html.
    cd tests && npm install && node site_check.js        (exit code 0 = pass)
Uses the Chrome or Edge installed on this computer (set CHROME_PATH to choose one).

Checks, in both themes (dark is the default; light = a user who toggled):
 1. No placeholder, TODO or lorem text. The page shows exactly the computed text from site/data.js (findings, winners, robustness takeaway,
    ADWIN sensitivity, limitations), and every results-table cell "mean +/- CI" equals the summary row.
    (tests/test_site_numbers.py checks that this text traces back to results/summary.csv.)
 2. Zero console messages, zero page errors and zero external requests, from file:// and from http,
    at 1920x1080, 1366x768, 768x1024 (tablet) and 390x844; no horizontal page scroll.
 3. axe accessibility check: zero violations (including colour contrast).
 4. Keyboard-only walkthrough: every Tab stop shows a focus ring; the dataset switcher works with arrow
    keys, chips with Space, the data table with Enter, the theme button with Enter.
*/
const { chromium } = require("playwright-core");
const http = require("http"), fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");

const SITE = path.resolve(__dirname, "..", "site");
const CHROME = process.env.CHROME_PATH || [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find(p => fs.existsSync(p));
const SIZES = [[1920, 1080], [1366, 768], [768, 1024], [390, 844]];
const KEY = "tabpfn-drift-theme";
const problems = [];
const fail = msg => { problems.push(msg); console.log("  FAIL", msg); };

function serve() {   // tiny static server for the http run (no extra dependency)
  const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
  return new Promise(res => {
    const srv = http.createServer((req, out) => {
      const f = path.join(SITE, decodeURIComponent(req.url.split("?")[0]).replace(/^\/$/, "/index.html"));
      if (!f.startsWith(SITE) || !fs.existsSync(f)) { out.writeHead(404); return out.end(); }
      out.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" });
      fs.createReadStream(f).pipe(out);
    }).listen(0, "127.0.0.1", () => res(srv));
  });
}

async function open(browser, url, w, h, theme, extra = {}) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, ...extra });
  if (theme === "light") await ctx.addInitScript(k => { try { localStorage.setItem(k, "light"); } catch (e) {} }, KEY);
  const page = await ctx.newPage();
  const log = { msgs: [], external: [] };
  page.on("console", m => log.msgs.push(m.type() + ": " + m.text()));
  page.on("pageerror", e => log.msgs.push("pageerror: " + e.message));
  page.on("request", r => { const u = r.url();
    if (!(u.startsWith("file://") || u.startsWith(url.split("/index.html")[0]) || u.startsWith("data:"))) log.external.push(u); });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForTimeout(250);
  return { ctx, page, log };
}

(async () => {
  if (!CHROME) { console.log("No Chrome/Edge found: set CHROME_PATH"); process.exit(2); }
  const browser = await chromium.launch({ executablePath: CHROME });
  const srv = await serve();
  const fileUrl = pathToFileURL(path.join(SITE, "index.html")).href;
  const httpUrl = `http://127.0.0.1:${srv.address().port}/index.html`;

  // 1. page text = data.js text; results table = summary rows
  console.log("1. page text against site/data.js");
  {
    const { ctx, page } = await open(browser, fileUrl, 1920, 1080, "dark");
    const r = await page.evaluate(() => {
      const R = window.RESULTS, F = window.FMT, norm = s => s.replace(/\s+/g, " ").trim();
      const body = norm(document.body.innerText), missing = [];
      const want = [];
      R.findings.forEach(f => want.push(f.headline, f.body, f.real, f.caveat, ...f.evidence.map(e => e.text)));
      if (R.synthetic_winners) want.push(R.synthetic_winners.text);
      want.push(R.robustness_takeaway);
      if (R.adwin_sensitivity) want.push(R.adwin_sensitivity.summary, R.adwin_sensitivity.caveat, ...R.adwin_sensitivity.points);
      R.limitations.forEach(l => want.push(l.text));
      want.filter(Boolean).forEach(t => { if (!body.includes(norm(t))) missing.push(t.slice(0, 90)); });
      return { n: want.filter(Boolean).length, missing };
    });
    console.log(`  ${r.n} computed texts checked, missing on the page: ${r.missing.length}`);
    const leftovers = await page.evaluate(() => (document.body.innerText.match(/\[[A-Z][A-Z ]{2,}\]|TODO|lorem ipsum/gi) || []));
    console.log(`  placeholder / TODO / lorem text on the page: ${leftovers.length ? leftovers.join(", ") : "none"}`);
    leftovers.forEach(t => fail("placeholder text on page: " + t));
    r.missing.forEach(m => fail("text not on page: " + m));
    for (const ds of await page.evaluate(() => window.RESULTS.datasets.map(d => d.key))) {
      await page.goto(fileUrl + "#results-" + ds); await page.waitForTimeout(150);
      const bad = await page.evaluate(ds => {
        const F = window.FMT, out = [];
        document.querySelectorAll("#results-table tbody tr").forEach(tr => {
          const name = tr.querySelector("th span").textContent;
          const m = window.RESULTS.methods.find(x => x.label === name);
          const row = F.summaryRow(ds, m.key), cells = tr.querySelectorAll("td");
          const shownRmse = [...cells].map(c => c.textContent).find(t => t.includes("\u00b1") || t.includes("no CI"));
          if (shownRmse !== F.withCI(row.rmse_mean, row.rmse_ci95, row.n_seeds)) out.push(name + ": " + shownRmse);
        });
        return out;
      }, ds);
      bad.forEach(b => fail(`results table ${ds}: ${b}`));
    }
    console.log("  results tables checked for all datasets");
    await ctx.close();
  }

  // 2. console, network, overflow
  console.log("2. console messages, external requests, horizontal scroll");
  for (const [kind, url] of [["file", fileUrl], ["http", httpUrl]]) for (const theme of ["dark", "light"]) for (const [w, h] of SIZES) {
    const { ctx, page, log } = await open(browser, url, w, h, theme);
    const st = await page.evaluate(() => ({ theme: document.documentElement.dataset.theme, over: document.documentElement.scrollWidth - innerWidth }));
    const tag = `${kind} ${theme} ${w}x${h}`;
    if (st.theme !== theme) fail(`${tag}: theme is ${st.theme}`);
    log.msgs.forEach(m => fail(`${tag}: console ${m}`));
    log.external.forEach(u => fail(`${tag}: external request ${u}`));
    if (st.over > 0) fail(`${tag}: horizontal scroll ${st.over}px`);
    await ctx.close();
  }
  console.log(`  ${2 * 2 * SIZES.length} page loads checked`);

  // 3. axe, both themes (CSP bypassed only so the checker can be injected)
  console.log("3. axe accessibility check");
  for (const theme of ["dark", "light"]) for (const [w, h] of [[1366, 768], [390, 844]]) {
    const { ctx, page } = await open(browser, fileUrl, w, h, theme, { bypassCSP: true });
    await page.addScriptTag({ path: require.resolve("axe-core/axe.min.js") });
    const res = await page.evaluate(async () => await window.axe.run(document, { resultTypes: ["violations"] }));
    console.log(`  ${theme} ${w}x${h}: ${res.violations.length} violations`);
    res.violations.forEach(v => fail(`axe ${theme} ${w}x${h}: ${v.id} (${v.nodes.length}) ${v.help}`));
    await ctx.close();
  }

  // 4. keyboard-only walkthrough
  console.log("4. keyboard-only walkthrough");
  {
    const { ctx, page } = await open(browser, fileUrl, 1366, 768, "dark");
    const stops = [];
    for (let i = 0; i < 150; i++) {
      await page.keyboard.press("Tab");
      const f = await page.evaluate(() => {
        const e = document.activeElement; if (!e || e === document.body) return null;
        const s = getComputedStyle(e), lab = e.matches("input[type=radio]") ? document.querySelector(`label[for="${e.id}"]`) : null;
        const ring = lab ? getComputedStyle(lab) : s;
        return { id: e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + "|" + (e.textContent || e.getAttribute("aria-label") || "").trim().slice(0, 30),
                 ring: ring.outlineStyle !== "none" && parseFloat(ring.outlineWidth) >= 2 };
      });
      if (!f) continue;
      if (stops.length && f.id === stops[0].id) break;          // wrapped around: every stop visited
      if (!stops.find(s => s.id === f.id)) stops.push(f);
    }
    const noRing = stops.filter(s => !s.ring);
    console.log(`  ${stops.length} tab stops, without a visible focus ring: ${noRing.length}`);
    noRing.forEach(s => fail("no focus ring: " + s.id));
    // dataset switcher with arrow keys
    await page.focus("#ds-sudden"); await page.keyboard.press("ArrowRight"); await page.waitForTimeout(100);
    const ds = await page.evaluate(() => document.querySelector("input[name=dataset]:checked").value + " " + location.hash);
    console.log("  ArrowRight on the dataset switcher ->", ds);
    if (!ds.startsWith("gradual #results-gradual")) fail("arrow keys do not move the dataset switcher");
    // chip with Space
    await page.focus("#method-chips button"); const before = await page.getAttribute("#method-chips button", "aria-pressed");
    await page.keyboard.press("Space"); await page.waitForTimeout(100);
    const after = await page.getAttribute("#method-chips button", "aria-pressed");
    console.log(`  Space on the first chip: aria-pressed ${before} -> ${after}`);
    if (before === after) fail("Space does not toggle a chip");
    // details with Enter
    await page.focus("#results-curve-details summary"); await page.keyboard.press("Enter"); await page.waitForTimeout(100);
    const opened = await page.evaluate(() => document.getElementById("results-curve-details").open);
    console.log("  Enter on 'Show the chart's numbers as a table' opens it:", opened);
    if (!opened) fail("Enter does not open the data table");
    // theme with Enter
    await page.focus("#theme-toggle"); await page.keyboard.press("Enter"); await page.waitForTimeout(100);
    const th = await page.getAttribute("html", "data-theme");
    console.log("  Enter on the theme button ->", th);
    if (th !== "light") fail("Enter does not switch the theme");
    await ctx.close();
  }

  srv.close();
  await browser.close();
  console.log(problems.length ? `\nFAILED: ${problems.length} problem(s)` : "\nALL SITE CHECKS PASSED");
  process.exit(problems.length ? 1 : 0);
})();
