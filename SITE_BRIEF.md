# SITE_BRIEF.md - presentation website for "TabPFN under concept drift"

Goal: a static, professional, deploy-ready website that explains the project to someone who has never
heard of it, then shows the real results honestly. It replaces the Gradio app as the PRESENTATION version
(keep app/app.py as a backup). Audience: course evaluators, on a projector (1920x1080) and on phones.

## 0. Non-negotiable rules
1. NO number, claim or label may be typed by hand. Every figure comes from site_data/results.json /
   results/summary.csv. If a claim cannot be computed from the data, it does not appear.
2. Keep the honesty framing of app/app.py: synthetic vs real is always labelled; "REPLAY of saved
   results"; seed counts shown; 1-seed values show "no CI"; capped recovery shown as "not recovered";
   real data has NO true-drift line; ADWIN over-triggering is stated as an observation until the
   ablation results exist.
3. Works OFFLINE and when opened by double-clicking index.html (file://). Therefore: no CDN, no web
   fonts from the internet, no fetch() of local JSON (blocked on file://). Generate site/data.js
   (window.RESULTS = {...}) from results.json with a new script: python -m eval.make_site. Vendor any
   chart library into site/vendor/ (pin the version), or draw charts with plain SVG/canvas.
4. Zero external network requests (verify in the browser network log). No analytics, no trackers.
5. Do NOT commit until I have reviewed screenshots. Build in stages (see section 6).

## 1. Design direction: "editorial light"
Calm, serious, research-paper-meets-data-story. Light background so it holds up on a projector.

Tokens (CSS variables on :root; also define a dark variant under prefers-color-scheme, same hues):
- --bg: #FAF8F4 (warm paper)      --surface: #FFFFFF      --ink: #1B1B1F     --ink-2: #4A4A55
- --line: #E4E0D8                  --accent-tabpfn: #2B6FD6 (blue)   --accent-baseline: #E8683A (orange)
- Use ONLY blue for TabPFN policies and orange for baselines, everywhere, matching results/figures.
  Policies are told apart by dash style + direct labels (not by colour alone). Colour-blind safe.
- Type: headings serif stack (Georgia, "Iowan Old Style", "Times New Roman", serif), body system sans
  (system-ui, -apple-system, "Segoe UI", Roboto, sans-serif), numbers in tabular-nums. Self-host any
  font files or just use these stacks.
- Scale: body 18px / line-height 1.6; max text width 68ch; section spacing 96px (desktop), 56px (mobile).
- Cards: 1px --line border, 14px radius, no heavy shadows. Subtle motion only (150-250ms); respect
  prefers-reduced-motion.
- Contrast WCAG AA minimum; visible focus rings; full keyboard operation; semantic landmarks;
  alt text / aria-labels on charts with a text summary or data table fallback.

## 2. Page structure (single page, sticky slim top nav with anchors)
1. HERO: one-sentence question "When data changes, which past examples should an AI model look at?"
   Sub-line: what TabPFN is in plain words. One headline result computed from data (e.g. vanilla error
   multiple under drift). Buttons: "See the problem" / "Jump to results". Small badges: "3 synthetic
   streams + 3 real datasets", "no retraining".
2. THE PROBLEM (interactive): mini line chart of vanilla TabPFN block error on the sudden stream with a
   scrubber; text beside it updates ("block 31: the rule just changed, error jumped from ~1 to ~6.5").
   Plain-language definition of concept drift with a tiny before/after illustration.
3. THE IDEA: how TabPFN uses "context rows" in 3 simple steps (diagram). Then 6 policy cards
   (vanilla, sliding window, recency, ADWIN, validation, kNN): one sentence each + a small ILLUSTRATION
   showing which blocks of past data each policy keeps. Label it "Illustration, not data".
4. HOW WE TESTED: test-then-train in blocks of 200 rows; the 3 synthetic streams (sudden, gradual,
   recurring, plus no-drift control) and the 3 real datasets (Bike Sharing, Appliances Energy, Beijing
   PM2.5) in one line each; baselines XGBoost, MLP, adaptive random forest.
5. RESULTS (the main interactive section): dataset switcher (6 + "no drift" control). For each: the
   error-over-time chart (policy toggles; synthetic shows true drift markers; real shows 5-block rolling
   RMSE with no drift line; ADWIN flags optional), the results table (mean +/- 95% CI, seeds, recovery
   time for synthetic, "not recovered" labels), and the bar chart. Seed count line + REPLAY label always.
6. WHAT WE FOUND: three findings computed from data (same logic as app Key findings), each with its
   caveat line. Plus the "robustness" table: for each real dataset, each policy's % gap to the best
   policy on that dataset (computed). Add an "ADWIN sensitivity" block ONLY if results_ablation/ data
   exists (leave this hook data-driven; hide the block otherwise).
7. LIMITATIONS: the same list as the app Limitations tab, generated from data where numbers appear.
8. REPRODUCE + CREDITS: repo layout, the exact commands (make_report, make_figures, make_site, run
   app), dataset citations with licences (Bike Sharing and Appliances: UCI; Beijing PM2.5: Chen 2015,
   CC BY 4.0), TabPFN non-commercial licence note, and an AI-assistance disclosure line.
Footer: project title, team, course, date.

## 3. Interaction details
- Charts: hover tooltip with exact values; legend = direct labels at line ends, no overlap; toggle chips
  per method (default on: vanilla, sliding window, validation, ADWIN, XGBoost).
- Dataset switcher is a segmented control; selection stored in the URL hash so a view can be shared.
- Responsive: desktop 1920 and 1366 wide, tablet, phone (390). No horizontal page scroll. Tables scroll
  inside their own container.
- Print stylesheet: clean black-on-white results section.

## 4. File layout
site/index.html, site/css/style.css, site/js/app.js (+ small modules), site/vendor/ (pinned libs),
site/data.js (generated; commit it), eval/make_site.py (generates data.js from results.json and
summary.csv; computes the findings and robustness table so the page does not recompute stats).
Add a README section "Website": how to regenerate and how to open/deploy (Netlify drag-and-drop, or any
static host; note GitHub Pages on a private repo may need a paid plan).

## 5. Quality gate (do all of this and show me evidence)
- Playwright (or similar) screenshots at 1920x1080, 1366x768 and 390x844 for every section and every
  dataset in the switcher; look at them yourself and fix overlaps, clipping, bad contrast.
- Open from file:// and from a local http server; both must work.
- Browser console: zero errors/warnings. Network log: zero external requests.
- Every number on the page traced to data: write a test that regenerates data.js and checks the page
  text against summary.csv for all findings.
- Keyboard-only walkthrough; run an accessibility check (axe or equivalent) and fix issues.
- No leftover mock data, TODO text or placeholder lorem ipsum.

## 6. Build stages (stop after each and show screenshots; wait for my OK)
Stage 1: make_site.py + data.js, page shell, theme, nav, hero, problem section.
Stage 2: idea + policy cards + how we tested.
Stage 3: results section (all datasets) + findings + limitations.
Stage 4: reproduce/credits, responsive polish, print styles, quality gate, README section.
Stage 5 (later, after ablation): ADWIN sensitivity block.
