# Editing this deck

Follow-up to `intro_to_logistic_regression`. Same engine, same design system.
It answers one question: **what does the age estimate `0.11683` mean, and how does it
relate to the `summary(glm(...))` output?**

Open `index.html` and search for `EDIT PANEL 1 HERE`, `EDIT PANEL 2 HERE`, etc.

## Rounding convention

Show the coefficient as **0.1168** (and 10 × it as **1.168**) so it lines up by eye with
the summary output's `age 0.11683`. The odds ratio is **1.124** (= e^0.1168) and the
per-decade odds ratio is **3.2** (= e^1.168). Keep these consistent everywhere.

## The panels

Layout: standard **2×2 grid**, read top-left → bottom-left → top-right → bottom-right.
Panel 1 recall (TL), Panel 2 transform (BL), Panel 3 odds ratio (TR), Panel 4 recap (BR).
Panel 4 lights up at the end (`PANEL_FIRST[4]=13`).

1. **Recall** (top-left) — reopen on the last deck's closing figure: teal age-band bars, a
   gold logistic curve, and the **actual 0/1 data points** (the raw `PTS` scatter from the
   intro deck, jittered near y=1 for weak grip / y=0 for not; colours match
   `intro_to_logistic_regression`). These are individual observations, NOT band midpoints.
   The per-year change in probability is not constant (said, not drawn); yet the summary
   gives one number, 0.1168, per year, which is a change in the *log-odds*.
2. **The transform** (bottom-left, a squished landscape plot) — combines the old "two
   representations" and "what is a log-odds" slides. The full algebra extracting
   `ln(p/(1−p)) = β₀+β₁·age` from the logistic form (with short italic `.step-note`
   instructions on each move), then **one morphing plot**: at `t=0` the probability S-curve
   inside a 0–1 box; press → and it straightens into a line as the y-axis becomes log-odds
   (`t=1`). The `PLV` probability grid-lines (`.05 .10 .25 .50 .75 .90 .95`) are **labelled
   from the start** (`g2lab` is on at step 4) with the `p → odds → ln` calc, and each label
   travels as its line slides to the log-odds position. The **y-axis line stays put but its
   numbers and title crossfade**: `0 / .5 / 1` + "probability" fade to `-4 … 4` + "log-odds"
   as `t` goes 0→1 (`probNums`/`logNums`/`yTitleP`/`yTitleL` in `drawMorph`). Add/remove a
   grid line by editing `PLV`; the log-odds axis range `M.lo` is set wide enough (`[-4,4]`)
   to show `p = .05 … .95` plus the whole fitted curve.
4. **Which form to use** (bottom-right) — recap of the two jobs of regression: predict a
   probability (use the probability form) vs understand the effect (use the log-odds form),
   then exponentiate the log-odds change to the odds ratio. No new derivation: the only
   subtlety, `e^{ln A − ln B} = A/B` (a difference of logs becomes a ratio), is stated in one
   line and already proved on Panel 3.
3. **β₁ is a (log) odds ratio** (bottom-left) — highlight β₁ in the log-odds formula; write
   logit at `a` and `a+1`, subtract, apply the log rule, exponentiate → `e^0.1168 = 1.124`.
   The odds bars show **×1.124 applied every year across a 10-year run** (70→80), noting the
   ten steps compound to ≈ ×3.2; `exp(coef)` / `exp(confint)` for reporting.

## Panel content

Everything is ordinary stacked HTML. Edit in place; keep `data-step` for when each piece
appears (element with `data-step="N"` shows once `step ≥ N-1`):

```html
<p class="frag bullet" data-step="2">Teaching text.</p>            <!-- add `bad` → red bullet -->
<pre class="frag out" data-step="1">#&gt; ...</pre>                  <!-- console output; `code` for R code -->
<div class="frag inline-eq" data-step="6">$\ln\frac{p}{1-p}=\beta_0+\beta_1\text{age}$</div>
```

`<strong>` = teal · `<span class="bad">` = red · `<span class="hi-or">` = odds-ratio orange.
Equations are KaTeX; `\textcolor{#b5451f}{...}` tints a symbol (used to flag β₁ in panel 4).

## Visuals

The `STEP_CHANGES` list near the bottom of `index.html` controls only the charts and the
active panel. Each entry inherits the previous one, so only write what changes:

- `vis:[...]` — which SVG groups are shown. Groups: `g1bars`, `g1dots`, `g1curve` (P1);
  `g2plot`, `g2lab` (P2 morph — plot vs. travelling calc labels); `g3bars`, `g3mult` (P3).
- `c1` / `t` / `b3` — 0→1 animation targets: P1 curve draw, **P2 morph** (0 = probability
  curve, 1 = log-odds line), P3 odds-bar grow. Stepping forward past step 6 drives `t:0→1`
  (the transform); stepping back reverses it — the morph is fully scrubbable.
- `panel` — which panel gets the gold "active" ring.

`PANEL_FIRST` sets the step at which each panel lights up. All chart numbers come from
`GLM`, `BANDS`, and the `sig`/`lod`/`odds` helpers at the top of the script — the same KLoSA
fit as the intro deck (`out_glm.txt`, `out_bands.txt`). Change a number there, not inline.

## Static export

`index.html?still=1&step=NN` renders one fully-built step with no animation, handy for
screenshots or dropping a frame into the video.
