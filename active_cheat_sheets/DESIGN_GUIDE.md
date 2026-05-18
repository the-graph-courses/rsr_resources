# Active Concept Sheet: Design Guide

A reference for building interactive, single-page cheat sheets for The Graph Courses lessons. The HTML files in this directory serve as the working technical template; this document covers the thinking behind the design decisions.

Nothing is set in stone. Feel free to deviate from the guide if it makes sense.

## What is an Active Concept Sheet?

It is a one-page summary of a lesson's key ideas where key terms, concepts, or examples have fill-in blanks that let students practice active recall. The blanks come with hint options. The goal is active recall with guardrails: students have to retrieve or reason about each concept, but the options keep them from getting stuck.

## Content and Visual Guidelines

* **Writing the content:** Write sections and questions in complete sentences. Err on the side of using many blanks rather than too few.
* **Sentence structure:** Structure sentences so the blank appears after the context. Instead of "[Blank] is the country at the bottom of South America," write "The country at the bottom of South America is [Blank]." This ensures the student has all the necessary context before reaching the blank.
* **Intelligent bolding:** Filled answers appear as bold text in the final document, so pick blanks for key terms that naturally warrant emphasis. Selectively bold other key concepts throughout the document to aid revision, like a standard cheat sheet, but avoid overdoing it.
* **Designing fill-in blanks:** Each blank should be answerable by someone who understands the topic well, even if they haven't watched our specific video. Establish and set up any examples used before asking about them.
* **Options:** Provide exactly three options per blank. This prevents students from trivially typing every option to guess. Distractors should be wrong but plausible.
* **Formatting options:** Options are displayed as plain grey text below the blank. This invites the user to actively type the answer, but gives them guardrails. The typing requirement may necessitate workarounds for complex math notation or subscripts (see "Matching and normalization" below).
* **Hint randomization:** Hint option order is shuffled at page load.
* **Sizing:** Blanks are sized via canvas-based text measurement of the answers; just make sure the longest possible answer is short enough that it does not break line wrapping.
* **Diagrams and visuals:** When doing statistical diagrams with SVG, do the math first to define where the actual labels should fall rather than guessing. For labels that need to be fillable, do not use `<text>` inside the SVG. Instead, position absolute `<div class="fig-lbl">` overlays above the SVG so each label can contain a real `.b` blank.
* **Color coding:** Use intelligent color coding. If an equation box color-codes `b₀` in blue and `b₁` in rose, the attached diagram should use the same colors for the intercept marker and slope triangle. Define per-metric semantic colors in `:root` (for example `--est-rose`, `--se-blue`, `--ci-navy`, `--t-green`, `--p-gold`) so the same metric uses the same color across boxes, vocab terms, metric cards, and figures.


## Page Architecture

Every sheet follows the same outer skeleton:

* **Toolbar** above the page with the cheat sheet title, short instructions, and the action buttons (zoom controls, Hide hints, Clear, Print / Save).
* **A `<section class="page" id="sheet">`** at exactly `210mm × 297mm` (A4), `display: flex; flex-direction: column;` with a small `gap` between children.
* **Header** with the `<h1>` lesson title, a one-line subtitle, and a gold `border-bottom` accent.
* **Body sections** built from a single primitive: a `.box` with a `.box-tag` strap and a tinted `.box-body`. Boxes can be stacked vertically or laid out side by side using `.row.cols-2`, `.row.split-40-60`, etc.
* **Footer** linking to `thegraphcourses.org`.

Keep the `.page` flex layout so children share vertical space predictably. Apply `min-width: 0` to flex and grid children that contain wrapping text or SVGs, otherwise long content forces the column wider than its track.

## Interactivity and Feedback

* **Typing requirements:** Students type their answers directly into the `<input class="fill">` blanks. Buttons are deliberately avoided.
* **Matching and normalization:** A `norm()` helper lowercases, trims, collapses whitespace, converts unicode subscripts (`₀₁₂₃₄`) to plain digits, replaces `β` with `beta`, and unifies the minus sign variants. This lets you accept "β₁" and "beta1" and "b1" as equivalent in a single comparison.
* **Canonical and alternate answers:** Use `data-answer="..."` for the canonical form and `data-alt="..."` for pipe-separated alternates. Example: `data-answer="0.386" data-alt="0.3863"` accepts both rounded and unrounded forms; `data-answer="not significant" data-alt="insignificant"` accepts two phrasings.
* **Correct answer feedback:** On the `input` event, normalize and compare. If it matches, add `.solved` to the wrapper (which hides the hint), recompute the blank width to fit the typed value, and trigger a `.correct-flash` animation: a green background fades in and out over ~0.75s while the underline briefly turns green. Then the user can keep typing without re-triggering the flash.
* **Wrong answer feedback:** On `blur`, if the value is non-empty and does not match, add `.wrong-pick` to apply a rose underline and color plus a `shake` animation. The wrong answer remains visible so students can see what they typed.
* **Width measurement:** Widths are computed in JavaScript using a hidden `<canvas>` and `measureText()` with the input's own font. Default (unsolved) width is `max(longestAnswerWidth + 12px, 22px)`. Solved width is `max(typedValueWidth + 6px, 12px)`. This keeps unfilled blanks consistent regardless of which answer is correct, and lets the line gently breathe when filled.
* **Verdict buttons:** For misconception rows, clicking a `.v-btn` adds `.picked` plus `.correct` or `.incorrect` to the button, marks the row `.revealed` (which displays the explanation), and disables both buttons. State persists under its own `localStorage` key.

## State Persistence

* **localStorage keys are per-sheet and content-fingerprinted.** Use a slug plus a short hash of the answer key, for example `"t-and-p-values-cheatsheet-1a2b3c4d"`. The fingerprint is computed at page load from each blank's `data-answer + "|" + data-alt` (and each verdict row's `data-correct`) joined in DOM order, hashed with a small inline `cyrb53` function. Any edit that adds, removes, reorders, or changes the correct answer for a blank changes the fingerprint, so old saves are silently ignored and the student starts fresh. Edits to surrounding prose, hint distractors, or styling do not invalidate progress.
* **Each sheet is self-contained:** the hash function lives inline in that sheet's `<script>` block, just like the rest of its JS. No shared helper file.
* **What is saved:** the array of typed values (in DOM order), the verdict selections (under `"<slug>-verdict-<fp>"` so it follows the same fingerprint), and the zoom index.
* **Zoom is not fingerprinted.** Its key is just `"<slug>-zoom"`. Zoom should survive content edits, since it is a user preference, not an answer.
* **What is not saved:** focus state, transient animations, or whether a row has been "revealed" beyond its picked answer (this is reconstructed on restore).

## Toolbar and Zoom

* **Toolbar:** Sticks above the page (not inside the printable area). Includes the title and short instructions on the left, and on the right a zoom group plus a Hide hints toggle, a Clear button, and the primary Print / Save button with a printer icon.
* **Hide hints:** Toggles `.hints-hidden` on the sheet, which sets `display: none` on every `.hint`. Useful for a final clean review pass.
* **Clear:** Resets all blanks, clears wrong/correct classes, resets blank widths, resets verdict rows, and writes the cleared state to `localStorage`.
* **Zoom:** A list of fixed zoom levels (`0.1, 0.25, 0.5, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0`), applied via CSS `transform: scale()` with `transform-origin: top center`. The `pageWrap` `min-height` is adjusted so the scaled page is fully scrollable. Zoom is hidden in print via `transform: none !important;`.

## Print and Layout Constraints

* **Single A4 page:** The sheet must print as exactly one A4 page (1122.5px at 96dpi). This is a hard constraint. Test with Playwright or a browser print preview at every stage.
* **Testing methodology:** Render the page in print media mode and measure the height of the page container. Do this for both the empty state (with hint text visible) and the filled state (with hints hidden). Both must be at or under 1122.5px. In print media, hide hint options with `display: none !important` so they do not affect page height.
* **Print media rules:** Inside `@media print`, the `.page` becomes `width: 210mm; height: 297mm; max-height: 297mm; overflow: hidden;`. The toolbar is hidden, `transform` is forced to `none !important`, transitions are removed, hints are hidden, the wrong-pick rose styling reverts to ink so a stray wrong answer does not print red, and verdict buttons collapse into a printed verdict badge.
* **`@page { size: A4; margin: 0; }`** is required so the browser does not add its own margins around the sheet.
* **Print color:** Add `print-color-adjust: exact;` and `-webkit-print-color-adjust: exact;` to every colored element (box tags, code blocks, diagrams, equation colors, metric cards, vocab terms, verdict rows). Without this, browsers strip backgrounds when printing.
* **Group dividers:** Use `border-top` for horizontal rules between sections, not `background-color`. Background-based dividers do not render in many print engines.
* **Tuning to fit:** When content overflows, you have several levers to adjust, roughly in order of least-disruptive to most:
  * **Page padding:** Can go as tight as `3mm top, 1.5mm bottom, 6mm left/right` in print. The screen version uses slightly more for visual breathing room.
  * **Gap between sections:** `1.2mm` on screen, `0.85mm` in print, is the practical floor before things look cramped.
  * **Box-body padding:** `1.5mm to 2mm` on screen, `0.9mm to 1.7mm` in print.
  * **Row gap:** `2mm` screen, `1.7mm` print is comfortable; go to `1.4mm` only if necessary.
  * **Font size:** `9pt` body text is the designed size; do not go below `8.4pt` screen / `8.2pt` print for body text. Headings, R code, hints, and small chip-style tags use smaller sizes by design.
* **Mobile responsive:** Under `@media (max-width: 820px)`, all `.row` and grid children collapse to a single column, the toolbar stacks, and the page padding shrinks. The sheet remains functional but no longer enforces A4 sizing.

## Theme Colors

Feel free to include other colors that are useful for visual clarity and pedagogical intent, but these are the foundational tokens:

| Token | Hex | Used for |
| --- | --- | --- |
| **Teal** | `#0d7377` | Primary brand, default box tags, best-fit line, blank underlines |
| **Gold** | `#f0c808` | Header accent, pitfall/warning boxes, data points |
| **Rose** | `#b83b5e` | Wrong-answer feedback, sample slope highlight |
| **Green** | `#2c8a4a` | Correct-answer flash |
| **Ink** | `#182326` | Body text |
| **Muted** | `#5d6d72` | Secondary text, axis labels |

