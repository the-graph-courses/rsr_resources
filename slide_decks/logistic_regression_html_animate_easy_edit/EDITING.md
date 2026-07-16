# Editing this deck

Open `index.html` and search for `EDIT PANEL 1 HERE` (or the panel 2–4 bullet lists).

## Panel 1 (left column)

Everything is ordinary stacked HTML. Edit in place; keep `data-step` for when each piece appears:

```html
<p class="frag bullet" data-step="2">Load the data in R.</p>
<pre class="frag code" data-step="2">klosa &lt;- read_csv("klosa_men_grip_45plus.csv")</pre>
<pre class="frag out" data-step="3"># A tibble: 4 × 2
...</pre>
```

- `frag bullet` = teaching text
- `frag code` = R code
- `frag out` = console output
- Same `data-step` value → appear together

## Panels 2–4

Still list bullets:

```html
<li data-step="5">Grip tends to be <strong>lower at older ages</strong>.</li>
```

- Use `<strong>...</strong>` for teal emphasis.
- Use `<span class="bad">...</span>` for red emphasis.

The `STEP_CHANGES` list near the bottom of `index.html` controls only the charts,
equations, and active panel (and the code boxes on panels 2 and 4).

## Images

Images can be ordinary HTML with a reveal step. The panel 1 photograph is an
example:

```html
<figure class="inset-photo" data-step="4">
  <img src="images/grip_dynamometer.jpg" alt="Description of the image">
  <figcaption>hand dynamometer</figcaption>
</figure>
```

Its size and position are controlled by `.inset-photo` in the CSS near the top
of `index.html`. No image-specific JavaScript is required.

The original deck remains unchanged in `../logistic_regression_html_animate/`.
