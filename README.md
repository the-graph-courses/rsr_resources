# Interactive Statistics Resources

Interactive teaching tools for statistics concepts, built by [The GRAPH Courses](https://thegraphcourses.org).

## Resources

| Resource | File | Description |
|---|---|---|
| Mean & SD Explorer | `explorers/mean-sd-explorer.html` | Click to add points and watch the mean and standard deviation update live |
| Height Mean & SD Explorer | `explorers/height-explorer.html` | Add stick-figure people and see mean height and SD update with horizontal bands |
| Least Squares Explorer | `explorers/least-squares-explorer.html` | Drag sliders to fit a regression line and minimize the sum of squared residuals |
| Slope Standard Error | `explorers/slope-standard-error.html` | Explore what affects the precision of the slope estimate through simulation |
| Regression Assumptions Explorer | `explorers/regression-diagnostics.html` | Explore regression diagnostics, assumptions, and practical fixes |
| Odds Ratio Explorer | `explorers/odds-ratio-explorer.html` | Enter an odds ratio and see how the probability change depends on the starting probability |

## Usage

Open `index.html` in a browser. Use the sidebar to navigate between resources.

## Development

Interactive implementations live in `explorers/`. The regression diagnostics explorer is built from `explorers/regression-diagnostics.jsx` into `explorers/regression-diagnostics.bundle.js`.

```bash
npm install
npm run build
```

Run `npm run build` after editing `explorers/regression-diagnostics.jsx`. The generated bundle is committed so the site can still be deployed as static files.
