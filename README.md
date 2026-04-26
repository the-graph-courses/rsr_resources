# Interactive Statistics Resources

Interactive teaching tools for statistics concepts, built by [The GRAPH Courses](https://thegraphcourses.org).

## Resources

| Resource | File | Description |
|---|---|---|
| Mean & SD Explorer | `mean-sd-explorer.html` | Click to add points and watch the mean and standard deviation update live |
| Least Squares Explorer | `least-squares-explorer.html` | Drag sliders to fit a regression line and minimize the sum of squared residuals |
| Slope Standard Error | `slope-standard-error.html` | Explore what affects the precision of the slope estimate through simulation |
| Regression Assumptions Explorer | `regression-diagnostics.html` | Explore regression diagnostics, assumptions, and practical fixes |

## Usage

Open `index.html` in a browser. Use the sidebar to navigate between resources.

## Development

Most resources are self-contained HTML files. The regression assumptions explorer is built from `regression-diagnostics.jsx` into `regression-diagnostics.bundle.js`.

```bash
npm install
npm run build
```

Run `npm run build` after editing `regression-diagnostics.jsx`. The generated bundle is committed so the site can still be deployed as static files.
