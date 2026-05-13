# Interactive Statistics Resources

Interactive teaching tools for statistics concepts, built by [The GRAPH Courses](https://thegraphcourses.org).

## Resources

| Resource | File | Description |
|---|---|---|
| Mean & SD Explorer | `explorers/mean-sd-explorer.html` (short URL: `mean-sd-explorer.html`) | Click to add points and watch the mean and standard deviation update live |
| Height Mean & SD Explorer | `explorers/height-explorer.html` (short URL: `height-explorer.html`) | Add stick-figure people and see mean height and SD update with horizontal bands |
| Least Squares Explorer | `explorers/least-squares-explorer.html` (short URL: `least-squares-explorer.html`) | Drag sliders to fit a regression line and minimize the sum of squared residuals |
| Slope Standard Error | `explorers/slope-standard-error.html` (short URL: `slope-standard-error.html`) | Explore what affects the precision of the slope estimate through simulation |
| Regression Assumptions Explorer | `explorers/regression-diagnostics.html` (short URL: `regression-diagnostics.html`) | Explore regression diagnostics, assumptions, and practical fixes |

## Usage

Open `index.html` in a browser. Use the sidebar to navigate between resources.

## Development

Interactive implementations live in `explorers/`. Root-level HTML files with the same names are thin full-page iframes so existing short links keep working. The regression assumptions explorer is built from `regression-diagnostics.jsx` into `explorers/regression-diagnostics.bundle.js`.

```bash
npm install
npm run build
```

Run `npm run build` after editing `regression-diagnostics.jsx`. The generated bundle is committed so the site can still be deployed as static files.
