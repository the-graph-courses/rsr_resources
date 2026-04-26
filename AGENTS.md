# Notes for AI Assistants

## Regression Diagnostics Build

- `regression-diagnostics.jsx` is the source of truth for the React explorer.
- `regression-diagnostics.bundle.js` is generated. Do not edit it by hand.
- After editing `regression-diagnostics.jsx`, run `npm run build`.
- `regression-diagnostics.html` loads the generated bundle directly. Do not reintroduce runtime Babel, `fetch`, or `new Function` loading.
- Use `npm run watch` while actively editing the React explorer.

## Static Explorers

The other `*-explorer.html` files are self-contained static HTML/JS resources and do not use this build step.
