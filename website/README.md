# Project page

[简体中文](README.zh-CN.md) · [Main guide](../README.md)

The site uses React, TypeScript, and Vite, independently of the Python
inference environment. It requires Node.js 22.12 or newer; the GitHub Pages
workflow uses Node.js 22.

```bash
cd website
npm ci
npm run dev
```

`npm test` runs type checking, a production build, and content checks. Build
output goes to `dist/`; `npm run preview` serves it locally. A push to `main`
or a manual run of `.github/workflows/deploy.yml` deploys to GitHub Pages.

- `src/content/project.ts`: paper metadata, authors, external links, and figure/table lists.
- `src/App.tsx`, `src/styles.css`: page components and styles.
- `public/`: generated images and static assets used directly by normal builds.
- `scripts/`: optional asset maintenance tools. `extract-paper-assets.py <PDF path>`
  crops fixed regions from the matching paper PDF and requires `pypdfium2`
  and Pillow. `build-favicon.ps1` generates the favicon from the logo in
  Windows PowerShell.

Run asset maintenance tools from `website/`. Building the site does not require
the paper PDF or these Python dependencies.
