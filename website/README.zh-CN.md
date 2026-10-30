# 项目主页

[English](README.md) · [主文档](../README.zh-CN.md)

使用 React、TypeScript 和 Vite，独立于 Python 推理环境。需要 Node.js
22.12 或更新版本；GitHub Pages 工作流使用 Node.js 22。

```bash
cd website
npm ci
npm run dev
```

`npm test` 执行类型检查、生产构建和内容检查，产物位于 `dist/`。
`npm run preview` 本地预览生产构建。推送 `main` 或手动触发
`.github/workflows/deploy.yml` 可部署 GitHub Pages。

- `src/content/project.ts`：论文信息、作者、外部链接和图表列表。
- `src/App.tsx`、`src/styles.css`：页面和样式。
- `public/`：已生成的图片和静态资源，普通构建直接使用。
- `scripts/`：可选的素材维护工具。`extract-paper-assets.py <PDF路径>`
  从对应版本的论文 PDF 按固定位置裁图，需要 `pypdfium2` 和 Pillow；
  `build-favicon.ps1` 在 Windows PowerShell 中从 logo 生成 favicon。

素材维护工具从 `website/` 目录运行；构建网站无需论文 PDF 或这些 Python 依赖。
