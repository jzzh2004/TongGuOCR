# TongGuOCR Project Page

The visual identity uses the official TongGu logo from the [SCUT-DLVCLab TongGu-LLM repository](https://github.com/SCUT-DLVCLab/TongGu-LLM/blob/main/images/%E9%80%9A%E5%8F%A4logo.png), stored locally as `public/tonggu-logo.png` for stable static deployment.

TongGuOCR 的英文论文展示页。页面包含完整摘要、4 张论文图、9 张论文表格、核心指标、外部 Demo 入口，以及可直接用于 GitHub Pages 的自动部署配置。

## 本地预览

需要 Node.js 22 或更高版本。

```bash
npm install
npm run dev
```

浏览器访问终端显示的本地地址。生产构建与完整内容审计：

```bash
npm test
```

构建结果位于 `dist/`。

## 修改链接与内容

所有论文内容和顶部按钮都集中在 `src/content/project.ts`：

```ts
links: [
  { label: "Paper", url: null, meta: "Coming soon" },
  { label: "Demo", url: "http://121.41.49.212:6767/", meta: "Open demo" },
  { label: "GitHub", url: "https://github.com/jzzh2004/TongGuOCR", meta: "Source code" },
  { label: "Hugging Face", url: null, meta: "Coming soon" },
]
```

- 将 `null` 替换为完整 URL，Paper 或 GitHub 按钮就会自动启用。
- 修改 `Demo` 的 `url` 即可替换外部 Demo 地址。
- 图表 caption、alt 文本、作者和摘要也在同一个文件维护。

## 从 PDF 重新导出图表

项目内保留了可复现的导出脚本。先安装依赖：

```bash
python -m pip install pypdfium2 pillow
python scripts/extract-paper-assets.py "/path/to/TongguOCR_journal.pdf"
```

脚本会覆盖 `public/paper-assets/` 中的 4 张图和 9 张表。若论文排版发生改变，需要同步调整脚本中的 PDF 裁切坐标。

## 部署到 GitHub Pages

1. 在 GitHub 创建仓库，并将本目录作为仓库根目录推送到 `main` 分支。
2. 打开仓库的 **Settings → Pages**。
3. 将 **Build and deployment → Source** 设置为 **GitHub Actions**。
4. 推送后，`.github/workflows/deploy.yml` 会自动构建并发布网站。

Vite 使用相对资源路径，因此既可部署到 `username.github.io`，也可部署到 `username.github.io/repository/`。

## 项目结构

```text
src/
  App.tsx                  页面结构与交互
  styles.css               完整视觉系统和响应式样式
  content/project.ts       可编辑论文内容与链接
public/
  paper-assets/            从论文原稿高分辨率导出的全部图表
  og.png                   社交分享预览图
scripts/
  extract-paper-assets.py  PDF 图表导出脚本
tests/
  content-audit.test.mjs   图表编号、链接和元数据审计
```

## 说明

- 本项目仅展示论文内容，不包含 OCR 推理后端。
- Paper 与 GitHub 在 URL 为空时显示为 `Coming soon`，点击不会跳转。
- 页面结构参考了 DreamStory 和 Academic Project Page Template，并在页脚保留致谢链接。
