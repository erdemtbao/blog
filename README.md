<h1 id="readme-top">Erdemt's Blog</h1>

<p align="center">
  <strong>Language / 语言</strong><br><br>
  <a href="#english-section"><img src="https://img.shields.io/badge/README-English-blue?style=for-the-badge" alt="English README"></a>
  &nbsp;
  <a href="#chinese-section"><img src="https://img.shields.io/badge/README-中文-red?style=for-the-badge" alt="中文 README"></a>
</p>

---

<h2 id="english-section">English</h2>

**Personal technical blog** by **Erdemt Bao**, focused on **robot learning**, research notes, tools, and longer-form writing.

**Live site:** [https://erdemtbao.github.io/blog/](https://erdemtbao.github.io/blog/)

Built with [Astro](https://astro.build/) and deployed to GitHub Pages via GitHub Actions (`base: /blog`).

### Tech stack

- [Astro](https://astro.build/) + [Tailwind CSS](https://tailwindcss.com)
- Theme and features from the **Fuwari** template (light/dark mode, search, RSS, extended Markdown, etc.)
- Search: [Pagefind](https://pagefind.app/)

### Local development

Requires **Node.js ≥ 20** and **pnpm ≥ 9**.

```sh
pnpm install
pnpm dev          # http://localhost:4321
pnpm build        # output to ./dist/
pnpm preview      # preview production build
```

Edit [`src/config.ts`](src/config.ts) for site title, nav, and profile. New posts: `pnpm new-post <filename>` → `src/content/posts/`.

### Post frontmatter example

```yaml
---
title: My Post
published: 2024-01-01
description: Short summary
image: ./cover.jpg
tags: [Tag1, Tag2]
category: Notes
draft: false
---
```

### Acknowledgments

This site is based on the open-source **[Fuwari](https://github.com/saicaca/fuwari)** template. Thanks to **[saicaca](https://github.com/saicaca)** and all contributors. For the latest template docs and updates, see the upstream repository.

### License

This project follows Fuwari’s **MIT License**; see the license file in the repository.

<p align="right"><a href="#readme-top">↑ Top</a> · <a href="#chinese-section">中文</a></p>

---

<h2 id="chinese-section">中文</h2>

**个人技术博客** — 由 **Erdemt Bao** 维护，主要记录与 **机器人学习（Robot Learning）**、科研笔记、工具与长文相关的内容。

**在线站点：** [https://erdemtbao.github.io/blog/](https://erdemtbao.github.io/blog/)

本仓库基于静态站点生成器 [Astro](https://astro.build/) 构建，通过 GitHub Actions 部署至 GitHub Pages（`base: /blog`）。

### 技术栈

- [Astro](https://astro.build/) + [Tailwind CSS](https://tailwindcss.com)
- 主题与功能源自 **Fuwari** 模板（亮色 / 暗色、搜索、RSS、Markdown 扩展语法等）
- 搜索：[Pagefind](https://pagefind.app/)

### 本地开发

环境要求：**Node.js ≥ 20**、**pnpm ≥ 9**。

```sh
pnpm install
pnpm dev          # http://localhost:4321
pnpm build        # 输出到 ./dist/
pnpm preview      # 本地预览构建结果
```

站点标题、导航、侧栏等请在 [`src/config.ts`](src/config.ts) 中修改。新文章可使用 `pnpm new-post <filename>`，正文位于 `src/content/posts/`。

### 文章 Frontmatter 示例

```yaml
---
title: My Post
published: 2024-01-01
description: Short summary
image: ./cover.jpg
tags: [Tag1, Tag2]
category: Notes
draft: false
---
```

### 致谢

本站使用开源博客模板 **[Fuwari](https://github.com/saicaca/fuwari)**，感谢作者 **[saicaca](https://github.com/saicaca)** 及所有贡献者的设计与维护。若你也想搭建类似站点，可从上游仓库获取最新文档与更新。

### 许可证

项目沿用 Fuwari 的 **MIT License**，详见仓库内许可文件。

<p align="right"><a href="#readme-top">↑ 顶部</a> · <a href="#english-section">English</a></p>
