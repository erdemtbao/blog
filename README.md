# Erdemt's Blog

> **中文 / English — 点按切换全文**  
> GitHub 上的 README 不能运行脚本，无法在仓库首页里「只显示一种语言再切换」。  
> **→ 在浏览器中打开：[readme-lang.html](https://erdemtbao.github.io/blog/readme-lang.html)**（部署后可用；本地预览：`pnpm dev` 后访问 `http://localhost:4321/readme-lang.html`）

---

**Personal technical blog** by **Erdemt Bao**, focused on **robot learning**, research notes, tools, and longer-form writing.

**Live site:** [https://erdemtbao.github.io/blog/](https://erdemtbao.github.io/blog/)

Built with [Astro](https://astro.build/) and deployed to GitHub Pages via GitHub Actions (`base: /blog`).

## Tech stack

- [Astro](https://astro.build/) + [Tailwind CSS](https://tailwindcss.com)
- Theme and features from the **Fuwari** template (light/dark mode, search, RSS, extended Markdown, etc.)
- Search: [Pagefind](https://pagefind.app/)

## Local development

Requires **Node.js ≥ 20** and **pnpm ≥ 9**.

```sh
pnpm install
pnpm dev          # http://localhost:4321
pnpm build        # output to ./dist/
pnpm preview      # preview production build
```

Edit [`src/config.ts`](src/config.ts) for site title, nav, and profile. New posts: `pnpm new-post <filename>` → `src/content/posts/`.

## Post frontmatter example

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

## Acknowledgments

This site is based on the open-source **[Fuwari](https://github.com/saicaca/fuwari)** template. Thanks to **[saicaca](https://github.com/saicaca)** and all contributors. For the latest template docs and updates, see the upstream repository.

## License

This project follows Fuwari’s **MIT License**; see the license file in the repository.
