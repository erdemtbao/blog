---
title: "Paper Notes: Robot Learning (3)"
published: 2026-03-08
description: Reading notes—language-grounded spatial manipulation (VoxPoser, CoPa, ReKep) and Robot_Learning_3 assets.
image: ''
tags: [Paper Notes, Robot Learning]
category: Paper Notes
draft: false
---

本页对应 `assets/paper_node/Robot_Learning_3/`（线上：`public/paper-node/Robot_Learning_3/`）。版式同 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/)、[Robot Learning (2)](/blog/posts/paper-notes-robot-learning-2/)：**卡片**内元信息 + **BibTeX 折叠** + 笔记；`::::paper` 四个冒号。**排序**：从早期到近期——**VoxPoser（CoRL 2023）→ CoPa（arXiv 2024）→ ReKep**；均为大模型接地 + 显式几何/优化。

:::tip[PDF / 图片放哪？]
`/blog/paper-node/Robot_Learning_3/<子目录>/<文件>`
:::

## 目录

| 简称 | 论文（短） | 跳转 |
|:---:|:---|:---:|
| VoxPoser | LLM 合成 3D value maps 操作 | [↓](#voxposer) |
| CoPa | 部件空间约束 + 基础模型操作 | [↓](#copa) |
| ReKep | 关系关键点时空约束 | [↓](#rekep) |

::::paper{tone="voxposer"}

## VoxPoser

**论文：** *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*

**主要机构：** Google DeepMind / Stanford 合作线（Huang 等，以论文为准）

**会议 / 年份：** Conference on Robot Learning (CoRL)，2023

:::note[一句话]
用 **LLM/VLM** 从自然语言推断 affordance 与约束，在观测空间上**组合 3D value map**，再在基于模型的规划里生成 **6-DoF 密集路点**；强调可组合、少预定义原语。
:::

**材料：** [Paper](https://arxiv.org/abs/2307.05973) · [Project](https://voxposer.github.io/) · [Code](https://github.com/huangwl18/VoxPoser)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{huang2023voxposer,
  title     = {{VoxPoser}: Composable {3D} Value Maps for Robotic Manipulation with Language Models},
  author    = {Huang, Wenlong and others},
  booktitle = {Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.05973}
}</code></pre>
</details>

**我记住的三点**

1.
2.
3.

**方法 / 实现（想写再写）**

—

**实验里印象最深**

—

**疑问或想继续看的**

—

::::paper{tone="copa"}

## CoPa

**论文：** *CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models*

**主要机构：** （以论文署名页为准；常用 VLM 接地与操作管线）

**会议 / 年份：** IEEE RO-MAN 等（正式信息以 IEEE Xplore / 论文页为准）；arXiv 2024

:::note[一句话]
用 **GPT-4V 等 VLM** 做**面向任务的抓取部位**与**任务相关部件的几何约束**，再规划抓取后 6-DoF 位姿；强调开放指令、少训练。
:::

**材料：** [Paper](https://arxiv.org/abs/2403.08248) · [Project](https://copa-2024.github.io/) · Code（以项目页为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{copa2024,
  title         = {{CoPa}: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models},
  year          = {2024},
  eprint        = {2403.08248},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2403.08248}
}</code></pre>
</details>

**我记住的三点**

1.
2.
3.

**方法 / 实现（想写再写）**

—

**实验里印象最深**

—

**疑问或想继续看的**

—

::::paper{tone="rekep"}

## ReKep

**论文：** *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation*

**主要机构：** Stanford 等（以论文为准）

**会议 / 年份：** 以论文最终录用为准（OpenReview / 会议 proceedings 核对）

:::note[一句话]
把任务写成 **关系关键点上的时空约束**（可微代价）；大模型从语言与观测生成约束，再 **分层优化** 末端位姿，支持多阶段、双手与反应式行为。
:::

**材料：** [Paper](https://arxiv.org/abs/2409.01652) · [Project](https://rekep-robot.github.io/) · [Code](https://github.com/huangwl18/ReKep)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{rekep2024,
  title         = {{ReKep}: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation},
  year          = {2024},
  eprint        = {2409.01652},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2409.01652}
}</code></pre>
</details>

**我记住的三点**

1.
2.
3.

**方法 / 实现（想写再写）**

—

**实验里印象最深**

—

**疑问或想继续看的**

—

**和本页其他论文**

- [VoxPoser](#voxposer) → [CoPa](#copa) → [ReKep](#rekep)：从 **3D value map 组合** 到 **部件约束** 再到 **关系关键点时空约束**；与 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/) 中端到端 policy / VLA **对照阅读**。

## 新增一篇时怎么写（极简）

1. `Robot_Learning_3` 下建子目录；复制 **卡片 + BibTeX + 笔记**。  
2. 新 `tone` 在 `markdown.css` 增加 `[data-paper-tone="…"]`。  
3. 目录表加一行。

---

_Changelog：2026-03 — 由 Imitation Learning (1) 改为 Robot Learning (3)，按 `Robot_Learning_3` 三篇模板化。_
