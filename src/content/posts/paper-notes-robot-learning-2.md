---
title: "Paper Notes: Robot Learning (2)"
published: 2026-03-20
description: Reading notes—VLA variants, π0 family, hierarchical / agentic robot learning (Robot_Learning_2).
image: ''
tags: [Paper Notes, Robot Learning]
category: Paper Notes
draft: false
---

本页对应本地资料目录 `assets/paper_node/Robot_Learning_2/`（线上请同步到 `public/paper-node/Robot_Learning_2/`）。版式与 [Paper Notes: Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/) 相同。**排序**：从简到繁、从早期到近期——**小/轻量 VLA → π 家族（Pi0→FAST→0.5→0.6）→ 世界模型式 VLA → 分层与长程智能体**；**同一系列相邻**（π 线、AgiBot 相关可对照）。**卡片内**元信息 + **卡片外** BibTeX 折叠 + 笔记；`::::paper` 须四个冒号。

:::tip[PDF / 图片放哪？]
链接形式：`/blog/paper-node/Robot_Learning_2/<子目录>/<文件名>`。
:::

## 目录

| 简称 | 论文（短） | 跳转 |
|:---:|:---|:---:|
| TinyVLA | Fast data-efficient small VLA | [↓](#tinyvla) |
| VLA-Adapter | Tiny-scale VLA + bridge attention | [↓](#vla-adapter) |
| Pi0 | Flow VLA generalist policy | [↓](#pi0) |
| Pi0 FAST | FAST action tokenization + VLA | [↓](#pi0-fast) |
| Pi0.5 | Open-world generalization VLA | [↓](#pi05) |
| Pi0.6 | Experience / RECAP (π*0.6) | [↓](#pi06) |
| WorldVLA | VLA + autoregressive world model | [↓](#worldvla) |
| Hi Robot | Hierarchical VLA open-ended instructions | [↓](#hi-robot) |
| ACoT-VLA | Action chain-of-thought for VLA | [↓](#acot-vla) |
| RoboClaw | Agentic long-horizon VLA framework | [↓](#roboclaw) |

::::paper{tone="tinyvla"}

## TinyVLA

**论文：** *TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation*

**主要机构：** （请按论文署名核对）

**会议 / 年份：** IEEE Robotics and Automation Letters (RA-L)，2025（见论文页）

:::note[一句话]
**小快省 VLA**：强调推理延迟与数据效率；微调阶段结合 **扩散策略式解码** 等，在真机上与 OpenVLA 等对比速度与效果。
:::

**材料：** [Paper](https://arxiv.org/abs/2409.12514) · [Project](https://tiny-vla.github.io/) · [Code](https://github.com/liyaxuanliyaxuan/TinyVLA)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{tinyvla2024,
  title   = {{TinyVLA}: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation},
  journal = {arXiv preprint arXiv:2409.12514},
  year    = {2024},
  url     = {https://arxiv.org/abs/2409.12514}
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

::::paper{tone="vlaadapter"}

## VLA-Adapter

**论文：** *VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model*

**主要机构：** （请按论文署名核对）

**会议 / 年份：** arXiv 2025（正式会议以论文页为准）

:::note[一句话]
**极小骨干 + 轻量 Policy 模块**（如 Bridge Attention）：少参数、快微调，在 LIBERO 等基准上对标大 VLA。
:::

**材料：** [Paper](https://arxiv.org/abs/2509.09372) · [Project](https://vla-adapter.github.io/) · [Code](https://github.com/OpenHelix-Team/VLA-Adapter)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{vlaadapter2025,
  title         = {{VLA-Adapter}: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  year          = {2025},
  eprint        = {2509.09372},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2509.09372}
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

::::paper{tone="pi0"}

## Pi0

**论文：** *π₀: A Vision-Language-Action Flow Model for General Robot Control*

**主要机构：** Physical Intelligence (π)

**会议 / 年份：** arXiv 2024

:::note[一句话]
**流匹配 (flow matching)** 的 VLA：VLM 骨干 + 动作专家，连续动作生成；与扩散策略并列的生成式控制路线。
:::

**材料：** [Paper](https://arxiv.org/abs/2410.24164) · [Blog](https://www.physicalintelligence.company/blog/pi0) · [Code](https://github.com/Physical-Intelligence/openpi)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{black2024pi0,
  title         = {pi0: A Vision-Language-Action Flow Model for General Robot Control},
  author        = {Black, Kevin and others},
  year          = {2024},
  eprint        = {2410.24164},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2410.24164}
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

::::paper{tone="pi0fast"}

## Pi0 FAST

**论文：** *FAST: Efficient Robot Action Tokenization*（与 π₀-FAST 模型配套）

**主要机构：** Physical Intelligence (π)

**会议 / 年份：** 技术报告 / 博客与 PDF（以 π 官网为准）

:::note[一句话]
**频域式动作离散化 (FAST)**：用 DCT 等压缩动作序列，使 **自回归 Transformer** 在高频灵巧操作上接近 flow/diffusion，并加速训练。
:::

**材料：** [Paper / PDF](https://www.pi.website/research/fast) · [Blog](https://www.pi.website/research/fast) · [Code](https://github.com/Physical-Intelligence/openpi)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{pi2024fast,
  title = {{FAST}: Efficient Robot Action Tokenization},
  note  = {Physical Intelligence; https://www.pi.website/research/fast},
  year  = {2024}
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

::::paper{tone="pi05"}

## Pi0.5

**论文：** *（产品级技术博文）π0.5 — 开放环境泛化*

**主要机构：** Physical Intelligence (π)

**会议 / 年份：** 博客 / 技术报告 2025

:::note[一句话]
在 π0 之上强调 **开放世界与新环境泛化**（如家庭等）；技术细节以官方博客与 openpi 发布说明为准。
:::

**材料：** [Blog](https://www.physicalintelligence.company/blog/pi05) · [openpi](https://github.com/Physical-Intelligence/openpi) · Code同上

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{pi2025pi05,
  title = {Physical Intelligence pi0.5 model release},
  note  = {https://www.physicalintelligence.company/blog/pi05},
  year  = {2025}
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

::::paper{tone="pi06"}

## Pi0.6

**论文：** *π*0.6 与 RECAP — 从经验中学习（官方博文）*

**主要机构：** Physical Intelligence (π)

**会议 / 年份：** 博客 / 技术报告 2025–2026

:::note[一句话]
在 π0.6 基座上引入 **RECAP** 等，从演示、纠正与自主试错中改进策略，超越纯模仿；多模态输出（子任务文本、FAST、连续动作块等以模型卡为准）。
:::

**材料：** [Blog](https://www.pi.website/blog/pistar06) · [openpi](https://github.com/Physical-Intelligence/openpi) · Code同上

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{pi2026pistar,
  title = {Physical Intelligence pi-star-0.6 and RECAP},
  note  = {https://www.pi.website/blog/pistar06},
  year  = {2026}
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

::::paper{tone="worldvla"}

## WorldVLA

**论文：** *WorldVLA: Towards Autoregressive Action World Model*

**主要机构：** 阿里巴巴达摩院等（以论文作者为准）

**会议 / 年份：** arXiv 2025

:::note[一句话]
把 **VLA 与世界模型**统一：自回归同时建模动作与未来视觉，用注意力掩码等缓解长序列误差累积；与「只输出动作」的 VLA 对照读。
:::

**材料：** [Paper](https://arxiv.org/abs/2506.21539) · Project（以论文 / Hugging Face 页为准） · Code（以官方为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{worldvla2025,
  title         = {{WorldVLA}: Towards Autoregressive Action World Model},
  year          = {2025},
  eprint        = {2506.21539},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.21539}
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

::::paper{tone="hirobot"}

## Hi Robot

**论文：** *Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models*

**主要机构：** Physical Intelligence (π) 等（以论文署名页为准）

**会议 / 年份：** International Conference on Machine Learning (ICML)，2025

:::note[一句话]
**分层 VLA**：先由 VLM 做高层推理与步骤分解，再执行底层动作，支持开放式指令与交互反馈（长程任务）。
:::

**材料：** [Paper](https://arxiv.org/abs/2502.19417) · [Project](https://www.pi.website/research/hirobot) · [Code](https://github.com/Physical-Intelligence/openpi)（实现以 π 官方仓库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{hirobot2025,
  title         = {{Hi Robot}: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models},
  year          = {2025},
  eprint        = {2502.19417},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.19417}
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

::::paper{tone="acot"}

## ACoT-VLA

**论文：** *ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models*

**主要机构：** 智元机器人（AgiBot）等（开源与 AgiBot World Challenge 基线相关）

**会议 / 年份：** arXiv 2026（录用信息以官网为准，如 CVPR 2026 等）

:::note[一句话]
在 **动作空间** 里做链式推理（而非仅语言 CoT）：显式轨迹参考 + 从 VLM 隐表示抽动作先验，缩小语义–运动鸿沟。
:::

**材料：** [Paper](https://arxiv.org/abs/2601.11404) · [Project](https://github.com/AgibotTech/ACoT-VLA) · [Code](https://github.com/AgibotTech/ACoT-VLA)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{acotvla2026,
  title         = {{ACoT-VLA}: Action Chain-of-Thought for Vision-Language-Action Models},
  year          = {2026},
  eprint        = {2601.11404},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2601.11404}
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

::::paper{tone="roboclaw"}

## RoboClaw

**论文：** *RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks*

**主要机构：** （AgiBot / 合作方，以论文为准）

**会议 / 年份：** arXiv 2026（预印本）

:::note[一句话]
**智能体式机器人框架**：用 VLM 统一数据采集、策略学习与长程执行；**纠缠动作对 (EAP)** 等机制支持自复位与闭环迭代。
:::

**材料：** [Paper](https://arxiv.org/abs/2603.11558) · [Project](https://roboclaw-agibot.github.io/) · Code（以项目页为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{roboclaw2026,
  title         = {{RoboClaw}: An Agentic Framework for Scalable Long-Horizon Robotic Tasks},
  year          = {2026},
  eprint        = {2603.11558},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.11558}
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

- [TinyVLA](#tinyvla) / [VLA-Adapter](#vla-adapter)：小模型、高效率 VLA 与 OpenVLA 大模型对照。  
- [Pi0](#pi0) → [Pi0 FAST](#pi0-fast) → [Pi0.5](#pi05) → [Pi0.6](#pi06)：π 家族 **flow → FAST 自回归 → 泛化 → 经验学习**。  
- [WorldVLA](#worldvla)：把 **世界模型**写进同一条自回归框架。  
- [Hi Robot](#hi-robot) / [ACoT-VLA](#acot-vla) / [RoboClaw](#roboclaw)：分层、**动作空间推理** 与长程智能体三条线。

## 新增一篇时怎么写（极简）

1. 在 `Robot_Learning_2` 下建子文件夹；复制 **卡片 + BibTeX + 笔记**。  
2. 新 `tone` 在 `markdown.css` 增加 `[data-paper-tone="…"]`。  
3. 目录表加一行；链接 `/blog/paper-node/Robot_Learning_2/...`。

---

_Changelog：2026-03 — 按 `Robot_Learning_2` 目录补全 10 篇模板。_
