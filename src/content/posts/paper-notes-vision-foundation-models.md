---
title: "Paper Notes: Vision Foundation Models"
published: 2026-02-21
description: Reading notes—CLIP, DINO, SAM, VLM backbones (Vision_Foundation_Models assets).
image: ''
tags: [Paper Notes, Vision Foundation Models]
category: Paper Notes
draft: false
---

对应 `assets/paper_note/Vision_Foundation_Models/`（线上：`public/paper-note/Vision_Foundation_Models/`）。卡片模板与 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/) 一致；此处多为**视觉/多模态基础模型**，不单篇机器人 policy。**排序**：从简到繁、从早期到近期，**同一系列相邻**（如 CLIP→SigLIP，DINOv2→DINOv3，SAM3→SAM3D）。

:::tip[PDF / 图片放哪？]
`/blog/paper-note/Vision_Foundation_Models/<子目录>/<文件>`
:::

## 目录

| 简称 | 说明 | 跳转 |
|:---:|:---|:---:|
| CLIP | 图文对比学习 | [↓](#clip) |
| SigLIP | 改进的图文对比训练 | [↓](#siglip) |
| DINOv2 | 自监督稠密特征 | [↓](#dinov2) |
| DINOv3 | 大规模 SSL 视觉基础模型 | [↓](#dinov3) |
| LLaVA | 视觉指令微调 LLM | [↓](#llava) |
| Qwen-VL | 通义千问视觉语言模型 | [↓](#qwen-vl) |
| InternVL | 开源大规模 VLM 系列 | [↓](#internvl) |
| FoundationPose | 统一 6D 位姿估计 | [↓](#foundationpose) |
| SAM3 | Segment Anything 3 | [↓](#sam3) |
| SAM3D | 图像到 3D（Meta） | [↓](#sam3d) |

::::paper{tone="clip"}

## CLIP

**论文：** *Learning Transferable Visual Models From Natural Language Supervision*

**主要机构：** OpenAI

**会议 / 年份：** International Conference on Machine Learning (ICML)，2021

:::note[一句话]
**图像–文本对比学习**得到可迁移视觉编码器；开放集分类、检索与大量 VLM 的视觉塔基础。
:::

**材料：** [Paper](https://arxiv.org/abs/2103.00020) · [Blog](https://openai.com/research/clip) · [Code](https://github.com/openai/CLIP)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{radford2021learning,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle = {International Conference on Machine Learning},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.00020}
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

::::paper{tone="siglip"}

## SigLIP

**论文：** *Sigmoid Loss for Language Image Pre-Training*

**主要机构：** Google Research

**会议 / 年份：** ICCV 2023 Workshop / arXiv 2023

:::note[一句话]
用 **sigmoid 损失**替代 softmax 对比学习，改善大规模图文训练稳定性与效率；**SigLIP** 常作新一代视觉–语言对齐骨干（如 OpenVLA 等）。
:::

**材料：** [Paper](https://arxiv.org/abs/2303.15343) · [Code](https://github.com/google-research/big_vision)（big_vision 中实现） · Project（以 Google 文档为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{zhai2023sigmoid,
  title   = {Sigmoid Loss for Language Image Pre-Training},
  author  = {Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal = {arXiv preprint arXiv:2303.15343},
  year    = {2023},
  url     = {https://arxiv.org/abs/2303.15343}
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

::::paper{tone="dinov2"}

## DINOv2

**论文：** *DINOv2: Learning Robust Visual Features without Supervision*

**主要机构：** Meta AI

**会议 / 年份：** Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / arXiv 2023

:::note[一句话]
**纯视觉自监督**，强稠密特征与对应关系；检测、分割、深度等下游常用 backbone，机器人里常作几何/关键点特征。
:::

**材料：** [Paper](https://arxiv.org/abs/2304.07193) · [Project](https://ai.meta.com/dinov2/) · [Code](https://github.com/facebookresearch/dinov2)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{oquab2023dinov2,
  title   = {{DINOv2}: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and others},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023},
  url     = {https://arxiv.org/abs/2304.07193}
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

::::paper{tone="dinov3"}

## DINOv3

**论文：** *DINOv3: Self-supervised learning for vision at unprecedented scale*（以官方论文题名为准）

**主要机构：** Meta AI

**会议 / 年份：** arXiv 2025

:::note[一句话]
进一步 **扩规模** 的 DINO 系模型（更大参数与数据）；强调稠密特征与多任务零样本表现，细节见 Meta 技术报告与代码。
:::

**材料：** [Paper](https://arxiv.org/abs/2508.10104) · [Project](https://ai.meta.com/dinov3/) · [Code](https://github.com/facebookresearch/dinov3)（以官方仓库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{dinov32025,
  title         = {{DINOv3}},
  year          = {2025},
  eprint        = {2508.10104},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2508.10104}
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

::::paper{tone="llava"}

## LLaVA

**论文：** *Visual Instruction Tuning*（LLaVA-1.5 等为后续扩展）

**主要机构：** UW–Madison / Microsoft 等

**会议 / 年份：** NeurIPS 2023（LLaVA）；后续版本见各自论文

:::note[一句话]
用 **合成指令数据** 将冻结 CLIP 式视觉与 LLM **端到端微调**，开启「聊天式视觉理解」范式；大量 VLM 与机器人高层模块沿用此路线。
:::

**材料：** [Paper](https://arxiv.org/abs/2304.08485) · [Project](https://llava-vl.github.io/) · [Code](https://github.com/haotian-liu/LLaVA)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{liu2023llava,
  title     = {Visual Instruction Tuning},
  author    = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.08485}
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

::::paper{tone="qwenvl"}

## Qwen-VL

**论文：** *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*

**主要机构：** Alibaba Cloud（通义实验室）

**会议 / 年份：** arXiv 2023；系列持续迭代（Qwen2-VL 等见新稿）

:::note[一句话]
中英双语与 **定位、OCR** 等能力突出的 VLM；多版本与 Qwen LLM 生态绑定，工程部署与微调资料全。
:::

**材料：** [Paper](https://arxiv.org/abs/2308.12966) · [Blog](https://github.com/QwenLM/Qwen-VL) · [Code](https://github.com/QwenLM/Qwen-VL)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{bai2023qwenvl,
  title   = {{Qwen-VL}: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author  = {Bai, Jinze and others},
  journal = {arXiv preprint arXiv:2308.12966},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.12966}
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

::::paper{tone="internvl"}

## InternVL

**论文：** *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*（系列有多篇，此为总览代表）

**主要机构：** OpenGVLab / 上海 AI Lab 等

**会议 / 年份：** arXiv 2024 起系列论文；以具体版本论文页为准

:::note[一句话]
**大规模视觉编码器 + 对齐 LLM** 的开源 VLM 家族；多版本（1.5、2、3…）覆盖高分辨率与长上下文，常作机器人 VLM 骨干候选。
:::

**材料：** [Paper](https://arxiv.org/abs/2404.01656) · [Project](https://internvl.readthedocs.io/) · [Code](https://github.com/OpenGVLab/InternVL)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{chen2024internvl,
  title   = {{InternVL}: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author  = {Chen, Zhe and others},
  journal = {arXiv preprint arXiv:2404.01656},
  year    = {2024},
  url     = {https://arxiv.org/abs/2404.01656}
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

::::paper{tone="foundationpose"}

## FoundationPose

**论文：** *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*

**主要机构：** NVIDIA Research 等

**会议 / 年份：** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)，2024

:::note[一句话]
**统一模型**对新物体做 **6D 位姿估计与跟踪**；可结合深度/合成数据，机器人抓取与对准常用基线。
:::

**材料：** [Paper](https://arxiv.org/abs/2312.08344) · [Project](https://nvlabs.github.io/FoundationPose/) · [Code](https://github.com/NVlabs/FoundationPose)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{wen2024foundationpose,
  title     = {{FoundationPose}: Unified {6D} Pose Estimation and Tracking of Novel Objects},
  author    = {Wen, Bowen and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
  url       = {https://arxiv.org/abs/2312.08344}
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

::::paper{tone="sam3"}

## SAM3

**论文：** *（技术报告 / 博客发布；正式 bib 以 Meta 论文页为准）*

**主要机构：** Meta AI

**会议 / 年份：** 2025 发布（Segment Anything 第三代）

:::note[一句话]
在 SAM 2 上扩展 **开放词汇检测与分割、视频跟踪** 等；文本/示例等多提示，面向「概念级」分割与产品集成。
:::

**材料：** [Blog](https://ai.meta.com/research/sam3/) · [Project](https://segment-anything.com/) · [Code](https://github.com/facebookresearch/sam3)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{sam3meta2025,
  title = {{SAM} 3},
  note  = {Meta AI; https://ai.meta.com/research/sam3/},
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

::::paper{tone="sam3d"}

## SAM3D

**论文：** *（单图 3D 重建；正式引用以 Meta 发布为准）*

**主要机构：** Meta AI

**会议 / 年份：** 2025 发布

:::note[一句话]
与 SAM3 协同的 **图像到 3D** 基础能力（物体/人体等）；强调真实图像下的几何与纹理重建，服务 AR/电商/机器人感知等。
:::

**材料：** [Blog](https://ai.meta.com/blog/sam-3d/) · [SAM 总览](https://ai.meta.com/research/sam3/) · Code（以 Meta 开源仓库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{sam3dmeta2025,
  title = {{SAM 3D}},
  note  = {Meta AI; https://ai.meta.com/blog/sam-3d/},
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

**和本页其他论文**

- **CLIP → SigLIP**：对比学习损失与训练配方演进。  
- **DINOv2 → DINOv3**：自监督视觉 scaling。  
- **LLaVA → Qwen-VL → InternVL**：指令微调范式 → 工业多语种 VLM → 大规模开源系列（视觉塔常接 CLIP / SigLIP）。  
- **FoundationPose**：几何位姿，与上文表征类基础模型互补。  
- **SAM3 → SAM3D**：分割与 3D 重建基础模型与机器人感知管线衔接。

## 新增一篇时怎么写（极简）

1. `Vision_Foundation_Models` 下建子目录；复制 **卡片 + BibTeX + 笔记**。  
2. 新 `tone` 在 `markdown.css` 追加。  
3. 目录表加一行。

---

_Changelog：2026-03 — 原 Vision Foundation Models (1) 合并为此单篇；按 `paper_note/Vision_Foundation_Models` 目录模板化。_
