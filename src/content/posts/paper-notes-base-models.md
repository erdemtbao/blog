---
title: "Paper Notes: Base Models"
published: 2026-02-21
description: Reading notes—VAE, normalizing flows, Transformer, ViT, diffusion, DiT, flow matching, Decision Transformer (Base assets).
image: ''
tags: [Paper Notes, Machine Learning, Foundation Models]
category: Paper Notes
draft: false
---

对应 `assets/paper_note/Base/`（线上：`public/paper-note/Base/`）。子目录与磁盘一致；其中 **`Diffuison` 为常见拼写笔误**，本篇卡片标题写 **Diffusion (DDPM)**，便于检索。版式同 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/)：`::::paper` + BibTeX 折叠 + 笔记。**排序**：从简到繁、从早期到近期——**潜变量与密度（VAE、NFs）→ 通用序列骨干（Transformer）→ 视觉中的 Transformer（ViT）→ 扩散系（Diffusion、DiT）→ 流匹配 → 决策序列（DT）**。

:::tip[PDF / 图片放哪？]
`/blog/paper-note/Base/<子目录>/<文件>`（若本地文件夹名为 `Diffuison`，线上仍可沿用同名路径。）
:::

## 目录

| 简称 | 说明 | 跳转 |
|:---:|:---|:---:|
| VAE | 变分自编码器 | [↓](#vae) |
| NFs | Normalizing flows | [↓](#nfs) |
| Transformer | 自注意力序列模型 | [↓](#transformer) |
| ViT | Vision Transformer | [↓](#vit) |
| Diffusion | DDPM 扩散生成框架 | [↓](#diffusion) |
| DiT | Diffusion Transformer 骨干 | [↓](#dit) |
| Flow | Flow matching 等 | [↓](#flow) |
| DT | Decision Transformer | [↓](#dt) |

::::paper{tone="vae"}

## VAE

**论文：** *Auto-Encoding Variational Bayes*

**主要机构：** Amsterdam（Kingma & Welling）

**会议 / 年份：** International Conference on Learning Representations (ICLR)，2014

:::note[一句话]
**变分下界**训练隐变量生成模型；潜空间与 **β-VAE、VQ-VAE** 等构成表示学习与生成前置知识。
:::

**材料：** [Paper](https://arxiv.org/abs/1312.6114) · Code（以 PyTorch 教程 / 实现为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{kingma2014autoencoding,
  title   = {Auto-Encoding Variational Bayes},
  author  = {Kingma, Diederik P. and Welling, Max},
  journal = {arXiv preprint arXiv:1312.6114},
  year    = {2014},
  url     = {https://arxiv.org/abs/1312.6114}
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

::::paper{tone="nfs"}

## NFs

**论文：** *Variational Inference with Normalizing Flows*（密度建模经典）

**主要机构：** DeepMind（Rezende & Mohamed）

**会议 / 年份：** International Conference on Machine Learning (ICML)，2015

:::note[一句话]
用 **可逆变换**堆叠成复杂密度；早期生成模型主线，与 VAE、扩散对比；现代仍用于 **精确似然**与部分机器人密度建模。
:::

**材料：** [Paper](https://arxiv.org/abs/1505.05770) · [Code](https://github.com/)（以教程/实现库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{rezende2015variational,
  title     = {Variational Inference with Normalizing Flows},
  author    = {Rezende, Danilo Jimenez and Mohamed, Shakir},
  booktitle = {International Conference on Machine Learning},
  year      = {2015},
  url       = {https://arxiv.org/abs/1505.05770}
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

::::paper{tone="transformer"}

## Transformer

**论文：** *Attention Is All You Need*

**主要机构：** Google Brain

**会议 / 年份：** Advances in Neural Information Processing Systems (NeurIPS)，2017

:::note[一句话]
**自注意力**取代 RNN/CNN 做序列 transduction；LLM、VLA、DiT、决策 Transformer 的共同祖先。
:::

**材料：** [Paper](https://arxiv.org/abs/1706.03762) · [Blog](https://research.google/pubs/pub46201/) · [Code](https://github.com/tensorflow/tensor2tensor)（历史实现）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
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

::::paper{tone="vit"}

## ViT

**论文：** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*

**主要机构：** Google Research

**会议 / 年份：** International Conference on Learning Representations (ICLR)，2021

:::note[一句话]
将图像切 **patch 序列** 送入 Transformer；现代视觉骨干与 **CLIP、VLM** 的视觉塔核心范式之一。可与 [Vision Foundation Models](/blog/posts/paper-notes-vision-foundation-models/) 对照读应用侧。
:::

**材料：** [Paper](https://arxiv.org/abs/2010.11929) · [Code](https://github.com/google-research/vision_transformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{dosovitskiy2021image,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.11929}
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

::::paper{tone="diffusion"}

## Diffusion

**论文：** *Denoising Diffusion Probabilistic Models*（DDPM）

**主要机构：** Google Research 等（Ho 等）

**会议 / 年份：** Advances in Neural Information Processing Systems (NeurIPS)，2020

:::note[一句话]
**前向加噪、反向去噪**的离散时间扩散；现代图像/视频生成与 **Diffusion Policy** 的默认数学起点之一。
:::

**材料：** [Paper](https://arxiv.org/abs/2006.11239) · [Blog](https://hojonathanho.github.io/diffusion/) · [Code](https://github.com/hojonathanho/diffusion)（教学实现；工程以所用库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{ho2020denoising,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.11239}
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

::::paper{tone="dit"}

## DiT

**论文：** *Scalable Diffusion Models with Transformers*

**主要机构：** UC Berkeley / OpenAI（Peebles & Xie）

**会议 / 年份：** IEEE/CVF International Conference on Computer Vision (ICCV)，2023

:::note[一句话]
用 **Transformer 替换 U-Net** 作为扩散的去噪网络（patchify latent）；与 **RDT、视频生成 DiT** 等同一条架构线。
:::

**材料：** [Paper](https://arxiv.org/abs/2212.09748) · [Project](https://www.wpeebles.com/DiT) · [Code](https://github.com/facebookresearch/DiT)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{peebles2023scalable,
  title     = {Scalable Diffusion Models with Transformers},
  author    = {Peebles, William and Xie, Saining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023},
  url       = {https://arxiv.org/abs/2212.09748}
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

::::paper{tone="flow"}

## Flow

**论文：** *Flow Matching for Generative Modeling*（代表工作之一）

**主要机构：** Meta AI / Weizmann 等（Lipman 等）

**会议 / 年份：** International Conference on Learning Representations (ICLR)，2023

:::note[一句话]
**流匹配**构造从噪声到数据的向量场，训练稳定、采样路径清晰；与扩散并列的生成建模范式，**π0** 等机器人策略采用。
:::

**材料：** [Paper](https://arxiv.org/abs/2210.02747) · [Code](https://github.com/atong01/conditional-flow-matching)（社区实现众多，以引用链为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{lipman2023flow,
  title     = {Flow Matching for Generative Modeling},
  author    = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.02747}
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

::::paper{tone="dt"}

## DT

**论文：** *Decision Transformer: Reinforcement Learning via Sequence Modeling*

**主要机构：** Google Brain / UC Berkeley 等

**会议 / 年份：** Advances in Neural Information Processing Systems (NeurIPS)，2021

:::note[一句话]
把 **RL 轨迹**看成序列，用 Transformer **自回归建模回报条件策略**；离线 RL 与行为克隆语境下与 BC、CQL 等对照。
:::

**材料：** [Paper](https://arxiv.org/abs/2106.01345) · [Project](https://sites.google.com/view/decision-transformer) · [Code](https://github.com/kzl/decision-transformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{chen2021decision,
  title     = {Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author    = {Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.01345}
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

- **VAE → NFs**：潜变量与 **可逆密度** 两条经典生成路线。  
- **Transformer → ViT → DiT**：序列注意力从 NLP 到视觉 patch，再到扩散骨干。  
- **Diffusion → DiT**：从通用去噪网络到 **Transformer 去噪**。  
- **Diffusion / Flow**：两种主流 **连续时间生成** 路径。  
- **DT**：同一 **Transformer** 壳用于 **序列决策** 而非生成像素。

## 新增一篇时怎么写（极简）

1. `Base` 下建子目录；复制 **卡片 + BibTeX + 笔记**。  
2. 新 `tone` 在 `markdown.css` 追加。  
3. 目录表加一行。

---

_Changelog：2026-03 — 原 Vision Foundation Models (2) 改为 Base Models；按 `paper_note/Base` 目录模板化。_
