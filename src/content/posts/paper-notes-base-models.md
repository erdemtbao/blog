---
title: "Paper Notes: Base Models"
published: 2026-02-12
description: 基础模型精读——VAE、Normalizing Flows、Transformer、ViT、Diffusion (DDPM)、DiT、Flow Matching、Decision Transformer。从潜变量生成与可逆密度，到通用注意力骨干，再到连续时间生成与序列决策的八块基石。
image: ''
tags: [Paper Notes, Machine Learning, Foundation Models]
category: Paper Notes
draft: false
---

## Overview

本篇是「基础模型」阅读笔记：把当代 AI / 机器人系统反复用到的**八块底层积木**放在一起精读。它们不是某个任务的解法，而是被 VLA、LLM、生成模型、离线 RL 反复复用的**通用组件**。

八块积木按四条线索组织（顺序从早到近、从简到繁）：

- **生成模型的两条早期路线** — [VAE](#vae)（潜变量 + 变分下界，近似似然）与 [NFs](#nfs)（可逆变换 + 精确似然）。它们定义了「怎么学一个能采样的分布」。
- **通用注意力骨干** — [Transformer](#transformer)（自注意力取代循环/卷积）→ [ViT](#vit)（把同一套骨干搬到视觉 patch）。后续几乎所有大模型都长在这上面。
- **连续生成的现代主线** — [Diffusion](#diffusion)（前向加噪 / 反向去噪）→ [DiT](#dit)（把扩散去噪网络从 U-Net 换成 Transformer）→ [Flow](#flow)（流匹配，免仿真地学噪声到数据的向量场）。
- **序列决策** — [DT](#dt)（把 Transformer 当作条件序列模型来做 RL）。

一条主线贯穿全篇：**如何用可微、可扩展的神经网络，去逼近一个复杂分布或一个复杂映射。** VAE/NFs 从「显式建模似然」入手，Diffusion/Flow 从「学一个逐步变换过程」入手；Transformer/ViT/DT 则回答「用什么架构承载这些学习」。它们彼此交叉——DiT = Diffusion + Transformer，π0 类策略 = Flow + Transformer，DT = RL + Transformer——正是这些交叉构成了今天的基础模型版图。

## Paper List

| 简称 | 年份 / Venue | 主题 | 一句话定位 |
|:--|:--|:--|:--|
| [VAE](#vae) | ICLR 2014 | 潜变量生成 | 变分下界 + 重参数化，摊销推断学生成模型 |
| [NFs](#nfs) | ICML 2015 | 可逆密度 | 可逆变换堆叠 + 变量替换，精确似然 |
| [Transformer](#transformer) | NeurIPS 2017 | 序列骨干 | 自注意力取代 RNN/CNN，可并行、建长程依赖 |
| [ViT](#vit) | ICLR 2021 | 视觉骨干 | 图像切 patch 当 token，纯 Transformer 做识别 |
| [Diffusion](#diffusion) | NeurIPS 2020 | 扩散生成 | 前向加噪 / 反向去噪，预测噪声即训练目标 |
| [DiT](#dit) | ICCV 2023 | 扩散骨干 | U-Net → Transformer，扩散随算力可扩展 |
| [Flow](#flow) | ICLR 2023 | 流匹配 | 免仿真回归向量场，OT 路径采样更直更快 |
| [DT](#dt) | NeurIPS 2021 | 序列决策 | RL 当序列建模，回报条件自回归预测动作 |

::::paper{tone="vae"}

## VAE

**Auto-Encoding Variational Bayes**

:::note[一句话]
用**变分下界（ELBO）**训练带连续潜变量的生成模型：一个**识别网络（编码器）**摊销地近似后验，配合**重参数化技巧**让采样可导，从而端到端用 SGD 学习。潜空间表示是后续 β-VAE、VQ-VAE、以及 [Diffusion](#diffusion)/[DiT](#dit) 潜空间的前置知识。
:::

**年份 / Venue** ICLR 2014（arXiv 1312.6114, 2013）｜ **机构** Universiteit van Amsterdam（Kingma & Welling）｜ **方向** 潜变量生成 / 变分推断 ｜ **基准** MNIST, Frey Faces

**材料** [Paper](https://arxiv.org/abs/1312.6114) · [Kingma thesis / 教程实现为准]

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

![VAE 计算流示意：Datapoint → 推断模型 q(z│x) → 采样 z → 生成模型 p(x,z)，目标为 ELBO（图源：Kingma & Welling, *An Introduction to Variational Autoencoders*, 2019, Fig. 2.2）](/blog/paper-note/Base/VAE/pipeline.png)

### Motivation

想学一个带潜变量的生成模型 $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$，难点在于**后验 $p_\theta(z\mid x)$ 不可解**，经典变分推断又要对每个数据点单独优化一套变分参数，无法随大数据集扩展。VAE 的目标：给出一个**对连续潜变量、大数据集都高效**的近似推断 + 学习算法。

### Method

- **摊销推断**：用一个神经网络（识别网络 / 编码器）$q_\phi(z\mid x)$ 直接输出近似后验的参数（如高斯的 $\mu,\sigma$），一套参数服务所有样本。
- **变分下界（ELBO）**：最大化对数似然的下界

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] - D_{\text{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
$$

前项是重构、后项把近似后验拉向先验 $p(z)=\mathcal N(0,I)$。

- **重参数化技巧（reparameterization）**：把 $z\sim\mathcal N(\mu,\sigma^2)$ 写成 $z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$，使采样对 $\phi$ 可导，得到低方差的 SGVB 梯度估计——这是全篇的关键。

### Experiments

- **基准**：MNIST、Frey Faces 等小图像数据集。
- **结果**：随潜维度 / 数据量增加，变分下界稳定提升；能从潜空间采样生成可辨识样本，并给出比朴素 wake-sleep 等更优的下界。**具体下界数值此处定性描述，不引用精确值。**

### Strengths and Limitations

**Strengths**：原理清晰、训练稳定、推断摊销后**采样与编码都是一次前向**；潜空间连续可插值，是表示学习与条件生成的通用底座。

**局限**：高斯后验/似然假设导致**样本偏模糊**；易出现**后验坍缩（posterior collapse）**（解码器过强时忽略 $z$）；下界与真实似然间有间隙。NFs、扩散、VQ-VAE 分别从「更灵活的后验 / 更强的生成过程 / 离散潜码」方向改进。

### Takeaways

VAE 给出「用神经网络 + 变分下界学潜变量生成模型」的范式。记住三个词：**ELBO、摊销推断、重参数化**。它的模糊样本与后验坍缩，正是 [NFs](#nfs)（灵活密度）与 [Diffusion](#diffusion)（渐进生成）要解决的问题；它的潜空间则被潜扩散 / [DiT](#dit) 直接复用。

::::paper{tone="nfs"}

## NFs

**Variational Inference with Normalizing Flows**

:::note[一句话]
把一个简单分布（如高斯）经过一串**可逆变换**逐步「拉扯」成复杂分布，用**变量替换公式**精确追踪密度变化。既可当 VAE 的**更灵活后验**，也可作为**精确似然**生成模型的一支主线。
:::

**年份 / Venue** ICML 2015（arXiv 1505.05770）｜ **机构** Google DeepMind（Rezende & Mohamed）｜ **方向** 可逆密度 / 变分推断 ｜ **基准** MNIST, CIFAR（作为 VAE 后验）

**材料** [Paper](https://arxiv.org/abs/1505.05770)

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

![NFs 推断与生成模型（原论文 Fig. 2）](/blog/paper-note/Base/NFs/pipeline.png)

### Motivation

VAE 的近似后验通常是**对角高斯**，表达力太弱，限制了下界的紧致度。NFs 的动机：构造**任意灵活但仍可精确求密度**的分布族，用来当更好的后验（也可独立作生成模型）。

### Method

- **可逆变换链**：从简单基分布 $z_0\sim q_0$ 出发，施加一串可逆映射 $z_K=f_K\circ\cdots\circ f_1(z_0)$。
- **变量替换（change of variables）**：密度按雅可比行列式修正

$$
\log q_K(z_K) = \log q_0(z_0) - \sum_{k=1}^{K}\log\Big|\det\frac{\partial f_k}{\partial z_{k-1}}\Big|
$$

- **廉价行列式的流**：论文提出 **planar flow** $f(z)=z+u\,h(w^\top z+b)$ 与 **radial flow**，利用矩阵行列式引理把 $\det$ 的计算降到 $O(D)$，使深层堆叠可行。

### Experiments

- **基准**：作为 VAE 的后验近似，在 MNIST / CIFAR 上评估。
- **结果**：随流长度 $K$ 增加，近似后验更贴近真实后验、下界更紧，优于对角高斯与早期 NICE 等基线。**具体下界数值此处定性描述，不引用精确值。**

### Strengths and Limitations

**Strengths**：**精确似然**（无下界间隙）、密度灵活、采样与密度评估都直接。

**局限**：为保证雅可比行列式可算，架构**受强约束**（planar/radial 单步表达力有限，需堆很多层）；后续 RealNVP、IAF、Glow 用耦合层 / 自回归结构增强表达力，但可逆性约束始终是流模型的根本张力。

### Takeaways

NFs 代表生成建模的**「可逆 + 精确似然」**一派，与 VAE 的「下界近似」、Diffusion 的「渐进变换」形成三足。它的核心公式——**变量替换 + 雅可比行列式**——也是理解连续归一化流（CNF）与 [Flow](#flow) 匹配的入口。

::::paper{tone="transformer"}

## Transformer

**Attention Is All You Need**

:::note[一句话]
用**自注意力**彻底取代 RNN 的循环与 CNN 的局部卷积：序列中每个位置直接与所有位置交互，既可**并行训练**又能建**长程依赖**。它是 LLM、[ViT](#vit)、[DiT](#dit)、[DT](#dt) 乃至几乎所有现代大模型的共同祖先。
:::

**年份 / Venue** NeurIPS 2017（arXiv 1706.03762）｜ **机构** Google Brain / Google Research ｜ **方向** 序列 transduction 骨干 ｜ **基准** WMT'14 EN-DE / EN-FR 机器翻译

**材料** [Paper](https://arxiv.org/abs/1706.03762) · [Code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}</code></pre>
</details>

<img
  src="/blog/paper-note/Base/Transformer/pipeline.png"
  alt="Transformer 编码器–解码器架构（原论文 Fig. 1）"
  style="display: block; width: 50%; margin: 0 auto;"
/>

### Motivation

RNN 的计算沿时间步**串行**，无法并行、且长程依赖随距离衰减；CNN 虽可并行但需堆很多层才能覆盖长程。Transformer 的问题意识：**能否只用注意力、完全去掉循环与卷积**，既拿到并行度又拿到「任意两位置常数距离交互」？

### Method

- **缩放点积注意力**：

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V
$$

除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。

- **多头注意力（multi-head）**：把 $Q,K,V$ 投影到多个子空间并行做注意力再拼接，让模型同时关注不同类型的关系。
- **位置编码**：注意力本身对顺序不敏感，用**正弦位置编码**注入位置信息。
- **整体**：编码器–解码器堆叠，每层 = 多头注意力 + 位置前馈网络（FFN），配 **残差连接 + LayerNorm**；解码器用掩码保证自回归。

### Experiments

- **基准**：WMT'14 英德 / 英法机器翻译。
- **结果（原文明确报告）**：大模型在 **EN-DE 达 28.4 BLEU、EN-FR 达 41.8 BLEU**，均为当时新 SOTA，且训练成本显著低于此前最好的循环 / 卷积模型。

### Strengths and Limitations

**Strengths**：高度并行、长程依赖直接、随数据与参数**极佳扩展**——这是它成为一切大模型骨干的根本原因。

**局限**：自注意力对序列长度是 **$O(n^2)$** 计算/显存；需显式位置编码；小数据上因缺乏归纳偏置不如 CNN/RNN。后续 sparse/linear attention、FlashAttention、RoPE 等围绕这些点改进。

### Takeaways

Transformer 是本篇的**枢纽**：[ViT](#vit) 把它搬到图像、[DiT](#dit) 把它搬进扩散去噪、[DT](#dt) 把它搬去做序列决策。读懂「注意力 + 多头 + 残差/LN + 位置编码」这套结构，本页后面一半的模型都只是它的变体。

::::paper{tone="vit"}

## ViT

**An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**

:::note[一句话]
把图像切成固定大小的 **patch 序列**，线性嵌入后当作 token 直接喂给标准 [Transformer](#transformer) 编码器——几乎不改架构就把 NLP 的骨干搬到视觉。关键结论：**够大规模预训练后，纯 Transformer 可匹敌乃至超越 CNN**。
:::

**年份 / Venue** ICLR 2021（arXiv 2010.11929）｜ **机构** Google Research（Brain Team）｜ **方向** 视觉骨干 ｜ **基准** ImageNet / ImageNet-21k / JFT-300M 预训练 + 下游迁移

**材料** [Paper](https://arxiv.org/abs/2010.11929) · [Code](https://github.com/google-research/vision_transformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{dosovitskiy2021image,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk
               and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias
               and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.11929}
}</code></pre>
</details>

![ViT 模型总览：切 patch → 线性嵌入 + 位置编码 → Transformer 编码器（原论文 Fig. 1）](/blog/paper-note/Base/ViT/pipeline.png)

### Motivation

CNN 长期垄断视觉，其**局部性 + 平移等变**是很强的归纳偏置。ViT 的问题：如果**数据足够多**，能否不要这些先验、直接用通用 Transformer 学视觉？动机是把 NLP 里「大数据 + 大模型 + 通用架构」的成功复制到图像。

### Method

- **patchify**：把 $H\times W$ 图像切成 $16\times16$ 的 patch，每个 patch 展平后线性投影成一个 token embedding。
- **序列构造**：patch embedding + **可学习位置嵌入**，前置一个 **[class] token** 作为分类表示（借鉴 BERT）。
- **主干**：标准 Transformer 编码器；[class] token 的输出接 MLP 头做分类。除切 patch 外，几乎不引入图像专属结构。

### Experiments

- **基准**：在 ImageNet-21k / JFT-300M 上大规模预训练，再迁移到 ImageNet 等下游。
- **结果**：**中小数据**上 ViT 逊于同级 ResNet（缺归纳偏置）；但**在大规模预训练后**，ViT 匹配或超过强 CNN 基线（BiT），且预训练算力更省。**具体各基准精度此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：全局感受野、随数据/规模扩展性好、与 NLP 骨干统一（利于多模态）；成为 CLIP、多数 VLM 视觉塔与视觉预训练的默认选择之一。

**局限**：**数据饥渴**（弱归纳偏置需大数据补偿）；$O(n^2)$ 注意力对高分辨率昂贵。后续 DeiT（蒸馏省数据）、Swin（层级 + 窗口注意力）针对性改进。

### Takeaways

ViT 证明「Transformer 是通用骨干」不止于文本。它与 [Transformer](#transformer) 是同一架构在两个模态的化身，也直接连向 [DiT](#dit)（同样先 patchify 再上 Transformer，只是目标从分类换成扩散去噪）。应用侧可对照 [Vision Foundation Models](/blog/posts/paper-notes-vision-foundation-models/)。

::::paper{tone="diffusion"}

## Diffusion

**Denoising Diffusion Probabilistic Models (DDPM)**

:::note[一句话]
定义一个**前向过程**逐步往数据里加高斯噪声直至纯噪声，再训练一个网络学**反向去噪**过程。关键简化：把变分目标化简成**预测每步噪声 $\epsilon$ 的均方误差**，训练稳定、样本质量高——现代图像/视频生成与 [Diffusion Policy](/blog/posts/paper-notes-robot-learning-1/) 的数学起点。
:::

**年份 / Venue** NeurIPS 2020（arXiv 2006.11239）｜ **机构** UC Berkeley（Ho, Jain, Abbeel）｜ **方向** 扩散生成 ｜ **基准** CIFAR-10, LSUN, CelebA-HQ

**材料** [Paper](https://arxiv.org/abs/2006.11239) · [Project/Blog](https://hojonathanho.github.io/diffusion/) · [Code](https://github.com/hojonathanho/diffusion)

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

![DDPM 前向加噪 / 反向去噪的马尔可夫链（原论文 Fig. 2）](/blog/paper-note/Base/Diffusion/pipeline.png)

### Motivation

GAN 样本质量高但训练不稳、易模式崩塌；VAE 稳但样本模糊。扩散模型（Sohl-Dickstein 2015）理论优雅但当时质量一般。DDPM 的目标：**把扩散模型的样本质量做到与 GAN 竞争**，同时保留似然模型的稳定训练与模式覆盖。

### Method

- **前向过程**（固定、无参数）：$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I)$，可闭式跳步：

$$
q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\quad \bar\alpha_t=\textstyle\prod_{s\le t}(1-\beta_s)
$$

- **反向过程**（学习）：$p_\theta(x_{t-1}\mid x_t)$ 用网络（原文用 U-Net）参数化。
- **简化目标**：将变分下界重参数化后，训练目标等价于**预测注入的噪声**：

$$
L_{\text{simple}}=\mathbb E_{t,x_0,\epsilon}\big[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\|^2\big]
$$

这一步把复杂的 KL 目标变成朴素回归，是 DDPM 好训练的关键；并揭示与去噪得分匹配 / Langevin 采样的联系。

### Experiments

- **基准**：CIFAR-10、LSUN、CelebA-HQ 等无条件 / 类条件图像生成。
- **结果（原文明确报告）**：CIFAR-10 无条件生成达 **FID 3.17、Inception 9.46**，为当时该设定下的领先结果；高分辨率 LSUN 样本质量与 ProgressiveGAN 相当。

### Strengths and Limitations

**Strengths**：样本质量高、训练稳定（纯回归损失）、模式覆盖好、易做条件生成。

**局限**：**采样慢**——需数百上千步反向迭代（后续 DDIM、少步蒸馏、一致性模型缓解）；原始 U-Net 骨干扩展性有限（[DiT](#dit) 换 Transformer 解决）；像素空间直接扩散昂贵（潜扩散 / Stable Diffusion 移到 VAE 潜空间）。

### Takeaways

DDPM 把「加噪—去噪」变成一个可训练、可扩展的生成范式。它的两处软肋——**采样慢**与**骨干选择**——分别通向 [Flow](#flow)（更直的路径、更少步）与 [DiT](#dit)（Transformer 骨干），也直接支撑机器人里的 [Diffusion Policy](/blog/posts/paper-notes-robot-learning-1/)。

::::paper{tone="dit"}

## DiT

**Scalable Diffusion Models with Transformers**

:::note[一句话]
把扩散模型里去噪网络的 **U-Net 换成 Transformer**：在 VAE 潜空间里把 latent 切 patch，用带 **adaLN-Zero** 条件注入的 Transformer 块做去噪。核心发现：**去噪网络的算力（Gflops）越大，生成质量越好**——扩散终于有了可预测的扩展律。
:::

**年份 / Venue** ICCV 2023（arXiv 2212.09748）｜ **机构** UC Berkeley / New York University（Peebles & Xie）｜ **方向** 扩散骨干 / 可扩展生成 ｜ **基准** ImageNet 256×256 / 512×512 类条件生成

**材料** [Paper](https://arxiv.org/abs/2212.09748) · [Project](https://www.wpeebles.com/DiT) · [Code](https://github.com/facebookresearch/DiT)

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

![DiT 架构：latent patchify + 条件 DiT 块（adaLN-Zero）（原论文 Fig. 3）](/blog/paper-note/Base/DiT/pipeline.png)

### Motivation

扩散模型一直默认用 **U-Net** 做去噪网络，其归纳偏置和可扩展性未被系统检验。而 [Transformer](#transformer) 在 NLP/视觉上已展现清晰扩展律。DiT 的问题：**用 Transformer 替换 U-Net，扩散能否获得同样干净的「越大越好」扩展性？**

### Method

- **潜空间 + patchify**：沿用潜扩散，在预训练 VAE 的 latent 上操作；把 latent 切成 patch token（呼应 [ViT](#vit)）。
- **DiT 块**：标准 Transformer 块处理 patch 序列做噪声预测。
- **条件注入 adaLN-Zero**：把时间步 $t$ 与类别标签经 MLP 映射为 **自适应 LayerNorm** 的缩放/平移参数注入每个块，并将残差分支初始化为零（zero-init），显著稳住训练——这是 DiT 的关键设计。
- **扩展轴**：以模型 Gflops（深度/宽度/patch 大小）为轴系统扫描。

### Experiments

- **基准**：ImageNet 类条件生成 256×256 与 512×512。
- **结果（原文明确报告）**：最大模型 **DiT-XL/2 在 256×256 达 FID 2.27**，刷新当时该基准；并观察到 **FID 随去噪网络 Gflops 单调下降**的清晰扩展规律。

### Strengths and Limitations

**Strengths**：干净的扩展律、去掉 U-Net 的手工归纳偏置、与 Transformer 生态（并行、大模型工程）统一；是 Stable Diffusion 3、Sora 等现代生成系统的架构基础。

**局限**：依赖预训练 VAE（两阶段）；大模型训练/采样算力高；与所有扩散一样受采样步数拖累。

### Takeaways

DiT = [Diffusion](#diffusion) 的过程 + [Transformer](#transformer)/[ViT](#vit) 的骨干，是「架构统一到 Transformer」这条大趋势在生成领域的落点。机器人里的 [RDT](/blog/posts/paper-notes-robot-learning-1/) 等大型扩散策略正是同一条架构线。

::::paper{tone="flow"}

## Flow

**Flow Matching for Generative Modeling**

:::note[一句话]
不学「逐步去噪」，而是**免仿真地回归一个向量场**：定义从噪声到数据的概率路径，训练网络去匹配沿该路径的目标速度场。用**最优传输（OT）式的直线路径**时，采样轨迹更直、步数更少——[π0](/blog/posts/paper-notes-robot-learning-2/) 类机器人策略与 SD3 等采用。
:::

**年份 / Venue** ICLR 2023（arXiv 2210.02747；Oral / Outstanding）｜ **机构** Meta AI (FAIR) / Weizmann（Lipman 等）｜ **方向** 连续时间生成 / 流匹配 ｜ **基准** ImageNet（图像生成、似然）

**材料** [Paper](https://arxiv.org/abs/2210.02747) · [Meta Flow Matching Guide](https://ai.meta.com/research/publications/flow-matching-guide-and-code/)

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

![Flow Matching（OT 路径）的概率路径 $p_t$：从高斯噪声（左）沿向量场逐步输运到数据分布（右）（原论文 2D checkerboard 示例）](/blog/paper-note/Base/Flow/pipeline.png)

### Motivation

连续归一化流（CNF）表达力强但训练要**仿真 ODE**、昂贵；扩散的 SDE / 得分匹配框架有效但推导绕、采样慢。Flow Matching 的目标：给出一个**免仿真、回归式**的目标，直接学出定义生成过程的**向量场**，并能自由选择概率路径。

### Method

- **目标**：学一个时间相关向量场 $v_\theta(t,x)$，其诱导的 ODE $\frac{dx}{dt}=v_\theta(t,x)$ 把噪声分布 $p_0$ 沿概率路径 $p_t$ 输运到数据分布 $p_1$。
- **Conditional Flow Matching (CFM)**：直接回归边缘向量场不可解，但**以单个数据点为条件**的路径 $p_t(x\mid x_1)$ 及其速度 $u_t(x\mid x_1)$ 可闭式写出，且二者的回归目标**梯度相同**：

$$
\mathcal L_{\text{CFM}}=\mathbb E_{t,\,q(x_1),\,p_t(x\mid x_1)}\big[\|v_\theta(t,x)-u_t(x\mid x_1)\|^2\big]
$$

- **OT 路径**：取高斯条件路径 + 线性插值（$x_t=(1-t)x_0+t x_1$）时，轨迹接近**直线**，采样所需 ODE 步数更少、训练更稳。扩散是它的一个特例（特定路径）。

### Experiments

- **基准**：ImageNet 图像生成与似然评估。
- **结果**：以 OT 路径的 Flow Matching 训练 CNF，在样本质量、似然与**采样效率**上匹配或优于扩散基线，且训练更稳定。**具体 FID / NLL 此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：免仿真训练、路径可自由设计（OT 路径**采样更快更直**）、把扩散/CNF 统一在一个回归框架下；已成机器人策略（π0）与新一代图像生成（SD3）的主流选择。

**局限**：采样仍需数值 ODE 求解；生态较扩散年轻；OT 路径的优势依赖路径/耦合设计。

### Takeaways

Flow Matching 把 [Diffusion](#diffusion) 的「学一个渐进变换」提炼成「**回归一个向量场**」，并用 OT 路径解决采样步数问题。它与 [NFs](#nfs) 一脉相承（都是连续变量输运），是当前连续生成与机器人 action 生成的当红范式。

::::paper{tone="dt"}

## DT

**Decision Transformer: Reinforcement Learning via Sequence Modeling**

:::note[一句话]
把 RL 轨迹当成一条序列 **(return-to-go, state, action, …)**，用 GPT 式 [Transformer](#transformer) **自回归地预测动作**，以「期望回报」为条件。不做价值自举、不做动态规划——把（离线）RL 直接化为监督式序列建模。
:::

**年份 / Venue** NeurIPS 2021（arXiv 2106.01345）｜ **机构** UC Berkeley / Facebook AI / Google Brain ｜ **方向** 离线 RL / 序列建模 ｜ **基准** Atari, D4RL, Key-to-Door

**材料** [Paper](https://arxiv.org/abs/2106.01345) · [Project](https://sites.google.com/view/decision-transformer) · [Code](https://github.com/kzl/decision-transformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{chen2021decision,
  title     = {Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author    = {Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya
               and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.01345}
}</code></pre>
</details>

![Decision Transformer 架构：return-to-go / state / action 作为 token 送入因果 Transformer（原论文 Fig. 1）](/blog/paper-note/Base/DT/pipeline.png)

### Motivation

传统 RL 靠 TD 自举 + 动态规划，面临 [致命三要素](/blog/posts/paper-notes-reinforcement-learning-1/#td)、离线分布偏移等稳定性难题。DT 的问题意识：既然 Transformer 在序列建模上如此强大，能否**绕开价值/策略梯度**，把 RL 直接当成「给定想要的回报，预测该做什么动作」的条件序列建模？

### Method

- **轨迹序列化**：把轨迹表示为 $(\hat R_1, s_1, a_1, \hat R_2, s_2, a_2, \dots)$，其中 **return-to-go** $\hat R_t=\sum_{t'\ge t} r_{t'}$ 是从当前起的剩余回报。
- **自回归建模**：GPT 式因果 Transformer 以过去 token 为条件预测下一个动作 $a_t$，训练是纯监督（对动作做回归/分类），**无 TD、无 bootstrapping**。
- **回报条件生成**：推理时把 $\hat R_1$ 设为期望目标回报，模型据此「演」出达成该回报的动作序列；注意力天然处理长程信用分配。

### Experiments

- **基准**：Atari（离线）、D4RL（连续控制）、Key-to-Door（长程信用分配）。
- **结果**：在多数任务上匹配或超过 [CQL](/blog/posts/paper-notes-reinforcement-learning-2/#cql) 等专门离线 RL 方法，尤其在需要长程信用分配的任务上占优。**具体分数此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：概念极简、复用 Transformer 的扩展性与工程、无 TD 不稳定；长程信用分配由注意力自然承担。

**局限**：性能受**数据集回报覆盖**限制，**难以「拼接」次优轨迹**超越数据（stitching 弱于动态规划类方法）；依赖设定合理的目标回报；本质离线。后续 Trajectory Transformer、Online DT、Q-learning DT 等改进。

### Takeaways

DT 展示了「Transformer 骨干 + 序列建模」可以吞下**决策**问题，是本页架构线与 [RL](/blog/posts/paper-notes-reinforcement-learning-1/) 线的交汇点。它与离线 RL（[CQL](/blog/posts/paper-notes-reinforcement-learning-2/#cql)/[IQL](/blog/posts/paper-notes-reinforcement-learning-2/#iql)）解决同一问题的两种哲学——**序列建模** vs. **保守价值学习**——值得对照。

## Cross-Paper Comparison

| 模型 | 年份 | 类别 | 核心机制 | 训练目标 | 主要贡献 / 对治问题 |
|:--|:--|:--|:--|:--|:--|
| VAE | 2014 | 潜变量生成 | 编码器 + 重参数化 | ELBO（下界） | 摊销变分推断，可导潜变量学习 |
| NFs | 2015 | 可逆密度 | 可逆变换 + 变量替换 | 精确对数似然 | 灵活后验 / 精确似然 |
| Transformer | 2017 | 序列骨干 | 自注意力 + 多头 | 序列预测（如翻译） | 去循环、并行、建长程依赖 |
| ViT | 2021 | 视觉骨干 | patch token + 编码器 | 分类交叉熵 | Transformer 通用于视觉（需大数据） |
| Diffusion | 2020 | 扩散生成 | 前向加噪 / 反向去噪 | 预测噪声 MSE | 稳定训练、高质量样本 |
| DiT | 2023 | 扩散骨干 | patch + adaLN-Zero | 扩散去噪 | U-Net→Transformer，可扩展扩散 |
| Flow | 2023 | 流匹配 | 回归向量场 (CFM) | 条件向量场 MSE | 免仿真、OT 路径少步采样 |
| DT | 2021 | 序列决策 | 回报条件自回归 | 动作监督 | RL 化为序列建模，无 TD |

## Discussion

1. **两大脉络：生成 vs 骨干。** 一条线是「怎么学一个分布」——[VAE](#vae)（下界）、[NFs](#nfs)（精确似然）、[Diffusion](#diffusion)/[Flow](#flow)（渐进变换 / 向量场）；另一条线是「用什么架构承载学习」——[Transformer](#transformer)→[ViT](#vit)→[DT](#dt)。二者在 [DiT](#dit)（扩散 × Transformer）交汇，也在机器人 [π0](/blog/posts/paper-notes-robot-learning-2/)（Flow × Transformer）交汇。

2. **架构在收敛到 Transformer。** ViT 把视觉、DiT 把扩散、DT 把决策统统改造成「切成 token 序列 + 注意力」。理解 [Transformer](#transformer) 一篇，等于拿到读懂本页一半模型的钥匙。

3. **生成建模的三种数学味道。** VAE 是**变分下界**、NFs 是**可逆 + 精确似然**、Diffusion/Flow 是**渐进随机 / 确定性变换**。三者互为参照：Flow Matching 甚至把扩散收作特例，NFs 与 CNF/Flow 一脉相承。

4. **「采样效率」是连续生成的暗线。** 扩散样本好但步数多；Flow 用 OT 直线路径压步数；这条线继续通向少步蒸馏 / 一致性模型。机器人实时控制尤其吃这一点。
