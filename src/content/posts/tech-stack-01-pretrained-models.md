---
title: "Technology Stack 01: Pre-trained Models"
published: 2026-07-11
description: 具身智能的预训练底座——从多模态感知的 VLM（Qwen2.5-VL / Qwen3-VL / Qwen3-VL-MoE），到加上动作的 VLA（π₀ / π₀.₅ / LingBot-VLA），再到会"想象未来"的 World Action Model（DreamZero / LingBot-VA），一张从感知到行动的模型地图。
image: ''
tags: [Technology Stack, Pre-training, VLM, VLA, World Model, Foundation Models]
category: Technology Stack
draft: false
---

具身AI 的技术栈拆成三层来讲：**预训练模型**、**后训练算法**、**数据与规模化**。这一篇讲最底层的地基——**预训练模型**。

不过我不打算泛泛地讲"什么是预训练"，而是把镜头对准**具身智能（embodied AI）** 这条线：一个机器人要动起来，底座模型是怎么从"看懂世界"一步步长到"决定动作"、再到"预测世界如何演化"的。顺着这条主线，预训练底座正好可以分成三代范式：

> **VLM（看懂）→ VLA（会动）→ World Action Model（能想象）**

- **VLM（Vision-Language Model）**：多模态感知与语义的通用底座，把图像/视频接进语言模型。它是一切的起点。
- **VLA（Vision-Language-Action）**：在 VLM 之上接一个"动作头"，直接把观测映射成机器人动作。
- **World Action Model（WAM，世界动作模型）**：更进一步，让模型**同时预测"未来世界会变成什么样"和"该做什么动作"**，用视频当稠密监督信号。

下面按这个顺序展开。每个家族挑几个代表模型，必要处配上算法框架图。

---

## 一、VLM：多模态感知的通用底座

VLM 解决的是"**让语言模型长出眼睛**"。典型骨架是：一个视觉编码器（常来自 CLIP/SigLIP/DINO 系）把图像编码成视觉 token，一个适配器把它对齐到语言空间，再喂给 LLM 联合建模。它是后面 VLA、WAM 的感知与语义来源——**视觉塔和 VLM 的上限，几乎决定了上层策略的上限**。

### Qwen2.5-VL

阿里通义的工业级开源多模态模型，是当前具身/多模态系统里被复用最多的底座之一。相比早期 Qwen-VL（用位置感知适配器把图像压成定长 token、并用坐标 token 支持 grounding 与 OCR），Qwen2.5-VL 的几个关键升级值得记住：

- **原生动态分辨率（native dynamic resolution）**：不再把图像强行缩放到固定尺寸，而是按原始长宽比处理，视觉 token 数随分辨率自然变化，细节与大图都不吃亏。
- **绝对时间编码的视频理解**：把时间信息显式编码进位置，能处理长视频并**定位到具体时刻**（事件发生在第几秒），这对机器人"看视频学动作"很关键。
- **强 grounding 与文档解析**：精确的框/点定位、表格与文档结构解析，天然适合把"语言指令"接地到"画面里的物体"。
- **Agent 能力**：能作为视觉 agent 操作界面、调用工具。

一句话：Qwen2.5-VL 把"看懂图 + 定位物体 + 读懂文档 + 理解长视频"打包成一个可直接微调的底座。

### Qwen3-VL

Qwen3-VL 是这一系的新一代，整体思路是在 Qwen2.5-VL 的多模态能力上**继续 scale、并强化"作为智能体去行动"的能力**：更强的长上下文与长视频理解、更好的空间/2D-3D grounding、更强的多语言 OCR，以及更成熟的 GUI/工具调用 agent 表现。对具身场景，它意味着一个**更会定位、更会看长序列、更会按指令行动**的感知底座。

![Qwen3-VL 架构：视觉编码器 + 文本/视觉位置对齐（含 DeepStack 多层视觉特征注入与时间对齐的位置编码）接入 Qwen3 LLM（图源：Qwen3-VL 官方仓库）](/blog/technology-stack/Qwen/qwen3vl_arc.jpg)

> 从 Qwen2.5-VL 到 Qwen3-VL，趋势很清楚：VLM 不再只是"图像问答器"，而是越来越像一个**能感知、能定位、能操作**的通用多模态智能体骨干。

### Qwen3-VL-MoE

MoE（Mixture-of-Experts，混合专家）是 Qwen3-VL 的**稀疏化**版本。它的核心思想是：把 FFN 层拆成很多个"专家"，每个 token 只**路由激活其中少数几个**，于是模型的**总参数量**可以做得很大（容量大、能力强），但每次前向的**激活参数量**却很小（推理更省）。

对多模态尤其划算——不同专家可以隐式地分工处理不同模态、不同任务。Qwen3-VL-MoE 就是想用"大总量、小激活"的方式，在**不成比例增加推理成本**的前提下把 VLM 的能力上限往上抬。这也是当下大模型的普遍路线：**用稀疏换规模**。

---

## 二、VLA：给底座接上"动作头"

VLA 在 VLM 之上再进一步：不止理解画面和指令，而是**直接输出机器人动作**。难点在于动作是**连续、高频、跨本体**的，怎么把它和离散的语言 token 统一进一个模型，是这一代的核心命题。

### π₀（pi-zero）

Physical Intelligence 的 VLA 底座，也是这条线的地基。它的做法是在一个 VLM（PaliGemma，约 3B）之上，接一个**从零训练的"动作专家"模块**，用 **flow matching**（流匹配，可看作扩散的一个变体）来生成**连续动作块**：

- **动作专家 + flow matching**：在噪声与真实动作块之间构造路径，网络回归"从含噪动作指向干净动作"的向量场；推理时从噪声出发做约 10 步积分得到动作块。这样能稳定输出 **50Hz 的连续高频动作**——这是离散 token 化很难做到的。
- **跨本体**：用超过 1 万小时、7 种机器人配置（单臂/双臂/移动）的真机数据预训练，一个模型横跨多种本体。
- **VLM 预训练的价值**：完整架构相对"无 VLM"变体约有 >2× 的增益，印证了互联网级语义先验对动作学习的帮助。

![π₀ 总览：PaliGemma VLM 骨干 + 从零训练的 flow-matching 动作专家，输出 50Hz 连续动作块](/blog/paper-note/Robot_Learning_2/π0/overview.png)

π₀ 之后的很多工作，本质都在回答"π₀ 还差什么"：训练慢、泛化窄、执行卡顿、只会模仿……下面的 π₀.₅ 就是补"泛化"这一刀。

### π₀.₅（pi-0.5）

π₀ 虽灵巧，但主要在与训练分布相近的受控环境里管用。π₀.₅ 主攻**开放世界泛化**——目标是走进**训练里从没见过的真实家庭**里干多分钟长活。它的关键配方有两点：

- **异构数据 co-training**：把图像、语言指令、物体检测、高层语义子任务、低层动作，连同多机器人示范 + web 数据一起联合训练，让知识跨抽象层级迁移（消融显示 **web 数据对处理新物体很关键**）。
- **分层推理（机器人版 CoT）**：推理时**先用自然语言预测一个高层语义子任务**（如"拿起枕头"），再据此产出低层电机动作；高层用离散自回归 token、低层用 flow matching。

![π₀.₅ 分层推理：先预测高层语义子任务，再生成低层连续动作，异构数据 co-training 支撑开放世界泛化](/blog/paper-note/Robot_Learning_2/π0.5/Figure_3.png)

π₀.₅ 把 π₀ 从"实验室灵巧"推向"真实家庭可用"，并把"先语义、后动作"的分层正式写进了底座。

### LingBot-VLA

LingBot-VLA 是蚂蚁 Robbyant（灵波科技）具身基础大模型矩阵里的 **VLA（视觉-语言-动作）** 成员。和 π 系一样，它属于"VLM 感知底座 + 动作生成"的范式：用强多模态编码器理解场景与语言指令，再产出可执行的机器人动作，目标是**跨任务、跨场景的通用操作**。作为 Robbyant "感知（LingBot-Vision/Depth）→ 行动（LingBot-VLA）→ 世界（LingBot-World/VA）"整套矩阵的一环，它承担的是"把理解直接变成动作"的角色。

![LingBot-VLA Teaser：从多模态感知到通用操作，覆盖多本体、多场景的视觉-语言-动作模型（图源：Robbyant 官方）](/blog/technology-stack/LingBot-VLA/teaser.png)

> 说明：LingBot-VLA 的具体指标以 [Robbyant 官方页面](https://technology.robbyant.com/lingbot-vla) 为准；本文只做范式定位，不逐字引用未核实的数值。

### LingBot-VLA 2.0

LingBot-VLA 2.0 是上一代的迭代版本（[官方页面](https://technology.robbyant.com/lingbot-vla-v2)）。沿着这一代 VLA 的普遍演进方向，可以预期它在几个维度上加强：**更强的通用性与泛化**（更多本体、更多场景）、**更高频/更稳的动作生成**、以及**更好的长指令与推理执行**。它和下面的 LingBot-VA 2.0 是 Robbyant 在"直接出动作（VLA）"与"先想象世界再出动作（WAM）"两条路线上的并行探索。

![LingBot-VLA 2.0 框架：统一多本体动作空间（Arm/EEF/Gripper/Move/Waist/Head/Hand），Understanding Expert + 带 MoE 层的 Action Expert，配合以 LingBot-Depth & DINO-Video 为目标的 Dual-Query（current/future）蒸馏（图源：Robbyant 官方）](/blog/technology-stack/LingBot-VLA-2.0/framework.png)

---

## 三、World Action Model：让模型"想象未来"

VLA 是"看到就动"，而 **World Action Model（WAM）** 多了一步：**先预测未来世界会变成什么样，再据此决定动作**。它把预训练的**视频生成模型**当作世界模型——视频天然记录了"世界如何演化"，是比稀疏奖励密集得多的监督信号。这条线赌的是：**大视频模型即世界模型**。

### DreamZero

NVIDIA 的工作，把 WAM 直接**当成 zero-shot 策略**。它在预训练视频扩散骨干上建了一个 **14B 的自回归世界动作模型**，通过**同时预测未来世界状态（视频）+ 动作**来学物理动力学：

- **视频当稠密监督**：当前 VLA 擅长语义泛化，却难泛化到新环境里没见过的**物理运动**；DreamZero 用视频把"世界怎么演化"学进模型。
- **仅视频/人类示范的跨本体迁移**：可以用其他机器人、甚至**人类的"仅视频"示范**做迁移，极大降低新本体的数据成本。
- **大模型也能实时**：靠模型 + 系统协同优化，让 14B 模型跑到约 **7Hz 实时闭环**，证明大 WAM 是可部署的。

![DreamZero 架构：预训练视频扩散骨干上的 14B 自回归世界动作模型，联合预测未来世界状态与动作（图源：DreamZero）](/blog/technology-stack/DreamZero/architecture.png)

DreamZero 把 WAM 从"一种策略架构"升格为"**zero-shot 泛化的来源**"，直指 VLA 最大的痛点——新本体/新运动的数据成本。

### LingBot-VA / LingBot-VA 2.0

LingBot-VA 是 Robbyant 的**视频动作（Video-Action）世界模型**，走的是 WAM 里的"**闭环鲁棒**"支线。它的第一代（*Causal World Modeling for Robot Control*）用**自回归扩散**把"视频世界模型"和"动作策略"统一：

- **双流 Mixture-of-Transformers**：视频流（初始化自 Wan2.2-5B）与动作流并行，各自 QKV，再跨模态注意力融合，避免模态互扰。
- **视频-动作交错 + 因果自回归**：把视频帧与动作交错成一个序列，用因果掩码做统一的 next-token 预测——**因果性**保证未来不影响过去，符合物理直觉。
- **逆动力学出动作**：世界模型先"想象"出期望的未来画面，**逆动力学**再反推"什么动作能得到这个画面"。
- **实时异步推理 + FDM grounding**：用 KV cache 加速、异步流水线让"想象"与"电机执行"并行，并用前向动力学模型（FDM）以最新真实反馈重新对齐，避免开环漂移。

它专治 chunk 式视频-动作生成的三个洞：**reactivity（实时反馈）、长时记忆、物理因果**。

![LingBot-VA 框架（第一代）：视频流（Wan2.2-5B）+ 动作流双 MoT，视频-动作交错因果自回归，逆动力学出动作，异步闭环 + FDM grounding（图源：LingBot-VA 论文 arXiv:2601.21998）](/blog/technology-stack/LingBot-VA/framework.png)

**LingBot-VA 2.0**（[官方页面](https://technology.robbyant.com/lingbot-va-v2)）是它的升级版。延续第一代"因果自回归 + 逆动力学 + 异步闭环"的骨架，2.0 进一步强化：**更大的规模与更强的世界建模**、**更长时程的记忆与一致性**、以及**更快更稳的实时闭环部署**。它和 LingBot-VLA 2.0 一起，代表了 Robbyant 在"想象驱动的行动"这条路线上的最新进展。

![LingBot-VA 2.0 框架（图源：Robbyant 官方）](/blog/technology-stack/LingBot-VA-2.0/framework.png)

---

## 收束：一条从感知到想象的主线

把这三代范式连起来看，就是具身预训练底座的演化主线：

1. **VLM**（Qwen2.5-VL / Qwen3-VL / Qwen3-VL-MoE）——先把"看懂世界、听懂指令"这件事做扎实，它是一切的感知底座；
2. **VLA**（π₀ / π₀.₅ / LingBot-VLA 系）——在感知底座上接"动作头"，把理解直接变成连续、高频、跨本体的动作；
3. **World Action Model**（DreamZero / LingBot-VA 系）——再往前一步，让模型**先想象世界如何演化、再决定动作**，用视频当稠密监督换取更强的物理泛化。

一个清晰的技术张力贯穿始终：**是靠结构先验（分层、因果、逆动力学），还是靠通用视频骨干 + 规模化？** 两条路线都在快速演进，而它们共享同一个底座逻辑——**先有一个见过大量世界的预训练模型，才谈得上让它行动。**

但无论哪代范式，模型能力的上限最终都由同一件事决定——**用什么算法把它对齐、以及用多少数据和算力去喂它**。这正是接下来两篇的主题。

---
