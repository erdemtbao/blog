---
title: "Paper Notes: Robot Learning (1)"
published: 2026-03-20
description: 八篇机器人学习里程碑工作的精读笔记——ACT、Diffusion Policy、DP3、RT-1、RT-2、Octo、OpenVLA、RDT-1B，覆盖模仿学习、扩散策略与 VLA。
image: ''
tags: [Paper Notes, Robot Learning]
category: Paper Notes
draft: false
---

## Overview

本篇整理 2022–2024 年机器人操作（manipulation）学习中的八篇代表性工作。它们看似分散，其实共同回答两个问题：**如何表达示范中的多模态动作分布（multimodal action distribution）**，以及**如何把机器人策略「做大」（scaling）**。按这两条线索，可以把八篇论文归为三组：

- **小规模模仿 + 表达力更强的动作模型**：ACT 用 action chunking + CVAE 处理复合误差与多模态；Diffusion Policy（DP）改用扩散过程直接建模动作分布；DP3 把观测从 2D 图像换成 3D 点云以提升泛化。
- **机器人 Transformer 的规模化与 VLA 的出现**：RT-1 证明「大规模真实数据 + Transformer」可行；RT-2 进一步把互联网级 VLM 的知识迁移进控制，形成 Vision-Language-Action（VLA）范式。
- **开源通才 / 基础模型**：Octo 与 OpenVLA 是两条开源路线（轻量扩散头 vs. 7B VLM）；RDT-1B 把扩散策略推到十亿参数级的双臂基础模型。

这些工作在方法上互相借用：action chunking 贯穿 ACT、DP、Octo、RDT；扩散动作头从 DP 扩散到 DP3、Octo、RDT；动作 token 化从 RT-1 延续到 RT-2、OpenVLA。

## Paper List

| 简称 | 年份 / Venue | 主题 | 机器人 / Benchmark | 核心思想 |
|:--|:--|:--|:--|:--|
| [ACT](#act) | RSS 2023 | 模仿学习 / action chunking | ALOHA 双臂真机 | 一次预测动作块 + CVAE，缓解复合误差 |
| [DP](#dp) | RSS 2023（后扩展 IJRR 2024） | 扩散策略 | Robomimic 等 4 类 + UR5/Franka | 用条件扩散建模动作分布 |
| [DP3](#dp3) | RSS 2024 | 3D 扩散策略 | 72 仿真任务 + Franka/Allegro | 稀疏点云 + 轻量 3D 编码器提升泛化 |
| [RT-1](#rt-1) | RSS 2023 | 机器人 Transformer | Everyday Robots 移动操作机 | 大规模真机数据 + 动作离散 token |
| [RT-2](#rt-2) | CoRL 2023 | VLA / 知识迁移 | Everyday Robots 移动操作机 | VLM 直接输出动作 token，涌现语义泛化 |
| [Octo](#octo) | RSS 2024 | 开源通才策略 | 9 套真机（WidowX/UR5/Franka） | 模块化 Transformer + 扩散头，快速微调 |
| [OpenVLA](#openvla) | CoRL 2024 | 开源 7B VLA | WidowX / Google robot / Franka | 双视觉编码器 + Llama 2，全开源可微调 |
| [RDT](#rdt) | ICLR 2025 | 双臂扩散基础模型 | ALOHA 双臂真机 | 1.2B DiT 扩散 + 统一动作空间 |


::::paper{tone="act"}

## ACT

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**（ACT，Action Chunking with Transformers）

:::note[一句话]
用 Transformer 一次性预测未来一段**动作块（action chunk）**，配合 CVAE 隐变量与时序集成，从约 50 条示范里学会低成本双臂硬件上的细粒度操作。
:::

**会议 / 年份** RSS 2023 ｜ **机构** Stanford · UC Berkeley · Meta ｜ **方向** Imitation learning, action chunking, CVAE ｜ **真机** ✅ ALOHA 双臂（14-DoF，4 相机）

**材料** [Paper](https://arxiv.org/abs/2304.13705) · [Project](https://tonyzhaozh.github.io/aloha/) · [Code (ACT)](https://github.com/tonyzhaozh/act) · [Code (ALOHA 硬件)](https://github.com/tonyzhaozh/aloha)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{zhao2023learning,
  title     = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author    = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle = {Robotics: Science and Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.13705}
}</code></pre>
</details>

### Motivation

细粒度双手操作（穿扎带、插电池、开调料杯盖）对硬件精度要求高，而低成本遥操作硬件本身不精确、示范带噪声。逐帧模仿学习在长时程上会累积**复合误差（compounding error）**：某一步偏差导致观测分布偏移，后续预测越漂越远。ACT 的目标是让这类精细任务在约 10 分钟示范、廉价硬件上也能学会。

### Method

ACT 以 CVAE 形式训练，核心是把预测粒度从「每帧」抬到「每段」：

- **Action Chunking**：策略一次预测未来 *k* 步动作，把有效决策步数缩短到原来的 1/*k*，直接压制复合误差。
- **CVAE + Transformer**：训练时一个 style encoder 把专家动作序列 + 关节状态压成隐变量 *z*；观测（ResNet-18 图像特征 + 关节 token）与 *z* 一起送入 Transformer decoder。推理时 **z 取先验均值（即 z = 0）**，而非从 N(0, I) 采样。
- **非自回归解码**：decoder 在**一次前向**中输出完整的 *k* × 14 动作块，而不是逐步自回归生成——这点原笔记写反了。
- **Temporal Ensemble**：相邻时刻预测的动作块在时间上重叠，对重叠部分做**指数加权平均**（权重 $w_i = \exp(-m\cdot i)$）平滑输出、抑制抖动。

结构规模：ResNet-18 编码每帧图像，4 层 encoder + 7 层 decoder，隐变量维度 32，**参数量约 80M**（原笔记「30M」有误），单 GPU 可训。

![ACT 算法伪代码](/blog/paper-note/Robot_Learning_1/ACT/ACT_algo.png)

![ACT 结构细节](/blog/paper-note/Robot_Learning_1/ACT/ACT_detail_architecture.png)

### Experiments

- **平台与数据**：真机为 ALOHA 双臂（约 2 万美元级、开源），6 个细粒度真实任务 + 2 个仿真任务（Transfer Cube、Bimanual Insertion）；每任务约 50 条示范（Thread Velcro 用 100 条）。
- **结果**：真实任务成功率大致落在 80–90%（单任务从 64% 到 96% 不等）。
- **Baseline**：对比 **BC-ConvMLP、BeT（Behavior Transformer）、RT-1、VINN**（原笔记写的 LSTM-GMM / IBC 不是本文 baseline）。这些方法多数明显更低，但并非一律低于 20%（BeT 部分任务可达约 60%）。
- **消融**：动作块长度 *k* = 100（对应 50Hz 下约 2s）是常用甜点；隐变量维度 32 足够。Temporal ensemble 有正向增益，但论文未给出可直接引用的固定百分点数值。
- **超参更正**：KL 权重 **β = 10**（原笔记「0.01」有误）。

### Strengths and Limitations

**Strengths**：样本高效（约 50 条示范即可）；action chunking + temporal ensemble 直击复合误差并平滑轨迹；非自回归单次解码，推理可跟上 50Hz 实时控制；低成本开源硬件降低了双臂研究门槛。

**局限（分析）**：评测集中在短时程桌面任务，*k* 需按域调参（反应性 vs. 平滑性权衡）；纯模仿、无在线纠错，性能上限受示范质量约束；单一 embodiment（ALOHA），跨机器人泛化非其目标。

### Takeaways

Action chunking 是这一批工作里最有生命力的设计之一，后续 DP、Octo、RDT 都在动作块上做文章；ACT 与 DP 分别用 CVAE 与扩散回答同一个「多模态」问题，为后面扩散路线埋下对照。

::::paper{tone="dp"}

## DP

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**

:::note[一句话]
把动作序列的生成建模成一个**条件去噪扩散（conditional denoising diffusion）**过程：给定观测，从高斯噪声迭代去噪出动作块，天然表达多模态分布。
:::

**会议 / 年份** RSS 2023（Best Paper，后扩展为 IJRR 2024）｜ **机构** Columbia · Toyota Research Institute · MIT ｜ **方向** Diffusion policy, imitation learning ｜ **真机** ✅ UR5 · Franka

**材料** [Paper](https://arxiv.org/abs/2303.04137) · [Project](https://diffusion-policy.cs.columbia.edu/) · [Code](https://github.com/real-stanford/diffusion_policy)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{chi2023diffusion,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  booktitle = {Robotics: Science and Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.04137}
}</code></pre>
</details>

> 注：代码仓库挂在 `real-stanford` 组织下（Shuran Song 实验室后迁至 Stanford），但**论文署名机构为 Columbia / TRI / MIT，不含 Stanford**。原笔记的「NeurIPS 2023」有误。

### Motivation

行为克隆（behavior cloning）面对多模态示范时会「取中间」：同一局面可以左推也可以右推，回归（MSE）或混合高斯（GMM）会把两个模式平均成一个都不对的动作。DP 的出发点是用扩散过程直接学习动作分布的（近似）梯度，通过迭代去噪表达任意复杂的多模态分布，同时保持训练稳定。

### Method

- **动作空间上的 DDPM**：训练一个噪声预测网络 $\epsilon_\theta(a_k, o, k)$，学习在扩散第 *k* 步加到动作上的噪声（标准 DDPM 目标），推理时从高斯噪声迭代去噪成动作序列。
- **两种去噪骨干**：CNN 版（1D temporal U-Net，观测经 FiLM 条件化）与 Transformer 版（改自 minGPT）。论文的结论不是简单的「CNN 更快」，而是：**CNN U-Net 是好调、鲁棒的默认选择，但对高频/快速变化的动作较弱；Transformer 在这类任务上更好，但对超参更敏感**。
- **视觉编码**：ResNet-18（从零训练），把全局平均池化换成 spatial-softmax、BatchNorm 换成 GroupNorm 以稳定 EMA。
- **Receding horizon**：观测窗口 $T_o$、预测视野 $T_p$、执行步数 $T_e$；常用 $T_o=2, T_p=16, T_e=8$，推理用 DDIM 10 步即可保持性能。

![Diffusion Policy Teaser](/blog/paper-note/Robot_Learning_1/DP/DP_teaser.png)

![Diffusion Policy 输入输出](/blog/paper-note/Robot_Learning_1/DP/policy_input_output.png)

### Experiments

- **仿真**：**12 个任务、4 类 benchmark**（Robomimic、Push-T、Multimodal Block Pushing、Franka Kitchen），相对已有方法平均成功率提升约 **46.9%**。
- **Push-T** 是刻意设计的多模态任务：MSE 回归几乎失败、GMM 也差，DP 大幅领先——直观说明为何需要多模态动作建模。
- **Baseline**：LSTM-GMM、IBC、BET 等，DP 全面更优且训练更稳（无 IBC 的能量模型不稳定问题）。
- **真机**：UR5（真实 Push-T）与 Franka（含多阶段「倒/抹酱」任务）。
- **推理**：DDIM 10 步约 0.1s（RTX 3080）。原笔记「DDIM 0.02s / DDPM 100 步 0.2s」不准确，论文未报 0.2s 这一数字。

### Strengths and Limitations

**Strengths**：自然表达多模态、避免模式平均；训练稳定；能处理高维、时间连贯的动作序列；跨 12 个仿真任务与两类真机验证充分。

**局限（分析）**：迭代去噪比单步回归更贵，需 DDIM 才能接近实时（仍约 0.1s/次）；骨干与视野选择敏感；纯行为克隆，受示范覆盖约束；主设定下视觉编码器从零训练，未用大规模预训练。

### Takeaways

DP 确立了「用扩散建模机器人动作分布」的范式，是 DP3、Octo 扩散头、RDT 的共同源头。它与 ACT 一同把「动作序列 + receding horizon」变成事实标准。

::::paper{tone="dp3"}

## DP3

**3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations**

:::note[一句话]
在**稀疏点云**上做扩散策略：用一个极简的 3D 编码器把点云压成紧凑特征来条件化扩散，换取更好的泛化与样本效率。
:::

**会议 / 年份** RSS 2024 ｜ **机构** 上海期智研究院 · 上海交通大学 · 清华大学 IIIS · 上海人工智能实验室 ｜ **方向** 3D diffusion policy ｜ **真机** ✅ Franka + Allegro 灵巧手

**材料** [Paper](https://arxiv.org/abs/2403.03954) · [Project](https://3d-diffusion-policy.github.io/) · [Code](https://github.com/YanjieZe/3D-Diffusion-Policy)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{ze2024dp3,
  title     = {3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  author    = {Ze, Yanjie and Zhang, Gu and Zhang, Kangning and Hu, Chenyuan and Wang, Muhan and Xu, Huazhe},
  booktitle = {Robotics: Science and Systems},
  year      = {2024},
  url       = {https://arxiv.org/abs/2403.03954}
}</code></pre>
</details>

> 注：原笔记把项目页 / 代码写成 `real-stanford.github.io/dp3` 与 `real-stanford/dp3`，均**无效**；机构也非 Stanford。正确链接与机构已如上更正。

### Motivation

DP 依赖 2D 图像，对视角、光照、纹理变化的泛化有限。DP3 的设想是：几何比纹理更稳定的泛化信号——用一台深度相机得到点云，直接在 3D 表征上条件化扩散策略，从而在更少示范下获得更好的泛化与安全性。

### Method

- **3D 输入**：单深度相机的稀疏点云，用**最远点采样（FPS）**下采到 **512 或 1024 点**。
- **DP3 Encoder（关键更正）**：并非 PointNet++ / Point Transformer，而是一个**三层 MLP + max-pooling（顺序等变）+ 投影头**，输出 64 维紧凑向量。「简单」正是其核心卖点。
- **去噪网络**：与 DP 相同的条件扩散 + 1D temporal U-Net，训练 100 步、推理 DDIM 10 步。与 DP 的差异**仅在观测编码**，故代码改动很小。

![DP3 方法概览](/blog/paper-note/Robot_Learning_1/DP3/method_v3.png)

### Experiments

- **仿真**：**72 个任务、跨 7 个域**（Adroit、Bi-DexHands、DexArt、DexDeform、DexMV、HORA、MetaWorld）；每任务仅 **10 条示范**时相对 baseline 约 **+24.2%**（相对提升）。
- **真机**：Franka 机械臂 + Allegro 灵巧手（RealSense L515），**4 个任务**、每任务 40 条示范：Roll-Up 90%、Dumpling 70%、Drill 80%、Pour 100%，**平均 85%**。
- **更正**：原笔记的「换马克杯 DP 30% / DP3 85%」中，**马克杯任务并不存在**；85% 是上述 4 个真机任务的平均。「相机位姿随机化 +25pp」「512/1024/4096 点消融」在论文中**未核到精确数值**，此处不引用。

### Strengths and Limitations

**Strengths**：编码器极轻（3 层 MLP），无重型 3D 骨干；样本高效（10 条示范即有效）；相对 2D DP 在视角/实例/外观变化下泛化更好、安全违规更少。

**局限（分析）**：依赖标定良好、干净的点云，对深度噪声、透明/反光物体敏感；单视角稀疏点云，性能与裁剪、点云质量强相关；扩散推理成本仍在（靠 10 步 DDIM 缓解）。

### Takeaways

DP3 的价值在于「简单 3D 表征也能显著提升泛化」，而不是堆更复杂的 3D backbone。它提示：当任务对几何敏感时，换观测表征往往比换策略更有效。

::::paper{tone="rt1"}

## RT-1

**RT-1: Robotics Transformer for Real-World Control at Scale**

:::note[一句话]
**可消化大规模异构真机数据**的机器人 Transformer：图像历史 + 语言指令 → 离散动作 token，体现机器人里的 scaling 思路，为 RT-2 铺路。
:::

**会议 / 年份** RSS 2023 ｜ **机构** Google（Robotics at Google + Everyday Robots）｜ **方向** Robot Transformer, imitation learning ｜ **真机** ✅ Everyday Robots 移动操作机

**材料** [Paper](https://arxiv.org/abs/2212.06817) · [Project](https://robotics-transformer1.github.io/) · [Code](https://github.com/google-research/robotics_transformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{brohan2023rt1,
  title     = {{RT-1}: Robotics Transformer for Real-World Control at Scale},
  author    = {Brohan, Anthony and others},
  booktitle = {Robotics: Science and Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2212.06817}
}</code></pre>
</details>

> 注：原笔记项目页 `robotics-transformer.github.io` 缺了「1」；官方页为 `robotics-transformer1.github.io`。

### Motivation

把视觉/NLP 里「大而多样的数据 → 通才模型」的配方搬到机器人：让单个 Transformer 策略吸收大规模、任务无关的真机数据，零样本泛化到新任务、新物体、新环境，并能融合仿真与异构机器人的数据，同时吞吐量要够在真机上实时运行。

### Method

- **动作离散为 token**：每个动作维度均匀离散为 **256 个 bin**，每维一个 token，用交叉熵训练。
- **11 维动作空间**（原笔记只提末端执行器，不完整）：7 维手臂（x/y/z、roll/pitch/yaw、夹爪）+ 3 维底盘（x/y/yaw）+ 1 维模式切换（控臂 / 控底盘 / 终止）。
- **高效视觉骨干**：EfficientNet-B3 提特征，语言经 Universal Sentence Encoder 编码并以 FiLM 注入，TokenLearner 把每帧压成 **8 个 token**；6 帧历史共 48 token，送入 8 层 decoder-only Transformer。
- **实时性**：约 **3 Hz**（原笔记「5Hz」有误）；参数量约 35M。

![RT-1 Teaser](/blog/paper-note/Robot_Learning_1/RT-1/rt1_teaser_model.png)

![RT-1 完整模型](/blog/paper-note/Robot_Learning_1/RT-1/rt1_full_model.png)

### Experiments

- **数据**：约 **13 万条示范**、**13 台机器人**、**700+（744）条指令**，历时 17 个月采集。
- **主结果与 baseline**（成功率）：

| 方法 | Seen | Unseen 指令 | Distractors | Backgrounds |
|:--|:--:|:--:|:--:|:--:|
| **RT-1** | 97% | 76% | 83% | 59% |
| BC-Z | 72% | 19% | 47% | 41% |
| Gato | 65% | 52% | 43% | 35% |

> 更正：原笔记「Gato 33%（unseen）/ 2%（backgrounds）」为杜撰值；实际 Gato 分别为 52% 与 35%。RT-1 的 76% 指的是 **unseen 指令/任务**，非「unseen 物体」。
- **数据规模消融**：论文的核心结论是**数据多样性比数据量更重要**（砍掉 25% 任务类型的伤害≈砍掉约一半数据量）。原笔记「10%→100%: 50%→97%」不是真实数据点（最小测试比例约 22%，对应 seen 59%），此处按论文结论重述。

### Strengths and Limitations

**Strengths**：吸收大规模、多样真机数据且有正向 scaling；对 unseen 指令、干扰物、背景的泛化明显强于 BC-Z / Gato；能融合仿真和异构（如 Kuka）数据而不损原有技能；3 Hz、<100ms 推理，可实时闭环。

**局限（分析）**：纯模仿，无法超过示范水平；泛化是「已见概念的新组合」，非真正的新动作/新技能；数据采集极昂贵（17 个月、13 台机器人）；灵巧度与时程有限，单一 embodiment。

### Takeaways

RT-1 证明了机器人领域「数据规模 + 简单架构」可行，但也暴露采集成本问题；它没用互联网级预训练——这正是 RT-2 要补的短板。

::::paper{tone="rt2"}

## RT-2

**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**

:::note[一句话]
让**大型 VLM 直接输出机器人动作**：把动作编码为文本 token 加入词表，在互联网 VQA 数据 + 机器人示范上 co-fine-tune，从而继承 VLM 的语义理解与推理。
:::

**会议 / 年份** CoRL 2023 ｜ **机构** Google DeepMind ｜ **方向** VLA, web-knowledge transfer ｜ **真机** ✅ Everyday Robots 移动操作机 ｜ **代码** 无官方开源

**材料** [Paper](https://arxiv.org/abs/2307.15818) · [Project](https://robotics-transformer2.github.io/)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{brohan2023rt2,
  title     = {{RT-2}: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control},
  author    = {Brohan, Anthony and others},
  booktitle = {Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.15818}
}</code></pre>
</details>

### Motivation

传统「感知 + 策略」栈难以把互联网级语义带进控制。RT-2 的洞见是：既然大型 VLM 已经会视觉推理和常识，就让它**直接以文本 token 的形式吐出动作**，用一个模型同时做 web VQA 与机器人控制，让语义能力自然迁移到操作上。

### Method

- **backbone**：两个变体，RT-2-PaLI-X（55B）与 RT-2-PaLM-E（12B）。
- **动作 token 化**：**8 维**（6-DoF 末端位移 + 夹爪 + **终止**指令，原笔记漏了终止维），每维离散 256 bin，一步 = **8 个整数 token**（例 `1 128 91 241 5 101 127`）。VLM 自回归生成这些 token，解码为连续动作。
- **co-fine-tuning**：混合互联网 VQA 数据与机器人示范，让 VLM 既保住语义能力又学会控制。混合比例并非统一 50/50——机器人数据在 PaLI-X 中约占 **50%**，在 PaLM-E 中约占 **66%**。

![RT-2 Teaser](/blog/paper-note/Robot_Learning_1/RT-2/rt2_teaser.png)

### Experiments

- **涌现能力评测**（RT-2-PaLI-X-55B）：Symbol Understanding 82%、Reasoning 46%、Person Recognition 53%，**平均约 60%**；对照 RT-1 约 17%、VC-1 约 11%。

> 更正：原笔记「RT-2 62% vs RT-1 与 VLM 微调均 0%」中，**baseline 不是 0%**（RT-1≈17%、VC-1≈11%）；论文的说法是在涌现/语义任务上约 **3×** 于 baseline。
- **泛化**：A/B 评测中约 **2×** 于 baseline；标准 RT-1 任务上与 RT-1 基本持平（未因 VLM 预训练损害操作能力）。
- **规模**：PaLI-X-55B（约 63%）与 PaLM-E-12B（约 62%）在 unseen 泛化上基本打平，且 **PaLM-E 在更难的场景上更好**——不宜简单断言「55B > 12B」。
- **真机**：约 6000 次评测试验，7-DoF 移动操作机；Language-Table 上刷新 SoTA（90% vs 77%）。

### Strengths and Limitations

**Strengths**：统一 VLA，一套模型兼做 VQA 与控制；强语义泛化与推理（涌现约 3×、泛化约 2×）；co-fine-tuning 保住 web 知识；支持链式推理执行多步语义指令；真机评测规模大。

**局限（分析）**：只带来语义泛化，**不产生新的物理技能**（动作仍受机器人数据限制）；模型巨大（至 55B），实时推理需云端 TPU，频率受限；闭源、无权重与代码；依赖 RT-1 级别的示范数据。

### Takeaways

RT-2 定义了 VLA 范式，但把两个问题留给后人：**部署成本**（催生 OpenVLA、TinyVLA 等轻量路线）与**开源**（催生 OpenVLA）。

::::paper{tone="octo"}

## Octo

**Octo: An Open-Source Generalist Robot Policy**

:::note[一句话]
开源的**跨 embodiment 通才策略**：模块化 Transformer + 扩散动作头，在 Open X-Embodiment 上预训练，可在消费级 GPU 上数小时内微调到新机器人。
:::

**会议 / 年份** RSS 2024 ｜ **机构** UC Berkeley · Stanford · CMU · Google DeepMind ｜ **方向** 开源通才策略, diffusion head ｜ **真机** ✅ 9 套真机 / 4 家机构（WidowX、UR5、Franka 等）

**材料** [Paper](https://arxiv.org/abs/2405.12213) · [Project](https://octo-models.github.io/) · [Code](https://github.com/octo-models/octo)
::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{octo2024,
  title     = {Octo: An Open-Source Generalist Robot Policy},
  author    = {{Octo Model Team} and Ghosh, Dibya and Walke, Homer and Pertsch, Karl
               and Black, Kevin and Mees, Oier and Dasari, Sudeep and Hejna, Joey
               and Xu, Charles and Luo, Jianlan and Kreiman, Tobias and Tan, You Liang
               and Sanketi, Pannag and Vuong, Quan and Xiao, Ted and Sadigh, Dorsa
               and Finn, Chelsea and Levine, Sergey},
  booktitle = {Robotics: Science and Systems},
  year      = {2024},
  url       = {https://arxiv.org/abs/2405.12213}
}</code></pre>
</details>

### Motivation

RT-2 这类强通才策略闭源且昂贵。Octo 想做一个**完全开源**、拿来即可高效微调的通才策略：在大规模异构机器人数据上预训练，让用户用少量目标域数据、在消费级 GPU 上几小时内适配到自己的机器人、传感器与动作空间。

### Method

- **模块化输入**：**task tokens**（语言 / 目标图像）+ **observation tokens**（图像 + 本体感觉），送入 Transformer backbone。
- **扩散动作头**：接在 readout token 上的扩散头，输出连续多模态动作（动作块，预测未来 4 步）。
- **readout token + 块状注意力（block-wise attention）**：readout token 只「读」前面的观测/任务 token 而不被它们读，从而可以增删观测/任务/头而不扰乱已训练 token 的语义——这是 Octo 易扩展的关键。
- **规模与组件**：Octo-Small 27M / Octo-Base 93M；语言用 **T5-base**；图像 tokenizer 是**浅层卷积 + patch**（并非 ResNet/ViT）。
- **微调（更正）**：论文推荐**更新全模型**，且其效果优于冻结部分参数；原笔记「冻结 backbone、只更新 readout + 动作头、类 LoRA」的说法不成立——readout/块状设计是为了「加头/加输入」方便，不是冻结方案。

![Octo 架构](/blog/paper-note/Robot_Learning_1/Octo/architecture.png)

### Experiments

- **预训练数据**：Open X-Embodiment，**25 个数据集**的精选混合，约 **80 万条 episode**（应表述为「25 个数据集、覆盖多种 embodiment」，而非「25 个 embodiment」）。
- **对比**：相对 RT-1-X（35M）平均高约 29%；与**大得多的 RT-2-X（55B）表现相当**（原笔记「落后约 10pp」无依据）；参数量差约 **600×**（近 3 个数量级，而非 2 个），且完全开源。
- **动作头消融**：扩散头在多模态任务上明显优于 MSE / 离散 token 头（方向明确；具体百分比以论文表格为准）。
- **真机**：9 套真机 / 4 家机构，含 WidowX、UR5、Franka；零样本 WidowX 约 0.50，微调后 6 任务平均约 0.72。

### Strengths and Limitations

**Strengths**：首个此规模的**全开源**（权重 + 数据管线 + 代码）跨 embodiment 通才策略；模块化设计使微调到新传感器/动作空间很快；扩散头擅长多模态动作；轻量（27M–93M），单张 RTX 4090 可跑（13–17 it/s）。

**局限（分析）**：相对超大的 RT-2-X 是「相当」而非碾压，难任务绝对成功率仍有限；对相机/观测配置敏感；语言 grounding 受冻结 T5-base 限制；异构动作空间的混合仍具挑战。

### Takeaways

Octo 与 OpenVLA 是开源双子星：前者轻量、微调快、偏几何/操作泛化；后者重 VLM、偏语义。Octo 的 readout/块状注意力是一种优雅的「可扩展 Transformer」工程范式。

::::paper{tone="openvla"}

## OpenVLA

**OpenVLA: An Open-Source Vision-Language-Action Model**

:::note[一句话]
开源 **7B 级 VLA**：SigLIP + DINOv2 双视觉编码器 + Llama 2 语言骨干，在 97 万条真机示范上做动作微调，可用 LoRA 在消费级 GPU 上适配。
:::

**会议 / 年份** CoRL 2024 ｜ **机构** Stanford · UC Berkeley · TRI · Google DeepMind · Physical Intelligence · MIT ｜ **方向** 开源 VLA ｜ **真机** ✅ WidowX · Google robot · Franka

**材料** [Paper](https://arxiv.org/abs/2406.09246) · [Project](https://openvla.github.io/) · [Code](https://github.com/openvla/openvla)
::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{kim2024openvla,
  title     = {OpenVLA: An Open-Source Vision-Language-Action Model},
  author    = {Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted
               and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan
               and Sanketi, Pannag and Vuong, Quan and Kollar, Thomas and Burchfiel, Benjamin
               and Tedrake, Russ and Sadigh, Dorsa and Levine, Sergey and Liang, Percy and Finn, Chelsea},
  booktitle = {Conference on Robot Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.09246}
}</code></pre>
</details>

> 更正：会议为 **CoRL 2024**（非 2025）；作者应为 **Ted Xiao**（非「Xiang」），且 **Pathak 并非作者**；机构补上 Physical Intelligence。

### Motivation

RT-2/RT-2-X 这类 VLA 闭源、难以适配。OpenVLA 提供一个开源 7B VLA，在大规模真机数据上训练，重点是**易微调**（含 LoRA、消费级 GPU）与跨 embodiment 泛化，作为闭源方案的开放替代。

### Method

- **双视觉编码器**：**SigLIP（偏语义）+ DINOv2（偏空间/几何）**的 patch 特征在通道维拼接，过 MLP 投影后送入 **Llama 2 7B**；整体基于 Prismatic-7B VLM。
- **动作 token 化（沿用 RT-2）**：每维 256 bin（取训练动作 1–99 分位内均匀离散），7-DoF → **7 个 token/步**，实现上**覆写 Llama 词表中最少用的 256 个 token**。
- **单帧输入**：仅当前一帧图像、无历史（论文明确列为局限）。

![OpenVLA 模型结构](/blog/paper-note/Robot_Learning_1/OpenVLA/openvla_model.png)

![OpenVLA Teaser](/blog/paper-note/Robot_Learning_1/OpenVLA/openvla_teaser.png)

### Experiments

- **数据/算力**：Open X-Embodiment 的 **97 万条**轨迹（70+ 数据集混合），64 张 A100 训练 14 天。
- **主结果**：在跨 29 个任务、多 embodiment 的通才评测上，OpenVLA（7B）以约 **16.5 个百分点（绝对值）** 超过 RT-2-X（55B），参数量约为其 1/7。
- **推理与量化**：约 **6 Hz on RTX 4090**（bf16，约 15GB；原笔记「A100」有误）。量化对比：bf16 71.3%、**INT4 71.9%（约 7GB，基本无损）**、**INT8 反而降到约 58%**——所以推荐的是 **4-bit**，而非原笔记强调的 INT8。
- **微调**：LoRA（约 97.6M 参数）可与全参数微调持平。
- **消融更正**：原笔记「去 DINOv2 −5pp / 去 SigLIP −10pp」的单编码器移除数值**在论文中未核到**，此处不引用（论文的相关消融是与其他 base VLM 的对比，以及分辨率、是否微调视觉编码器等）。
- **真机**：WidowX（BridgeData V2，170 rollouts/17 任务）、Google robot（60/12）、Franka（5Hz/15Hz 两套）。

### Strengths and Limitations

**Strengths**：全开源（权重/代码/数据混合），可复现；7B 超过 55B 的 RT-2-X 达 16.5pp；LoRA 微调 + 4-bit 量化让消费级 GPU 可用（约 7GB）；跨 embodiment。

**局限（分析）**：仅单帧观测（无历史、无多视角、无本体感觉融合）；约 6 Hz 吞吐对高频（如 50Hz）控制偏慢；多数任务成功率仍 <90%，尚不足生产级。

### Takeaways

OpenVLA 把 VLA 拉进开源生态，成为社区默认基线之一；其单帧、离散动作、约 6Hz 的三个约束，正是后续工作（连续/扩散动作头、加入历史、轻量化）的改进方向。

::::paper{tone="rdt"}

## RDT

**RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation**

:::note[一句话]
把扩散策略推到十亿参数级的**双臂基础模型**：1.2B 的 DiT 类去噪 Transformer + 统一动作空间，衔接异构机器人数据，强调零样本与少样本语言跟随。
:::

**会议 / 年份** ICLR 2025 ｜ **机构** 清华大学（TSAIL）｜ **方向** 双臂扩散基础模型 ｜ **真机** ✅ ALOHA 双臂

**材料** [Paper](https://arxiv.org/abs/2410.07864) · [Project](https://rdt-robotics.github.io/rdt-robotics/) · [Code](https://github.com/thu-ml/RoboticsDiffusionTransformer)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{liu2025rdt,
  title     = {{RDT-1B}: a Diffusion Foundation Model for Bimanual Manipulation},
  author    = {Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu
               and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.07864}
}</code></pre>
</details>

### Motivation

双臂操作缺大规模数据，且两臂协调难。RDT 想把扩散 Transformer 扩到 1.2B，用**统一动作空间**在最大规模的多机器人数据上预训练，再用自采双臂数据微调，追求对未见物体/场景的零样本泛化、语言跟随，以及 1–5 条示范的少样本学习。

### Method

- **1.2B DiT 类去噪网络**：以图像、语言、本体状态为多模态条件，对噪声动作 token 去噪。
- **扩散形式（更正）**：是 **DDPM（离散时间扩散）**，但目标是**预测干净动作（sample-prediction）而非预测噪声（ε-prediction）**；采样用 **DPM-Solver++**（约 6 Hz/动作块）。它不是 flow matching。
- **条件编码器（关键更正）**：**语言用 T5-XXL，图像用 SigLIP**；**没有 DINOv2**。原笔记把语言/图像编码器写反并虚构了 DINOv2。
- **统一动作空间**：物理可解释的统一动作向量，异构机器人以 zero-pad + mask 对齐（论文常引最大维度为 128；正文未核到精确值时不作硬断言）。
- **条件注入（更正）**：以 **cross-attention + Alternating Condition Injection（ACI）** 为主（图文 token 在相邻层交替注入），并用 QKNorm、RMSNorm、非线性 MLP 解码头等 DiT 改造；并非「AdaLN-zero 承载条件」。动作块长度 64。
- **微调（更正）**：**全模型微调**，无冻结 backbone / 局部头方案。

![RDT 框架](/blog/paper-note/Robot_Learning_1/RDT/framework.png)

![RDT 微调数据集](/blog/paper-note/Robot_Learning_1/RDT/ft_dataset.png)

### Experiments

- **数据**：预训练 **46 个数据集、100 万+ 轨迹**；微调 **6000+ 条**自采 ALOHA 双臂 episode。
- **Baseline（更正）**：对比 **ACT、OpenVLA、Octo**（非原笔记的 DP）。真机结果举例：

| 任务 | RDT | ACT | OpenVLA |
|:--|:--:|:--:|:--:|
| Wash Cup | 87.5% | 0% | 0% |
| Pour Water | 62.5% | 37.5% | 0% |
| 机器狗灵巧任务 | 48% | 32% | 0% |

> 更正：原笔记「叠衣服/拧瓶盖 RDT 70–85% vs ACT 60–75% / DP 55–70%」为杜撰值，已替换为论文实际任务与数值。
- **消融**：小模型为 **166M**（非「200M」）；小→1.2B 时，Unseen Object 37.5%→50%、Instruction Following 25%→100%（Unseen Scene 持平）。
- **零样本**：对未见物体/场景有一定零样本能力。

### Strengths and Limitations

**Strengths**：当时最大的扩散操作基础模型（1.2B）；统一物理动作空间支持异构多机预训练（46 数据集、100 万+ 轨迹）；少样本（1–5 条）与零样本泛化强，难双臂任务上大幅领先 ACT/OpenVLA/Octo；DPM-Solver++ 支持近实时（约 6 Hz/块）。

**局限（分析）**：真机评测局限于单一 ALOHA 类双臂平台、任务集不大（每任务试验数少，如 n=8 → 12.5% 粒度较粗）；全模型微调成本高、无轻量适配；最难任务的绝对成功率仍中等（如灵巧任务 48%）。

### Takeaways

RDT 代表「DP 路线的 scaling up」：把轻量扩散头换成十亿级 DiT，并用统一动作空间打通异构数据。它与 π0（Flow Matching + VLM）形成扩散基础模型的两种技术选型对照。

## Cross-Paper Comparison

| 论文 | 核心思想 | 学习范式 | 优势 | 局限（分析） |
|:--|:--|:--|:--|:--|
| ACT | 动作块 + CVAE | Imitation（CVAE） | 样本高效、抑制复合误差 | 短时程、单 embodiment |
| DP | 扩散建模动作分布 | Imitation（Diffusion） | 多模态、训练稳 | 迭代去噪较慢 |
| DP3 | 点云 + 轻量 3D 编码器 | Imitation（Diffusion, 3D） | 几何泛化、极轻编码器 | 依赖深度质量 |
| RT-1 | 大数据 + 动作 token | Imitation（Transformer） | 真机 scaling 验证 | 采集贵、无 web 知识 |
| RT-2 | VLM 输出动作 token | VLA（co-fine-tune） | 语义泛化/推理涌现 | 巨大、慢、闭源 |
| Octo | 模块化 Transformer + 扩散头 | 开源通才（Diffusion head） | 全开源、微调快、轻量 | 语言 grounding 受限 |
| OpenVLA | 双视觉编码器 + Llama 2 | 开源 VLA（离散 token） | 7B 超 55B、可 LoRA/量化 | 单帧、约 6Hz 偏慢 |
| RDT | 十亿级 DiT + 统一动作空间 | 双臂扩散基础模型 | 大规模、少/零样本强 | 单平台评测、微调贵 |

## Discussion

把八篇放在一起看，可以梳理出几条趋势：

1. **动作表示是主线**。从回归 → CVAE（ACT）→ 扩散（DP/DP3/Octo/RDT）→ 离散 token（RT-1/RT-2/OpenVLA），本质都在解决同一件事：如何表达示范里的多模态分布。扩散与离散 token 目前是两大主流，各有取舍——扩散连续、精度高但迭代慢，token 化契合自回归 VLM 但受 bin 精度限制。

2. **action chunking + receding horizon 已成事实标准**。ACT 提出、DP 系统化，之后几乎所有策略都预测动作块再滚动执行，兼顾前瞻与反应。

3. **Scaling 有两条路**。一条是「数据规模化」（RT-1 用真机数据、RDT 用异构数据 + 统一动作空间）；另一条是「知识迁移」（RT-2/OpenVLA 复用互联网级 VLM）。两者互补：前者教会「怎么动」，后者带来「懂语义」。

4. **开源与部署成本正在收敛差距**。Octo、OpenVLA 证明中小模型 + 开源 + 高效微调可逼近甚至超过闭源大模型（OpenVLA 7B > RT-2-X 55B）。真正的瓶颈从「能不能做」转向「能不能便宜、实时、可复现地做」。

对 long-horizon manipulation / skill learning / embodied foundation model 的启发：**几何/3D 表征**（DP3）对接触密集任务的泛化值得深挖；**统一动作空间**（RDT）是打通异构数据的实用工程；而**VLA + 扩散/连续动作头**的融合（OpenVLA 之后的连续动作头、RDT 与 π0 的对照）很可能是下一阶段基础模型的主战场。这些线索在本系列后续（Robot Learning 2/3，含 π0 系列、TinyVLA、WorldVLA 等）继续展开。
