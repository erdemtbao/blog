---
title: "Paper Notes: Robot Learning (1)"
published: 2026-03-20
description: Reading notes—ACT, Diffusion Policy, DP3, RT-1, RT-2, Octo, OpenVLA, RDT and related robot learning papers.
image: ''
tags: [Paper Notes, Robot Learning]
category: Paper Notes
draft: false
---

Robot learning 论文阅读记录，对应本地资料目录 `assets/paper_note/Robot_Learning_1/`（与 `paper_node` 二选一；**线上可访问**请同步到 `public/paper-node/Robot_Learning_1/`）。**排序**：从简到繁、从早期到近期，**同一系列相邻**（如 DP→DP3，RT-1→RT-2）。**圆角卡片**内：简称、题名、主要机构、**会议 / 年份**、一句话、`Paper · Project · Code`；**卡片外**先 **BibTeX 折叠**，再写笔记。`::::paper{tone="…"} … ::::` 须四个冒号以嵌套 `:::note`。

:::tip[PDF / 图片放哪？]
放到 `blog/public/paper-node/Robot_Learning_1/<简称>/`，文中链接：`/blog/paper-node/Robot_Learning_1/<简称>/xxx.pdf`（与 `assets/...` 二选一时，以 `public` 为准才能被网站访问）。
:::

## 目录

| 简称 | 论文（短） | 跳转 |
|:---:|:---|:---:|
| ACT | Bimanual + action chunking | [↓](#act) |
| DP | Diffusion visuomotor policy | [↓](#dp) |
| DP3 | 3D diffusion policy | [↓](#dp3) |
| RT-1 | Robotics Transformer at scale | [↓](#rt-1) |
| RT-2 | VLA transfer web knowledge | [↓](#rt-2) |
| Octo | Open generalist robot policy | [↓](#octo) |
| OpenVLA | Open-source 7B VLA | [↓](#openvla) |
| RDT | Diffusion Transformer 双手基础模型 | [↓](#rdt) |

::::paper{tone="act"}

## ACT

**论文：** *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware by Imitating Short Unconstrained Workflows*（ACT / Action Chunking with Transformers）

**主要机构：** Stanford University（ALOHA 等与本文联用的典型设置）

**会议 / 年份：** Robotics: Science and Systems (RSS)，2023

:::note[一句话]
用 Transformer 对**动作块**建模，从短时程示范里学细粒度双手操作；常与 ALOHA 等低成本真机联用。
:::

**材料：** [Paper](https://arxiv.org/abs/2304.13705) · [Project](https://tonyzhaozh.github.io/aloha/) · [Code](https://github.com/tonyzhaozh/aloha) · 本地摘录：[算法伪代码](/blog/paper-node/Robot_Learning_1/ACT/ACT_algo.pdf) · [详细结构图](/blog/paper-node/Robot_Learning_1/ACT/ACT_detail_architecture.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{zhao2023learning,
  title     = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware by Imitating Short Unconstrained Workflows},
  author    = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle = {Robotics: Science and Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.13705}
}</code></pre>
</details>

**我记住的三点**

1. **Action Chunking**：策略一次预测未来 *k* 步动作（动作块），而非逐步输出。这缓解了自回归策略中**复合误差**的问题——一旦某步偏差导致分布偏移，后续帧会快速漂移。动作块将时间粒度从「每帧」提升到「每段」，使策略更鲁棒。
2. **CVAE 训练 + Transformer 解码**：训练阶段使用一个额外的 **Style Encoder**（类似 CVAE encoder），把专家动作序列压缩为一个隐变量 *z*，与当前观测拼接后输入 Transformer decoder 预测动作块。推理时 *z* 从标准正态分布采样，让模型能覆盖**多模态**的示范分布。
3. **Temporal Ensemble**（时序集成）：连续时间步预测的动作块有重叠区，对重叠部分做**指数加权平均**可平滑输出、减少抖动。这比简单「取最新预测」效果好很多，尤其在需要高精度末端控制时（如插入、拧盖等任务）。

**方法 / 实现（想写再写）**

![ACT 算法伪代码](/blog/paper-node/Robot_Learning_1/ACT/ACT_algo.png)

![ACT 结构细节](/blog/paper-node/Robot_Learning_1/ACT/ACT_detail_architecture.png)

整体架构可分为两部分：

- **训练**：观测（关节位置 + 图像 token）和专家动作序列分别编码。专家动作经 Style Encoder 得到隐变量 *z*（CVAE 瓶颈），与观测 token 一起送入 Transformer Decoder（cross-attention），输出完整动作块。KL 散度项约束 *z* 靠近先验。
- **推理**：去掉 Style Encoder，*z* 从先验采样；Transformer Decoder 输入仅有观测，自回归生成 *k* 步动作。

关键实现：
- 图像走 ResNet-18 出 patch feature map，展平后加可学习位置编码。
- 关节位置直接线性投射成 token。
- Transformer 层数不深（4 层 encoder + 7 层 decoder 为典型），参数量约 30M，在单 GPU 上即可训练。

**实验里印象最深**

- 在 ALOHA 双臂低成本硬件上，ACT 仅用 **50 条示范**即可学会将积木放入另一只手、剥黄瓜等细粒度双手任务，成功率 80%+。对比 LSTM-GMM 与 IBC 等基线掉到 20% 以下。
- **动作块长度 *k*** 的消融：*k*=1 退化为逐帧策略，*k* 过大则块间衔接不畅。*k*=100（对应约 2s @ 50Hz）是多任务的甜点。
- Temporal Ensemble 提升约 15 pp 成功率；CVAE 的 *z* 维度较低即可（32）。

**疑问或想继续看的**

- CVAE 的 *z* 在推理时从先验采样，是否会出现 **"posterior collapse"**？论文用的 KL 权重较小 (β=0.01) 来缓解，但在更复杂分布下效果存疑。
- 与 Diffusion Policy 的对比：两者都解决多模态问题，ACT 用 CVAE 瓶颈 + Transformer，DP 用迭代去噪。后续 ACT++ 尝试在动作块上叠加扩散，是否能两全？
- Action Chunking 的思路后续被 π0-FAST、RDT 等大模型采用（FAST token 实际是频域压缩版动作块），说明这一设计有持续影响。

::::paper{tone="dp"}

## DP

**论文：** *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*

**主要机构：** Columbia University · MIT · Toyota Research Institute（按论文作者单位常见组合）

**会议 / 年份：** Advances in Neural Information Processing Systems (NeurIPS)，2023

:::note[一句话]
把**扩散模型**当动作分布：给定观测，通过去噪生成动作序列（或动作块），适合多峰分布与模仿学习。
:::

**材料：** [Paper](https://arxiv.org/abs/2303.04137) · [Project](https://diffusion-policy.cs.columbia.edu/) · [Code](https://github.com/real-stanford/diffusion_policy) · 本地摘录：[Teaser](/blog/paper-node/Robot_Learning_1/DP/DP_teaser.pdf) · [策略输入输出](/blog/paper-node/Robot_Learning_1/DP/policy_input_output.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{chi2023diffusion,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.04137}
}</code></pre>
</details>

**我记住的三点**

1. **核心洞见——用扩散建模动作分布**：将 DDPM 的去噪过程直接用于动作空间，从高斯噪声逐步去噪为条件于观测的动作序列。相比显式回归（MSE loss）或混合高斯（GMM），扩散天然能表达**任意复杂的多模态分布**，不会出现模式平均（"抹平两个模式取中间"）的灾难性失败。
2. **两种骨干实现等效**：论文提出 **CNN-based**（1D temporal U-Net）和 **Transformer-based** 两种去噪网络。前者把观测 embedding 通过 FiLM 条件化注入 U-Net 各层，后者把观测 token 与噪声动作 token 拼接做 cross-attention。二者在 11 个 benchmark 上表现相当，但 CNN 版推理更快。
3. **观测历史窗口 + 动作预测视野（action horizon）**：策略接收最近 *T_o* 步观测，预测未来 *T_a* 步动作，但只执行前 *T_e* 步后重新规划。这一 **receding-horizon** 设计与 ACT 的动作块 + temporal ensemble 精神一致，使得策略既有前瞻又有反应性。

**方法 / 实现（想写再写）**

![Diffusion Policy Teaser](/blog/paper-node/Robot_Learning_1/DP/DP_teaser.png)

![Diffusion Policy 输入输出](/blog/paper-node/Robot_Learning_1/DP/policy_input_output.png)

训练：
- 从数据集 *(o, a)* 采样一对，给 *a* 加 *k* 步噪声得到 *a_k*，训练去噪网络 *ε_θ(a_k, o, k)* 预测噪声（标准 DDPM 目标）。
- 观测侧用 ResNet-18/34 提取图像特征，低维状态直接拼接。

推理：
- 初始化 *a_K ~ N(0, I)*，迭代去噪 *K* 步（典型 100 步 DDPM 或 10–20 步 DDIM）。
- 取预测动作序列的前 *T_e* 步执行，滑窗前进。

关键超参：*T_o*=2, *T_a*=16, *T_e*=8 是多任务常见设置；去噪步数用 DDIM 缩到 10 步仍保持性能。

**实验里印象最深**

- **Push-T 任务**上 MSE 回归完全失败（因为多模态——同一局面可左推或右推），GMM 也很差，而 Diffusion Policy 的覆盖率和成功率大幅领先。这个例子非常直观地说明了为什么需要**多模态动作建模**。
- 在 **RobotMimic 和真机**（Franka + UR5）上全面优于 IBC、BET、LSTM-GMM，且不需要额外的回放或 DAgger。
- 推理延迟：DDPM 100 步约 0.2s，DDIM 10 步约 0.02s，足够 10–50Hz 控制循环。

**疑问或想继续看的**

- 扩散的去噪步数与动作维度的关系：对高自由度机器人（如双臂 14-DoF + 手指）是否需要更多去噪步或更大网络？
- 与 Flow Matching 的对比：π0 用了 Flow Matching 替代 DDPM，声称训练更稳、采样路径更短。两者在机器人策略上的系统性对比还不够。
- DP 的采样噪声可以看作一种**内在探索**，但论文场景全是离线模仿，没有在线 RL；后续 DPPO 等开始在扩散策略上做策略梯度。

::::paper{tone="dp3"}

## DP3

**论文：** *3D Diffusion Policy: Generalizable Visuomotor Policy Learning with Simple 3D Representations*

**主要机构：** Stanford University（REAL Robotics）

**会议 / 年份：** Robotics: Science and Systems (RSS)，2024

:::note[一句话]
在**点云等 3D 表征**上做扩散策略，强调泛化；可看作 DP 在几何感知上的延伸（常写作 DP³）。
:::

**材料：** [Paper](https://arxiv.org/abs/2403.03954) · [Project](https://real-stanford.github.io/dp3/) · [Code](https://github.com/real-stanford/dp3) · 本地摘录：[方法总览](/blog/paper-node/Robot_Learning_1/DP3/method_v3.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{ze2024dp3,
  title     = {3D Diffusion Policy: Generalizable Visuomotor Policy Learning with Simple 3D Representations},
  author    = {Ze, Yanjie and Zhang, Gu and Zhang, Kangning and Hu, Chenyuan and Wang, Muhan and Xu, Huazhe},
  booktitle = {Robotics: Science and Systems},
  year      = {2024},
  url       = {https://arxiv.org/abs/2403.03954}
}</code></pre>
</details>

**我记住的三点**

1. **从 2D 到 3D 的核心动机**：DP 依赖 2D 图像，对**视角变化、光照变化、遮挡**的泛化有限。DP3 的关键是用深度相机获取点云，直接在 **3D 表征**上条件化扩散策略。点云天然对视角不变（相机外参补偿后），也包含了接触面的几何信息。
2. **简洁的 3D 编码器**：不使用复杂的 NeRF 或体素网格，而是用 **PointNet++ / Point Transformer** 对原始点云（下采样到 ~1024 点）编码为 compact 特征向量。这个设计刻意保持**轻量**，训练数据需求与 DP 相当。
3. **泛化性的跃升**：相比 2D DP 在「换相机视角」「换物体颜色/纹理」上成功率暴跌，DP3 在这些设置下几乎不掉点。更关键的是，DP3 在**新物体形态**（训练时没见过的杯子/瓶子）上仍能完成抓取和放置，说明 3D 几何是比 2D 纹理更好的泛化信号。

**方法 / 实现（想写再写）**

![DP3 方法概览](/blog/paper-node/Robot_Learning_1/DP3/method_v3.png)

整体流程：
1. **感知**：RGBD 相机 → 点云（去背景、下采样）→ PointNet++ 编码器 → 3D 特征 *f_3d*。
2. **策略**：与 DP 相同的条件扩散框架：去噪网络 *ε_θ(a_k, f_3d, k)*，1D temporal U-Net 骨干。
3. **执行**：同 DP 的 receding-horizon，取前 *T_e* 步动作执行。

与 DP 的差异仅在**观测编码**——图像换成点云 + PointNet++。去噪网络、噪声调度、训练目标完全相同，因此代码改动很小。

**实验里印象最深**

- **MetaWorld + Adroit** 仿真上 72 个任务，DP3 在 **camera pose 随机化** 下平均成功率比 DP 高 25+ pp。
- 在真机 **Franka** 上做「抓不同颜色/形状马克杯」实验：DP 换杯子后成功率掉到 30%，DP3 保持 85%+。
- 点云数量的消融：512 → 1024 → 4096 点提升微弱，说明 1024 点已经足够表示桌面操作场景。

**疑问或想继续看的**

- 点云质量高度依赖深度相机精度。在工业场景（金属反光、透明物体）下深度缺失严重时，DP3 的鲁棒性会否退化？是否需要深度补全预处理？
- DP3 的 3D 特征是全局特征向量，没有显式的**关键点或部件对应**。与 ReKep、CoPa 等显式约束方法结合后是否能进一步提升？
- 能否用 DINOv2/DINOv3 的稠密特征做「伪 3D」代替真点云？减少对深度相机的依赖。

::::paper{tone="rt1"}

## RT-1

**论文：** *RT-1: Robotics Transformer for Real-World Control at Scale*

**主要机构：** Google DeepMind / Google Robotics（大规模机器人数据团队）

**会议 / 年份：** Robotics: Science and Systems (RSS)，2023

:::note[一句话]
**可吃大规模异构示范**的机器人 Transformer：图像历史 + 语言指令 → 离散动作 token；体现机器人里的 **scaling** 思路，为 RT-2 铺路。
:::

**材料：** [Paper](https://arxiv.org/abs/2212.06817) · [Project](https://robotics-transformer.github.io/) · [Code](https://github.com/google-research/robotics_transformer) · 本地摘录：[完整模型](/blog/paper-node/Robot_Learning_1/RT-1/rt1_full_model.pdf) · [Teaser 模型](/blog/paper-node/Robot_Learning_1/RT-1/rt1_teaser_model.pdf)

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

**我记住的三点**

1. **动作离散化为 token**：将连续动作空间（末端位移 Δx/y/z、旋转、夹爪开合等）各维度分别**均匀离散化为 256 个 bin**，每维一个 token，用分类交叉熵训练。这比回归更稳定，且能直接复用 Transformer 的 softmax token 预测。代价是精度受 bin 数限制，但 256 bin 在厘米级位移下已足够。
2. **大规模真实数据的价值**：RT-1 收集了来自 **13 台移动操作机器人**、**130k+ 条真实示范** 的数据集，涵盖 700+ 个语言指令和抓取、放置、开关抽屉等任务。数据规模是关键——同样架构在小数据上表现一般，但在 130k 示范上泛化到新指令和新物体的能力质变式提升。
3. **EfficientNet + TokenLearner 的高效视觉骨干**：用预训练 **EfficientNet-B3** 提取图像特征，再通过 **TokenLearner** 将特征图压缩为少数 token（8 个），大幅降低 Transformer 序列长度。这使得模型能处理 6 帧图像历史而不爆显存。

**方法 / 实现（想写再写）**

![RT-1 Teaser](/blog/paper-node/Robot_Learning_1/RT-1/rt1_teaser_model.png)

![RT-1 完整模型](/blog/paper-node/Robot_Learning_1/RT-1/rt1_full_model.png)

架构流程：
1. **语言编码**：指令经 Universal Sentence Encoder 编码为固定向量，FiLM 方式注入每帧的视觉特征。
2. **视觉编码**：每帧图像 → EfficientNet-B3 → TokenLearner → 8 个 token。6 帧共 48 token。
3. **Transformer**：8 层 self-attention，输出 → 线性头 → 各维度动作 bin 的分类 logits。
4. **推理**：取 argmax 或加温度采样，解码为连续动作值（bin 中心值）。

关键设计选择：
- 6 帧历史（间隔为最近 6 步 @ 3Hz）提供短期运动信息。
- 推理约 5Hz（Transformer 部分 ~30ms，加视觉约 200ms），足够桌面操作。

**实验里印象最深**

- RT-1 在 Google 内部办公环境中部署，在 **200+ 个长程任务链** 上（如「把可乐放进抽屉」→「关抽屉」）达到 97% 单步成功率。
- 与 Gato、BC-Z 等对比：RT-1 在 **unseen 物体**上成功率 76%（Gato 33%），**unseen 背景**上 59%（Gato 2%），差距悬殊。
- 数据量消融：从 10% 数据到 100% 数据，成功率从 ~50% 上升到 ~97%，几乎线性。

**疑问或想继续看的**

- 动作离散化的精度瓶颈：256 bin 对粗操作够用，但对灵巧操作（如毫米级插入）可能不够。RT-2 之后的 π0-FAST 转向**频域离散化**来提升精度。
- RT-1 是纯机器人数据训练，没有利用互联网级 VLM 预训练。RT-2 的核心改进就是把 PaLI-X / PaLM-E 的知识迁移进来。
- 这种「大量真实数据 + 简单架构」的路线在工业界可复制性有限——130k 条真实示范的采集成本极高。

::::paper{tone="rt2"}

## RT-2

**论文：** *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*

**主要机构：** Google DeepMind

**会议 / 年份：** Conference on Robot Learning (CoRL)，2023

:::note[一句话]
把 **VLM 与机器人动作**联训：动作表示为文本 token，继承互联网级视觉–语言知识，强调对新物体与语义指令的泛化与涌现行为。
:::

**材料：** [Paper](https://arxiv.org/abs/2307.15818) · [Project](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) · Code（以官方发布为准） · 本地摘录：[Teaser](/blog/paper-node/Robot_Learning_1/RT-2/rt2_teaser.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{zitkovich2023rt2,
  title     = {{RT-2}: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control},
  author    = {Zitkovich, Brianna and others},
  booktitle = {Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.15818}
}</code></pre>
</details>

**我记住的三点**

1. **VLM 即 VLA 的思路**：RT-2 的关键洞见——**大型 VLM（PaLI-X 55B / PaLM-E 12B）本身就可以输出机器人动作**，只要把动作编码为文本 token 加入词表（如 `<action_bin_128>`），然后在混合的「互联网 VQA 数据 + 机器人示范数据」上做 co-fine-tuning。这使得模型继承了 VLM 的**语义理解**和**视觉推理**能力。
2. **涌现的推理与泛化**：RT-2 展示了 RT-1 无法完成的能力——如理解「把垃圾扔进垃圾桶」（需要识别哪个物体是垃圾）、「移动到名画旁边」（需要视觉常识）、甚至**链式推理**（先推断目标再执行动作）。这些能力来自 VLM 预训练，并非机器人数据中显式存在的。
3. **动作 token 化的方案**：每个维度（x, y, z, roll, pitch, yaw, gripper）离散化为 256 bin，用特殊 token 表示。一步动作 = 7–8 个 token 的文本序列。VLM 的自回归生成直接输出这些 token，解码后变成连续动作。

**方法 / 实现（想写再写）**

![RT-2 Teaser](/blog/paper-node/Robot_Learning_1/RT-2/rt2_teaser.png)

核心流程：
1. **输入**：当前图像 + 语言指令（如 "pick up the bag on the left"）。
2. **VLM backbone**：PaLI-X 或 PaLM-E 做 vision-language encoding。
3. **输出**：自回归生成动作 token 序列（7–8 个 token/步）。
4. **训练**：多任务混合——约 50% 互联网 VQA 数据 + 50% 机器人示范数据。VQA 数据保持 VLM 能力不退化，机器人数据教会模型输出动作。

与 RT-1 对比：
- RT-1 的视觉骨干（EfficientNet）从零训练或 ImageNet 预训练；RT-2 的骨干是 **55B 参数的 VLM**，知识量差若干数量级。
- RT-1 只理解训练集中见过的指令模板；RT-2 能处理**自由形式的语言**和**隐含推理**。

**实验里印象最深**

- **涌现能力的 eval**：在 RT-2 独有的「语义推理」任务上（如 "pick the fruit that's not an apple"），RT-2 成功率约 62%，而 RT-1 和直接微调的 VLM 均为 0%。
- 在 RT-1 数据集的标准任务上，RT-2 (PaLI-X 55B) 成功率略高于 RT-1 (~1–3 pp)，说明 VLM 预训练没有**损害**原有操作能力。
- **规模律**：PaLI-X 55B > PaLM-E 12B > 5B 版本，在涌现任务上差距尤为明显。

**疑问或想继续看的**

- 55B 参数模型的**推理延迟与部署成本**：论文没给实时性数据，但推测单步推理需要数百毫秒到秒级。与 DP 的 10–50Hz 相比差距很大。后续 OpenVLA (7B)、TinyVLA 正是为解决这个问题。
- 动作 token 化的精度问题与 RT-1 相同——256 bin 在灵巧任务上是否够？FAST 方法（频域 token）就是为了在自回归框架下提升动作精度。
- **数据混合比例**的敏感性：互联网数据太多会让模型「忘记」怎么控制机器人，太少则不能迁移知识。最优比例与任务分布相关。

::::paper{tone="octo"}

## Octo

**论文：** *Octo: An Open-Source Generalist Robot Policy*

**主要机构：** UC Berkeley · Stanford · CMU · Google DeepMind 等（Open X-Embodiment 合作线）

**会议 / 年份：** arXiv 2024 / 机器人学习社区常作通才策略基线（正式收录请以论文页为准）

:::note[一句话]
基于 Transformer 的**通才操作策略**，在 Open X-Embodiment 等大规模数据上预训练；支持语言/目标图像条件与多机形态，**扩散式动作头**生成动作分布。
:::

**材料：** [Paper](https://arxiv.org/abs/2405.12213) · [Project](https://octo-models.github.io/) · [Code](https://github.com/octo-models/octo) · 本地摘录：[架构图](/blog/paper-node/Robot_Learning_1/Octo/architecture.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{octo2024,
  title         = {Octo: An Open-Source Generalist Robot Policy},
  author        = {Mees, Oier and Walters, Dibya and Jayaraman, Dinesh and Mandlekar, Ajay and Li, Yifeng and Li, Pieter and Chen, Yiming and Gupta, Animesh and Ye, Yuke and Hausman, Karol and others},
  year          = {2024},
  eprint        = {2405.12213},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2405.12213}
}</code></pre>
</details>

**我记住的三点**

1. **开源通才策略的定位**：Octo 不追求单一任务 SOTA，而是要做一个「拿来就能微调」的**预训练基础模型**。在 Open X-Embodiment（25 个 embodiment、800k+ 条 episode）上预训练后，用户只需少量目标域数据微调即可部署到自己的机器人上。
2. **模块化 Transformer 架构**：把输入分为 **task tokens**（语言 / 目标图像）和 **observation tokens**（当前图像 + 本体感觉），分别编码后送入 Transformer backbone 做 cross-attention。输出端接 **diffusion action head**（小型去噪网络），生成连续动作。这种模块化使得切换 embodiment 时只需改动 action head 的维度。
3. **微调效率**：Octo 设计了「**readout token**」机制——在 Transformer 序列中插入可学习 token，微调时只更新这些 token + action head 参数，冻结 backbone。这类似 LoRA 的精神，使得微调 Octo 到新任务只需几分钟 GPU 时间。

**方法 / 实现（想写再写）**

![Octo 架构](/blog/paper-node/Robot_Learning_1/Octo/architecture.png)

架构要点：
- **视觉编码**：每个相机的图像经 ResNet / ViT 编码为 token 序列。支持多相机（如腕部 + 第三人称）。
- **语言编码**：T5 encoder 输出的语言 embedding 作为 task token 拼入。
- **Transformer backbone**：标准 decoder-only，约 27M / 93M 两个版本。
- **Diffusion action head**：接在 readout token 上，小型 MLP 去噪网络，预测 action chunk。

训练数据处理：
- Open X-Embodiment 数据统一到「image + language + action」三元组，不同 embodiment 的动作空间通过 padding + mask 处理。
- 训练约在 64 TPU 上跑数天。

**实验里印象最深**

- **微调对比**：Octo 预训练 → 微调 vs 从零训练 DP / RT-1。在 WidowX（桌面操作）、Franka（双臂）上，Octo 微调用 100 条示范即可达到从零训练 1000 条的效果。
- 与 RT-2-X（闭源 55B）对比：在 Open X-Embodiment 评估上 Octo 落后 ~10 pp，但参数量小两个数量级，且完全开源。
- Diffusion head vs MSE head vs GMM head 消融：扩散头在多模态任务上明显更好，单模态任务上三者持平。

**疑问或想继续看的**

- Octo 的训练数据量（800k+）足够大，但数据质量参差。Open X-Embodiment 中一些旧数据集的标注质量差，是否拖后腿？
- 与后续的 OpenVLA 对比：OpenVLA 用 VLM（Llama 2 + SigLIP + DINOv2）替代了 Octo 的 ResNet + T5，语义理解更强。Octo 的优势在于轻量和微调速度。
- 能否把 Octo 的 diffusion head 换成 FAST head？π0-FAST 证明自回归 token 在灵巧任务上可与扩散匹敌。

::::paper{tone="openvla"}

## OpenVLA

**论文：** *OpenVLA: An Open-Source Vision-Language-Action Model*

**主要机构：** Stanford 等（Open X-Embodiment / 开源 VLA 代表）

**会议 / 年份：** Conference on Robot Learning (CoRL)，2025

:::note[一句话]
开源 **7B 级 VLA**：DINOv2 + SigLIP 视觉、Llama 2 语言，经大规模真实机器人示范微调；常与闭源 RT-2-X 等对照。
:::

**材料：** [Paper](https://arxiv.org/abs/2406.09246) · [Project](https://openvla.github.io/) · [Code](https://github.com/openvla/openvla) · 本地摘录：[模型架构](/blog/paper-node/Robot_Learning_1/OpenVLA/openvla_model.pdf) · [Teaser](/blog/paper-node/Robot_Learning_1/OpenVLA/openvla_teaser.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{kim2025openvla,
  title     = {OpenVLA: An Open-Source Vision-Language-Action Model},
  author    = {Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiang, Ted and Pathak, Deepak and Finn, Chelsea},
  booktitle = {Conference on Robot Learning},
  year      = {2025},
  url       = {https://arxiv.org/abs/2406.09246}
}</code></pre>
</details>

**我记住的三点**

1. **双视觉编码器**：OpenVLA 同时使用 **SigLIP**（侧重语义对齐）和 **DINOv2**（侧重空间 / 几何特征）的视觉编码器，两者的 patch token 拼接后通过投影层送入 Llama 2 7B 的输入。这种设计来自 Prismatic VLM 的洞见——**语义 + 几何特征互补**比单用一种好。
2. **动作 token 化沿用 RT-2 方案**：每个动作维度离散化为 256 bin，用 Llama 的词表中的特殊 token 表示。自回归生成 7 个 token（Δx/y/z, Δrx/ry/rz, gripper），解码为连续动作。整体与 RT-2 一致，但 backbone 换成开源的 Llama 2 + SigLIP。
3. **全开源与微调便利性**：OpenVLA 是首个 **完全开源**、**可从 HuggingFace 直接下载**、**有标准微调脚本** 的 7B VLA。社区可以像微调 LLaMA 一样微调 OpenVLA 到自己的机器人上。权重、数据、代码全部公开。

**方法 / 实现（想写再写）**

![OpenVLA 模型结构](/blog/paper-node/Robot_Learning_1/OpenVLA/openvla_model.png)

![OpenVLA Teaser](/blog/paper-node/Robot_Learning_1/OpenVLA/openvla_teaser.png)

训练流程：
1. **阶段一**：Prismatic VLM 预训练（SigLIP + DINOv2 + Llama 2 7B），在 LLaVA-1.5 的视觉指令数据上训练视觉–语言对齐。
2. **阶段二**：在 Open X-Embodiment 的 970k 条机器人 episode 上做 **action fine-tuning**。输入格式 = 图像 + 语言指令，输出 = 7 个动作 token。
3. **微调到新任务**：标准的 LoRA 或全参数微调，少量数据（~50–100 demos）即可。

推理：
- 7B 模型在 A100 上约 5–6 Hz（包含视觉编码 + 自回归生成 7 token），勉强满足桌面操作。
- 量化到 INT8 后可加速到 ~8 Hz。

**实验里印象最深**

- 在 **WidowX** 真机上的「Bridge V2」评估中，OpenVLA 微调后成功率与 RT-2-X（55B，闭源）持平甚至略高，参数量却只有后者的 1/8。
- 与 Octo 对比：OpenVLA 在**需要语义理解的任务**上（如区分相似物体、理解复杂指令）明显更好，因为 VLM 基座更强。在简单拾放任务上二者接近。
- 消融双视觉编码器：去掉 DINOv2（只留 SigLIP）掉约 5 pp，去掉 SigLIP（只留 DINOv2）掉约 10 pp。印证了「语义 + 几何」互补的假设。

**疑问或想继续看的**

- 7B 的推理延迟（5–6 Hz）对灵巧操作仍然太慢。TinyVLA 和 VLA-Adapter 正在探索 <1B 的轻量 VLA。
- 动作精度问题：256 bin 的离散化与 RT-2 共享同样的精度瓶颈。OpenVLA v2 开始探索扩散式 / FAST 式的连续动作头。
- OpenVLA 目前是**单帧输入**（无历史），能否加入 temporal 信息（如 ACT 的多帧观测）来处理需要运动信息的任务？

::::paper{tone="rdt"}

## RDT

**论文：** *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*

**主要机构：** 清华大学等（RDT-robotics 发布方）

**会议 / 年份：** International Conference on Learning Representations (ICLR)，2025

:::note[一句话]
**大规模双手操作扩散基础模型**：Transformer + 扩散建模多模态动作；统一动作空间衔接异构机器人数据，强调零样本与语言跟随。
:::

**材料：** [Paper](https://arxiv.org/abs/2410.07864) · [Project](https://rdt-robotics.github.io/rdt-robotics) · [Code](https://github.com/thu-ml/RoboticsDiffusionTransformer) · 本地摘录：[架构](/blog/paper-node/Robot_Learning_1/RDT/framework.pdf) · [微调数据](/blog/paper-node/Robot_Learning_1/RDT/ft_dataset.pdf)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{liu2024rdt,
  title   = {{RDT-1B}: a Diffusion Foundation Model for Bimanual Manipulation},
  author  = {Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  journal = {arXiv preprint arXiv:2410.07864},
  year    = {2024}
}</code></pre>
</details>

**我记住的三点**

1. **Diffusion + Transformer at scale**：RDT 不是小型去噪 MLP，而是一个 **1.2B 参数的 Transformer**（与 DiT 同架构）作为去噪网络。输入是噪声动作 token + 多模态条件（图像、语言、本体状态），通过标准 DDPM 训练。这是目前把扩散策略推到**十亿参数级**的代表。
2. **统一动作空间 (Unified Action Space)**：为解决不同机器人（单臂 / 双臂 / 不同自由度）的动作维度不一致问题，RDT 定义了一个 **128 维的最大公约数动作空间**，所有机器人的动作 zero-padding 到 128 维，并用 mask 标记有效维度。这使得预训练可以混用异构机器人数据。
3. **预训练 + 微调范式**：在来自 46 个数据集、多种 embodiment 的数据上预训练后，只需 **少量目标域数据**（50–100 demos）微调即可适配新机器人。微调时只更新 action head 的部分参数（类似 Octo 的 readout 思路），backbone 大部分冻结。

**方法 / 实现（想写再写）**

![RDT 框架](/blog/paper-node/Robot_Learning_1/RDT/framework.png)

![RDT 微调数据集](/blog/paper-node/Robot_Learning_1/RDT/ft_dataset.png)

架构：
- **条件编码**：SigLIP 编码语言，SigLIP + DINOv2 编码图像（多相机），本体状态线性投射。所有条件 token 拼接。
- **去噪 Transformer**：类 DiT 的 Transformer，每层用 AdaLN-Zero 注入 diffusion timestep。输入 = 条件 token 拼接 噪声动作 token。
- **训练**：标准 DDPM 的 ε-prediction。128 维动作 token，chunk size 64（即预测未来 64 步动作）。
- **推理**：DDIM 10–20 步去噪。

关键差异 vs DP：
- DP 的去噪网络是轻量 1D U-Net（~30M），RDT 是 **1.2B Transformer**——后者的表达能力和数据吞吐量大得多。
- RDT 支持**多模态条件**（语言 + 多相机 + 本体），DP 通常只有图像。

**实验里印象最深**

- 在 ALOHA 双臂真机上，RDT 用 50 条示范微调后，在叠衣服、拧瓶盖等**精细双手任务**上成功率 70–85%，优于 ACT（60–75%）和 DP（55–70%）。
- **零样本能力**：在某些简单任务（如抓取已知物体）上，预训练的 RDT 不经微调即可执行，说明大规模预训练带来了泛化。
- 消融 1.2B vs 200M 参数：大模型在**多步长程任务**上优势明显（约 +15 pp），在单步任务上差距较小。

**疑问或想继续看的**

- 1.2B 的推理成本：RDT 在 A100 上约 3Hz（DDIM 10 步），对快速反应场景可能不够。能否结合**一致性蒸馏 (consistency distillation)** 把去噪步压到 1–2 步？
- 128 维统一动作空间的 padding 是否引入了无用梯度？是否有更优雅的方案（如 per-embodiment adapter）？
- RDT 与 π0 的对比：π0 用 Flow Matching + VLM backbone，RDT 用 DDPM + DiT backbone。两者在同样数据量下谁更优？π0.5/π0.6 的结果似乎更强，但闭源难以公平对比。

**和本页其他论文**

- [ACT](#act)：小规模模仿 + 动作块的先驱；其后为 **规模化** Transformer / 扩散 / VLA。ACT 的 action chunking 思想在 DP、RDT 中均有体现。
- [DP](#dp) → [DP3](#dp3)：2D 图像观测 → 3D 表征 + 扩散策略。DP 建立了用**扩散建模动作分布**的范式，DP3 引入**3D 泛化**，RDT 则将扩散推到**十亿参数级基础模型**。
- [RT-1](#rt-1) → [RT-2](#rt-2)：从纯机器人 Transformer 到 **VLA** 与网络知识迁移。RT-1 展示了大规模真实数据的价值，RT-2 展示了 VLM 预训练知识的迁移，OpenVLA 则将此路线**开源化**。
- [Octo](#octo) ↔ [OpenVLA](#openvla)：两条开源路线——Octo 轻量、微调快，侧重**操作泛化**；OpenVLA 重 VLM、语义强，侧重**语言理解**。选择取决于任务需要更多「几何泛化」还是「语义理解」。
- [RDT](#rdt) ↔ [DP](#dp) / [DP3](#dp3)：**扩散 + Transformer + 大规模预训练** vs 单任务轻量扩散策略。RDT 是 DP 路线的「scaling up」版本。

## 新增一篇时怎么写（极简）

1. 在 `Robot_Learning_1` 下建子文件夹，复制 **卡片块** + **BibTeX 折叠** + **笔记小节**。  
2. `tone` 见 `markdown.css` 中 `[data-paper-tone="…"]`；新色仿照追加一条。  
3. 目录表加一行，`[↓](#锚点)` 与标题生成 slug 一致。  
4. 静态资源进 `public/paper-node/Robot_Learning_1/...`。

> [!TIP]
> 需要更细的拆解时，在卡片下加 `###` 小节即可。

---

_Changelog：2026-03 — 扩充至与 `Robot_Learning_1` 目录一致的 8 篇；资源路径改为 `Robot_Learning_1`。_
_Changelog：2026-04 — 全部填充深度笔记文字、插入论文摘录 PDF。_
