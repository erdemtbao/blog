---
title: "Paper Notes: Reinforcement Learning (2)"
published: 2026-03-17
description: 深度强化学习算法精读——DQN、DDPG、PPO、SAC、CQL、IQL、QRL（Quasimetric RL）、DPO、GRPO。从深度价值/连续控制到离线 RL 再到 LLM 后训练。
image: ''
tags: [Paper Notes, Reinforcement Learning]
category: Paper Notes
draft: false
---

## Overview

本篇是 [Reinforcement Learning (1)](/blog/posts/paper-notes-reinforcement-learning-1/) 的延续。如果说第一篇讲的是**形式化基础**（MDP/POMDP、MC/TD、Q-learning、策略梯度），那么这一篇讲的是把这些基础**落到深度网络与大规模场景**里的九个代表算法。

九个算法按三条线索组织（顺序与命名略作重排，便于连贯阅读）：

- **深度价值与连续控制** — [DQN](#dqn)（离散动作的深度价值）→ [DDPG](#ddpg)（确定性策略搬到连续动作）→ [PPO](#ppo)（稳定好调的 on-policy 默认）→ [SAC](#sac)（最大熵 off-policy，样本高效）。
- **离线 / 目标条件 RL** — [CQL](#cql)（保守下界压制 OOD 高估）→ [IQL](#iql)（expectile 隐式回避 OOD 查询）→ [QRL](#qrl)（用**拟度量**直接学最优目标可达距离）。
- **LLM 后训练** — [DPO](#dpo)（免显式奖励模型的偏好优化）→ [GRPO](#grpo)（组内相对优势，去掉 critic）。

一条主线贯穿：**如何在「函数逼近 + 自举」的不稳定性、以及「数据分布 ≠ 策略分布」的偏移之间，找到能稳定训练的目标函数。** 从 DQN 的目标网络、DDPG/TD 系的过估计抑制、离线 RL 的保守约束，到 LLM 后训练把「奖励」重参数化掉，本质都在回答同一个问题。

## Paper List

| 简称 | 年份 / Venue | 主题 | 底座 / 前身 | 核心思想 |
|:--|:--|:--|:--|:--|
| [DQN](#dqn) | Nature 2015 | 深度价值 | Q-learning | 神经网络逼近 Q + 经验回放 + 目标网络 |
| [DDPG](#ddpg) | ICLR 2016 | 连续控制 | DPG + DQN | 确定性 actor-critic + 软目标更新 |
| [PPO](#ppo) | arXiv 2017 | on-policy | TRPO | 裁剪替代目标限制更新幅度 |
| [SAC](#sac) | ICML 2018 | 最大熵 off-policy | DDPG/TD3 | 熵正则 + 双 Q + 重参数化随机策略 |
| [CQL](#cql) | NeurIPS 2020 | 离线 RL | Q-learning | 保守项压低 OOD 动作的 Q，得下界 |
| [IQL](#iql) | ICLR 2022 | 离线 RL | Q-learning | expectile 回归隐式取 max，不查 OOD |
| [QRL](#qrl) | ICML 2023 | 目标条件 RL | — | 拟度量建模最优目标可达距离 |
| [DPO](#dpo) | NeurIPS 2023 | LLM 对齐 | RLHF | 语言模型即隐式奖励模型，免 RL |
| [GRPO](#grpo) | arXiv 2024 | LLM 推理后训练 | PPO | 组内相对优势估计，去掉 value 网络 |


::::paper{tone="dqn"}

## DQN

**Human-level Control Through Deep Reinforcement Learning**

:::note[一句话]
用**卷积网络逼近 Q 函数**，配合**经验回放**与**目标网络**稳定离策略学习，在 49 个 Atari 游戏上用同一套架构与超参达到接近专业人类玩家的水平——开启深度价值方法的时代。
:::

**年份 / Venue** Nature 2015（前身 NIPS 2013 Deep Learning Workshop）｜ **机构** DeepMind ｜ **方向** 深度价值方法, 离散动作 ｜ **基准** Atari 2600（ALE），49 games

**材料** [Paper (Nature)](https://www.nature.com/articles/nature14236) · [arXiv 前身](https://arxiv.org/abs/1312.5602) · [Code (Dopamine)](https://github.com/google/dopamine)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{mnih2015human,
  title   = {Human-level Control Through Deep Reinforcement Learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A.
             and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin
             and others},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015},
  url     = {https://www.nature.com/articles/nature14236}
}</code></pre>
</details>

### Motivation

表格型 Q-learning 在小状态空间可证明收敛，但面对 Atari 这种**高维像素输入**必须用函数逼近。而「非线性逼近 + 自举 + 离策略」正是 Sutton 所说的**致命三要素（deadly triad）**，直接训练极易发散：相邻帧高度相关、目标随参数漂移。DQN 的贡献不是提出 Q-learning，而是给出**让它在深度网络下稳定收敛的工程配方**。

### Method

- **网络**：把连续 4 帧灰度图（84×84）堆叠为输入，卷积网络一次前向输出每个离散动作的 $Q(s, \cdot)$。
- **经验回放（experience replay）**：转移 $(s, a, r, s')$ 存入容量约 1M 的缓冲区，训练时**均匀随机采小批**，打破时间相关性、提高数据复用。
- **目标网络（target network）**：单独维护一份参数 $\theta^-$ 计算 TD 目标，每隔 $C$ 步才从在线网络同步一次，避免「目标随预测一起动」。
- **损失**：

$$
L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i)\big)^2\Big]
$$

### Experiments

- **基准**：49 个 Atari 游戏，**同一套网络结构与超参数**，仅以原始像素与得分为输入/信号。
- **结果（原文明确报告）**：DQN 达到「与专业人类游戏测试者相当」的水平；在 **49 个游戏中的 29 个**上得分**高于人类的 75%**。
- **消融**：去掉经验回放或目标网络都会显著掉分，验证这两件稳定化设计的必要性。

### Strengths and Limitations

**Strengths**：首次证明「像素到动作」的端到端深度 RL 可行且通用；经验回放 + 目标网络成为后续几乎所有 off-policy 深度 RL 的标配。

**局限**：只处理**离散、低维动作**；$\max$ 算子带来系统性**Q 过估计**（后由 Double DQN 缓解）；样本效率低（需上亿帧）；对奖励尺度与超参敏感。后续 Double DQN、Dueling、Prioritized Replay、Rainbow 等逐一改进。

### Takeaways

DQN 把「[Q-learning](/blog/posts/paper-notes-reinforcement-learning-1/#q-learning) + 深度网络」变得可训练，是本篇一切深度价值/演员-评论家方法的起点。它遗留的两个问题——**过估计**与**只能离散动作**——分别由 [SAC](#sac)/TD3 的双 Q 和 [DDPG](#ddpg) 的确定性连续策略接手。

::::paper{tone="ddpg"}

## DDPG

**Continuous Control with Deep Reinforcement Learning**

:::note[一句话]
把 DQN 的稳定化配方（回放 + 目标网络）搬到**连续动作**：用一个**确定性 actor** 输出动作、一个 critic 估 Q，按**确定性策略梯度**沿 $\nabla_a Q$ 更新 actor。
:::

**年份 / Venue** ICLR 2016（arXiv 2015）｜ **机构** DeepMind ｜ **方向** 确定性 actor-critic, 连续控制 ｜ **基准** 20+ 仿真物理任务（含从像素学习）

**材料** [Paper](https://arxiv.org/abs/1509.02971) · [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) · [Code (baselines)](https://github.com/openai/baselines)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{lillicrap2016continuous,
  title     = {Continuous Control with Deep Reinforcement Learning},
  author    = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander
               and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  booktitle = {International Conference on Learning Representations},
  year      = {2016},
  url       = {https://arxiv.org/abs/1509.02971}
}</code></pre>
</details>

### Motivation

DQN 靠 $\max_{a'} Q(s', a')$ 选动作，但连续动作空间无法枚举这个 $\max$。DDPG 借用 Silver 等（DPG, ICML 2014）的**确定性策略梯度**：既然 critic 对动作可微，就让一个确定性 actor $\mu(s)$ 直接沿 $\nabla_a Q$ 爬坡，绕开对连续动作求 $\max$。

### Method

- **双网络**：actor $\mu(s; \theta^\mu)$ 输出确定动作，critic $Q(s, a; \theta^Q)$ 用 DQN 式 TD 损失训练。
- **确定性策略梯度**：

$$
\nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \rho^\beta}\Big[\nabla_a Q(s, a; \theta^Q)\big|_{a=\mu(s)} \cdot \nabla_{\theta^\mu}\mu(s; \theta^\mu)\Big]
$$

- **软目标更新（soft update）**：$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$，取很小的 $\tau$（原文 0.001）让目标网络缓慢跟随，比 DQN 的硬同步更平滑。
- **探索**：确定性策略本身不探索，训练时叠加 **Ornstein–Uhlenbeck 时间相关噪声**（后续实践常改用高斯噪声）；同样用经验回放。

### Experiments

- **基准**：20 多个仿真物理控制任务（cartpole 摆起、reacher、gripper、walker、腿式移动、以及仿真驾驶 Torcs 等），部分**直接从像素**学习。
- **结果**：能在这些连续控制任务上学到有效策略，并与规划基线 iLQG 对比归一化性能。**具体每任务回报数值原文以学习曲线呈现，此处定性描述，不引用精确分数。**

### Strengths and Limitations

**Strengths**：首个在高维连续控制上可用的深度 off-policy actor-critic，样本效率优于当时的 on-policy 方法。

**局限**：**对超参与随机种子极其敏感**，训练脆弱、易发散；继承 DQN 的 **Q 过估计**且被确定性 $\max$ 放大。这两点正是 [TD3](/blog/posts/paper-notes-reinforcement-learning-1/) 与 [SAC](#sac) 的直接改进动机（双 critic、延迟更新、目标平滑、熵正则）。

### Takeaways

DDPG 打通了「DQN 稳定化技巧 → 连续动作」的路径，但它的脆弱性说明**光有确定性策略梯度不够**，还需要系统性抑制过估计——这条线在 [SAC](#sac) 收束。

::::paper{tone="ppo"}

## PPO

**Proximal Policy Optimization Algorithms**

:::note[一句话]
用**裁剪替代目标（clipped surrogate）**把每步策略更新限制在旧策略附近，用一阶优化近似 TRPO 的信赖域效果——稳定、易实现、易调参，成为工程上 on-policy 的默认选择。
:::

**年份 / Venue** arXiv 2017（**技术报告，无同行评审 venue**）｜ **机构** OpenAI ｜ **方向** on-policy 策略优化 ｜ **基准** MuJoCo（7 连续控制）+ Atari

**材料** [Paper](https://arxiv.org/abs/1707.06347) · [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) · [Code (SB3)](https://github.com/DLR-RM/stable-baselines3)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{schulman2017proximal,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017},
  url     = {https://arxiv.org/abs/1707.06347}
}</code></pre>
</details>

### Motivation

策略梯度对步长极敏感：走大了会把策略推到回报塌陷的区域，且 on-policy 数据一旦策略变差就作废。TRPO 用硬 KL 信赖域约束解决稳定性，但要算二阶量、实现复杂。PPO 想用**只需一阶 SGD** 的目标达到类似的「不要一步走太远」效果。

### Method

- **概率比**：$r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$。
- **裁剪替代目标**：

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
$$

裁剪区间常取 $\epsilon = 0.2$。当更新使 $r_t$ 越出 $[1-\epsilon, 1+\epsilon]$ 且方向对目标有利时，$\min$ 与 clip 联手**移除其改进激励**，从而抑制过大更新。

- **优势估计**：用 GAE 计算 $\hat{A}_t$；完整目标再加价值函数回归项与熵奖励：$L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2 S[\pi_\theta]$。
- **训练循环**：采一批 rollout，对同一批数据做 **K 个 epoch 的小批 SGD**（在 TRPO 的「一批一更新」上提高数据利用）。

### Experiments

- **基准**：MuJoCo 的 7 个连续控制任务与 Atari（ALE）。
- **结果**：在连续控制上匹配或超过 A2C/ACER/TRPO/vanilla PG；在 Atari 上按「训练全程平均回报」与「最终表现」两种口径评估时，PPO 相对 A2C/ACER 胜出的游戏数最多。**具体每游戏分数此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：实现简单、对超参相对宽容、跨任务鲁棒，是仿真机器人、游戏、以及 **RLHF/LLM 后训练**中最常用的强基线。

**局限**：**on-policy → 样本效率低**（每次更新后旧数据基本作废）；裁剪是启发式而非严格信赖域，理论保证弱于 TRPO；对优势归一化、学习率、epoch 数等实现细节仍较敏感。

### Takeaways

PPO 是「稳」的代名词，也是 [GRPO](#grpo) 的直接前身——GRPO 保留 PPO 的裁剪目标，只把「优势怎么算」从 critic 换成组内相对回报。

::::paper{tone="sac"}

## SAC

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor**

:::note[一句话]
在回报目标里加上**策略熵**：既最大化回报又保持随机性，配合**双 Q（取 min）**抑制过估计、**重参数化**的随机 actor——off-policy、样本高效，是连续控制的另一根默认基线。
:::

**年份 / Venue** ICML 2018（自动温度版 arXiv 1812.05905）｜ **机构** UC Berkeley ｜ **方向** 最大熵 off-policy actor-critic ｜ **基准** MuJoCo（含 21-DoF Humanoid）

**材料** [Paper](https://arxiv.org/abs/1801.01290) · [Project](https://sites.google.com/view/soft-actor-critic) · [Code](https://github.com/haarnoja/sac)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{haarnoja2018soft,
  title     = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author    = {Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle = {International Conference on Machine Learning},
  pages     = {1861--1870},
  year      = {2018},
  url       = {https://arxiv.org/abs/1801.01290}
}</code></pre>
</details>

### Motivation

DDPG 脆弱、TD3 靠若干技巧修补，但确定性策略天生不利于探索。SAC 换一个目标：**最大熵 RL**——在最大化回报的同时最大化策略熵，鼓励「在能拿到高回报的前提下尽量随机」，天然改善探索、对多模态最优更鲁棒，且推导出的更新更稳。

### Method

- **最大熵目标**：

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\big[r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\big]
$$

- **随机 actor + 重参数化**：策略为高斯并经 tanh 压缩到动作边界；用 reparameterization trick 让策略梯度低方差可导。
- **双 Q（clipped double-Q）**：两个 critic，目标取 $\min(Q_1, Q_2)$ 抑制过估计（这一点 SAC 原文即含）。
- **温度 $\alpha$**：原文为固定超参；后续 [1812.05905](https://arxiv.org/abs/1812.05905) 引入**自动温度调节**——按目标熵 $\bar{\mathcal{H}}$ 做约束优化自动学 $\alpha$，免去逐任务手调。

### Experiments

- **基准**：MuJoCo 连续控制（Hopper、Walker2d、HalfCheetah、Ant，以及高维 Humanoid ~21-DoF）。
- **结果**：在样本效率与稳定性上匹配或超过 DDPG/PPO/TD3，在最难的 Humanoid 上优势明显。**具体每任务回报此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：样本高效、训练稳定、对超参相对宽容（尤其自动温度版）；随机策略利于探索，是真机与仿真连续控制的常用 off-policy 首选。

**局限**：只适用于连续（或需特殊处理的离散）动作；双 Q + 目标网络 + 温度调节使实现比 PPO 复杂；最大熵目标引入的 $\alpha$ 改变了「最优策略」的定义，与纯回报最优略有偏差。

### Takeaways

SAC 把 DQN 以来悬而未决的**过估计**（双 Q min）与**探索**（熵正则）在一个框架内一并解决，是深度价值/演员-评论家这条线的集大成者。接下来问题从「怎么在线学好」转向「**只有固定数据集时怎么办**」——进入离线 RL。

::::paper{tone="cql"}

## CQL

**Conservative Q-Learning for Offline Reinforcement Learning**

:::note[一句话]
离线 RL 的致命伤是对**数据集没覆盖的动作**给出虚高的 Q 并被自举放大。CQL 在贝尔曼误差外加一个**保守项**：压低 OOD 动作的 Q、抬高数据内动作的 Q，得到真实价值的**下界**。
:::

**年份 / Venue** NeurIPS 2020 ｜ **机构** UC Berkeley / Google ｜ **方向** 离线 RL（offline / batch RL）｜ **基准** D4RL, Atari

**材料** [Paper](https://arxiv.org/abs/2006.04779) · [Project](https://sites.google.com/view/cql-offline-rl) · [Code](https://github.com/aviralkumar2907/CQL)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{kumar2020conservative,
  title     = {Conservative Q-Learning for Offline Reinforcement Learning},
  author    = {Kumar, Aviral and Zhou, Aurick and Tucker, George and Levine, Sergey},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.04779}
}</code></pre>
</details>

### Motivation

在线 RL 靠与环境交互纠正高估；离线 RL 只有一批固定数据，无法验证。策略一旦被虚高的 Q 诱导去选**数据集外动作**，误差会经贝尔曼自举不断放大——这是离线 RL 发散的根因（分布偏移 / 外推误差）。CQL 的思路：**宁可保守**，让学到的 Q 成为真实值的下界，从根上堵住「幻想出的高价值动作」。

### Method

- 在标准贝尔曼误差外，加一个保守正则（CQL(H) 变体）：

$$
\alpha \cdot \mathbb{E}_{s \sim D}\Big[\log \textstyle\sum_a \exp Q(s, a) - \mathbb{E}_{a \sim D}[Q(s, a)]\Big]
$$

- 前一项（logsumexp）对**所有动作**取软 $\max$ 并压低，后一项抬高**数据集内**动作；两者相减即「压 OOD、抬 in-dist」。
- $\alpha$ 平衡保守程度与贝尔曼拟合。可证在合适条件下得到价值下界。

### Experiments

- **基准**：D4RL（MuJoCo locomotion、AntMaze、Adroit、kitchen 等）与 Atari 离线数据集。
- **结果**：在较难任务（AntMaze、Adroit、kitchen）上相对 BEAR/BRAC/BCQ 等先前离线方法有明显优势。**具体 D4RL 分数因数据集版本（-v0/-v2）而异，此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：思想直接、理论上给出价值下界、对数据集覆盖差的场景鲁棒；是离线 RL 与机器人离线策略学习的常用强基线。

**局限**：保守可能**过头**，在数据本就近最优时压制真实高价值动作、拖累性能；$\alpha$ 需调；logsumexp 在大/连续动作空间需采样近似。

### Takeaways

CQL 用「压低 OOD」直接对抗高估，但代价是可能过度保守。[IQL](#iql) 换一条更温和的路：**根本不去查 OOD 动作**。

::::paper{tone="iql"}

## IQL

**Offline Reinforcement Learning with Implicit Q-Learning**

:::note[一句话]
既然查询 OOD 动作是离线 RL 出错的源头，IQL 干脆**不查**：用 **expectile 回归**只在**数据分布内**近似「对动作取 max」，再用优势加权回归提取策略。
:::

**年份 / Venue** ICLR 2022 ｜ **机构** UC Berkeley / Google ｜ **方向** 离线 RL ｜ **基准** D4RL（locomotion / AntMaze / kitchen / Adroit）

**材料** [Paper](https://arxiv.org/abs/2110.06169) · [Code](https://github.com/ikostrikov/implicit_q_learning)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{kostrikov2022offline,
  title     = {Offline Reinforcement Learning with Implicit Q-Learning},
  author    = {Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  url       = {https://arxiv.org/abs/2110.06169}
}</code></pre>
</details>

### Motivation

CQL 通过惩罚压制 OOD，但仍要在 OOD 上算 Q。IQL 的观察更彻底：**TD 目标里的 $\max_a Q(s', a)$ 才是 OOD 查询的来源**。若能只用**数据里出现过的动作**来近似这个 max，就再也不碰 OOD，训练自然稳定。

### Method

- **expectile 回归学 V**：用不对称的 expectile 损失（$\tau \to 1$ 时上偏，逼近数据内的 max）拟合状态价值：

$$
L_V = \mathbb{E}\big[L_2^\tau(Q(s,a) - V(s))\big], \quad L_2^\tau(u) = |\tau - \mathbb{1}(u<0)| \cdot u^2
$$

- **Q 拟合**：$Q \leftarrow r + \gamma V(s')$，目标用 $V$ 而非 $\max_a Q$，**全程不查 OOD 动作**。
- **策略提取**：优势加权回归（AWR 式）$\max_\pi \mathbb{E}\big[\exp(\beta(Q(s,a)-V(s))) \cdot \log\pi(a\mid s)\big]$。

### Experiments

- **基准**：D4RL 全套（locomotion、AntMaze、FrankaKitchen、Adroit）。
- **结果**：与 CQL 相当或更优，尤其在 **AntMaze** 上表现突出，且**计算更省**。**具体分数此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：训练稳定、实现相对简单、算力友好；「不查 OOD」从设计上回避了离线 RL 的主要发散源。

**局限**：expectile 的 $\tau$ 与 AWR 的温度 $\beta$ 需调；本质上性能被数据集覆盖上界所限（无法超出数据里出现过的动作组合太多）。

### Takeaways

CQL 与 IQL 代表离线 RL 的两种哲学——**显式保守** vs. **隐式回避**。二者都止步于「学好数据内的策略」；[QRL](#qrl) 则从另一角度切入：当任务是**目标可达**时，最优价值本身有特殊结构可利用。

::::paper{tone="qrl"}

## QRL

**Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning**

:::note[一句话]
目标条件 RL 里，「从 $s$ 到目标 $g$ 的最优代价」天然是一个**拟度量**（quasimetric：满足三角不等式但**不对称**）。QRL 直接用拟度量网络建模这个距离，绕开自举式 TD。
:::

**年份 / Venue** ICML 2023 ｜ **机构** MIT / UC San Diego 等 ｜ **方向** 目标条件 / 目标可达 RL ｜ **基准** 离线 + 在线 goal-reaching（GCRL）任务

**材料** [Paper](https://arxiv.org/abs/2304.01203) · [PMLR](https://proceedings.mlr.press/v202/wang23al.html)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{wang2023optimal,
  title     = {Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning},
  author    = {Wang, Tongzhou and Torralba, Antonio and Isola, Phillip and Zhang, Amy},
  booktitle = {International Conference on Machine Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.01203}
}</code></pre>
</details>

### Motivation

在目标可达（goal-reaching）问题里，最优价值 $V^*(s, g)$ 等价于「从 $s$ 到 $g$ 的最短到达步数/代价」的负值。这个量满足**三角不等式**（经中间点绕行不会更近），但**方向不对称**（$s\to g$ 与 $g\to s$ 代价可不同）——这正是**拟度量**的定义。QRL 主张：与其用普适逼近器盲学 $V$，不如把这个已知的几何结构**直接编码进网络**。

### Method

- **拟度量参数化**：用满足拟度量公理（非负、$d(x,x)=0$、三角不等式、允许不对称）的可微模型（如 Interval Quasimetric Embeddings, IQE）表示状态间距离 $d(s, g)$。
- **训练目标（约束优化）**：在「相邻可达状态对的距离 $\le$ 单步代价」这一**局部约束**下，**最大化**其余状态对之间的拟度量距离。三角不等式由结构保证，于是这个约束优化的解恰好逼近**最优代价函数**——无需 TD 自举。
- **策略**：由学到的距离场诱导目标可达策略（选使 $d(\cdot, g)$ 下降的动作）。

### Experiments

- **基准**：离线与在线的目标可达（GCRL）控制任务。
- **结果**：在这些任务上，利用拟度量结构相较不带该结构的价值逼近基线更能学到接近最优的到达策略。**具体每任务成功率/回报此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：把「最优目标可达价值是拟度量」这一**结构先验**直接注入模型，避免自举式 TD 的不稳定；三角不等式自动满足带来更好的组合/长程泛化。

**局限**：**只适用于目标可达式**任务（奖励=到达），不覆盖一般稠密奖励 MDP；依赖专门的拟度量嵌入结构；相较通用 actor-critic 生态更小众。

### Takeaways

QRL 与 CQL/IQL 都属「跳出朴素 TD」，但角度不同：离线 RL 修的是**分布偏移**，QRL 换的是**价值的结构假设**。至此价值/策略学习的经典脉络告一段落；最后两篇把 RL 的目标搬到**语言模型后训练**。

::::paper{tone="dpo"}

## DPO

**Direct Preference Optimization: Your Language Model is Secretly a Reward Model**

:::note[一句话]
RLHF 要先训奖励模型再用 PPO，链路长且不稳。DPO 证明：在 KL 约束的 RLHF 目标下，**语言模型自身就是隐式奖励模型**，于是偏好学习可化为一个**简单的分类损失**，无需奖励模型、无需 RL 采样循环。
:::

**年份 / Venue** NeurIPS 2023 ｜ **机构** Stanford ｜ **方向** LLM 对齐 / RL-free 偏好优化 ｜ **基准** 情感生成、摘要、单轮对话等

**材料** [Paper](https://arxiv.org/abs/2305.18290) · [Code](https://github.com/eric-mitchell/direct-preference-optimization)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{rafailov2023direct,
  title     = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author    = {Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano
               and Manning, Christopher D. and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.18290}
}</code></pre>
</details>

### Motivation

标准 RLHF = 训练奖励模型 + 用 PPO 优化「奖励 − KL 到参考策略」。这条管线有三处痛点：奖励模型可能被 reward hacking、PPO 训练不稳且工程复杂、要同时持有策略/参考/奖励/价值四个模型。DPO 问：能否**跳过奖励模型与 RL**，直接在偏好数据上优化策略？

### Method

- **关键恒等式**：带 KL 约束的 RLHF 最优策略有闭式 $\pi^*(y\mid x) \propto \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$。反解得隐式奖励 $r(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$。
- **配分函数抵消**：代入 Bradley–Terry 偏好模型时，$Z(x)$ 在「优/劣」两项之差中**自动消去**，剩下一个对偏好对 $(y_w, y_l)$ 的 sigmoid 分类损失：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log\sigma\big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big)\Big]
$$

- $\beta$ 控制与参考策略的偏离强度（等价于 KL 约束强度）。全程无显式奖励模型、无在线采样。

### Experiments

- **基准**：受控情感生成、摘要（TL;DR）、单轮对话等偏好对齐任务。
- **结果**：在这些任务上匹配或超过基于 PPO 的 RLHF，同时**训练更稳、实现更简单、算力更省**。**具体胜率/指标此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：把 RLHF 从「四模型 + RL 循环」压成「一个分类损失」，稳定、易复现，是开源对齐的主流方案之一。

**局限**：性能受**偏好数据质量与覆盖**限制；纯离线、不采样，可能不如在线 RLHF 探索到新行为；对 $\beta$ 与参考策略选择敏感；后续有 IPO/KTO/ORPO 等针对其过拟合与长度偏置的改进。

### Takeaways

DPO 代表 LLM 对齐里「**免 RL**」的一派：把奖励重参数化掉。而当奖励**可自动验证**（如数学答案对错）时，另一派选择保留 RL 但简化它——这就是 [GRPO](#grpo)。

::::paper{tone="grpo"}

## GRPO

**DeepSeekMath / DeepSeek-R1（Group Relative Policy Optimization）**

:::note[一句话]
GRPO 是 PPO 的变体：对同一提示**采样一组 $G$ 个回答**，用**组内相对回报（减均值、除标准差）**当优势，**去掉独立的 value/critic 网络**——为大规模、可验证奖励的 LLM 推理后训练大幅省显存。
:::

**年份 / Venue** 首次提出于 DeepSeekMath（arXiv 2024），经 DeepSeek-R1（arXiv 2025）规模化推广 ｜ **机构** DeepSeek-AI ｜ **方向** LLM 推理后训练 / RLVR ｜ **基准** 数学推理（GSM8K、MATH 等）

**材料** [DeepSeekMath (原始)](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1 (推广)](https://arxiv.org/abs/2501.12948)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{shao2024deepseekmath,
  title   = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author  = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao
             and Bi, Xiao and others and Guo, Daya},
  journal = {arXiv preprint arXiv:2402.03300},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.03300},
  note    = {Introduces Group Relative Policy Optimization (GRPO)}
}

@misc{deepseek2025r1,
  title         = {{DeepSeek-R1}: Incentivizing Reasoning Capability in {LLMs} via Reinforcement Learning},
  author        = {{DeepSeek-AI}},
  year          = {2025},
  eprint        = {2501.12948},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2501.12948},
  note          = {Scales GRPO for large-model reasoning post-training}
}</code></pre>
</details>

### Motivation

PPO 在 LLM 上要额外训一个与策略同规模的 **value 网络**估基线——显存翻倍、且对逐 token 价值的估计在长序列上噪声大。但 LLM 推理任务的奖励往往是**序列级、可验证**的（答案对/错）。既然如此，何不用「**同一题多采几个答案、相互比较**」来得到基线，省掉 critic？

### Method

- **组采样**：对每个提示，用旧策略采一组 $G$ 个输出 $\{o_1, \dots, o_G\}$。
- **组内相对优势**（关键，替代 critic 基线）：

$$
A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

（outcome 监督下，该优势赋给输出 $i$ 的所有 token。）

- **目标**：PPO 式裁剪替代目标 + **直接加在损失里的 KL 惩罚**（而非并进奖励）：

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Big[\tfrac{1}{G}\textstyle\sum_i \min(\rho_i A_i,\ \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\Big], \quad \rho_i = \tfrac{\pi_\theta(o_i)}{\pi_{\theta_{\text{old}}}(o_i)}
$$

### Experiments

- **基准**：数学推理（GSM8K、MATH 等）；DeepSeek-R1 进一步在大模型上用 GRPO 激发长链推理。
- **结果**：DeepSeekMath 报告 GRPO 相对基线在数学推理上有效提升；DeepSeek-R1 展示纯 RL（GRPO）即可**涌现**较强推理与自我反思行为。**具体分数此处定性描述，不引用精确数值。**

### Strengths and Limitations

**Strengths**：**去掉 value 网络**显著省显存/算力；实现比完整 PPO 简单；与可验证奖励（RLVR）天然契合，是当前开源推理模型后训练的主流之一。

**局限**：依赖**每题多次采样**（推理开销大）；组内标准化在奖励方差极小/极大时不稳；主要适配**可验证/稀疏结果奖励**，对开放式生成的适配仍在探索；理论性质弱于经典策略优化。

### Takeaways

GRPO 与 [DPO](#dpo) 是 LLM 后训练的两种当红路线：DPO 免 RL、吃**偏好对**；GRPO 保留 RL、吃**可验证奖励**并用组内比较省掉 critic。二者都可视作把经典 RL（[PPO](#ppo) / RLHF）为语言生成场景做的**目标函数简化**。

## Cross-Paper Comparison

| 算法 | 类别 | on/off-policy | 动作空间 | 是否需 critic/价值网络 | 主要对治的问题 |
|:--|:--|:--|:--|:--|:--|
| DQN | 深度价值 | off | 离散 | 是（Q） | 深度下的 deadly triad（回放+目标网络稳定化） |
| DDPG | 确定性 AC | off | 连续 | 是（Q） | 连续动作无法枚举 max |
| PPO | 策略优化 | on | 连续/离散 | 是（V） | 策略更新步长过大 → 塌陷 |
| SAC | 最大熵 AC | off | 连续 | 是（双 Q） | 过估计 + 探索 + 稳定性 |
| CQL | 离线 RL | offline | 连续/离散 | 是（Q） | OOD 动作 Q 高估（显式压制→下界） |
| IQL | 离线 RL | offline | 连续/离散 | 是（Q,V） | OOD 查询本身（expectile 隐式回避） |
| QRL | 目标条件 | 皆可 | 连续/离散 | 否（拟度量距离场） | 最优目标可达价值的结构建模 |
| DPO | LLM 对齐 | offline | 离散 token | 否 | RLHF 链路长/不稳（重参数化掉奖励） |
| GRPO | LLM 后训练 | on（近） | 离散 token | 否（组内基线） | PPO 的 value 网络开销 |

## Discussion

1. **一条不断简化的主线。** DQN → DDPG → SAC 是「往框架里加技巧」（回放、目标网络、双 Q、熵）；而 IQL、QRL、DPO、GRPO 却在**做减法**——IQL 减掉 OOD 查询，QRL 减掉 TD 自举，DPO 减掉奖励模型与 RL 循环，GRPO 减掉 value 网络。成熟往往体现为**知道哪个组件可以拿掉**。

2. **过估计是一根暗线。** DQN 的 $\max$、DDPG 的确定性放大、离线 RL 的 OOD 自举，本质都是「对没充分验证的动作过度乐观」。SAC/TD3 用双 Q min、CQL 用保守下界、IQL 用不查 OOD——都是同一病的不同药方。

3. **on-policy vs off-policy vs offline 的取舍。** off-policy/offline 样本效率高但易受分布偏移之害；on-policy（PPO/GRPO）稳但每次更新后数据作废、样本贵。选择取决于「交互便宜还是数据便宜」。

4. **经典 RL 与 LLM 后训练同源。** DPO 是带 KL 约束 RLHF 的闭式解，GRPO 是 PPO 去 critic 的变体——LLM 后训练不是另起炉灶，而是把 [策略梯度](/blog/posts/paper-notes-reinforcement-learning-1/#policy-gradient) 这套东西**为「奖励可验证/只有偏好」的语言场景重新裁剪**。理解经典 RL 仍是理解它们的前提。
