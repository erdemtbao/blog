---
title: "Paper Notes: Reinforcement Learning (1)"
published: 2026-03-01
description: Reading notes—MDP/POMDP, MC & TD, Q-learning, policy gradient, DDPG/DQN/PPO/SAC/TD3, offline RL (CQL/IQL/BC), quantile Q, DPO/GRPO/RLT.
image: ''
tags: [Paper Notes, Reinforcement Learning]
category: Paper Notes
draft: false
---

强化学习阅读笔记，对应本地资料目录 `assets/paper_node/Reinforcement_Learning_1/`（线上请同步到 `public/paper-node/Reinforcement_Learning_1/`）。**排序**：从形式化基础 → 表格型方法 → 深度策略/价值 → 离线 RL → 分位/分布式价值 → **偏好与 LLM 后训练（DPO / GRPO 等）**。版式与 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/) 一致：**圆角卡片**内为简称、题名、机构、会议/年份、一句话、材料链接；**卡片外**为 BibTeX 折叠与个人笔记。`::::paper{tone="…"} … ::::` 须四个冒号以嵌套 `:::note`。

:::tip[PDF / 图片放哪？]
`/blog/paper-node/Reinforcement_Learning_1/<简称>/`；文中链接示例：`/blog/paper-node/Reinforcement_Learning_1/MDP/xxx.pdf`。
:::

## 目录

| 简称 | 说明 | 跳转 |
|:---:|:---|:---:|
| MDP | 马尔可夫决策过程 | [↓](#mdp) |
| POMDP | 部分可观测 MDP | [↓](#pomdp) |
| MC | 蒙特卡洛方法 | [↓](#mc) |
| TD | 时序差分 | [↓](#td) |
| Q-learning | 离策略表格型 Q | [↓](#q-learning) |
| Policy gradient | 策略梯度 / REINFORCE | [↓](#policy-gradient) |
| DDPG | 深度确定性策略梯度 | [↓](#ddpg) |
| DQN | 深度 Q 网络 | [↓](#dqn) |
| PPO | 近端策略优化 | [↓](#ppo) |
| SAC | 软演员–评论家 | [↓](#sac) |
| TD3 | 双延迟 DDPG | [↓](#td3) |
| CQL | 保守 Q 学习（离线） | [↓](#cql) |
| IQL | 隐式 Q 学习（离线） | [↓](#iql) |
| BC | 行为克隆 | [↓](#bc) |
| QRL | 分位 / 分布式 Q（QR-DQN 代表） | [↓](#qrl) |
| DPO | 直接偏好优化 | [↓](#dpo) |
| GRPO | 组相对策略优化 | [↓](#grpo) |
| RLT | （待绑定具体文献） | [↓](#rlt) |

::::paper{tone="mdp"}

## MDP

**论文 / 专著：** *Markov Decision Processes: Discrete Stochastic Dynamic Programming*

**主要机构：** Martin L. Puterman（Wiley 专著）

**会议 / 年份：** Wiley，1994（经典教材级参考）

:::note[一句话]
用 **状态–动作–转移–回报** 形式化序贯决策；**马尔可夫性**下最优策略可通过值迭代 / 策略迭代等求解，是后续 POMDP、近似动态规划与深度 RL 的共同语言。
:::

**材料：** [Google Books / ISBN 检索](https://www.wiley.com/en-us/Markov+Decision+Processes%3A+Discrete+Stochastic+Dynamic+Programming-p-9780471619772) · [Sutton & Barto 在线书（对照）](http://incompleteideas.net/book/the-book-2nd.html) · Code（按课程实现为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@book{puterman1994markov,
  title     = {Markov Decision Processes: Discrete Stochastic Dynamic Programming},
  author    = {Puterman, Martin L.},
  year      = {1994},
  publisher = {John Wiley \& Sons}
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

::::paper{tone="pomdp"}

## POMDP

**论文：** *Planning and Acting in Partially Observable Stochastic Domains*

**主要机构：** Brown University 等（Kaelbling, Littman, Cassandra）

**会议 / 年份：** Artificial Intelligence，1998

:::note[一句话]
在 **观测仅为状态的随机函数** 时，最优决策需依赖 **信念状态**（对隐状态的分布）；精确求解难，机器人感知噪声、部分可见任务常建模为 POMDP 或其近似。
:::

**材料：** [Paper](https://www.sciencedirect.com/science/article/pii/S000437029800023X) · Project（经典 POMDP 求解器众多，按引用链选） · Code（如 pomdp-solve 等）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{kaelbling1998planning,
  title   = {Planning and Acting in Partially Observable Stochastic Domains},
  author  = {Kaelbling, Leslie Pack and Littman, Michael L. and Cassandra, Anthony R.},
  journal = {Artificial Intelligence},
  volume  = {101},
  number  = {1--2},
  pages   = {99--134},
  year    = {1998}
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

::::paper{tone="mc"}

## MC

**论文 / 教材：** *Reinforcement Learning: An Introduction*（第 5 章：蒙特卡洛方法）

**主要机构：** Richard S. Sutton & Andrew G. Barto（MIT Press）

**会议 / 年份：** 第 2 版，2018（免费在线版持续更新）

:::note[一句话]
用 **完整回合的回报样本**估计值函数或策略梯度；**无模型**、高方差，是理解 **首次访问 / 每次访问 MC** 与 **探索起点** 的基础，再过渡到 TD。
:::

**材料：** [Book (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) · Code（教学实现见书友仓库） · Project—

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@book{sutton2018reinforcement,
  title     = {Reinforcement Learning: An Introduction},
  author    = {Sutton, Richard S. and Barto, Andrew G.},
  edition   = {Second},
  publisher = {MIT Press},
  year      = {2018},
  url       = {http://incompleteideas.net/book/the-book-2nd.html}
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

::::paper{tone="td"}

## TD

**论文：** *Learning to Predict by the Methods of Temporal Differences*

**主要机构：** Richard S. Sutton

**会议 / 年份：** Machine Learning，1988

:::note[一句话]
**自举（bootstrapping）**：用当前价值估计更新估计，**无需等到回合结束**；连接 MC 与动态规划，是 **TD(0)、TD(λ)、n-step TD** 的思想源头。
:::

**材料：** [Paper (Springer)](https://link.springer.com/article/10.1007/BF00115009) · [Book 第 6–12 章](http://incompleteideas.net/book/the-book-2nd.html) · Code—

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{sutton1988learning,
  title   = {Learning to Predict by the Methods of Temporal Differences},
  author  = {Sutton, Richard S.},
  journal = {Machine Learning},
  volume  = {3},
  pages   = {9--44},
  year    = {1988}
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

::::paper{tone="qlearning"}

## Q-learning

**论文：** *Q-learning*

**主要机构：** Cambridge（Watkins）；Gatsby（Dayan）

**会议 / 年份：** Machine Learning，1992

:::note[一句话]
**离策略**表格算法：用 **贝尔曼最优算子** 的采样近似更新 Q；收敛条件与探索策略有关，是 **DQN** 的直接祖先。
:::

**材料：** [Paper (Springer)](https://link.springer.com/article/10.1007/BF00992698) · [Watkins 1989 thesis (Cambridge)](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) · Code—

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{watkins1992q,
  title   = {Q-learning},
  author  = {Watkins, Christopher J. C. H. and Dayan, Peter},
  journal = {Machine Learning},
  volume  = {8},
  pages   = {279--292},
  year    = {1992}
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

::::paper{tone="policygrad"}

## Policy gradient

**论文：** *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*（REINFORCE）

**主要机构：** Ronald J. Williams（Northeastern / 连接主义时代典型署名）

**会议 / 年份：** Machine Learning，1992

:::note[一句话]
对 **参数化策略** 直接求 **期望回报梯度** 的蒙特卡洛估计；**高方差**，常配合 **基线（baseline）** 与 **演员–评论家** 减方差，是 **PPO、SAC** 等现代算法的根。
:::

**材料：** [Paper (Springer)](https://link.springer.com/article/10.1007/BF00992696) · [Sutton et al. PG 定理（ICML 2000）](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) · Code—

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{williams1992simple,
  title   = {Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
  author  = {Williams, Ronald J.},
  journal = {Machine Learning},
  volume  = {8},
  pages   = {229--256},
  year    = {1992}
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

::::paper{tone="ddpg"}

## DDPG

**论文：** *Continuous Control with Deep Reinforcement Learning*

**主要机构：** Google DeepMind 等（Lillicrap, Hunt, Silver…）

**会议 / 年份：** ICLR，2016（arXiv 2015）

:::note[一句话]
**确定性策略** + **双网络（actor / critic）** + **目标网络** + **经验回放**，把 Q-learning 思想搬到**连续动作**；对 **TD3、SAC** 等连续控制基线影响极大。
:::

**材料：** [Paper](https://arxiv.org/abs/1509.02971) · Project— · [Code (OpenAI baselines 等)](https://github.com/openai/baselines)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{lillicrap2016continuous,
  title     = {Continuous Control with Deep Reinforcement Learning},
  author    = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  booktitle = {International Conference on Learning Representations},
  year      = {2016},
  url       = {https://arxiv.org/abs/1509.02971}
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

::::paper{tone="dqn"}

## DQN

**论文：** *Human-level Control Through Deep Reinforcement Learning*

**主要机构：** DeepMind（Mnih 等）

**会议 / 年份：** Nature，2015

:::note[一句话]
**神经网络逼近 Q** + **经验回放** + **目标网络** 稳定离策略学习，Atari 上达到人类水平；开启 **深度价值方法** 主流时代，与 **Double DQN、Rainbow** 等改进一脉相承。
:::

**材料：** [Paper](https://www.nature.com/articles/nature14236) · [arXiv 扩展版](https://arxiv.org/abs/1312.5602) · [Code (官方/Dopamine 等)](https://github.com/google/dopamine)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{mnih2015human,
  title   = {Human-level Control Through Deep Reinforcement Learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and others},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015}
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

::::paper{tone="ppo"}

## PPO

**论文：** *Proximal Policy Optimization Algorithms*

**主要机构：** OpenAI（Schulman 等）

**会议 / 年份：** arXiv 2017（工程上广泛作为 on-policy 默认）

:::note[一句话]
**裁剪替代目标**限制策略更新幅度，实现稳定、易调的 on-policy 训练；机器人仿真与游戏基准常用 **PPO** 作强基线。
:::

**材料：** [Paper](https://arxiv.org/abs/1707.06347) · [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) · [Code (Stable-Baselines3 等)](https://github.com/DLR-RM/stable-baselines3)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{schulman2017proximal,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Praphulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017},
  url     = {https://arxiv.org/abs/1707.06347}
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

::::paper{tone="sac"}

## SAC

**论文：** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*

**主要机构：** UC Berkeley / BAIR（Haarnoja, Zhou, Abbeel, Levine）

**会议 / 年份：** ICML，2018

:::note[一句话]
在目标中加入 **熵正则**，随机策略 + 双 Q 网络减轻过估计；**样本效率高**，连续控制与机器人学习常用 **SAC** 作 off-policy 默认之一。
:::

**材料：** [Paper](https://arxiv.org/abs/1801.01290) · [Project](https://sites.google.com/view/sac-initial-results/) · [Code](https://github.com/haarnoja/sac)

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

::::paper{tone="td3"}

## TD3

**论文：** *Addressing Function Approximation Error in Actor-Critic Methods*

**主要机构：** Fujimoto, van Hoof, Meger（McGill / Alberta）

**会议 / 年份：** ICML，2018

:::note[一句话]
**双 critic**、**延迟策略更新**、**目标策略平滑** 抑制 Q 过估计；在 **DDPG** 框架上小改即显著稳态，常与 **SAC** 对照读连续控制。
:::

**材料：** [Paper](https://arxiv.org/abs/1802.09477) · [Project](https://spinningup.openai.com/en/latest/algorithms/td3.html) · [Code (作者 PyTorch)](https://github.com/sfujim/TD3)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{fujimoto2018addressing,
  title     = {Addressing Function Approximation Error in Actor-Critic Methods},
  author    = {Fujimoto, Scott and van Hoof, Herke and Meger, David},
  booktitle = {International Conference on Machine Learning},
  pages     = {1587--1596},
  year      = {2018},
  url       = {https://arxiv.org/abs/1802.09477}
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

::::paper{tone="cql"}

## CQL

**论文：** *Conservative Q-Learning for Offline Reinforcement Learning*

**主要机构：** UC Berkeley / Google（Kumar, Zhou, Tucker, Levine）

**会议 / 年份：** NeurIPS，2020

:::note[一句话]
在贝尔曼备份外加 **保守项**，压低数据集未覆盖动作的 Q，缓解 **离线 RL 的过估计与外推误差**；与 **BC、IQL** 同为机器人离线数据集策略学习的常用对照。
:::

**材料：** [Paper](https://arxiv.org/abs/2006.04779) · [Project](https://sites.google.com/view/cql-offline-rl) · [Code](https://github.com/aviralkumar2907/CQL)

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

::::paper{tone="iql"}

## IQL

**论文：** *Offline Reinforcement Learning with Implicit Q-Learning*

**主要机构：** UC Berkeley / Google Brain（Kostrikov, Nair, Levine）

**会议 / 年份：** ICLR，2022

:::note[一句话]
用 **期望分位回归（expectile）** 隐式逼近最优价值，**避免显式查询数据集未出现动作的 Q**，训练稳、实现相对简单；离线机器人模仿改进常用 **IQL**。
:::

**材料：** [Paper](https://arxiv.org/abs/2110.06169) · Project— · [Code](https://github.com/ikostrikov/implicit_q_learning)

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

::::paper{tone="bc"}

## BC

**论文：** *ALVINN: An Autonomous Land Vehicle in a Neural Network*

**主要机构：** CMU（Pomerleau）

**会议 / 年份：** NeurIPS（前身 NIPS），1989

:::note[一句话]
**监督学习拟合专家状态–动作对**，无显式环境模型；**复合误差**与分布偏移是经典痛点（见 DAgger 等），仍是机器人模仿与 **离线 RL 行为先验** 的基线。
:::

**材料：** [Paper (NIPS 1989)](https://papers.nips.cc/paper/1989/hash/89f0fd5c927d466b6a9fc097cad76a0d-Abstract.html) · [Florence et al. BC 综述 (2021)](https://arxiv.org/abs/2105.06981) · Code—

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{pomerleau1989alvinn,
  title     = {{ALVINN}: An Autonomous Land Vehicle in a Neural Network},
  author    = {Pomerleau, Dean A.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {1989}
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

::::paper{tone="qrl"}

## QRL

**论文：** *Distributional Reinforcement Learning with Quantile Regression*（**QR-DQN**；口语中常与 IQN、分位价值统称「分位 / 分布式 Q」）

**主要机构：** DeepMind（Dabney, Rowland, Bellemare, Munos）

**会议 / 年份：** AAAI，2018

:::note[一句话]
用 **分位数回归** 逼近回报分布的 **Wasserstein 距离**，不只估计均值 Q；**QRL** 在部分教材/讨论中指该 **Quantile / 分布式** 一脉（若你指其它缩写请替换本卡片题名与 bib）。
:::

**材料：** [Paper (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/11791) · [IQN 后续 (ICML 2018)](https://arxiv.org/abs/1806.06923) · Code（Dopamine / CleanRL 等）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{dabney2018distributional,
  title     = {Distributional Reinforcement Learning with Quantile Regression},
  author    = {Dabney, Will and Rowland, Mark and Bellemare, Marc G. and Munos, R{\'e}mi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  year      = {2018}
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

::::paper{tone="dpo"}

## DPO

**论文：** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*

**主要机构：** Stanford 等（Rafailov, Sharma, Mitchell, Ermon, Manning, Finn）

**会议 / 年份：** NeurIPS，2023

:::note[一句话]
**跳过显式奖励模型**，直接在偏好对上优化策略，使目标等价于带 KL 约束的 RLHF 类问题；大模型对齐与 **RL-free 偏好学习** 的代表作之一。
:::

**材料：** [Paper](https://arxiv.org/abs/2305.18290) · [Project](https://github.com/eric-mitchell/direct-preference-optimization) · [Code](https://github.com/eric-mitchell/direct-preference-optimization)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{rafailov2023direct,
  title     = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author    = {Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D. and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.18290}
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

::::paper{tone="grpo"}

## GRPO

**论文：** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（技术报告；**GRPO** 作为组内相对优势估计广泛用于后续开源复现）

**主要机构：** DeepSeek-AI

**会议 / 年份：** arXiv，2025

:::note[一句话]
对同一提示 **采样一组完整回答**，用 **组内相对回报（减均值/标准化）** 构造优势，**弱化独立 critic**；大模型 **推理链后训练** 与开源 **RLVR** 管线常讨论 **GRPO**。
:::

**材料：** [Paper](https://arxiv.org/abs/2501.12948) · [OpenReview / 社区解读](https://openreview.net/) · Code（TRL、verl、OpenRLHF 等实现以仓库为准）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{deepseek2025r1,
  title         = {{DeepSeek-R1}: Incentivizing Reasoning Capability in {LLMs} via Reinforcement Learning},
  author        = {{DeepSeek-AI}},
  year          = {2025},
  eprint        = {2501.12948},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2501.12948},
  note          = {Describes GRPO-style group-relative policy optimization for LLM RL}
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

::::paper{tone="rlt"}

## RLT

**论文：** *（请替换为你所指的具体论文题名；**RLT** 缩写多义）*

**主要机构：** （待补）

**会议 / 年份：** （待补）

:::note[一句话]
占位卡片：**RLT** 可能对应课程/实验室内部的特定方法缩写；请把 **Paper 链接、机构、年份、BibTeX** 换成正式文献，并在 `Reinforcement_Learning_1/RLT/` 存放 PDF。
:::

**材料：** Paper · Project · Code（待补）

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@misc{rltPlaceholder2026,
  title = {RLT: replace with your canonical citation},
  note  = {User-defined abbreviation; update url, author, and year when bound to a specific paper},
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

**和本页其他论文**

- [MDP](#mdp) → [POMDP](#pomdp)：完全可观测 → **信念状态** 上的序贯决策。  
- [MC](#mc) → [TD](#td) → [Q-learning](#q-learning)：回合回报估计 → **自举** → **离策略** 最优 Q。  
- [Policy gradient](#policy-gradient) → [DDPG](#ddpg) / [TD3](#td3) / [SAC](#sac)：**随机/确定性策略** + 深度 critic 的演进；[DQN](#dqn) 与 [PPO](#ppo) 分属 **深度价值** 与 **稳定 on-policy** 主线。  
- [BC](#bc) → [CQL](#cql) / [IQL](#iql)：纯模仿 → **保守或隐式 Q** 缓解离线外推。  
- [QRL](#qrl)：在期望 Q 之外建模 **回报分布 / 分位数**。  
- [DPO](#dpo) / [GRPO](#grpo)：从 **偏好优化** 到 **组相对优势** 的 LLM 后训练；与经典机器人 RL 对照读 **目标函数与数据形态**。

## 新增一篇时怎么写（极简）

1. `Reinforcement_Learning_1` 下建子目录；复制 **卡片 + BibTeX + 笔记**。  
2. 新 `tone` 在 `markdown.css` 的 `.paper-note-card[data-paper-tone="…"]` 追加左边框色。  
3. 目录表加一行；链接 `/blog/paper-node/Reinforcement_Learning_1/...`。

---

_Changelog：2026-03 — 按 Robot Learning 系列模板补全 MDP→GRPO 等条目；RLT 保留待绑定文献。_
