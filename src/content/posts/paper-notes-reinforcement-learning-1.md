---
title: "Paper Notes: Reinforcement Learning (1)"
published: 2026-03-12
description: 强化学习的形式化基础精读——MDP、POMDP、Monte Carlo、Temporal Difference、Q-learning、Policy Gradient。从序贯决策的形式语言到无模型价值/策略学习的三条主线。
image: ''
tags: [Paper Notes, Reinforcement Learning]
category: Paper Notes
draft: false
---

## Overview

本篇是强化学习阅读笔记的第一篇，聚焦**形式化基础**：把「智能体在环境中通过试错最大化长期回报」这件事，用数学语言精确地写下来，并给出最基础的求解思路。第二篇 [Reinforcement Learning (2)](/blog/posts/paper-notes-reinforcement-learning-2/) 才进入 DQN、PPO、SAC、离线 RL 与 LLM 后训练等深度算法——那些算法的每一个符号，都建立在这一篇的六个概念之上。

六个主题分成两组：

- **问题的形式化** — [MDP](#mdp)（完全可观测的序贯决策语言）→ [POMDP](#pomdp)（观测有噪、需在**信念状态**上决策）。它们定义「问题是什么」。
- **求解的思路** — [MC](#mc)（用完整回合的回报估值）、[TD](#td)（用**自举**在线更新）、[Q-learning](#q-learning)（离策略学最优 Q）、[Policy Gradient](#policy-gradient)（直接对策略求回报梯度）。它们定义「怎么学」。

一条主线：**当环境模型（转移与奖励）未知时，如何仅凭采样的经验，估计价值、改进策略、并保证收敛。** MC 用完整回报（无偏、高方差），TD 用自举（有偏、低方差、可在线），二者的张力贯穿整个 RL；Q-learning 与 Policy Gradient 则分别代表**价值派**与**策略派**两大阵营，后续深度算法几乎都是它们的后代。

## 概念清单

| 简称 | 主题 | 归属 | 核心思想 |
|:--|:--|:--|:--|
| [MDP](#mdp) | 马尔可夫决策过程 | 问题形式化 | (S,A,P,R,γ) + 马尔可夫性；Bellman 方程与值/策略迭代 |
| [POMDP](#pomdp) | 部分可观测 MDP | 问题形式化 | 观测有噪，最优决策依赖**信念状态** |
| [MC](#mc) | 蒙特卡洛方法 | 求解思路 | 用完整回合回报估值，无偏高方差 |
| [TD](#td) | 时序差分 | 求解思路 | **自举**：用估计更新估计，可在线、低方差 |
| [Q-learning](#q-learning) | 离策略表格 Q | 求解思路 | 采样贝尔曼最优算子，学最优 Q |
| [Policy Gradient](#policy-gradient) | 策略梯度 / REINFORCE | 求解思路 | 直接对参数化策略求期望回报梯度 |

::::paper{tone="mdp"}

## MDP

**Markov Decision Process（序贯决策的形式语言）**

:::note[一句话]
用 **(S, A, P, R, γ)** 加**马尔可夫性**形式化序贯决策：最优价值满足 **Bellman 最优方程**，可由**值迭代 / 策略迭代**求解。它是后续一切 RL 的共同语言。
:::

**年份 / Venue** Bellman 1957（动态规划）· Howard 1960（策略迭代）· Puterman 1994（标准专著）｜ **方向** 序贯决策形式化 ｜ **性质** 有限折扣 MDP 存在最优确定性平稳策略

**材料** [Bellman, Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) · [Sutton & Barto (2nd ed.) ch.3–4](http://incompleteideas.net/book/the-book-2nd.html)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@book{bellman1957dynamic,
  title     = {Dynamic Programming},
  author    = {Bellman, Richard E.},
  year      = {1957},
  publisher = {Princeton University Press}
}

@book{puterman1994markov,
  title     = {Markov Decision Processes: Discrete Stochastic Dynamic Programming},
  author    = {Puterman, Martin L.},
  year      = {1994},
  publisher = {John Wiley \& Sons}
}</code></pre>
</details>

### 直觉与动机

要谈「最优决策」，先得把「决策问题」写清楚。MDP 给出的抽象是：智能体处于某个**状态**，选一个**动作**，环境按**转移概率**跳到新状态并给一个**奖励**，如此往复。关键的简化假设是**马尔可夫性**——「未来只取决于现在，与如何到达现在无关」。这个假设让「最优」变得可计算：无需记住整段历史，只看当前状态即可。

### 形式化

一个（有限、折扣）MDP 是元组 **$(S, A, P, R, \gamma)$**：状态集 $S$、动作集 $A$、转移核 $P(s'\mid s,a)$、奖励 $R(s,a)$、折扣 $\gamma\in[0,1)$。策略 $\pi(a\mid s)$ 把状态映到动作分布。**马尔可夫性**：

$$
P(S_{t+1}, R_{t+1} \mid S_t, A_t, S_{t-1}, \dots) = P(S_{t+1}, R_{t+1} \mid S_t, A_t)
$$

**Bellman 期望方程**（评估固定策略 $\pi$）与**Bellman 最优方程**（刻画最优 $v_*$）：

$$
v_\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big]
$$

$$
v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_*(s')\big]
$$

### 关键结果与性质

- **收缩性**：Bellman 最优算子是 sup-范数下的 **$\gamma$-收缩**，$v_*$ 是其唯一不动点。
- **值迭代（Value Iteration）**：反复应用最优算子，**以速率 $\gamma$ 几何收敛**到 $v_*$（渐近，非有限步）。
- **策略迭代（Policy Iteration）**（Howard 1960）：交替「精确策略评估」与「贪心改进」，对有限 MDP **有限步内收敛到最优**（单调改进 + 确定性策略只有有限个）。
- **最优策略存在性**：有限折扣 MDP 总存在一个**确定性平稳**最优策略。

### 局限与延伸

MDP 假设**状态完全可观测**且**模型已知**。放松「完全可观测」得到 [POMDP](#pomdp)；放松「模型已知」——即 $P, R$ 未知只能采样——正是 [MC](#mc)/[TD](#td) 等**无模型**方法要解决的问题；状态空间过大则需函数逼近（深度 RL）。

### Takeaways

MDP 是 RL 的「坐标系」。后面每一个算法，本质都是在「模型未知 / 状态部分可见 / 状态空间巨大」这三种放松下，近似地求解 Bellman 方程。

::::paper{tone="pomdp"}

## POMDP

**Planning and Acting in Partially Observable Stochastic Domains**

:::note[一句话]
当**观测只是状态的带噪函数**时，最优决策不能只看当前观测，而要依赖对隐状态的**信念（belief）分布**。POMDP 等价于在信念空间上的一个连续状态 MDP，精确求解极难。
:::

**年份 / Venue** Artificial Intelligence 1998（vol. 101, 99–134）｜ **机构** Brown / Duke 等（Kaelbling, Littman, Cassandra）｜ **方向** 部分可观测序贯决策 ｜ **复杂度** 有限时程 PSPACE-complete，无限时程不可判定

**材料** [Paper (AIJ)](https://www.sciencedirect.com/science/article/pii/S000437029800023X)

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

### 直觉与动机

现实里智能体几乎从不「看见完整状态」：机器人只有带噪传感器、遮挡、有限视场。此时当前观测**不满足马尔可夫性**——两个真实状态可能给出相同观测却需要不同动作。POMDP 的回答是：把「对隐状态的概率分布」当作真正的决策依据。

### 形式化

POMDP 在 MDP 元组上增加观测集 $\Omega$ 与观测模型 $O(o\mid s',a)$，成为 **$(S, A, P, R, \Omega, O, \gamma)$**。智能体看不到 $s$，转而维护**信念状态** $b\in\Delta(S)$——状态上的概率分布，它是最优控制的**充分统计量**。执行动作 $a$、收到观测 $o$ 后按贝叶斯更新：

$$
b'(s') = \frac{O(o\mid s',a)\sum_s P(s'\mid s,a)\,b(s)}{\Pr(o\mid b,a)}
$$

于是 POMDP 等价于在连续信念空间 $\Delta(S)$ 上的一个「**信念 MDP**」。

### 关键结果与性质

- **值函数结构**：有限时程下，信念上的最优值函数是**分段线性且凸（PWLC）**，可由有限组 α-向量表示（Smallwood & Sondik 1973）。
- **复杂度**：有限时程精确求解 **PSPACE-complete**；无限时程判定最优/近优策略**不可判定**——这也是实践中普遍采用近似（point-based value iteration、QMDP、粒子滤波 + 规划、或直接端到端学循环策略）的原因。
- KLC 1998 本身是**信念-MDP 框架与 Witness 算法**的标准出处；上述硬度结论出自 Papadimitriou–Tsitsiklis、Madani 等另一支文献。

### 局限与延伸

精确 POMDP 只在很小的问题上可解。现代做法要么用**近似信念表示**，要么在深度 RL 里用 **RNN / Transformer 隐状态**隐式地承担「信念」的角色（把历史压进隐藏状态）。理解 POMDP 的价值在于：它解释了**为什么部分可观测任务需要记忆**。

### Takeaways

MDP → POMDP 是「完全可观测 → 信念状态上的序贯决策」的一步跨越。凡是感知有噪、有遮挡、需要记忆的任务，其形式底座都是 POMDP 或其近似。

::::paper{tone="mc"}

## MC

**Monte Carlo Methods（Sutton & Barto 第 5 章）**

:::note[一句话]
无模型价值估计的最直接办法：**跑完整回合、平均观测到的回报**。无需知道转移/奖励，估计**无偏**但**方差高**，且只适用于会终止的**回合式**任务。
:::

**年份 / Venue** Sutton & Barto, *Reinforcement Learning: An Introduction*（2nd ed. 2018）ch.5 ｜ **方向** 无模型、基于回报的价值估计 ｜ **性质** first-visit 无偏；需回合终止

**材料** [Book (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)

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

### 直觉与动机

当转移与奖励未知时，最朴素的估值方式是「多试几次，看平均能拿多少回报」。这就是 MC：不建模环境、不做自举，纯靠**采样的完整回报**的样本均值逼近价值。它是理解无模型 RL 的起点，也是 TD 的对照面。

### 形式化

用回合内从 $t$ 起的**完整回报** $G_t = R_{t+1} + \gamma R_{t+2} + \dots$ 的样本均值估计 $v_\pi(s)$ 或 $q_\pi(s,a)$：

- **首次访问 MC（first-visit）**：每回合只取 $s$ **首次**出现之后的回报平均。样本 i.i.d.、**无偏**，按大数定律收敛，方差约 $\propto 1/n$。
- **每次访问 MC（every-visit）**：对 $s$ 的**每次**出现都计入。有限样本**有偏**（同回合内回报相关），但**一致**（访问数 → ∞ 时收敛）。

### 关键结果与性质

- **无偏但高方差**：整条轨迹的回报噪声大。
- **需要回合终止**：必须观测到完整回报才能更新，不适用连续不终止任务。
- **探索**：控制时需保证所有 $(s,a)$ 被采到——用**探索起点（exploring starts）**，或改用 **on-policy 的 ε-soft** 策略维持探索。
- **off-policy MC**：用行为策略 $b$ 评估目标策略 $\pi$ 需**重要性采样**——普通 IS 无偏但方差可能无限；加权 IS 有偏但方差远低，实践更常用。

### 局限与延伸

高方差 + 必须等回合结束，是 MC 的两大痛点。**TD 用自举同时解决这两件事**：不等回合结束、方差更低（代价是引入偏差）。MC 与 TD 的取舍——偏差 vs 方差、在线 vs 回合——是贯穿 RL 的核心张力。

### Takeaways

MC 给出「无模型估值」最干净的定义。记住它的两个标签：**无偏、高方差、需回合**。接下来的一切改进，几乎都在和这三点做交易。

::::paper{tone="td"}

## TD

**Learning to Predict by the Methods of Temporal Differences**

:::note[一句话]
**自举（bootstrapping）**：用「当前估计」去更新「当前估计」，无需等回合结束。TD 结合了 MC 的采样与 DP 的自举，可**在线、增量**学习，方差低于 MC——是现代价值方法的思想源头。
:::

**年份 / Venue** Machine Learning 1988（vol. 3, 9–44）｜ **机构** Sutton（当时 GTE Labs）｜ **方向** 无模型、自举式价值预测 ｜ **性质** TD(λ)：λ=0 即 TD(0)，λ=1 即 MC

**材料** [Paper (Springer)](https://link.springer.com/article/10.1007/BF00115009) · [Book ch.6–12](http://incompleteideas.net/book/the-book-2nd.html)

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

### 直觉与动机

MC 要等回合结束才知道回报，太慢也太吵。但其实我们不必等：走一步、看到即时奖励和下个状态，就已经对「这一步好不好」有了信息——因为下个状态的价值估计 $V(S_{t+1})$ 已经包含了对未来的预测。用这个「一步真实 + 后续估计」去更新当前估计，就是**自举**。

### 形式化

**TD(0) 更新**：

$$
V(S_t) \leftarrow V(S_t) + \alpha\big[\underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t,\ \text{TD 误差}}\big]
$$

TD 误差 $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 是「新信息与旧估计之差」。**TD(λ)** 用几何权重 $(1-\lambda)\lambda^{n-1}$ 融合所有 n-step 回报；后向视角引入**资格迹（eligibility trace）**：

$$
e_t(s) = \gamma\lambda\, e_{t-1}(s) + \mathbb{1}[S_t = s], \qquad V(s) \leftarrow V(s) + \alpha\,\delta_t\, e_t(s)\ \ \forall s
$$

**$\lambda=0$ 退化为 TD(0)，$\lambda=1$ 等价于 MC**——TD(λ) 是连接二者的连续谱。

### 关键结果与性质

- **有偏、低方差、可在线**：与 MC 恰好互补。
- **收敛性**：Sutton (1988) 证明 TD(λ) 对吸收马尔可夫链**均值收敛**；表格 TD(0) 的**几乎必然收敛**由 Dayan (1992)、Jaakkola–Jordan–Singh (1994) 给出；**线性逼近 + on-policy** 收敛见 Tsitsiklis & Van Roy (1997)。
- **致命三要素（deadly triad）**：**函数逼近 + 自举 + 离策略**三者同时出现时可能发散——这是深度 RL 稳定性问题的理论根源（[DQN](/blog/posts/paper-notes-reinforcement-learning-2/#dqn) 的目标网络正是为缓解它）。
- 标准步长条件：Robbins–Monro，$\sum_t\alpha_t=\infty,\ \sum_t\alpha_t^2<\infty$。

### 局限与延伸

自举引入偏差，且在离策略 + 函数逼近下可能不稳。但「可在线、低方差」的优势太大，使 TD 成为主流：[Q-learning](#q-learning)（离策略 TD 控制）与几乎所有深度价值方法都建立在 TD 之上。

### Takeaways

TD 是 RL 里最核心的一个 idea：**用估计更新估计**。它既是 MC 与 DP 的桥梁，也埋下了「致命三要素」这颗后续深度 RL 反复要处理的雷。

::::paper{tone="qlearning"}

## Q-learning

**Q-learning（Watkins 1989 thesis; Watkins & Dayan 1992）**

:::note[一句话]
**离策略（off-policy）**表格控制算法：用 TD 采样近似**贝尔曼最优算子**，直接学最优动作价值 $Q^*$。TD 目标里的 $\max_a Q$ 与实际行为策略无关——这正是它「离策略」的来源，也是 [DQN](/blog/posts/paper-notes-reinforcement-learning-2/#dqn) 的直接祖先。
:::

**年份 / Venue** Machine Learning 1992（vol. 8, 279–292；原始为 Watkins 1989 Cambridge 博士论文）｜ **机构** Cambridge（Watkins）· Gatsby（Dayan）｜ **方向** 无模型、离策略、最优控制 ｜ **性质** 表格下 w.p.1 收敛到 $Q^*$

**材料** [Paper (Springer)](https://link.springer.com/article/10.1007/BF00992698) · [Watkins 1989 thesis](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

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

### 直觉与动机

TD 能评估一个**给定**策略；但我们真正想要的是**最优**策略。Q-learning 的巧妙在于：把 TD 目标里的「按当前策略取下一动作」换成「取**贪心**动作 $\max_a Q$」。这样无论行为策略如何探索，学到的都是**最优** Q——评估的对象（目标策略）与产生数据的对象（行为策略）解耦，即**离策略**。

### 形式化

**更新规则**（离策略 TD 控制）：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\big]
$$

目标中的 $\max_a Q$ 是**贝尔曼最优算子**的采样近似；行为上常用 ε-greedy 保证探索，但它不进入目标——故为离策略。

### 关键结果与性质

**表格情形**下，$Q$ 以概率 1 收敛到 $Q^*$，当且仅当：

1. **每个状态–动作对 $(s,a)$ 被无限次访问**；
2. 步长满足 **Robbins–Monro**：$\sum_t\alpha_t(s,a)=\infty,\ \sum_t\alpha_t^2(s,a)<\infty$；
3. 奖励有界，$\gamma<1$（或问题为恰当的回合式 MDP）。

原始证明见 Watkins & Dayan (1992)；更一般的随机逼近证明见 Jaakkola–Jordan–Singh (1994)、Tsitsiklis (1994)。**注意：保证仅对表格成立，函数逼近下不再有收敛保证。**

### 局限与延伸

表格 Q-learning 无法扩展到高维/连续状态；$\max$ 算子还带来系统性**高估**。用神经网络逼近 $Q$ 即 [DQN](/blog/posts/paper-notes-reinforcement-learning-2/#dqn)，它用经验回放 + 目标网络稳定训练、用 Double DQN 缓解高估；连续动作则无法枚举 $\max$，催生 [DDPG](/blog/posts/paper-notes-reinforcement-learning-2/#ddpg)。

### Takeaways

Q-learning 是**价值派**的代表：先学好 $Q^*$，再贪心导出策略。它的两处软肋——**只能表格 / 只能离散**——直接定义了下一篇 DQN、DDPG、SAC 等一系列工作的问题域。

::::paper{tone="policygrad"}

## Policy Gradient

**Simple Statistical Gradient-Following Algorithms（REINFORCE）+ 策略梯度定理**

:::note[一句话]
不学价值再导出策略，而是**直接对参数化策略 $\pi_\theta$ 求期望回报的梯度**并上升。**策略梯度定理**表明该梯度里**不含状态分布对 $\theta$ 的导数**，因而可由采样估计——这是 PPO、SAC 等现代策略方法的根。
:::

**年份 / Venue** Williams 1992（REINFORCE, Machine Learning 8:229–256）· Sutton, McAllester, Singh, Mansour（策略梯度定理, NIPS 1999）｜ **方向** 无模型、直接策略优化 ｜ **性质** 无偏、高方差，需基线/critic 减方差

**材料** [Williams 1992 (Springer)](https://link.springer.com/article/10.1007/BF00992696) · [PG Theorem (NIPS 1999)](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)

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
}

@inproceedings{sutton1999policy,
  title     = {Policy Gradient Methods for Reinforcement Learning with Function Approximation},
  author    = {Sutton, Richard S. and McAllester, David and Singh, Satinder and Mansour, Yishay},
  booktitle = {Advances in Neural Information Processing Systems 12 (NIPS 1999)},
  year      = {1999}
}</code></pre>
</details>

### 直觉与动机

价值派（Q-learning）先估价值、再贪心导出策略，对连续/高维动作或随机最优策略并不方便。**策略派**换个思路：把策略本身参数化为 $\pi_\theta$，直接调 $\theta$ 让期望回报变大。难点在于——回报既取决于策略选的动作，也取决于策略诱导的**状态分布**，后者对 $\theta$ 的依赖看似难算。策略梯度定理解决了这个难点。

### 形式化

**策略梯度定理**：对目标 $J(\theta)$（如期望回报），

$$
\nabla_\theta J(\theta) = \sum_s d^\pi(s)\sum_a \nabla_\theta \pi(a\mid s;\theta)\, q_\pi(s,a) = \mathbb{E}_\pi\big[\nabla_\theta \log \pi(A_t\mid S_t;\theta)\, q_\pi(S_t, A_t)\big]
$$

其中 $d^\pi$ 是策略诱导的（折扣）状态分布。**关键：梯度中不出现 $\nabla_\theta d^\pi$**——状态分布对 $\theta$ 的依赖恰好消去，于是梯度可纯由采样估计。

**REINFORCE 更新**（蒙特卡洛策略梯度，Williams 1992）：

$$
\theta \leftarrow \theta + \alpha\, G_t\, \nabla_\theta \log \pi(A_t\mid S_t;\theta)
$$

用完整回报 $G_t$，无偏但**高方差**。

### 关键结果与性质

- **基线（baseline）减方差**：减去任意与动作无关的 $b(S_t)$ 不引入偏差（因 $\mathbb{E}[\nabla_\theta\log\pi\cdot b]=0$），常取 $b(s)\approx V(s)$，得优势 $G_t - V(s)$。
- **演员–评论家（actor-critic）**：把 $G_t$ 换成**自举的 critic 估计**（如 TD 误差 $\delta_t$ 作优势），actor 更新策略、critic 学价值——用偏差换更低方差、可在线。
- **兼容函数逼近**：Sutton et al. (1999) 证明用**兼容特征**逼近 $q_\pi$ 可得**精确无偏**的策略梯度。

### 局限与延伸

纯 REINFORCE 方差太大、样本效率低。现代策略方法几乎都在此基础上加约束与减方差：[PPO](/blog/posts/paper-notes-reinforcement-learning-2/#ppo)（裁剪限制更新幅度）、[SAC](/blog/posts/paper-notes-reinforcement-learning-2/#sac)（最大熵 + 双 Q）、[DDPG](/blog/posts/paper-notes-reinforcement-learning-2/#ddpg)（确定性策略梯度）；乃至 LLM 后训练的 RLHF/[GRPO](/blog/posts/paper-notes-reinforcement-learning-2/#grpo) 也都是策略梯度的直系后代。

### Takeaways

Policy Gradient 是**策略派**的根。记住两点：**梯度可采样估计**（定理保证）、**必须减方差**（基线 / critic）。第二篇的 PPO、SAC、GRPO 全是「如何更稳、更省地做策略梯度」的不同答案。

## Cross-Topic Comparison

| 主题 | 需要模型? | 需回合终止? | 自举? | 偏差 / 方差 | on/off-policy | 学什么 |
|:--|:--|:--|:--|:--|:--|:--|
| MDP | 是（P,R 已知） | — | DP 自举 | 精确 | — | 规划求解 $v_*/\pi_*$ |
| POMDP | 是 | — | DP 自举 | 精确但难解 | — | 信念上的 $v_*$ |
| MC | 否 | **是** | 否 | 无偏 / 高 | 皆可 | 价值（回报均值） |
| TD | 否 | 否 | **是** | 有偏 / 低 | 皆可 | 价值 |
| Q-learning | 否 | 否 | 是 | 有偏 / 中 | **off** | 最优 $Q^*$ |
| Policy Gradient | 否 | REINFORCE 需；AC 否 | AC 用 | 无偏(MC)/有偏(AC) | 多为 on | 策略 $\pi_\theta$ |

## Discussion

1. **两条主线：价值 vs 策略。** [Q-learning](#q-learning) 学价值再导出策略；[Policy Gradient](#policy-gradient) 直接优化策略。深度时代前者长出 [DQN](/blog/posts/paper-notes-reinforcement-learning-2/#dqn)、后者长出 [PPO](/blog/posts/paper-notes-reinforcement-learning-2/#ppo)/[SAC](/blog/posts/paper-notes-reinforcement-learning-2/#sac)；actor-critic 则是两者的合流。

2. **一个永恒的取舍：偏差 vs 方差。** [MC](#mc) 无偏高方差、[TD](#td) 有偏低方差，TD(λ) 是二者的连续谱。选 λ、选 n-step、选是否用 critic，本质都是在这条轴上选点。

3. **致命三要素是暗线。** 「函数逼近 + 自举 + 离策略」同时出现会破坏收敛保证。本篇的收敛定理**只在表格 / 线性 on-policy 下成立**；第二篇的目标网络、双 Q、保守约束，很大程度都是在**工程上驯服**这个理论隐患。

4. **形式化决定了「需要记忆吗」。** [MDP](#mdp) 完全可观测 → 无需历史；[POMDP](#pomdp) 部分可观测 → 必须靠信念/记忆。深度 RL 里的 RNN/Transformer 隐状态，正是在隐式地做 POMDP 的信念更新。
