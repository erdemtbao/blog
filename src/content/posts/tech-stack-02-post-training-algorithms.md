---
title: "Technology Stack 02: Post-training Algorithms"
published: 2026-07-26
updated: 2026-09-11
description: 比较监督微调、模仿学习、偏好优化和强化学习等后训练路线，说明 PPO、GRPO、DPO、SAC、IQL、RLPD、DSRL 与 RECAP 的数据和交互要求。
tags: [Technology Stack, Post-training, RLHF, RL, Alignment, VLA, Machine Learning]
category: Technology Stack
draft: false
---

预训练造出一个"底座模型"，但底座只是见过世界，还不知道该怎么与人协作、也不知道怎么在真实环境里做决策。**后训练（post-training）** 就是把这块底座对齐（alignment）、并进一步塑造成一个有用、可控、能干活的模型或策略的过程。

下面把语言模型对齐与机器人策略学习放在同一张地图中比较，但两者的数据、反馈信号和部署约束不同，不能把同名算法的使用方式简单等同。

## 后训练的主要反馈形式

按训练信号和是否需要环境交互，可以区分为三类：

- **监督学习 / 模仿学习**：使用目标回答、专家动作或专家查询作为监督，例如 SFT、行为克隆和 DAgger。
- **基于强化学习的优化**：通过奖励、价值或优势更新策略，例如 PPO、GRPO、SAC、IQL 与 RLPD；其中 offline RL 只使用固定数据集，不进行在线试错。
- **直接偏好优化**：直接从成对偏好数据优化策略，例如 DPO；它与 RLHF 目标有关，但训练阶段不执行 RL rollout，也不训练显式奖励模型。

“先监督、再强化”是常见流程之一，因为监督数据能提供较好的初始策略并降低早期探索成本；它不是所有任务的固定模板。数据充足时可以只做监督或直接偏好优化，已有离线数据时也可以从 offline RL 开始。

---

## 一、监督微调与模仿学习（SFT / Imitation）

### Full-parameter SFT（全参数监督微调）

全参数 SFT 使用“输入—目标输出”对更新模型全部参数。在具身任务中，对专家示范做行为克隆常用于初始化后续 RL，因为它可以减少从随机策略开始的探索；收益大小取决于示范覆盖、任务奖励和策略容量，不能保证提高所有任务的最终上限。

### LoRA SFT（参数高效微调）

全参数微调需要为每个版本保存完整权重。**LoRA（Low-Rank Adaptation）** 冻结原始权重，用低秩矩阵 $A,B$ 表示增量 $\Delta W=BA$，从而减少可训练参数和优化器状态。它在不少任务上接近全参数微调，但效果取决于秩、插入层、数据和分布偏移，不能视为所有资源受限场景的默认最优方案。[LoRA](https://arxiv.org/abs/2106.09685)

### VLM SFT（视觉—语言模型的监督微调）

VLM SFT 在图像、视频与文本序列上优化条件生成目标，常见实现仍采用 next-token likelihood。评测不能只看训练 loss 或单一准确率，还应覆盖开放式生成、视觉定位、幻觉和任务泛化。它可作为 VLA 的语义初始化来源，但机器人动作学习还需要动作监督与闭环评测。

### DAgger（数据集聚合）

模仿学习最大的坑是**分布漂移（distribution shift）**：单纯的行为克隆（behavior cloning）只在专家轨迹上训练，模型一旦稍有偏差，就会走到训练时没见过的状态，然后错误**层层累积（compounding error）**，越走越偏。

> **DAgger 的核心洞察**：不要只在专家去过的地方学，要在**学生自己会走到的状态**上向专家请教。

它的循环是：按混合策略运行学生与专家 → 对运行过程中访问到的状态查询专家动作 → 聚合数据并重新训练学生 → 逐步降低专家策略的混合比例。这样可以降低纯行为克隆的训练—部署分布偏移，但效果依赖专家质量、查询策略与优化误差，并不保证两个分布始终一致。[DAgger](https://arxiv.org/abs/1011.0686)

### HG-DAgger（人在环路的门控 DAgger）

Human-gated DAgger 让人类在判断策略可能失败时介入，并把介入附近的纠正数据加入训练集。这可以减少无效专家标注并降低危险探索的频率，但不等同于形式化安全保证。是否只保留介入帧、是否使用 $\beta$ 衰减以及具体衰减率都属于实现选择，应随所引用论文或代码配置说明，不能写成算法固定步骤。

---

## 二、语言模型的 RL 与直接偏好优化

这一支解决的是"**答案没有唯一标准、但人能判断好坏**"的问题。经典链路是：**SFT → 奖励建模 → 用 RL 优化策略**。奖励建模让人类对同一提示的多个回答排序，训练一个**奖励模型**把"好回答"变成可优化的标量信号；剩下的就是用什么 RL 算法去优化它。

### PPO（近端策略优化）

RLHF 的主力算法。它是 **actor–critic** 结构：actor（策略）生成动作，critic（价值模型）评估好坏。PPO 的关键是**限制每次策略更新的步长**，避免一步走太大导致训练崩溃。它用重要性采样比 $r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ 衡量新旧策略的差异，并对它做**裁剪（clip）**：

$$
J_{\text{PPO}}(\theta)=\mathbb{E}\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_t\big)\Big]
$$

裁剪参数 $\varepsilon$ 控制概率比进入裁剪区间的范围；0.2 是常见但非固定取值。优势 $\hat{A}_t$ 常用 GAE 估计。在 LLM 场景中，critic 可以独立部署、共享部分骨干或使用更小模型，取决于实现与资源约束。[PPO](https://arxiv.org/abs/1707.06347)

> PPO-RLHF 需要价值估计、rollout 与策略更新等组件，工程和显存开销通常较高；critic 是否与 actor 同量级取决于具体系统。

### GRPO（组相对策略优化）

GRPO 的动机很直接：**能不能不要 critic？** 它对同一个 prompt **采样一组（group）回答**，用组内的相对表现来当优势，而不再训练价值网络。某个回答的优势就是它的奖励相对组内均值、标准差的归一化：

$$
\hat{A}_{i,t}=\frac{R_i-\mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})}
$$

其余部分沿用 PPO 式的裁剪目标。DeepSeekMath 的原始 GRPO 目标包含相对参考策略的 KL 项；后续实现可能改变或移除该项，应注明具体版本。去掉价值模型可以降低一部分训练开销，但 rollout 生成仍可能是主要成本。[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

### DAPO

DAPO 是在 GRPO 基础上、面向**大规模长链推理（long-CoT）** 的一套改进，主要修四个问题：

- **Clip-Higher**：把裁剪的上下界做成**不对称**（$\varepsilon_{high}>\varepsilon_{low}$），给"变好"的方向更大空间，鼓励探索、防止**熵坍塌**（策略过早变得死板）。
- **动态采样（Dynamic Sampling）**：丢弃"全对"或"全错"的组（它们优势全为 0、没有梯度信号），保证每组里**既有对的也有错的**，不浪费算力。
- **Token 级损失**：把策略梯度损失放到 **token 级**而非序列级，避免长回答被稀释、短回答被放大的**长度偏置**——这对长推理尤其关键。
- **超长惩罚（Overlong Reward Shaping）**：对超过安全长度的回答做线性惩罚，抑制那种又臭又长、没营养的输出。

### Async PPO（异步 PPO）

标准 PPO 是"**采样 → 停下 → 训练 → 同步 → 再采样**"的串行节奏，一旦环境步进慢、推理贵、或者 rollout 很长，各环节就会互相等待、GPU 闲置。Async PPO 把**环境交互、rollout 推理、actor 训练**拆成三条**长期运行、彼此重叠**的流水线，去掉全局同步屏障，显著提升吞吐。

代价是采样可能不再来自最新策略，从而引入数据陈旧与 off-policy 偏差。以 **RLinf 的一种实现**为例，系统记录生成样本的策略版本，并用 `staleness_threshold` 与 `on_policy_min_ratio` 管理样本新鲜度。这些字段是框架级配置，不是所有 Async PPO 算法都共享的定义。

### DPO（直接偏好优化）

DPO 在固定参考策略、KL 正则化奖励目标和成对偏好模型等假设下，对奖励函数与最优策略进行重参数化，从而把 chosen/rejected 偏好数据转成分类式监督损失：

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\big)\Big]
$$

训练阶段不需要显式奖励模型、critic 或在线 rollout，因此实现链路较短。它只能利用给定偏好数据的覆盖范围；与 PPO-RLHF 的优劣会随数据、奖励模型和评测任务变化，不能概括成统一的“效果上限”。[DPO](https://arxiv.org/abs/2305.18290)

---

## 三、面向连续控制与具身智能的 RL

机器人、连续控制这一支，动作是连续向量、交互样本很贵，因此算法更看重**样本效率**和**能否复用离线数据**。

### SAC（软演员—评论家）

SAC 属于**最大熵 RL**：在最大化期望回报的同时鼓励策略保持较高熵，即动作分布的随机性。常用版本使用可学习温度系数 $\alpha$、双 Q 网络和经验回放；这些组件用于平衡奖励与熵、缓解价值过估计并复用历史交互。策略熵不是模型对环境的认知不确定性。[SAC](https://arxiv.org/abs/1801.01290)

### CrossQ

CrossQ 移除目标网络，并利用 Batch Normalization 以及当前/下一状态的联合前向计算来改善训练稳定性。论文在所测试连续控制基准上，以较低的 update-to-data ratio 获得了有竞争力的样本效率；更宽 critic 是实验配置之一，不是算法定义。[CrossQ](https://arxiv.org/abs/1902.05605)

### RLPD（带先验数据的强化学习）

RLPD 研究如何用离线先验数据加速在线 RL。论文配方基于 SAC，分别从离线数据集和在线 replay 中采样，并使用 LayerNorm、critic ensemble 与较高更新频率。50/50 采样比例及 ensemble 大小是论文实验中的配置，应随实现调整，而不是算法在所有任务中的固定常数。[RLPD](https://arxiv.org/abs/2302.02948)

### IQL（隐式 Q 学习）

IQL 在训练价值函数时不查询数据集之外的动作：它用 **expectile regression** 学习偏向高价值动作的 $V$，再通过优势加权行为克隆提取策略。常见权重形式为 $\exp(\beta A)$，因此负优势动作通常仍有非零权重，只是权重更低；它不是简单地只保留正优势样本。[IQL](https://arxiv.org/abs/2110.06169)

### 让 RL 驱动"生成式策略"

许多连续控制算法使用可直接采样和计算密度的高斯策略。扩散与流匹配策略具有多步生成过程，策略梯度、熵计算和在线更新更昂贵，但并非与 RL 原理上“不兼容”。下面的方法分别选择端到端训练、噪声空间适配或 advantage conditioning 来降低这些困难：

- **SAC-Flow**：论文指出多步 flow rollout 在代数上等价于残差递归计算，因此可能出现类似 RNN 的梯度消失或爆炸。作者提出门控速度的 Flow-G、使用 decoder 的 Flow-T，以及 noise-augmented rollout，使 SAC 能端到端训练 flow policy。具体真机学习时间应连同硬件、示范数据和任务设置阅读。[论文](https://arxiv.org/abs/2509.25756)
- **DSRL（在扩散噪声空间里做 RL）**：冻结预训练扩散策略，在其潜在噪声空间训练小型 SAC agent，用学习到的噪声分布引导冻结策略适配新奖励。参数规模和适配效率来自论文特定实现，不能视为所有扩散策略的固定成本。[论文](https://arxiv.org/abs/2506.15799)
- **RECAP**：Physical Intelligence 的原始流程先进行 advantage-conditioned offline RL 预训练，再用任务示范微调，并继续从真机自主 episode、专家纠正和奖励反馈中学习。因此 RECAP 整体不是“无需在线交互”的纯离线方法。若引用 RLinf 的 offline adaptation 示例，应明确那是框架实现的离线子流程，而不是原方法的全部阶段。[官方报告](https://www.pi.website/blog/pistar06)
- **STEAM**：从专家轨迹的时间顺序构造进展信号，并用 critic ensemble 与保守聚合降低对陌生状态的乐观估计。原论文摘要报告的是相对基线的任务级提升，而不是一个可脱离任务定义理解的统一成功率区间；正文只保留能够从表格逐项对应的指标。[论文](https://arxiv.org/abs/2606.29834)
- **RLT（RL Token）**：两阶段框架。阶段一在示范数据上联合训练 VLA 和 token transformer，从 VLA 隐状态提取紧凑的 $z_{rl}$；阶段二冻结表征模型，在 $z_{rl}$、本体状态与参考动作上训练轻量 actor-critic。官方报告称该设计降低了在线 RL 的视觉计算开销，具体增益限于其机器人和精密操作任务设置。[官方页面](https://www.pi.website/research/rlt)

---

## 关键概念速查

为了不打断上面的叙述，这里把反复出现的几个概念集中解释一下：

- **重要性采样比（importance ratio）**：$r_t=\pi_\theta/\pi_{old}$，衡量新策略相对旧策略在某个动作上的"偏好变化"，是 PPO 系裁剪的对象。
- **优势函数（advantage）**：$A(s,a)=Q(s,a)-V(s)$，表示"这个动作比平均水平好多少"。优势为正就强化、为负就抑制。
- **KL 约束**：$D_{KL}(\pi_\theta\|\pi_{ref})$，限制新策略别偏离参考/初始策略太远，防止 RL 把模型带崩、"胡说八道"。
- **熵正则**：鼓励策略保持一定随机性，避免过早收敛到次优的确定性行为（SAC 的核心思想）。
- **分布漂移（distribution shift）**：训练分布与部署分布不一致，是模仿学习和离线 RL 共同的核心难题。
- **On-policy vs Off-policy**：on-policy 方法主要使用当前或近邻策略生成的数据，一批 rollout 通常可做有限轮更新；off-policy 方法可长期复用历史数据，但需要处理分布偏移。IQL 是纯 offline 方法，不应仅作为普通 off-policy 在线算法理解。
- **CFG（无分类器引导）**：本是扩散生成里的技术，这里被借来做"用优势标签引导策略"——高优势当条件、低优势当无条件，放大二者差异来偏好好动作。

---
