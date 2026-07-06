---
title: "Paper Notes: Robot Learning (3)"
published: 2026-04-02
description: 训练无关（training-free）的语言/视觉基础模型接地一线——VoxPoser、CoPa、ReKep：用 LLM/VLM 把自由语言接地成显式的 3D 几何/约束，再交给优化器/规划器解出 6-DoF 动作。
image: ''
tags: [Paper Notes, Robot Learning, Foundation Models]
category: Paper Notes
draft: false
---

## Overview

本篇聚焦一条与 [Robot Learning (1)](/blog/posts/paper-notes-robot-learning-1/)、[(2)](/blog/posts/paper-notes-robot-learning-2/) 的**端到端 VLA**（π0 系列那种「图像+语言→动作」直连）**正相反**的路线：

> **不训练策略网络**，而是把预训练基础模型（LLM / VLM / 大视觉模型）当成「会写代码、会推理几何」的接地器（grounder），让它从自由语言里推断出**显式的空间表示**（3D value map / 部件约束 / 关键点约束），再交给**经典的规划器或优化器**解出 6-DoF 末端位姿。

这条线的三篇代表作，恰好是一条**表示越来越结构化**的演进：

- **VoxPoser** — LLM 写代码调用 VLM，在体素化观测空间上**组合 3D value map**（affordance 吸引 + constraint 排斥），再用基于模型的规划合成密集 6-DoF 路点。
- **CoPa** — 把接地下沉到**物体部件**：粗到细定位任务相关部件，VLM 生成**部件的空间几何约束**，解出抓取后的一串位姿。
- **ReKep** — 把任务写成**关系关键点上的时空约束**（Python 代价函数），大模型自动生成，再用**分层优化**实时解 SE(3) 位姿，支持多阶段/双臂/反应式。

一条主线贯穿：**基础模型负责「语义→几何」的接地，经典优化负责「几何→动作」的求解**——用零样本、可解释、免数据的方式，绕开 VLA 对海量真机数据的依赖。代价是依赖深度感知/关键点跟踪的精度，且动作质量受限于所选优化器而非学习到的技巧。

## Paper List

| 简称 | 年份 / Venue | 表示 | 接地器 | 求解器 | 核心思想 |
|:--|:--|:--|:--|:--|:--|
| [VoxPoser](#voxposer) | CoRL 2023 | 3D value map（体素） | LLM 写码 + VLM 检测 | 基于模型的运动规划 | 组合 affordance/constraint 体素图，规划密集 6-DoF 路点 |
| [CoPa](#copa) | arXiv 2024 | 部件空间约束 | GPT-4V + 抓取/分割模型 | 约束求解 | 粗到细定位部件，VLM 出几何约束，解抓取后位姿 |
| [ReKep](#rekep) | arXiv 2024 | 关系关键点约束 | DINOv2 + GPT-4o | 分层优化（实时闭环） | 关键点代价函数 + 跟踪，实时解 SE(3)，多阶段/双臂 |


::::paper{tone="voxposer"}

## VoxPoser

**VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models**

:::note[一句话]
LLM 用**写代码**的能力调用 VLM，在体素化的观测空间上**组合 3D value map**（affordance 吸引 + constraint 排斥），把自由语言接地成几何目标，再用基于模型的规划**零样本**合成密集 6-DoF 路点，不依赖预定义原语。
:::

**年份 / Venue** CoRL 2023｜ **机构** Stanford（Fei-Fei Li 组）｜ **方向** Training-free LLM grounding, model-based planning ｜ **真机** ✅ Franka Emika + 仿真大规模研究

**材料** [Paper](https://arxiv.org/abs/2307.05973) · [Project](https://voxposer.github.io/) · [Code](https://github.com/huangwl18/VoxPoser)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@inproceedings{huang2023voxposer,
  title     = {{VoxPoser}: Composable {3D} Value Maps for Robotic Manipulation with Language Models},
  author    = {Huang, Wenlong and Wang, Chen and Zhang, Ruohan and Li, Yunzhu
               and Wu, Jiajun and Fei-Fei, Li},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.05973}
}</code></pre>
</details>

### Motivation

LLM 里装着大量可用于操作的「可执行知识」（推理 + 规划），但多数工作把这些知识落地时**仍靠预定义的运动原语**（pick/place/push…），这成了主要瓶颈——原语库有多大，机器人的行为空间就有多窄。VoxPoser 想直接**合成机器人轨迹本身**（密集的 6-DoF 末端路点），面向**开放指令 + 开放物体**，而不局限于某套原语。关键观察是：LLM 擅长从自由语言里**推断 affordance 与 constraint**，且能靠**写代码**去调用感知，把这些知识接地到智能体的观测空间。

### Method

- **LLM 写码接地**：LLM（GPT-4）接到指令后**生成 Python 代码**，代码里调用 VLM / 开放词表检测器定位相关物体，并在**体素网格**上填出数值——这一步把「语言里的语义」变成「观测空间里的数」。
- **3D value map（可组合）**：两类核心图——**affordance map**（该靠近/去往的区域，正值吸引）与 **constraint map**（该避开的区域，负值排斥）；此外还可组合旋转、速度、夹爪等图。多张图**可组合**成一个总的 value landscape。
- **基于模型的规划**：把组合后的 value map 当作代价/奖励地形，用**基于模型的运动规划**在其上搜索，合成**密集 6-DoF 路点**序列。**闭环**执行、每步重规划 → 对动态扰动鲁棒（人手推、移动目标都能纠）。
- **在线学习动力学（可选）**：对接触密集（contact-rich）的场景，可在线**高效学一个动力学模型**做 MPC，提升如开抽屉这类需要接触推理的任务。
- **全程零样本**：无策略训练、无预定义原语、无任务专属数据。

![VoxPoser 管线：(a) LLM+VLM 组合 3D value map，(b) 在 value map 上做运动规划](/blog/paper-note/Robot_Learning_3/VoxPoser/method.jpg)

### Experiments

- **规模**：在**仿真 + 真机（Franka）**上做了大规模研究，覆盖大量以自由语言指定的日常操作任务（如开抽屉、把物体挪开、避障移动等）。
- **对照**：与「LLM + 预定义原语」及去掉 value map 组合的变体对比，验证「组合式 value map + 规划」的价值；具体任务成功率百分比**未逐一核到**，此处只引用「零样本合成 6-DoF 轨迹、对扰动鲁棒」这一头条结论。
- **涌现能力（官方展示）**：常识推理（「我渴了」→取饮料）、细粒度语言纠正（「往左挪一点」）、多步视觉编程、行为常识（摆餐具）、估计物体物理属性等。

### Strengths and Limitations

**Strengths**：真正**零样本、免训练、免原语**，靠 LLM 写码 + VLM 接地就能合成密集 6-DoF 轨迹；value map 可组合、可解释；闭环重规划对扰动鲁棒；可选在线动力学学习覆盖接触任务。

**局限（分析）**：依赖 VLM/检测的接地精度与深度观测质量，感知错则满盘皆错；value map 表达的是「往哪去/避什么」，对**精细朝向/接触力**这类语义刻画有限（CoPa 正是补这块）；规划质量受体素分辨率与代价地形设计影响；对高度灵巧、多指接触的任务，纯规划难敌学习到的策略。

### Takeaways

VoxPoser 立起了这条线的范式：**LLM 写码把语言接地成 3D value map，再交给经典规划器**。它证明了不训练策略也能做开放世界操作。后两篇都在追问「value map 还不够刻画什么」——CoPa 补**部件级几何约束**，ReKep 补**关系关键点上的时空约束**。

::::paper{tone="copa"}

## CoPa

**CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models**

:::note[一句话]
把接地下沉到**物体部件**：用粗到细的 grounding 定位「任务相关部件」，让 VLM（GPT-4V）生成这些部件的**空间几何约束**（如锤头要在钉子正上方、轴线竖直），再解出抓取之后的一串 6-DoF 位姿；开放世界、免训练。
:::

**年份 / Venue** arXiv 2024（正式发表以官方页为准）｜ **机构** Tsinghua IIIS · Shanghai Qi Zhi · SJTU · Shanghai AI Lab（Yang Gao 组）｜ **方向** Part-level VLM grounding, task-aware planning ｜ **真机** ✅ Franka，开放世界桌面操作

**材料** [Paper](https://arxiv.org/abs/2403.08248) · [Project](https://copa-2024.github.io/)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{huang2024copa,
  title   = {{CoPa}: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models},
  author  = {Huang, Haoxu and Lin, Fanqi and Hu, Yingdong and Wang, Shengjie and Gao, Yang},
  journal = {arXiv preprint arXiv:2403.08248},
  year    = {2024},
  url     = {https://arxiv.org/abs/2403.08248}
}</code></pre>
</details>

### Motivation

基础模型的 web 级常识对**任务规划**很有用，但把计划**物理落地**往往还得靠任务专属学习——要采数据、难泛化。VoxPoser 那种 value map 擅长「去哪/避什么」，却难精确表达**朝向与部件级几何**（比如「用锤头砸钉子」要求锤头朝下、对准钉子）。CoPa 想只用基础模型的常识，**零训练**地生成一串 6-DoF 末端位姿，尤其把「任务相关部件之间的空间几何关系」显式刻画出来。

### Method

CoPa 把一次操作拆成**两个模块**（都由基础模型驱动、无需训练）：

1. **Task-Oriented Grasping Module（任务导向抓取）**：**粗到细 grounding**——先用分割 + Set-of-Mark 标注让 GPT-4V 在**粗粒度**选出要抓的物体，再在**细粒度**选出具体抓取部件的 mask；抓取候选由抓取基础模型（GraspNet/AnyGrasp 类）在点云上生成，用 grounded 的部件筛出合适抓取位姿。
2. **Task-Aware Motion Planning Module（任务感知运动规划）**：抓取后，模型**识别任务相关部件**并在 3D 中建模，把部件几何投影/标注回图像，让 VLM 生成这些部件的**空间约束**（如「瓶口朝下」「锤头在钉子正上方」），再由**求解器**据约束算出抓取后的位姿序列 $\{P_1,\dots,P_N\}$。

- **基础模型班底**：GPT-4V（选择与约束生成）、GroundingDINO / SAM（检测与分割）、GraspNet/AnyGrasp 类（抓取候选）、SoM（Set-of-Mark 提示）。
- **可接高层规划**：对长时程任务，可在上层接一个 LLM 任务规划器，把复杂指令拆成子任务再逐个交给 CoPa。

![CoPa 管线：任务导向抓取模块 → 任务感知运动规划模块 → 抓取后位姿序列](/blog/paper-note/Robot_Learning_3/CoPa/teaser.png)

![粗到细 grounding：分割 → Set-of-Mark → VLM 逐级选出任务相关部件](/blog/paper-note/Robot_Learning_3/CoPa/grounding.png)

![任务感知运动规划：对任务相关部件建模空间几何约束，解出抓取后位姿](/blog/paper-note/Robot_Learning_3/CoPa/behavior.png)

### Experiments

- **设置**：真机（Franka）**开放世界桌面操作**，任务如「锤钉子、把花插进花瓶、按按钮、把橡皮放进抽屉、倒水、把勺子放进杯子、扫坚果」等——多数**强依赖朝向/部件几何**，正是 value-map 式方法的短板。
- **对照**：与直接让 VLM 出位姿、以及缺乏部件约束的基线相比，CoPa 在需要精确朝向的任务上更稳；**逐任务成功率数值未从原文表格逐一核到**，此处只引用「部件空间约束显著改善朝向敏感任务」这一定性结论。
- **长时程**：接高层 LLM 规划器后可完成多步任务（如做简单料理/整理），验证模块化可组合。

### Strengths and Limitations

**Strengths**：把接地精度提到**部件级**，能刻画 value map 难表达的**朝向/几何约束**；两模块解耦（抓取 / 抓取后规划）清晰可组合；纯基础模型、零训练、开放指令与开放物体；可与高层规划器堆叠做长时程。

**局限（分析）**：管线较长（分割 + 抓取模型 + 多次 VLM 调用），环节多则误差累积、时延高；依赖 GPT-4V 对部件与几何约束的判断，复杂/遮挡场景易错；约束是**几何规则**而非学习到的接触技巧，对柔性物、力控精细任务能力有限；正式 benchmark 数值需回原文核对。

### Takeaways

CoPa 把 VoxPoser 的「场景级 value map」推进到「**部件级几何约束**」，专治朝向敏感的操作。它与 ReKep 的关系很近——都在用 VLM 生成**几何约束**再优化位姿；区别在于 ReKep 把约束统一成**关系关键点上的可微代价**并做**实时闭环**，而 CoPa 更偏「抓取 + 抓取后位姿」的分段规划。

::::paper{tone="rekep"}

## ReKep

**ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation**

:::note[一句话]
把操作任务写成**关系关键点上的时空约束**——一组 Python 代价函数，把若干 3D 关键点映射到标量代价；大视觉模型 + VLM 从语言与 RGB-D **自动生成**这些约束，再用**分层优化**实时解出 SE(3) 位姿序列，支持多阶段、双臂、反应式。
:::

**年份 / Venue** arXiv 2024（会议录用以官方为准）｜ **机构** Stanford · Columbia（Fei-Fei Li / Yunzhu Li）｜ **方向** Keypoint constraints, hierarchical optimization ｜ **真机** ✅ 移动单臂 + 固定双臂，in-the-wild

**材料** [Paper](https://arxiv.org/abs/2409.01652) · [Project](https://rekep-robot.github.io/) · [Code](https://github.com/huangwl18/ReKep)

::::

<details class="paper-bibtex-fold">
<summary>BibTeX</summary>
<pre><code>@article{huang2024rekep,
  title   = {{ReKep}: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation},
  author  = {Huang, Wenlong and Wang, Chen and Li, Yunzhu and Zhang, Ruohan and Fei-Fei, Li},
  journal = {arXiv preprint arXiv:2409.01652},
  year    = {2024},
  url     = {https://arxiv.org/abs/2409.01652}
}</code></pre>
</details>

### Motivation

把操作任务表达成「关联机器人与环境的**约束**」是一条有前途的路，但约束怎么写才能同时满足三点：① **通用**于各种任务、② **免人工标注**、③ 能被**现成求解器实时优化**出动作。此前的约束表示往往顾此失彼。ReKep 想给出一种**视觉接地**的约束表示，让大模型自动生成、求解器实时求解，从而在真实世界里做多阶段、双臂、反应式操作。

### Method

- **Relational Keypoint Constraints（ReKep）**：把任务写成一组 **Python 函数**，输入是环境里一组 **3D 关键点**，输出是一个**标量代价**（满足→0，违反→正）。用「关系关键点」既能表达机器人与物体、也能表达物体与物体之间的几何关系。
- **两类约束**：
  - **sub-goal 约束**：刻画某个**阶段末**要达成的关键点关系（如「壶嘴对准杯口」）。
  - **path 约束**：刻画阶段**全程**要保持的关系（如「端平不洒」）。
  - 一个任务 = 一串（分阶段的）sub-goal + path 约束。
- **自动生成**：**DINOv2**（+ SAM）在细粒度有意义区域上**提议关键点候选**并做 Set-of-Mark；**GPT-4o** 读语言指令 + 标注图像，**写出上述 Python 约束程序**——无需为每个新任务手工指定。
- **分层优化**：高层在离散的阶段/子目标间决策，低层用现成求解器把约束优化成 **SE(3) 末端位姿序列**；因关键点被**实时跟踪**，系统能在**感知-动作闭环**里以实时频率**重规划** → 反应式（物体被挪动也能跟上）。
- 全程**免任务专属训练、免环境模型**。

![ReKep 管线：RGB-D + 语言 → DINOv2 关键点 → GPT-4o 写关系关键点约束 → 约束优化求解器 → SE(3) 动作](/blog/paper-note/Robot_Learning_3/ReKep/method.jpg)

### Experiments

- **平台**：**移动单臂**（wheeled single-arm）与**固定双臂**（stationary dual-arm）两套真机。
- **任务**：in-the-wild（收书、倒茶、封箱、回收易拉罐）、双臂协作（装鞋、协同叠衣、叠毛衣）、以及**用新策略叠各类衣物**（毛衣/衬衫/连帽衫/背心/裙/裤/短裤/围巾）；展示多阶段与反应式（关键点被扰动后闭环纠正）。
- **数值**：官方页与摘要未给出统一的成功率表，**具体百分比未核到**；可靠的是「统一约束表示 + 分层优化 + 实时关键点跟踪 → 多阶段/双臂/反应式，零样本免训练」这一定性结论。

### Strengths and Limitations

**Strengths**：约束表示**统一而通用**（一个 Python 代价框架涵盖 sub-goal/path、机器人-物体/物体-物体、单/双臂）；大模型自动生成、**免人工标注**；用现成求解器**实时闭环**、天然反应式；覆盖 in-the-wild 与双臂等硬场景。

**局限（分析）**：强依赖**关键点检测与实时跟踪**的稳定性——关键点漂移/遮挡会直接坏掉约束；GPT-4o 写错约束程序则行为错，且难自查；关键点是**稀疏几何**，对纹理/形变复杂（如布料精细褶皱）的刻画有上限；求解质量与实时性受求解器与关键点数量制约。

### Takeaways

ReKep 是这条线的集大成：把 VoxPoser 的「value map」、CoPa 的「部件约束」统一成**关系关键点上的可微代价**，并首次把它做到**实时闭环 + 双臂**。它把「大模型接地 → 经典优化求解」这套范式，从「规划一次」推到了「**实时反应**」，是 training-free 路线里最接近可用系统的一篇。

## Cross-Paper Comparison

| 论文 | 空间表示 | 谁来接地 | 谁来求解 | 闭环/反应式 | 强项 | 局限（分析） |
|:--|:--|:--|:--|:--|:--|:--|
| VoxPoser | 3D value map（体素） | LLM 写码 + VLM 检测 | 基于模型的运动规划 | 每步重规划 | 零样本合成 6-DoF、可组合、可解释 | 难刻画精细朝向/接触 |
| CoPa | 部件空间约束 | GPT-4V + 分割/抓取模型 | 约束求解（抓取后位姿） | 分段规划 | 部件级几何、朝向敏感任务强 | 管线长、误差累积、时延 |
| ReKep | 关系关键点约束 | DINOv2 + GPT-4o | 分层优化（现成求解器） | **实时闭环、反应式** | 统一约束、双臂、实时 | 依赖关键点跟踪、稀疏几何 |

## Discussion

把三篇连起来看，它们共享**同一套范式**，却在「表示的结构化程度」和「求解的实时性」上逐步进化：

1. **共同信念：接地 ≠ 训练策略**。三篇都拒绝「端到端学一个 policy」，而是让基础模型只做它最擅长的**语义→几何接地**，把「几何→动作」交给几十年积累的**规划/优化**。好处是**零样本、可解释、免真机数据**；这与 [Robot Learning (2)](/blog/posts/paper-notes-robot-learning-2/) 的 π0 系列（堆万小时真机数据学连续策略）形成鲜明对照——**一条省数据、一条省先验**。

2. **表示越来越结构化**。VoxPoser 的 value map 是「稠密标量场」，擅长「去哪/避什么」；CoPa 把它下沉到**部件几何约束**，补上朝向；ReKep 再抽象成**关系关键点上的可微代价**，用少数关键点统一表达机器人-物体、物体-物体、单臂-双臂的关系。表示越结构化，越能表达精细任务，但也越依赖对应的感知（检测 / 部件分割 / 关键点跟踪）。

3. **从「规划一次」到「实时反应」**。VoxPoser/CoPa 更偏「解出一条轨迹/一串位姿再执行」，ReKep 靠**关键点实时跟踪 + 分层优化**做到感知-动作闭环，物体被挪也能跟——这是把 training-free 路线推向真实动态环境的关键一步。

4. **与端到端 VLA 的互补**。这条线的天花板是**感知精度**（深度/关键点错则全错）与**优化器的动作质量**（不会「学」出灵巧接触技巧）；VLA 的天花板是**数据**（要海量真机示范）与**可解释性**。可预见的融合是：**用这类几何约束/关键点作为 VLA 的中间监督或推理脚手架**（如 π0.5 的「先语义子任务再动作」、ReKep 的关键点可作 affordance 提示），让「大模型的语义」「显式的几何」「学习到的连续控制」三者各取所长。

对后续 embodied foundation model 的启发：**接地表示的选择（value map / 部件约束 / 关键点）本身就是一种强归纳偏置**——它决定了系统能表达什么任务、需要什么感知、能否实时。training-free 一线证明了「不学策略也能开放世界操作」，但要走向高灵巧与鲁棒，几乎必然要与学习到的策略（VLA）缝合。
