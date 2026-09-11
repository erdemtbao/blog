---
title: "Technology Stack 03: Data and Scaling"
published: 2026-08-15
updated: 2026-09-11
description: 从经验 scaling law 到 Chinchilla 配比：如何联合分析参数、数据与训练计算，以及这些结论在具身智能中的适用边界。
tags: [Technology Stack, Data, Scaling Laws, Compute, Machine Learning]
category: Technology Stack
draft: false
---

这是 **Technology Stack** 系列的第三篇。前两篇讨论模型结构和后训练，本篇关注训练预算如何在参数、数据与计算之间分配，以及这些结论在哪些条件下成立。

## Scaling law 预测的是什么

[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 在一组自回归语言模型实验中观察到：验证集交叉熵 loss 与模型规模、数据规模和训练计算量之间，在所研究范围内近似满足幂律关系。一个简化写法是：

$$
L(N,D) \approx L_\infty + A N^{-\alpha} + B D^{-\beta},
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。若采用常见的稠密 Transformer 训练成本近似，还可写出计算约束 $C \approx kND$；常数 $k$ 取决于模型结构和 FLOP 口径。实际拟合会根据实验设计采用不同形式，因此这不是跨架构不变的物理定律。

这里需要区分三个对象：

- scaling law 直接拟合的是 **loss 或明确指定的评测量**；
- 下游准确率可能与 loss 相关，但关系并不总是线性；
- 推理、工具使用、安全性等复杂能力，不能只凭一条 loss 曲线可靠外推。

因此，scaling law 的工程价值是帮助估计同一训练体系内的趋势和预算分配，而不是准确预言一个更大模型的所有能力。

## Chinchilla：固定训练算力下的配比

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 在固定训练计算预算下研究参数量与 token 数的联合选择。论文在其模型族和实验范围内得到的经验结论是：计算最优训练通常应让参数量与训练 token 数近似同比增长；仅增加参数而保持数据量不足，会使模型训练不充分。

这不是跨架构、跨数据分布都不变的常数。上下文长度、优化器、数据重复率、推理预算和硬件效率发生变化时，最优点也会移动。因此实际项目通常要先训练一组小模型，拟合本项目自己的 loss–compute 曲线，再决定大规模训练配置。

## 数据质量不能压缩成一句口号

“数据质量高”至少包括四个可分别测量的维度：

- **有效信息密度**：低信息、模板化和重复样本会消耗训练计算；
- **覆盖范围**：数据是否覆盖目标语言、任务、场景和长尾情况；
- **正确性与一致性**：标签、文本、动作和时间对齐是否可靠；
- **配比**：不同来源在训练 mixture 中的采样权重。

因此不能笼统断言“较小的干净数据一定胜过更大的脏数据”。结果取决于模型容量、去重策略、目标任务、数据覆盖以及训练 token 预算。更可靠的做法是固定计算预算，分别进行数据过滤和 mixture 消融。

## 具身智能中的 scaling 更难

机器人数据不仅是 token，还包含观测、动作、控制频率和本体定义。规模化时至少要处理：

- **本体异构**：不同机器人具有不同自由度、动作单位和控制接口；
- **时间对齐**：图像、语言、状态和动作必须共享可靠时间戳；
- **行为质量**：成功示范、纠正数据和自主失败轨迹的价值不同；
- **覆盖与安全**：真实机器人难以低成本覆盖危险状态和长尾失败；
- **评测口径**：训练 loss 下降不等于闭环成功率、恢复能力或跨本体泛化提高。

这也是为什么机器人基础模型不能只报告参数量和小时数，还应说明机器人配置数、任务分布、episode 质量、控制频率和未见场景评测。

## 一套可执行的预算流程

1. 确定主要指标，例如验证 loss、闭环成功率和单位成功任务成本。
2. 训练多个小规模配置，拟合参数、数据和计算量的经验曲线。
3. 对去重、过滤和数据 mixture 做独立消融，避免把结构改动误认为数据收益。
4. 在拟合范围内选择计算最优点，并为分布变化保留安全余量。
5. 放大训练后重新验证曲线；不要假设小规模规律会无限延伸。

## 系列总结

- [Technology Stack 01: Pre-trained Models](/blog/posts/tech-stack-01-pretrained-models/) 讨论底座结构与训练目标；
- [Technology Stack 02: Post-training Algorithms](/blog/posts/tech-stack-02-post-training-algorithms/) 讨论如何用监督、偏好和环境反馈调整模型；
- 本篇讨论参数、数据和计算预算如何约束前两步。

三者相互影响：结构决定计算利用方式，数据决定可学习信号，后训练决定模型最终针对什么目标优化。高质量系统需要对这三个层面分别评测，而不能用参数量或单一 benchmark 代替整体能力。

---

*系列回顾：[Technology Stack 01: Pre-trained Models](/blog/posts/tech-stack-01-pretrained-models/) · [Technology Stack 02: Post-training Algorithms](/blog/posts/tech-stack-02-post-training-algorithms/) · Technology Stack 03: Data and Scaling（本篇）*
