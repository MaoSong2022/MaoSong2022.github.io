---
title: 2022 Tesla AI day
date: 2023-04-15
description: 2022 Tesla AI day
math: false
img_path: /assets/images/
categories: []
tags: [speaking, autonomous_driving]
---
# Introduction
[2022 Tesla AI Day](https://youtu.be/ODSJsviD_SU?t=3481) 介绍了Tesla 最新的自动驾驶技术。里面首先提到了occupancy network的概念


# Framework
Tesla 自动驾驶的架构如下图所示：

![Tesla FSD framework](2022Tesla-ai-day-framework.png)

在该架构中，首先使用occupancy network 来构建自车周围的可行驶区域， 

# Perception
Tesla使用了纯视觉的方案来进行感知，其感知主要包括了两部分：Occupancy network和lane network
## Occupancy network
Occupancy network的架构如下图所示：

![occupancy](2022Tesla-ai-day-occupancy.png)

对感知不是很了解，这一块可以看[赵行老师的知乎回答](https://zhuanlan.zhihu.com/p/570824078). 

## Lane network
Lane network的架构如下图所示：

![lane network](2022Tesla-ai-day-lane-network.png)


# Planning
自动驾驶中，planning面对的问题主要有两个：
1. 可行域是非凸的，这意味着当我们使用优化的方法去求解轨迹规划问题时，很可能会得到局部最优解，并且求解过程会非常耗时
2. 高维度问题，轨迹规划问题是一个高维度优化问题，随着状态变得更加精细，问题的维度呈指数级增长。

Tesla使用交互搜索来进行轨迹规划，交互搜索包括三部分：
1. 使用树搜索进行决策规划，对可能的交互进行评估并剪枝
2. neural planner来进行启发式搜索生成轨迹
3. 最后由controller对搜索得到的轨迹进行smooth

## Interactive search

![interactive search](2022Tesla-ai-day-interactive-search.png)

在遇到比较复杂的交通状况时，决策往往会变得很困难，比如我们在左转的时候需要考虑是cut-in还是yield。Tesla使用了交互搜索来解决这个难题。交互搜索包含以下步骤
1. 使用occupancy network，lane network以及prediction来得到当前的环境信息（第一行）
2. 列出可能的目标区域，用作备选(第二行)
3. 使用neural planner生成一系列轨迹（第三行）
4. 我们对生成的轨迹进行评分，然后删除掉比较危险或者不符合条件的轨迹（第四行，第五行）。
5. 我们就得到了最终的轨迹。

> 注意到我们需要考虑多个障碍物，因此我们需要重复2-4的步骤来确定最终的轨迹。
{: .prompt-info }

这里Tesla还介绍了一下neural planner与optimization-based planner的对比：

![neural planner](2022Tesla-ai-day-neural-planner-optimization-planner.png)

optimization-based planner是通过逐步增加约束来求解轨迹规划问题的，速度为1-5ms； 而neural planner是同时使用人类的专家轨迹和离线轨迹规划算法生成的轨迹，速度远快于optimization-based planner.

轨迹评分部分，Tesla给了四个标准：
1. collision-checks
2. comfort analysis
3. intervention likelihood
4. human-like discriminator


# Final ArchiArchitecture
最终，Tesla给出的FSD架构图如下所示：

![architecture](2022Tesla-ai-day-final-architecture.png)


# Data
## Auto labeling
auto labeling的步骤如下：
第一步，收集高精度的轨迹信息，这一步主要是应用视觉惯性里程计来完成的

![high precision trajectory](2022Tesla-ai-day-high-precision-trajectory.png)

第二步，使用多辆车的轨迹信息来重新构建地图信息，通过将多辆车的轨迹在空间上进行对齐，并对场景信息进行联合优化，我们可以更好地恢复场景信息

![reconstruction](2022Tesla-ai-day-multi-trip-reconstruction.png)

第三步，使用自动标注来标记新的轨迹，当我们得到了地图的信息之后，我们就可以使用匹配，对齐，联合优化等方法来获得新的轨迹的标注。

![auto labeling](2022Tesla-ai-day-auto-labeling.png)

## Simulation
Tesla介绍的simulation主要包括以下几步：
1. 使用真实的数据标签来生成道路(road)边界
2. 使用道路的连接和几何关系来生成道路(lane)级别信息
3. 随机生成建筑物，路旁静态障碍物的信息
4. 根据道路信息生成交通信号灯
5. 建立lane之间的链接关系
6. 随机生成交通参与者


## Data engine
data engine这一块，Tesla主要介绍了一个例子：
可以对objects进行re-label的操作来生成不同的场景。

![relabel](2022Tesla-ai-day-data-engine-relabel.png)

如上图所示，一辆停在路边的车，我们可以通过给其赋予不同的标签（预测结果）来产生不同的场景。

然后，通过收集相似类别的场景，我们就可以建立场景库。
