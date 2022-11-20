---
title: PlanningComponent
date: 2022-09-06
description: The start of planning module of Apollo.
math: true
img_path: /assets/images/
categories: [learning notes]
tags: [apollo, autonomous driving, c++]
---

# Background

planning模块从`PlanningComponent`类开始，`PlanningComponent`继承了`cyber:Component`这个类，`cyber::Component`这个类初始化时可以接受最多四个channel的信息，我们在创建自己的模块时，可以选择我们需要的channel。然后，我们需要自己去override `cyber::Component`以下两个函数：

- `Init()`,这个函数使用`protobuf`这个对象来初始化component
- `Proc()`定义了这个component的逻辑
以上两个函数我们只需要定义好即可，它们实际上是被`CyberRT Frame`所调用的。



# 类图

![](PlanningComponent.png)





# PlanningComponent

planning模块接受的channel message为(输入)：

- `prediction::PredictionObstacles`
- `canbus::Chassis`
- `localization::LocalizationEstimate`

planning模块的输出为`ADCTrajectory`，代表了自车行驶的轨迹。

## Init

接下来我们先看`PlanningComponent`是如何初始化的。

首先，我们需要判断使用哪种planner,这由`FLAGS_use_navigation_mode`来确定。apollo提供了两种planning模式：

- `navigation`, 即巡航模式
- `on_lane`， 即开放道路的自动驾驶

```c++
if (FLAGS_use_navigation_mode) {
    planning_base_ = std::make_unique<NaviPlanning>(injector_);
  } else {
    planning_base_ = std::make_unique<OnLanePlanning>(injector_);
  }
```

这里`planning_base_`是一个指向`PlnningBase`类的多态指针，我们在后面分析这个类。

```
> 我们仅关注`OnLanePlanning` mode，因此后面我们默认讨论该模式
{: .prompt-info }
```

决定好planner之后，函数调用了`planning_base`对象的初始化方法

```c++
planning_base->Init(config_);
```

接下来，就是设置读取的信息和输出的信息，读取的信息有：

- RoutingResponse
- TrafficLightDetection
- PadMessage
- Stories
- [navigation mode]MapMsg

输出的信息有：

- ADCTrajectory
- RoutingRequest

## Proc

初始化结束之后，我们需要设定planning模块的执行逻辑。

首先，我们先处理reroute请求。这部分逻辑在`CheckRerouting`这个函数中。

然后我们将接受的channel message以及前面`config`读取的信息加入到`local_view_`这个变量中。

接下来，我们确认输入是否正常，这部分逻辑在`CheckInput`这个函数中。

确认输入无误之后，我们将planning的主体逻辑委托给planner进行，即

```c++
planning_base_->RunOnce(local_view_, &adc_trajectory_pb);
```

这里`loca_view`就包含了我们的输入，而`adc_trajectory_pb`就是我们的规划的轨迹。

最后，我们输出得到的轨迹，以及在历史信息中加入规划的轨迹：

```c++
planning_writer_->Write(adc_trajectory_pb);
history->Add(adc_trajectory_pb);
```

到这里，我们`PlanningComponent`的任务就结束了。

# PlanningBase
由前面对`PlanningComponent`的分析我们知道，planning的主体任务实际上是通过多态委托给`PlanningBase`这个基类的派生类来实现的。

`PlanningBase`是一个接口，它主要定义了派生类必须实现的功能

- `Init`，主要是调用`TaskFactory`的初始化方法，以及初始化需要用到的数据。
- `RunOnce`，这就是前文讲到的planning模块的主体逻辑
- `Plan`，计算`ADCTrajecotry`,将其输出到`adc_trajectory_pb`中，然后返回给`PlanningComponent`类

`PlanningBase`这个基类里比较重要的数据有：

- `local_view_`，即`PlanningComponent`传递进来的`local_view`变量
- `frame_`， 存储了一个planning周期中所需要的全部数据
- `hdmap_`, 高精地图，主要与参考线`ReferenceLine`有关系
- `planner`,  指向一个planner的多态指针，由`planner_dispatcher`生成
- `planner_dispatcher`, 根据configuration指定规划器

由于具体的实现逻辑是由planning模式决定的，因此，我们在下一节分析`PlanningBase`功能的实现。

# 总结

1. planning模块的输入包括：prediction的障碍物，localization，chassis和hdmap
2. planning模块只负责实现功能，调用planning模块的是系统
