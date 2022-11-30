---
title: OnLanePlanning
date: 2022-11-30
description: OnLanePlanning.cc
math: false
img_path: /assets/images/
categories: [learning notes]
tags: [apollo, autonomous driving, c++]
---

# Introduction
在第1节中，我们介绍了`planningcomponent`， 可以看到planning的主体逻辑是委托给一个`PlanningBase`类的多态指针来实现的。

# OnLanePlanning
我们在本节中将介绍`OnLanePlanning`的运行逻辑。从第1节的分析，`OnLanePlanning`需要实现以下三个函数：
- `Init`, 初始化
- `RunOnce`, planning模块的主体逻辑
- `Plan`, 输出`ADCTrajectory`.

我们接下来着重分析这个类。

## Init
`Init`这个函数负责初始化，首先，其调用了基类的初始化函数`PlanningBase::Init` 而基类的初始化函数又调用了`TaskFactory::Init`, 这个初始化函数的作用是注册各种`Decider`。
然后，`Init`调用`PlannerDisPatcher::Init`, 这个初始化函数调用了它自己的`RegisterPlanners`， 用来注册各种`Planner`， 包括：
- RRT
- public_road
- lattice
- navi
接下来，其加载了高精地图`hdmap_`，然后使用高精地图初始化`ReferenceLineProvider`。然后启动`ReferenceLineProvider`:
```c++
reference_line_provider_->Start();
```
再接下来，其根据configuration来选择合适的planner：
```c++
planner_ = planner_dispatcher_->DispatchPlanner(config_, injector_);
```
最后，调用对应的`Planner`的`Init`方法
```c++
planner_->Init(config_);
```

## RunOnce
方法一开始做了一大堆检查，我们略过这堆检查部分。
首先，函数使用自车状态更新`ReferenceLineProvider`
```c++
reference_line_provider_->UpdateVehicleState(vehicle_state);
```
其次，方法调用`TrajectoryStitcher`类生成`stitching_trajectory`
```c++
std::vector<TrajectoryPoint> stitching_trajectory =
	TrajectoryStitcher::ComputeStitchingTrajectory(
	vehicle_state, start_timestamp, planning_cycle_time,
	FLAGS_trajectory_stitching_preserved_length, true,
	last_publishable_trajectory_.get(), &replan_reason);
```
这里`stitching_trajectory`是`Plan`函数的输入。
然后，方法调用`InitFrame`方法对当前帧进行初始化：
```c++
status = InitFrame(frame_num, stitching_trajectory.back(), vehicle_state);
```
我们在后面分析`InitFrame`这个方法，
再然后，方法计算了一个安全距离：
```c++
injector_->ego_info()->CalculateFrontObstacleClearDistance(frame_->obstacles());
```
接下来，方法执行了`TrafficDecider`的`Init`和`Excute`方法，我们在后续介绍这个类，其目的就是检查当前的`referenceline`是否是可以通行的。
```c++
for (auto& ref_line_info : *frame_->mutable_reference_line_info()) {
	TrafficDecider traffic_decider;
	traffic_decider.Init(traffic_rule_configs_);
	auto traffic_status = traffic_decider.Execute(frame_.get(), &ref_line_info, injector_);
	if (!traffic_status.ok() || !ref_line_info.IsDrivable()) {
		ref_line_info.SetDrivable(false);
	}
}
```

再接下来，方法将输出trajectory的任务交给`Plan`方法执行，
```c++
status = Plan(start_timestamp, stitching_trajectory, ptr_trajectory_pb);
```
这里的`ptr_trajectory_pb`就是一个指向`ADCTrajectory`的指针，存储了`planning`模块的结果。
最后又是一大堆检查，我们略过。

## Plan
从`RunOnce`的方法我们可以看到，实际上我们planning模块的核心从`RunOnce`方法转移到了`Plan`方法。
`Plan`方法的输入为
- 当前时刻`current_time_stamp`
- `RunOnce`生成的`stitching_trajectory`
输出为`ADCTrajectory`.  
方法首先调用`Planner`的`Plan`方法：
```c++
auto status = planner_->Plan(stitching_trajectory.back(), frame_.get(),
	ptr_trajectory_pb);
```

> 我们在本文中考虑结构化道路的自动驾驶。

接下来，方法创造了两个`ReferenceLineInfo`类的变量：
```c++
const auto* best_ref_info = frame_->FindDriveReferenceLineInfo();
const auto* target_ref_info = frame_->FindTargetReferenceLineInfo();
```
`best_ref_info`是cost最小的一个，而`target_ref_info`则是不在当前`segments`上的那个。
然后，方法将当前帧规划的path加入到当前帧的信息里。

-------
## InitFrame
方法的第一步是更新当前`frame_`的内容
```c++
frame_.reset(new Frame(sequence_num, local_view_, planning_start_point,
					   vehicle_state, reference_line_provider_.get()));
```
然后获取了`reference_lines`和`segements`信息：
```c++
reference_line_provider_->GetReferenceLines(&reference_lines, &segments);
```
接下来根据当前速度计算`reference_lines`应该延长的距离
```c++
auto forward_limit = hdmap::PncMap::LookForwardDistance(vehicle_state.linear_velocity());
```
根据`forward_limit`判断`reference_lines`是否最少有两个点，否的话报错。再对`segments`执行`RouteSegments::Shrink`方法来压缩routing segments
最后，调用`Frame::Init`方法进行当前帧的初始化。