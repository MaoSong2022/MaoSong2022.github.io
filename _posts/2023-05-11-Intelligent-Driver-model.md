---
title: Intelligent Driver model (IDM) 
date:  2023-05-11 
description: 
math: true
img_path: /assets/images/
categories: [autonomous_driving]
tags: [model]
---

# 介绍
在交通流模型中，intelligent driver model (IDM) 是一个时间连续的跟车模型，该模型主要用在高速公路和城市道路上。

# 模型定义
对于车辆 $\alpha$, 令 $x_\alpha$, $v_\alpha$ , $\ell_\alpha$分别表示其位置，速度和车长。我们定义两车间距为 $s_\alpha=x_{\alpha-1}-x_{\alpha}-\ell_{\alpha-1}$, 其中 $\alpha-1$ 表示车辆 $\alpha$正前方的车辆，定义速度差为 $\Delta v_\alpha=v_\alpha-v_{\alpha-1}$. 
作为一个跟车模型，IDM描述了单车的动力学模型。IDM可以被以下公式表述：

$$ \begin{aligned}
\dot{x}_ {\alpha}&=\frac{dx_ {\alpha}}{dt}=v_ {\alpha}\\
\dot{v}_ {\alpha}&=\frac{dv_ {\alpha}}{dt}=a\left[1-\left(\frac{v_ \alpha}{v_ 0}\right)^\delta-\left(\frac{s^*(v_ \alpha,\Delta v_ \alpha)}{s_\ alpha}\right)^2\right]
\end{aligned} $$

其中

$$
s^*(v_\alpha,\Delta v_\alpha)=s_0+v_\alpha T+\frac{v_\alpha\Delta v_\alpha}{2\sqrt{ab}}
$$

$v_0$, $s_0$, $T$, $a$和 $b$的意义如下：
- 理想速度 $v_0$: 自车在自由交通下的行驶速度
- 最小车间距 $s_0$: 最小的车间距，如果车间距小于$s_0$, 那么自车将无法移动
- 理想车头时距 $T$: 最小车头时距
- 加速度 $a$: 自车最大加速度
- 舒适刹车减速度 $b$: 正数
指数 $\delta$ 通常设定为 $4$.

# 模型特征
自车的加速度 $a$可以被分成自由项

$$ \dot{v}_ {\alpha}^{free}=a [1-( \frac{v_ \alpha}{v_0} )^\delta ] $$

和交互项：

$$ \dot{v}_ {\alpha}^{init}=-a\left(\frac{s^*(v_ \alpha,\Delta v_ \alpha)}{s_ \alpha}\right)^2=-a\left(\frac{s_ 0+v_ \alpha T}{s_ \alpha}+\frac{v_ \alpha\Delta v_ \alpha}{2\sqrt{ab}s_ \alpha}\right)^2 $$

其中，
- 自由项代表了自车前方没有其他车辆，或者与其他车辆的较远的情况，此时自车的加速度由 $\dot{v}_ {\alpha}^{free}$ 主导，自车的速度会由 $v_ 0$逐渐递增到 $v_ \alpha$, 加速度则逐渐减小。
- 交互项代表了追尾，特别是速度差 $\Delta v_\alpha$比较大时的情况，此时 $\dot{v}_ {\alpha}^{init}$ 由第二项主导：

 $$ -a( \frac{v_ \alpha\Delta v_ \alpha}{2\sqrt{ab}s_ \alpha})^2=-\frac{v_ \alpha\Delta v_ \alpha}{4bs_ \alpha^2} $$
 
即自车的加速度会让自车在进行紧急刹车时，刹车加速度不会超过 $b$
- 当自车与前车非常接近，并且速度差 $\Delta v_\alpha$比较小时， 此时 $\dot{v}_ {\alpha}^{init}$ 由第一项主导：

 $$ -a\left(\frac{s_ 0+v_ \alpha T}{s_ \alpha}\right)^2 $$
 
即自车倾向于减速与前车拉开距离，来保证安全。 
