---
title: Frenet Cartesian Conversion
date: 2022-10-30
description: Conversion between frenet frame and Cartesian frame
math: true
img_path: /assets/images/
categories: [learning notes]
tags: [math, autonomous driving, planning]
---

# 介绍

我们一般使用Cartesian坐标系，也就是直角坐标系来描述物体运动规律。但是对于车辆来说，使用Cartesian坐标系存在以下问题：

1. 无法准确描述道路的形状，特别是复杂场景下的道路可能很难被一条曲线所描述
2. 不符合人类开车直觉，比如某一时刻我们可能是沿y轴负方向移动，但是实际上我们一般只有向前开和向后开的概念。
3. 很难确定两辆车的相对位置，我们需要将不再同车道的车辆也考虑进来，这会加大轨迹规划求解的难度。

为了解决Cartesian的问题，我们可以使用基于参考线，也就是道路中心线的frenet坐标系。在frenet坐标系中有$s$坐标和$l$坐标，其中$s$坐标代表了当前点离道路中心线起点的线距离（曲线长度），$l$坐标表示了当前点距离参考线的偏移位置（横向位移）。使用frenet坐标系的优势如下：

1. 道路始终是一条直线，我们只在乎开了多久（累积距离），和是否在当前车道上（横向偏移距离）
2. 自车始终是向前开的，不存在倒退问题。
3. 可以依据道路宽度和其他障碍物相对于车道偏移距离来轻松筛选出距离自车较远的障碍物。

基于以上考虑，现有的轨迹规划基本都建立在frenet坐标系上的。因此，我们需要了解frenet坐标系和Cartesian坐标系之间的关系。

以上是我个人思考，李柏老师在其论文里阐述了frenet坐标系的缺点，具体请参考[3]。

# Frenet坐标系

Frenet坐标系的理论基础是Frenet-Serret公式。在微分几何中，Frenet-Serret公式描述了三维欧式空间中的粒子在连续可微曲线上的运动，如下图所示。具体来讲，Frenet-Serret公式描述了曲线的切向（tangent），法向（normal）和副法向（binormal）之间的关系。

![A Space Curve](frenet.png)

图中，$T$称为单位切向量（unit tangent vector），表示沿曲线运动方向的单位向量；$N$称为单位法向量（unit normal vector），表示当前曲线运动平面内垂直于$T$的单位向量，$N$ 是$T$关于曲线长度参数的导数归一化之后的向量；$B$ 是副单位法向量，它是 $T$ 和 $N$ 的外积（cross product）：$B=T\times N$。由三者的定义我们知道它们互相正交：

$$ N^TT=N^TB=T^TB=0\tag{1} $$

假设$\vec{r}(t)$是欧式空间里随时间$t$变化的的非退化曲线，这里非退化可以理解为曲线具有非零曲率。令$s(t)$表示时刻$t$的累积曲线长度，其定义如下：

$$ s(t) = \int_0^t\|\vec{r}'(\sigma)\|d\sigma\tag{2} $$

在这里我们假设了$\vec{r}'(\sigma)\neq0$， 因此$s(t)$是一个严格单调递增的函数。从而我们可以将$t$表示为$s$的函数，即$\vec{r}(s)=\vec{r}(s(t))$.

在非退化的曲线$\vec{r}(s)$上，我们可以将 Frenet-Serret公式表述为以下形式：

$$ \begin{aligned}
\frac{dT}{ds}&=\kappa N\\
\frac{dN}{ds}&=-\kappa T+\tau B\\
\frac{dB}{ds}&=-\tau N
\end{aligned}\tag{3} $$

其中，$\kappa$是曲线的曲率，$\tau$是曲线的挠率。令$\frac{dT}{ds}=T'$, $\frac{dN}{ds}=N'$, $\frac{dB}{ds}=B'$, 则Frenet-Serret公式可以表示为矩阵形式：

$$ \begin{bmatrix}
T'\\
N'\\
B'
\end{bmatrix}=\begin{bmatrix}
0& \kappa &0\\
-\kappa & 0 &\tau\\
0 & -\tau & 0
\end{bmatrix}\begin{bmatrix}
T\\
N\\
B
\end{bmatrix}\tag{4} $$

可以看到，系数矩阵是反对称矩阵。

在自动驾驶领域中，我们一般假设车辆在平面内运动，即$\tau=0$，这样Frenet-Serret公式$(4)$可以简化为：

$$ \begin{bmatrix}
T'\\
N'\\
\end{bmatrix}=\begin{bmatrix}
0& \kappa\\
-\kappa & 0\\
\end{bmatrix}\begin{bmatrix}
T\\
N\\
\end{bmatrix}\tag{5} $$

# 几何概念

接下来我们介绍两种坐标系之间的转换，为此，我们需要给定参考线(reference line)和轨迹$\tau$上的一个点$\vec{x}$的相关信息。我们假设参考线的参数方程为：

$$ \tau:[0,s_f]\to\mathbb{R}^2,\tau(s)=(x(s),y(s)) \tag{6} $$

为了简便起见，我们将关于$t$的导数写为$\cdot$，如$\dot{l}=\frac{dl}{dt}$, 将关于$s$的导数写为$'$, 如$l'=\frac{dl}{ds}$.我们的公式推导基于下图：

![](frenet-cartesian-coordinate.svg)

其中$\theta_x,T_x,N_r$是轨迹在点$\vec{x}$处的航向角(yaw)，速度方向和法向；$\theta_r,T_r,N_r$是点$\vec{x}$在参考线上投影点$\vec{r}$处的航向角(yaw)，速度方向和法向。给定$\vec{r}=(x(s),y(x))$，我们可以通过下式得到$\theta_r$:

$$ \theta_r = \arctan\frac{y’(s)}{x’(s)}\in[-\frac{\pi}{2},\frac{\pi}{2}] \tag{7} $$

并且：

$$ T_r=(\sin\theta_r,\cos\theta_r), N_r=(-\cos\theta_r,\sin\theta_r)\tag{8} $$

同理，我们有

$$ T_x=(\sin\theta_x,\cos\theta_x), N_x=(-\cos\theta_x,\sin\theta_x)\tag{9} $$

> 不同的planner使用的Frenet坐标系表示量不一样，如Apollo使用的就是$(s,l,\dot{s},l',\ddot{s},l'')$, 这样表示的原因是车辆是nonholonomic model，横纵向不可能分开运动，因此使用$l',l''$来体现这一约束。

根据物理学定律，我们有以下基本关系式：

$$ \frac{d\vec{x}}{dt}=vT_x, \frac{d\vec{r}}{dt}=\dot{s}T_r,\frac{ds}{dt}=\dot{s},\dot{v}=\frac{dv}{dt}=a,\frac{d\theta}{ds}=\kappa\tag{10} $$

由曲率定义，我们有

$$ \dot{\theta_r}=\frac{d\theta_r}{dt}=\frac{d\theta_r}{ds}\frac{ds}{dt}=\kappa_r\dot{s} \tag{11} $$

同理，我们有

$$ \dot{\theta_x}=\frac{d\theta_x}{dt}=\frac{d\theta_x}{ds_x}\frac{ds_x}{dt}=\kappa_xv\tag{12} $$

# Cartesian到Frenet坐标系

## $(x,y)$到$(s,l)$

给定点$\vec{x}$的Cartesian坐标$\vec{x}=(x,y)$，我们需要找到$\vec{x}$在frenet坐标系下的坐标$(s,l)$.

我们可以遍历参考线上的点，找到与点$\vec{x}$最近的点$\vec{r}=(s,0)$，即

$$ \vec{r}=proj_\tau(\vec{x})\tag{13} $$

不妨假设点$\vec{r}$在Cartesian坐标系下的坐标为$\vec{r}=(x(s),y(s))$.则$(13)$等价于以下优化问题：

$$ s=\arg\min_{s\in[0,s_f]}\sqrt{(x-x(s))^2+(y-y(s))^2}\tag{14} $$

由$\vec{r}$的定义我们知道，

$$ [s,l]^T= \vec{x}-proj_\tau(\vec{x})=\vec{x}-\vec{r}\tag{15} $$

由投影的定义$(13)$我们有：

$$ \begin{aligned}
&proj_\tau(\vec{x})=\vec{x}-\vec{r}=lN_r\\
\Rightarrow& l = (\vec{x}-\vec{r})^TN_r
\end{aligned}\tag{16} $$

因此，由$(14)$和$(16)$可以得到$(s,l)$.

## $(x,y,\theta_x,v)$到$(s,l,\dot{s},\dot{l})$

在进行速度转换时，我们需要进一步假设：

> 1. 参考线的曲率远小于车辆的转弯极限
> 2. 车辆偏离参考线不会太远
> 3. 车辆在frenet坐标系下的航向角（yaw angle）不会太大
>
> 由以上假设1， 2可以认为$1-\kappa_r l>0$.由假设3可以认为 $\vert\theta_x-\theta_r\vert<\pi/2$ .

由式$(16)$，我们可以得到

$$ \dot{l}=\left[\dot{\vec{x}}-\dot{\vec{r}}\right]^TN_r + (\vec{x}-\vec{r})^T\dot{N_r}\tag{17} $$

由式$(5)$以及链式法则可知：

$$ \dot{N_r}=\frac{dN_r}{dt}=\frac{dN_r}{ds}\frac{ds}{dt}=-\kappa_r\dot{s}T_r\tag{18} $$

由定义$(10)$我们可以得到：

$$ \dot{\vec{x}}-\dot{\vec{r}}=vT_x-\dot{s}T_r\tag{19} $$

将$(5)$和$(19)$代入到$(17)$式中，并注意到$T_r,N_r$的正交性质$(1)$，我们得到：

$$ \dot{l}=[vT_x-\dot{s}T_r]^TN_r + lN_r^T(-\kappa_r\dot{s}T_r)=vT_x^TN_r\tag{20} $$

将$(8)$和$(9)$带入到$(20)$式我们有：

$$ \dot{l}=v\sin(\theta_x-\theta_r)\tag{21} $$

而由$(16)$和$(10)$可知

$$ \begin{aligned}
\dot{\vec{x}}&=\frac{d\vec{x}}{dt}=\frac{d}{dt}(\vec{r}+lN_r)\\
&\Rightarrow vT_x=\frac{d\vec{r}}{dt}+\dot{l}N_r+l\dot{N_r}\\
&\Rightarrow vT_x=\dot{s}T_r+\dot{l}N_r-l\kappa\dot{s}T_r\\
&\Rightarrow vT_x=\dot{s}(1-l\kappa_r)T_r+\dot{l}N_r
\end{aligned}\tag{22} $$

由于$T_x$是一个单位向量，我们有

$$ v = \sqrt{v^2T_x^TT_x}=\sqrt{[\dot{s}(1-\kappa_rl)]^2+(\dot{l})^2}\tag{23} $$

将$(20)$代入到$(23)$得到：

$$ \begin{aligned}
v &=\sqrt{[\dot{s}(1-\kappa_rl)]^2+v^2\sin^2(\theta_x-\theta_r)}\\
\Rightarrow\dot{s}&=\frac{\sqrt{v^2-(v\sin(\theta_x-\theta_r))^2}}{(1-\kappa_rl)\sin(\theta_x-\theta_r)}\\
&=\frac{v\cos(\theta_x-\theta_r)}{1-\kappa_rl}.
\end{aligned}\tag{24} $$



由 $(21)$ 和 $(24)$ 我们就得到了速度信息。

## $(x,y,\theta_x,v，a,\kappa_x)$到$(s,l,\dot{s},\dot{l},\ddot{s},\ddot{l})$

由前述内容，我们已经得到了$s,l,\dot{s},\dot{l}$ .接下来我们求$\ddot{s}$, $\ddot{l}$:

由 $(21)$ 我们有：

$$ \ddot{l}=\frac{d\dot{l}}{dt}=\dot{v}\sin(\theta_x-\theta_r)+v\cos(\theta_x-\theta_r)(\dot{\theta_x}-\dot{\theta_r})\tag{25} $$

将 $(10)$ 带入到 $(25)$ 中，有：

$$ \ddot{l} = a\sin(\theta_x-\theta_r) + v\cos(\theta_x-\theta_r)(\kappa_xv-\kappa_r\dot{s})\tag{26} $$

由 $(14)$ ,我们有

$$ \ddot{s} =\frac{\frac{d}{dt}[v\cos(\theta_x-\theta_r)](1-\kappa_rl)-v\cos(\theta_x-\theta_r)\frac{d}{dt}(1-\kappa_rl)}{(1-\kappa_rl)^2}\tag{27} $$

而由 $(10)$，我们有

$$\frac{d}{dt}[v\cos(\theta_x-\theta_r)]=a\cos(\theta_x-\theta_r) + v\sin(\theta_x-\theta_r)(\kappa_r\dot{s}-\kappa_xv), \frac{d}{dt}(1-\kappa_rl)=-(\dot{\kappa_r}l+\kappa_r\dot{l})\tag{28} $$

将 $(28)$ 带入到 $(27)$, 我们有：

$$ \begin{aligned}
\ddot{s}&=\frac{(1-\kappa_rl)[a\cos(\theta_x-\theta_r) + v\sin(\theta_x-\theta_r)(\kappa_r\dot{s}-\kappa_xv)]+v\cos(\theta_x-\theta_r)(\dot{\kappa_r}l+\kappa_r\dot{l})}{(1-\kappa_rl)^2}\\
&=\frac{a\cos(\theta_x-\theta_r)+\dot{l}(\kappa_r\dot{s}-\kappa_xv)+\dot{s}(\dot{\kappa_r}l+\kappa_r\dot{l})}{1-\kappa_rl}
\end{aligned}\tag{29} $$

# Frenet到Cartesian

我们知道cartesian坐标系到frenet坐标系的转换之后，只需要求解逆转换就可以了。因此我们略过这部分的计算。

# 总结

Cartesian到Frenet的转换为 $(14)$, $(16)$, $(20)$, $(24)$, $(26)$, $(29)$, 总结如下：

$$ \begin{aligned}
s&=\arg\min_{s\in[0,s_f]}\sqrt{(x-x(s))^2+(y-y(s))^2}\\
l &= (\vec{x}-\vec{r})^TN_r\\
\dot{s} & =\frac{v\cos(\theta_x-\theta_r)}{1-\kappa_rl}\\
\dot{l} &=v\sin(\theta_x-\theta_r)\\
\ddot{s}&=\frac{a\cos(\theta_x-\theta_r)+\dot{l}(\kappa_r\dot{s}-\kappa_xv)+\dot{s}(\dot{\kappa_r}l+\kappa_r\dot{l})}{1-\kappa_rl}\\
\ddot{l} &= a\sin(\theta_x-\theta_r) + v\cos(\theta_x-\theta_r)(\kappa_xv-\kappa_r\dot{s})
\end{aligned}\tag{30} $$

Frenet到Cartesian的转换总结如下：
$$ \begin{aligned}
x &= s- l\sin(\theta_r)\\
y &= s + l\cos(\theta_r)\\
\theta_x &= \arctan\frac{\dot{s}}{\dot{l}(1-\kappa_rl)}+\theta_r\in[-\pi,\pi]\\
v_x &= \sqrt{[\dot{s}(1-\kappa_rl)]^2+(\dot{l})^2}
\end{aligned}\tag{31} $$

$a$和$\kappa_x$可以联立 $\ddot{s}$ 和 $\ddot{l}$ 求解线性方程组得到。



> Remark: 注意到如下关系式，我们可以比较容易就能获取到$\dot{l}$与$l'$, $\ddot{l}$与$l''$之间的转换关系：
> $$\begin{aligned}
> l' &= \frac{dl}{ds} = \frac{dl}{dt}\frac{dt}{ds} = \frac{\dot{l}}{\dot{s}}\\
> l''&= \frac{dl'}{ds} = \frac{dl'}{dt}\frac{dt}{ds}=\frac{1}{\dot{s}}\frac{d}{dt}\left(\frac{\dot{l}}{\dot{s}}\right)=\frac{\ddot{l}\dot{s}-\ddot{s}\dot{l}}{(\dot{s})^3}
> \end{aligned}\tag{32}$$

# 参考文献

[1]https://zhuanlan.zhihu.com/p/136379544

[2]https://liuxiaofei.com.cn/blog/cartesian%E4%B8%8Efrenet%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%85%AC%E5%BC%8F%E6%8E%A8%E5%AF%BC/

[3]https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas

[4]https://ieeexplore.ieee.org/document/9703250
