---
title: Frenet and Cartesian Conversion
date: 2022-09-07
description: Learning notes on frenet and Cartesian coordinates system conversion.
math: true
img_path: /assets/images/
categories: [learning notes]
tags: [math, Autonomous Driving]
---

# Frenet and Cartesian Conversion

最近在从事轨迹规划相关工作，期间需要用到Frenet坐标系和Cartesian坐标系之间的相互转换。在学习了已有的博客经验之后，我进行了总结。



## Background

[Why we need transformation]



### Frenet coordinate system

Frenet坐标系是为了解决



## Problem formulation

We will use the following figure to derive all the formulas we need.  the dynamics of an vehicle can be described via $6$ variables: position, heading angle, velocity and curvature.

<img src="frenet-cartesian-coordinate.svg"  />



## Cartesian to Frenet

Given a reference line $\tau:[a,b]\to\mathbb{R}^2$, $\tau(s)=(x(s),y(s))$, we first study how to convert a cartesian coordinate $(x,y,\theta_x, v,a,\kappa_x)$ to a frenet coordinate $(s,l,\dot{s},\dot{l},\ddot{s},\ddot{l})$.



> Remark: another representation of frenet coordinate is $(s,l,\dot{s},l',\ddot{s},l'')$, this representation, according to comment in Apollo code, is to achieve better satisfaction of nonqholonomic constraints.

### Step 1: Compute $s$

The very first step is to compute $s$ :
$$
s = \arg\min_{z\in[a,b]}\sqrt{(x-x(z))^2+(y-y(z))^2}
$$
Geometrically, we can find a straight line with slope $\tan\theta_x$ that intersects $\tau$ for the first time, the intersection point is exactly $(x_r,y_r)$. In implementation, the reference line is discretized as $\{s_0,\dots,s_N\}$, so the computation can be simplified via brute search:
$$
s\approx\arg\min_{i\in\{0,\dots,N\}}\sqrt{(x-x(z_i))^2+(y-y(z_i))^2}
$$
After obtaining $s$, we find the reference point $\vec{r}=(s,0)$ with respect to current location $\vec{x}=(x,y)$. The cartesian coordinate $(x_r, y_r)$,  the heading angle  $\theta_r$, and the curvature $\kappa_r$ are accessible. Now we are ready for computing other members.

### Step 2: Compute $l$

From the vector calculation, we have:
$$
\vec{x} = \vec{r}+l\mathbf{N}_r
$$
note that $\mathbf{N}_r$ is a unit vector, we have
$$
\begin{aligned}
&\vec{x} = \vec{r}+l\mathbf{N}_r\\
\Rightarrow& (\vec{x} - \vec{r})^T = l\mathbf{N}_r^T\\
\Rightarrow& l = (\vec{x} - \vec{r})^T\mathbf{N}_r\\
\Rightarrow& l = \|\vec{x} - \vec{r}\|_2\cos\langle\vec{x} - \vec{r},\mathbf{N}_r\rangle
\end{aligned}
$$
Since the $\vec{r}$ is the projection of $\vec{x}$ onto $\tau$, we have $\cos\langle\vec{x} - \vec{r},\mathbf{N}_r\rangle=\pm1$.

## Frenet To Cartesian



## Reference

[1]https://liuxiaofei.com.cn/blog/cartesian%E4%B8%8Efrenet%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%85%AC%E5%BC%8F%E6%8E%A8%E5%AF%BC/

[2]

