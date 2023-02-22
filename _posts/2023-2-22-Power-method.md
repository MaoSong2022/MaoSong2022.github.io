---
title: Power iteration
date: 2022-12-08
description: notes on power iteration
math: true
img_path: /assets/images/
categories: [learning notes]
tags: [algorithm]
---

# Power method

## Introduction
Power method 是线性代数中用来求矩阵最大特征值的一个算法。
一般来说，给定一个矩阵 $\mathbf{A}=(a_{ij})\in\mathbb{R}^{n\times n}$, 我们通过求解多项式 $|\lambda \mathbf{I}-\mathbf{A}|$ 来找到 $\mathbf{A}$ 的最大特征值，但是这种方法一般非常耗时，因此，power method就可以用来解决这个问题。Power iteration算法表述如下：

1. 随机选取 $\mathbf{q}_0\in\mathbb{R}^n $
2. 对 $k=0,1,\dots,$, 重复以下过程:
3. $\mathbf{q}_{k+1}=\mathbf{A}\mathbf{q}_k/\|\mathbf{A}\mathbf{q}_k\|$
4. $\lambda_k=\mathbf{q}_k^T\mathbf{A}\mathbf{q}_k$.

## Algorithm Analysis
假设 $\mathbf{A}=\mathbf{V}\Lambda\mathbf{V}^{-1}$, 其中 $\Lambda=\mathbf{diag}(\lambda_1,\dots,\lambda_n)$, 并且 $|\lambda_1| > |\lambda_2| \geq \cdots \geq\cdots |\lambda_n|$. 我们记 $\mathbf{V}=[\mathbf{v}_1,\dots,\mathbf{v}_n]$. 

注意到 

$$ \mathbf{A}^k=\mathbf{V}\Lambda^k\mathbf{V}^{-1} \Rightarrow \mathbf{A}^k\mathbf{V} = \mathbf{V}\Lambda^k $$

由于 $\mathbf{range}(\mathbf{V})=\mathbb{R}^{n}$, 因此存在$\tilde{\mathbf{x}}\in\mathbb{R}^n$ 使得 $\mathbf{q}_0 = \mathbf{V}\tilde{\mathbf{x}}$. 这样我们就有：

$$ \mathbf{A}^k\mathbf{q}_0 = \mathbf{A}^k\mathbf{V}\tilde{\mathbf{x}}=\mathbf{V}\Lambda^k\tilde{\mathbf{x}}=\sum_{i=1}^n\lambda_i^k\tilde{x}_i\mathbf{v}_i $$

将上式简单的改写一下：

$$ \mathbf{A}^k\mathbf{q}_0 = \lambda_1^k\left(\sum_{i=1}^n\left(\frac{\lambda_i}{\lambda_1}\right)^k\tilde{x}_i\mathbf{v}_i\right) $$

由前面的假设 $|\lambda_1| > |\lambda_2| \geq \cdots \geq\cdots |\lambda_n|$, 我们有 $(\lambda_i/ \lambda_1)^k\to0$ for $i\neq 1$. 因此对于充分大的 $k$, 我们可以认为 $\mathbf{A}^k\mathbf{q}_0$ 与 $\mathbf{v}_1$ 平行， **假设** $\tilde{x}_1\neq0$, 则我们有：

$$ \mathbf{v}_1\approx \mathbf{q}_k = \frac{\mathbf{A}^k\mathbf{q}_0}{\|\mathbf{A}^k\mathbf{q}_0\|_2} $$

相应的特征值 $\lambda_1$ 可以通过一下方法计算：

$$ \lambda_1 \approx \lambda_k = \frac{\mathbf{q}_k^T\mathbf{A}\mathbf{q}_k}{\mathbf{q}_k^T\mathbf{q}_k} $$

并且，通过以上计算过程，我们有：

$$ \mathbf{dist}(\mathbf{span}(\mathbf{q}_k), \mathbf{span}(\mathbf{x}_1)) = \dfrac{\mathbf{q}_k^T\mathbf{x}_1}{\|\mathbf{q}_k\|_2\|\mathbf{x}_1\|_2}=\mathcal{O}\left(\left|\dfrac{\lambda_2}{\lambda_1}\right|^k\right) $$

同理，我们有：

$$ |\lambda_1-\lambda_k| =\mathcal{O}\left(\left|\dfrac{\lambda_2}{\lambda_1}\right|^k\right) $$

> Remark: 我们假设了 $\tilde{x}_1\neq0$, 实际上，由于 $\mathbf{x}$ 是随机选取的，因此 $\mathbf{V}^{-1}\mathbf{x}$ 的第一个元素等于 $0$ 的概率为 $0$. 另一方面，如果 $\tilde{x}_1=0$, 则由于我们在计算过程中因为浮点数运算产生的误差，这个问题也会被避免。

# Extensions

## Spectral transformation and shift-invert
由前面分析，我们知道 $\mathbf{A}^k=\mathbf{V}\Lambda^k\mathbf{V}^{-1}$, 假设 $f(\cdot)$ 是任意一个收敛级数， 那么只要特征值在收敛半径以内， 根据谱定理(spectral mapping theorem), 我们就有：

$$ f(\mathbf{A}) = \mathbf{V}f(\Lambda)\mathbf{V}^{-1} $$

特别地，我们考虑 $f(z) = (z-\sigma)^{-1}$. 则：

$$ (\mathbf{A}-\sigma\mathbf{I})^{-1} = \mathbf{V}f(\Lambda-\sigma\mathbf{I})^{-1}\mathbf{V}^{-1} $$

注意到 $(\mathbf{A}-\sigma\mathbf{I})^{-1}$ 的最大特征值为 

$$ \max_{i}(\lambda_i-\sigma)^{-1} $$

如果我们使用power iteration来求 $(\mathbf{A}-\sigma\mathbf{I})^{-1}$ 的特征向量的话， 最终将会得到如下特征值

$$ \max_i \dfrac{1}{|\lambda_j-\sigma|} $$

即我们可以求得与 $\sigma$ （复平面上）最近的特征值。



##  Rayleigh quotient iteration

求得一个近似的特征值 $\sigma$ 有时候也是一件很难的事情，我们可以考虑找一个近似的特征值，假设为 $\hat{\mathbf{v}}$, 然后我们找一个近似的特征值使其满足：

$$ \mathbf{A}\hat{\mathbf{v}}-\hat{\lambda}\hat{\mathbf{v}}\approx 0 $$

上面的定义不是很清晰，一个比较合理的选择是在两边左乘 $\hat{\mathbf{v}}^T$, 然后将约等于改为等于：

$$ \hat{\mathbf{v}}^T\mathbf{A}\hat{\mathbf{v}}-\hat{\mathbf{v}}^T\hat{\mathbf{v}}\hat{\lambda}=0. $$

这种近似方式称为Rayleigh quotient:

$$ \hat{\lambda} = \dfrac{\hat{\mathbf{v}}^T\mathbf{A}\hat{\mathbf{v}}}{\hat{\mathbf{v}}^T\hat{\mathbf{v}}} $$

如果我们动态的更新 $\hat{\mathbf{v}}$ 和 $\hat{\lambda}$ 的话，我们就得到了Rayleigh quotient iteration算法：

$$ \hat{\lambda}_{k+1} = \dfrac{\hat{\mathbf{v}}_{k}^T\mathbf{A}\hat{\mathbf{v}}_{k}}{\hat{\mathbf{v}}_{k}^T\hat{\mathbf{v}}_{k}} $$

$$ \hat{\mathbf{v}}_{k+1} = \dfrac{(\mathbf{A}-\hat{\lambda}_{k+1}\mathbf{I})^{-1}\hat{\mathbf{v}}_k}{\|(\mathbf{A}-\hat{\lambda}_{k+1}\mathbf{I})^{-1}\hat{\mathbf{v}}_k\|}$$

与power iteration算法不同，Rayleigh quotient iteration算法是一个局部二次收敛的算法。



### Finding other eigenvectors

一个比较简单的做法就是将最大特征值对应的特征向量从矩阵 $\mathbf{A}$ 中移去，然后重复求最大特征值，直到我们求得所有的特征值：

1. 用power iteration算法找到最大的特征值及其对应的特征向量： $\lambda_1$ 和 $\mathbf{v}_1$;
2. 令 $\mathbf{A}\gets \mathbf{A} - \lambda_1\mathbf{v}_1\mathbf{v}_1^T$
3. 用power iteration算法找到最大的特征值及其对应的特征向量： $\lambda_2$ 和 $\mathbf{v}_2$;
4. 重复步骤1-3，直到我们求得所有的特征值和特征向量。

### Inverse iteration

如果我们想要求得最小的特征值的话，我们可以通过以下迭代方式：

$$ \mathbf{A}\mathbf{y}_{k+1}=\mathbf{x}_k, \quad \mathbf{x}_{k+1} = \frac{\mathbf{y}_{k+1}}{\|\mathbf{y}_{k+1}\|_{\infty}} $$ 

实际上，我们求得的就是 $\mathbf{A}^{-1}$ 最大的特征值。



# orthogonal iteration

从对power iteration算法的分析来看，当 $\lambda_1$ 和 $\lambda_2$ 非常接近时，算法的收敛速率会很慢。因此，我们就需要改进算法。

orthogonal iteration的基本思想为同时求得 $\lambda_1$ 和 $\lambda_2$ 及这两个特征值对应的子空间。算法可以表示如下：

1. 令 $\mathbf{Q}_0\in\mathbb{R}^{n\times p}$ 的列之间相互正交 ($\mathbf{Q}_0^T\mathbf{Q}_0=\mathbf{I}_p$)
2. 对 $k=1,2,\dots,$ 重复以下过程直到收敛：
3. $\mathbf{Z}_k=\mathbf{A}\mathbf{Q}_{k-1}$
4. $\mathbf{Q}_k\mathbf{R}_k=\mathbf{Z}_k$ (QR factorization)
5. $\lambda(\mathbf{Q}_k\mathbf{A}\mathbf{Q}_k)=\{\lambda_1^{(k)},\dots,\lambda_p^{(k)}\}$

当 $p=1$ 时， 算法就退化为power iteration算法。我们在此略过算法收敛性的证明，其收敛速率近似可以表示为

$$ \mathrm{dist}(D_q(\mathbf{A}), \mathrm{ran}(\mathbf{Q}_k))\approx c\left|\frac{\lambda_{k+1}}{\lambda_{k}}\right|^k $$

其中 $D_q(\mathbf{A})$ 是矩阵 $\mathbf{A}$ 的 $q$ 维 dominant invariant space.



## Implementation
See Github

## Applications
1. PCA
2. PageRank

## Conclusion



## Reference

