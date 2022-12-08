---
title: Secretary Problems
date: 2022-12-08
description: notes on secretary problems
math: true
img_path: /assets/images/
categories: [learning notes]
tags: [algorithm]
---

# Introduction
假设我们是一个经理，我们想招聘一位秘书。
现在假设一共有 $n$ 位候选人，我们希望每次招聘的秘书，都比上一位秘书更优秀。我们用 $a_i$ 来评价这个优秀程度， $a_i$ 越大代表候选人越优秀。

由于招聘不是我们的强项，因此我们找一位猎头帮我们对候选人进行面试。猎头满足如下条件：
1. 每天都会给我们面试一位秘书候选人
2. 每面试一次，收取我们手续费 $c_i$

假设雇佣一位秘书的成本是 $c_h$，一般来说，雇佣秘书的成本远高于面试成本, 即 $c_h >> c_i$.
现在我们希望能够花费最小的代价来找到最优秀的秘书（我们假设除了手续费和雇佣成本之外没有其他开支）。问：如何分析这个最小总代价？

# Primary analyais
我们首先可以写出这个问题的求解算法：
```
HIRE-SCECRETARY(n)
best = 0 // dummy candidate
for i = 1 to n
	interview candidate i
	if a[i] > a[best]
		best = i
		hire candidate i
```
我们假设一共雇佣了 $m$ 位秘书，现在 $m$ 是未知的，并且满足 $1\leq m \leq n$. 
则根据问题介绍，我们总代价为 $O(c_in+c_hm)$, 由于面试成本总是不变的，一直是 $c_in$, 因此变化的地方在于雇佣代价 $O(c_hm)$.

## Wost case analysis
由于 $m \leq n$, 因此雇佣代价一个显然的下界是 $O(c_hn)$. 该下界出现的条件为候选人按照优秀程度严格升序的次序进行面试，即 $a_1<a_2<\cdots< a_n$.  
但是事实上，候选人出现的次序可能不受控制，即候选人的优秀程度排序是“随机”的，因此，我们需要关注一般场景下的雇佣代价。

## Probabilistic analysis
根据前面的分析，我们可以给根据 $a_i$ 给所有候选人进行排名，得到一个排名序列 
$$ \langle rank(1),\dots, rank(n)\rangle $$
其中 $rank(i)$ 代表了第 $i$ 名候选人在所有 $n$ 名候选人中的分数，分数越高，候选人越优秀。
这个排名序列是
$$ \langle 1,\dots,n \rangle $$
的一个置换(permutation). 由于一共有 $n!$ 种可能的置换，因此这里“随机”代表均匀分布，即 $n!$ 种置换出现的概率是相等的。

在进行分析之前，我们首先引入指示变量的定义。
>给定一个事件空间 $S$ 和一个事件 $A$ , 我们将事件 $A$ 的*指示变量*(indicator random variable) $I\{A\}$ 记为
$$ I\{A\} = \begin{cases} 1, &\text{if }A\text{ occurs}\\ 0, &\text{otherwise} \end{cases} $$

我们记事件 $X_i$ 为前 $i$ 位候选者中，第 $i$ 位候选者被雇佣这个事件的指示变量，则总的雇佣次数为
$$ X=X_1+\cdots+X_n $$
我们想要计算随机变量 $X$ 的期望 $\mathbb{E}[X]$.
由于第 $i$ 位候选者被雇佣的必要条件为第 $i$ 位候选者比前 $i-1$ 位候选者要优秀，而由均匀分布的定义，每个候选者出现在第 $i$ 位的概率是相等的，因此第 $i$ 位候选者排名第 $i$ 位的概率是 $1/i$， 即：
$$ \mathbb{E}[X_i] = \frac{1}{i}, i=1,\dots,n. $$
因此，由期望的线性性质，我们有：
$$ \mathbb{E}[X] = \mathbb{E}\left[\sum_{i=1}^nX_i\right]=\sum_{i=1}^n\mathbb{E}[X_i]=\sum_{i=1}^n\frac{1}{i}=\ln n + O(1) $$
其中最后一个等式我们使用了调和级数(harmonic series)的[性质](https://en.wikipedia.org/wiki/Harmonic_series_(mathematics))。我们将结论总结为以下引理：
> 引理：假设所有候选者出现的次序是随机的，则算法`HIRE-SCECRETARY(n)` 在平均意义下的雇佣总代价为 $O(c_h\ln n)$.



# Random algorithm
可以看到，我们前述的分析假设了“候选者出现的次序是随机的”， 但是实际上的输入可能并不是随机输入。
我们需要在算法中对输入进行随机化处理，即人为的实现随机化输入。因此，我们就有了如下的随机化算法：
```
RANDOMIZED-HIRE-SCECRETARY(n)
randomly permute the list of candidates
best = 0 // dummy candidate
for i = 1 to n
	interview candidate i
	if a[i] > a[best]
		best = i
		hire candidate i
```
通过转换之后，我们可以得到和之前一样的结果：
> 引理： 算法`RANDOMIZED-HIRE-SCECRETARY(n)`的期望雇佣总代价为$O(c_h\ln n)$. 

### Randomly permuting arrays
可以看到，为了实现的随机化，我们对输入的数组进行了一次随机置换。我们在本小节介绍如何实现随机置换。
第一个算法的实现方式如下：
```
PERMUTE-BY_SORTING(A)
n = A.length
let P[1, ..., n] be a new array
for i = 1 to n
	P[i] = RANDOM(1, n^3)
sort A, using P as sort keys
```
这里我们给`P[i]`随机赋值$1$到$n^3$之间的一个数，选择这个范围是为了让数组`P`的元素尽可能唯一。我们有如下结果
> 引理： 由算法`PERMUTE-BY_SORTING(A)`产生的数组`P`中，所有元素都唯一（即$P[i]\neq P[j]$,  $i\neq j$）的概率至少为$1-\frac{1}{n}$.
> 证明：我们记$X_i$为事件“数组P第i个元素是唯一的”的指示变量，那么”所有元素都唯一“这个事件的指示变量$X$可以记为：
> 
> $$ X=X_1\cap X_2\cap\cdots\cap X_n $$
因此，我们有：
$$ \begin{aligned} Pr\{X\} &=Pr\{X_1\cap X_2\cap\cdots\cap X_n\}\\ &= Pr\{X_1\}Pr\{X_2\mid X_1\}\cdots Pr\{X_n\mid X_{n-1}\cap\cdots\cap X_{1}\}\\ &=\frac{n^3}{n^3}\frac{n^3-1}{n^3}\cdots\frac{n^3-(n-1)}{n^3}\\ &\geq \frac{n^3-n}{n^3}\frac{n^3-n}{n^3}\cdots\frac{n^3-n}{n^3}\\ &=\left(1-\frac{1}{n^2}\right)^{n-1}\geq1-\frac{n-1}{n^3}\\&\geq 1-\frac{1}{n} \end{aligned} $$

这里我们利用了不等式$(1-a)(1-b)>(1-a-b)$, $a,b\geq0$. 

> 如果确实出现了元素重复的情况，我们可以重新调用一次`PERMUTE-BY-SORTING`.
{: .prompt-info }

第二个算法不需要使用额外空间，其算法如下：
```
RANDOMIZE-IN-PLACE(A)
n = A.length
for i = 1 to n
	swap A[i] with A[RANDOM(i, n)]
```
我们有以下理论保证
> 定理：算法`RANDOMIZE-IN-PLACE(A)`产生的随机排列服从均匀分布。

我们在这里忽略这个定理的证明。


# Online scecretary problem
我们最后来看一个秘书问题的一个变形。
假设我们现在希望减少面试成本，即不希望面试所有的候选人，而是选出候选人中相对较好的一位。现在我们增加两个要求：
- 我们只能雇佣一次。
- 每次面试结束之后，我们必须马上告诉面试者结果，即我们不能在多位候选者中做选择。
现在的问题是：如何在最小化面试次数和最大化候选者的优秀程度之间做出平衡？

我们首先对这个问题进行建模。在面试一位候选人之后，我们能够给每位候选者一个分数`score(i)`, 不妨假设没有两位候选人没有相同的分数。在面试过`k`位候选者之后，我们知道这`k`位候选者哪一位的分数最高，但是我们不知道在剩余的`n-k`位候选者会不会有更优秀的（分数更高）的候选者。
我们采取如下策略：
- 选择一个正整数`k<n`, 我们面试然后拒绝前`k`位候选者，记录前`k`位候选人的最高分数`bestscore`
- 面试剩下的`n-k`位候选者，选择第一个比`bestscore`更高的候选人，停止面试
- 如果最优秀的候选人在前`k`位中，雇佣第`n`位候选人
算法的伪代码如下：
```
bestscore = -inf
for i = 1 to k
	if score(i) > bestscore
		bestscore = score(i)
for i = k + 1 to n
	if score(i) > bestscore
		return i
return n
```

现在我们需要确定`k`的值，使得雇佣最好的候选人的概率尽可能高。
我们首先固定`k`， 然后令

$$ M(j)=\max_{1\leq i\leq j} \mathrm{score}(i) $$

表示前`k`位候选人中的最高分数。
我们令$S$表示“我们成功雇佣最优秀的候选人”这个事件，令$S_i$表示“第i位候选人是最优秀的，且我们成功雇佣第i位候选人”这个事件。显然，当$i\neq j$时，$S_i\cap S_j=\emptyset$. 因此：

$$ Pr\{S\} = \sum_{i=1}^nPr\{S_i\}=\sum_{i=k+1}^nPr\{S_i\} $$

这里第二个等式是因为我们不会雇佣前`k`位候选人，因此$Pr\{S_i\}=0$, $i=1,\dots,k$.
注意$S_i$发生的必要条件为：
- 第i位候选人是最优秀的，我们用事件$B_i$来表示
- 我们不能雇佣$k+1\sim i-1$ 其中任何一位候选人，我们用时间$O_i$来表示
我们可以证明上述条件也是$S_i$发生的充分条件。因此，$S_i=B_i\cap O_i$
显然事件$B_i$和事件$O_i$是不相交的，因此：

$$ Pr\{S_i\}=Pr\{B_i\cap O_i\}=Pr\{B_i\}Pr\{O_i\} $$

我们显然有$Pr\{B_i\}=1/n$, 这是因为最优秀的候选人可能是`n`位候选人中的任意一位。
为了计算$Pr\{O_i\}$, 注意到这等价于前$i-1$位候选人的`score`的最大值在前`k`位候选人中，即

$$ M(i-1) = \max_{1\leq j\leq i-1} \mathrm{score}(j) = \max_{1\leq j \leq k} \mathrm{score}(j)  $$

因此，我们有：$Pr\{O_i\} = k/(i-1)$.  总结起来，就是

$$ Pr\{S\} = \sum_{i=k+1}^nPr\{S_i\}=\sum_{i=k+1}^n\frac{k}{n(i-1)}=\frac{k}{n}\sum_{i=k}^{n-1}\frac{1}{i} $$

我们可以用积分来近似这个求和公式：

$$ \int_{k}^n\frac{1}{x}dx\leq \sum_{i=k}^{n-1}\frac{1}{i}\leq \int_{k-1}^{n-1}\frac{1}{x}dx $$

即

$$ \frac{k}{n}(\ln n-\ln k)\leq Pr\{S\} \leq \frac{k}{n}(\ln(n-1)-\ln(k-1))$$

因为我们希望$Pr\{S\}$尽可能大，因此我们对其下界进行最大化。经过计算，我们得到当

$$ k = \frac{n}{e} $$

时，概率下界达到最大值$1/e$, 因此，当我们选择$k=n/e$时，我们有至少$1/e$的概率能够找到最优秀的候选人。
