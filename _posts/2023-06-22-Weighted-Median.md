---
title: Weighted Median of Data
date: 2023-06-22
math:  true
mermaid: false
img_path: /assets/images/
categories: []
tags: [math, algorithm]
---

This post is to show median and Weighted median of data.

## Weighted-Median of Data
---
# Problem definition
假如我们要为一家大公司的某项商品制定定价策略，但是这家大公司在很多地方都有分店，且每个分店的价格都不一样。为了管理方便，我们希望制定一个统一的价格方便管理。在对价格进行调整时，我们需要额外支出一部分费用以补贴分店，分店降价时，我们补贴其利润；分店涨价时，我们补充其流量。现在的问题是，我们如何制定定价策略，从而使得价格统一之后，我们给分店的补贴最小呢？

我们可以将这个问题抽象为下面的问题：
给定两个包含正整数的数组`nums` 和 `costs`, 每次我们可以对`nums`中的一个元素`nums[i]`执行一次操作，这个操作使得`nums[i]`增加`1`或者减少`1`， 即`nums[i]=nums[i]+1`或`nums[i]=nums[i]-1`. 对元素`nums[i]`的一次操作需要花费`costs[i]` 。如何使得总的花费最小，而使得`nums`的所有元素经过若干次操作之后均相等？


# Method
我们定义如下函数：

$$ f(x) = \sum_{i=1}^nw_i|x_i-x| $$

其中，$x_i$ 和 $w_i\geq0$ 分别代表`nums[i]`和`costs[i]`.  我们希望找到 $f(x)$ 的最小值。

> Property 1: $f(x)$ 是一个凸函数，因而 $f(x)$ 存在唯一的最小值。

性质1非常容易证明，因为函数 $|x-x_i|$ 是一个凸函数， 而 $f(x)$ 是若干个凸函数的非负凸组合，因此 $f(x)$ 也是一个凸函数。

我们将 $x_1,\dots,x_n$ 由小到大排序后的结果记为：

$$ x_{(1)} \leq x_{(2)} \leq\cdots \leq x_{(n)} $$

则我们进一步有以下结论：

> Property 2: $f(x)$ 最小值在区间 $\[x_ {(1)}, x_ {(n)}\]$ 中取得。

性质2可以通过简单的运算得到，假如 $x \leq x_{(1)}$, 则

$$ f(x)= \sum_{i=1}^nw_i(x_i-x)=\sum_{i=1}^nw_i(x_i-x_{(1)})+\sum_{i=1}^nw_i(x_{(1)}-x)\geq f(x_{(1)}). $$

同理可证当 $x\geq x_{(n)}$ 时，我们有 $f(x) \geq f(x_{(n)})$.

接下来，我们证明以下定理，定理表明 $f(x)$ 的最小值就是 $x_1,\dots,x_n$, $w_1,\dots,w_n$的带权中位数(weighted median). 

> Theorem: 定义如下集合：
> 
>  $$ \mathcal{L}=\left\{j\mid \sum_{i=1}^jw_{(i)}\leq \frac{W}{2} \right\}, W=\sum_{i=1}^nw_i $$
> 
>  则：
>  1. 如果 $\mathcal{L}=\emptyset$, 则 $\arg\min f(x)=x_{(1)}$.
>  2. 如果 $\mathcal{L}\neq \emptyset$, 令 $k=\max \mathcal{L}$, 则
> 	 1. 如果 $\sum_{i=1}^kw_{(i)}\neq \frac{W}{2}$, 则 $\arg\min f(x)=x_{(k+1)}$.
> 	 2. 如果 $\sum_{i=1}^kw_{(i)}= \frac{W}{2}$, 则 $\arg\min f(x)=(1-\lambda)x_{(k)}+\lambda x_{(k+1)}$, $\lambda\in[0,1]$.

首先，我们知道 $f(x)$ 是一个分段线性函数，其断点为 $x_{(1)},\dots,x_{(n)}$. 我们将 $\kappa_i$, $i\in\{0,\dots,n\}$ 分别定义为以下区间上的斜率：

$$ (-\infty, x_{(1)}], \[x_{(1)}, x_{(2)}\],\dots,[x_{(n)}, \infty) $$

我们很容易得到：

$$ \kappa_0=-W,\kappa_m=W, \kappa_j=2\sum_{i=1}^jw_{(i)}-W, j\in\{1,\dots,n-1\} $$

如果 $\mathcal{L}=\emptyset$, 那么 

$$ \sum_{i=1}^jw_{(i)}>W, \forall j\in\{1,\dots,n\} $$

从而 $\kappa_0\<0\<\kappa_j, j\in\{1,\dots,n\}$, 即 $f(x)$ 在区间 $(-\infty, x_{(1)}]$ 上单调下降，在区间 $[x_{(1)}, \infty)$ 上单调上升，因此 $f(x)$ 的最小值在 $x=x_{(1)}$ 处取得。

当 $\mathcal{L}\neq \emptyset$, 我们有：$\kappa_k\leq 0$,  并且 $\kappa_j\geq0, \forall j\in\{k+1,\dots,n\}$. 
1. 当 $\kappa_v\neq0$ 时，$f(x)$ 在区间 $(-\infty, x_{(k+1)}]$ 上单调下降，在区间 $[x_{(k+1)}, \infty)$ 上单调上升，因此 $f(x)$ 的最小值在 $x=x_{(k+1)}$ 处取得。
2. 当 $\kappa_v=0$ 时，$f(x)$ 在区间 $\[x_{(k)}, x_{(k+1)}\]$ 上是一个常数，因此 $f(x)$ 的最小值在$\[x_{(k)}, x_{(k+1)}\]$ 上任一点达到最小值。
这样，我们就完成了定理的证明。

我们有一个比较简单的推论
> Remark : 当 $w_1=\dots=w_n$, $f(x)$ 的最小值恰好就是 $x_1,\dots,x_n$ 的中位数。


# Implementation
```c++
#include <algorithm>
#include <vector>

using namespace std;

/**
 * @brief compute the minimum cost to make the elements in an array euqal.
 *
 * @param nums arrays to be opterated on
 * @param costs operation cost of corresponding element in nums
 * @return int minimum tototal cost
 */
int MinCostToMakeArrayEqual(const vector<int> &nums, const vector<int> &costs) {
  int n = nums.size();
  vector<pair<int, int>> data;  // (num, cost)
  int sum_weights = 0;
  for (int i = 0; i < n; ++i) {
    data.emplace_back(nums[i], costs[i]);
    sum_weights += costs[i];
  }
  sort(data.begin(), data.end());

  // find the weighted median
  int median = 0;
  int index = 0;
  int current_sum_weights = 0;
  while (index < n && current_sum_weights < (sum_weights + 1) / 2) {
    current_sum_weights += data[index].second;
    median = data[index].first;
    ++index;
  }

  // compute the total cost
  int total_cost = 0;
  for (int i = 0; i < n; ++i) {
    total_cost += abs(nums[i] - median) * costs[i];
  }
  return total_cost;
}```

# References
[Leetcode minimum cost to make array equal](https://leetcode.com/problems/minimum-cost-to-make-array-equal/)
[The Component Weighted Median Absolute Deviations Problem](https://www.ejpam.com/index.php/ejpam/article/view/3808/927)

