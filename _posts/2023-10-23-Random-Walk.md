---
title: 2023-10-23-Random-Walk
author: Mao Song
date: 2023-10-23
math: true
mermaid: false
img_path: /assets/images/
categories: 
tags:
  - math
  - markov_chain
---

"A drunk man will find his way home, but a drunk bird may get lost forever". This fact indicates that in 2D, the random walk is recurrent; but in 3D, the random walk is transient

# Introduction
To answer this question, we need to use Markov chain to model this problem as a mathematical problem, then we need to analysis the convergence of series. 
So, this blog is divided into three parts: the first part is Markov chain; the second part is mathematical analysis, the third part is the analysis of the random walk.


# Markov Chains
To model a Markov chain, we need to define three things:
1. state space, it indicates where we can go
2. transition function, 
3. initial distribution

## State Space
We first need to use a discrete state space. For simplicity, we write the state space as the integer lattice points, for 2D space, this means 
$$ S = \{(x,y)\in\mathbb{R}^2\mid x\in \mathbb{Z},y\in\mathbb{Z}\} $$
where $\mathbb{R}$ is the set of real numbers and $\mathbf{Z}$ is the set of all integers.


## Transition function
Let $s_{t-1},s_t\in S$ be two states, the transition function defines how we can go from $s_{t-1}$ to $s_t$. 

In 2D case, we define the neighbor states of state $s_t=(i,j)$ as 
$$ N(s_t)  = N((i,j))=\{(x,y)\in S\mid 0<|x-i|+|y-j|\leq 1\} $$
In 3D case, we define the neighbor states of state $s=(i,j,k)$ as
$$ N(s_t) = N((i,j,k))=\{(x,y,z)\in S\mid 0<|x-i|+|y-j|+|z-k|\leq 1\} $$

For simplicity, we assume that state $s$ transits to its neighbor state $t\in N(s)$ with equal probability, we write this probability as $P(s_j=i\mid s_0=i)$:
$$ P(s_{t}=j\mid s_{t-1}=i) = \frac{1}{\#N(i)} $$
in 2D case, this means
$$ P(s_{t}=j\mid s_{t-1}=i) = \frac14 $$
in 3D case, this means 
$$ P(s_{t}=j\mid s_{t-1}=i) = \frac16, j\in N(i) $$

## initial distribution
The initial distribution $p(s_0)$ defines where we start our random walk. For simplicity, we assume that  for a given state $s_0$
$$ p(s) = \begin{cases}1, \text{ if }s=s_0\\
0,\text{otherwise}
\end{cases} $$

## Markov property
this property is the key to reduce the complex computation, the Markov property assume that the next state only depends on the current state, that is:
$$ P(s_{t+1}\mid s_t,s_{t-1,\dots,s_0}) = P(s_{t+1}\mid s_t) $$

# Recurrence and Transience
Now we need to define what does "go home" and "lost" means actually. In Markov chain, we use recurrence and transience to define these two terms. 

> Definition: For $i\in S$, let $r$ be the return probability:
> $$ r=P(s_n=i \text{ for some } n\geq1 \mid s_0=i) $$
> if $m_i=1$, then we say that the state $i$ is **recurrent**; if $m_i<1$, then we say that the state $i$ is **transient**. 

Now the problem is to determine the value of $r$ for 2D and 3D cases.  Computing $r$ directly is difficult, we first define the expected number of returns as:
$$ \mathbb{E}(\# \text{visits to }i\mid s_0=i)=\sum_{j=1}^\infty P(s_j=i\mid s_0=i) $$

We first use the following theorem to make the return probability computable:
> Theorem: 
>  1. If the state $i$ is recurrent, then $\mathbb{E}(\# \text{visits to }i\mid s_0=i)=\infty$, and we return to state $i$ infinitely many times with probability $1$.
>  2. If the state $i$ is transient, then $\mathbb{E}(\# \text{visits to }i\mid s_0=i)<\infty$, and we return to state $i$ infinitely many times with probability $0$.

This theorem is equivalent to say, if we can return to state $i$ with probability $i$, then we will return to state $i$ infinitely many times.

Now, we define the Bernoulli random variable $p_{ii}^{(j)}$ to indicates if we return to the state $i$ in $j$-th step, that is,
$$ p_{ii}^{(j)}=\begin{cases} 1,\text{if we didn't return in } j \text{-th step}\\ 0, \text{ otherwise} \end{cases} $$
then we can rewrite the expectation value as:
$$ \mathbb{E}(\# \text{visits to }i\mid s_0=i)=\sum_{j=0}^\infty p_{ii}^{(j)}  $$
So, this gives us a second way to determine whether the random variable is recurrent or transient:
1. If the series $\sum_{j=0}^\infty p_{ii}^{(j)}$ diverges, then the state $i$ is recurrent
2. If the series $\sum_{j=0}^\infty p_{ii}^{(j)}$ converges, then the state $i$ is transient


# Random Walk
Now we come back to random walk problem, we need to compute $p_{ii}^{(j)}$.

Notice that for any state $i$, the state space is symmetric in four directions, that is, if we go in one direction for $k$ steps, then we need another $k$ steps in reverse direction to return to the current state. So, it is easily to see that if state $i$ is recurrent, then  $p_{ii}^{(j)}=0$ when $j$ is odd, then we can write
$$ \sum_{j=0}^\infty p_{ii}^{(j)}=\sum_{j=1}^\infty p_{ii}^{(2j)}  $$
when state $i$ is recurrent.

Now consider the 2D case, we have four direction: up, down, left, right, we must have the equation: $\#$ steps goes left $=\#$ steps goes right and  $\#$ steps goes up $=\#$ steps goes down. We can write 
$$ p_{ii}^{(2j)} = \frac{1}{4^{2j}}\sum_{n=0}^{j} \binom{(2j)!}{n!n!(j-n)!(j-n)!} $$
where $4^{2j}$ is the total number of paths we can go in $2j$ steps. the second combinatorial term is equivalent to say that goes in one direction for $n$ steps, and $n$ steps in reverse of this direction; then we goes in second direction for $j-n$ steps and goes back.

Similarly, we can derive the formula for 3D case:
$$ p_{ii}^{(2j)} = \frac{1}{6^{2j}}\sum_{n\geq0,m\geq0}^{n+m\leq j} \binom{(2j)!}{n!n!m!m!(j-n-m)!(j-n-m)!}  $$

Note that we can simplify the combinatorial form as:
$$ \binom{(2j)!}{n!n!(j-n)!(j-n)!}=\frac{(2j)!}{(n!)^2((j-n)!)^2}=\binom{2j}{j}\binom{j}{n}\binom{j}{j-n} $$
this leads to 
$$ p_{ii}^{(2j)} = \frac{1}{4^{2j}}\binom{2j}{j}\sum_{n=0}^{j}\binom{j}{n}\binom{j}{j-n} $$
the second sum can be further simplified: we go in first direction (e.g., up) for $n$ steps, and  we go in second direction (e.g., left) for $j-n$ steps, and counting the total number of paths, this is equivalent to say the number of paths we go for $j$ steps, which is exactly $\binom{2j}{j}$. So, we have
$$\begin{align}p_{ii}^{(2j)} &= \frac{1}{4^{2j}}\binom{2j}{j}\binom{2j}{j}\\
&=\frac{1}{4^{2j}}\left(\frac{(2j)!}{(j!)^2}\right)^2\\ 
&\sim \frac{1}{4^{2j}}\left(\frac{\sqrt{4\pi j}(2j/e)^{2j}}{2\pi j(j/e)^{2j}}\right)^2\\
&= \frac{1}{\pi j}(1+o(1))\end{align}  $$
where we use the [Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation).

Thus we conclude that the state $i$ in 2D random walk is recurrent since 
$$\sum_{j=0}^\infty p_{ii}^{(j)} = \sum_{j=0}^\infty p_{ii}^{(2j)}\sim \sum_{j=0}^\infty \frac{1}{\pi j}=\infty $$

Similarly, we can derive that in 3D random walk,
$$ p_{ii}^{(2j)} \sim \left(\frac{1}{\pi j}\right)^{3/2}(1+o(1)) $$
Thus the state $i$ in 3D random walk is  transient since
$$ \sum_{j=0}^\infty p_{ii}^{(j)} = \sum_{j=0}^\infty p_{ii}^{(2j)}\sim \sum_{j=0}^\infty \frac{1}{\pi j^{3/2}}<\infty  $$


# Reference
[Random walks in 2D and 3D are fundamentally different](https://www.youtube.com/watch?v=iH2kATv49rc)
[Recurrence and transience](https://mpaldridge.github.io/math2750/S09-recurrence-transience.html)
[Two-Dimensional Random Walk](https://www.ime.unicamp.br/~popov/2srw.pdf)

