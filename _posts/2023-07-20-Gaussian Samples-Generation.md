---
title: Gaussian Samples Generation
date: 2023-07-20
math:  true
img_path: /assets/images/
categories: [math, algorithms]
tags: []
---

This post reviews some classic methods that are used to generate Gaussian samples.

# Background
Before we dive into the topics, we need the following fact:
> Theorem: For a continuous random variable $x$, its *cumulative distribution function* (CDF), which is defined as
>
>  $$ F(x) = P(X\leq x) $$
>  is uniformly distributed, that is, $F(x)\sim \text{uniform}(0,1)$.

To prove this theorem, let $z=F(x)$, then $z\in[0,1]$. Note that $F(x)$ is a continuous and monotonically increasing function, we have:

$$ F_z(z)=P(F(x)\leq z)=P(x\leq F^{-1}(z))=F(F^{-1}(z))=z $$
which indicates that $z$ obeys the uniform distribution.

# Inverse Gaussian CDF
The first way to generate a Gaussian sample is to use the above Theorem directly. 
Note that 

$$ X\sim \mathcal{N}(0,1) \Leftrightarrow P(X\leq x)=\int_{-\infty}^x \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt $$

Since $z=F(x)\sim \text{uniform}(0,1)$, we can first generate a random sample $z$ from the uniform distribution, then we find the corresponding $x$, which is given by $x=F^{-1}(z)$.
Our goal now becomes how to find $F^{-1}(z)$. 

Now, notice that

$$ z = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt = \frac{1}{2} + \int_{0}^x \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt $$

The integral on the right hand side is related to [error function](https://en.wikipedia.org/wiki/Error_function)

$$ \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}}\int_{0}^x e^{-{t^2}}dt $$

thus we have

$$ z = \frac{1}{2} + \frac{1}{2}\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) \Rightarrow x = \sqrt{2}\mathrm{erf}^{-1}(2z-1)  $$

Now we need to find the inverse of error function $\mathrm{erf}^{-1}(x)$.

## Inverse error function
First we note that $\mathrm{erf}(x)$ is an odd function, then $\mathrm{erf}^{-1}(x)$ is also odd([proof](https://proofwiki.org/wiki/Function_is_Odd_Iff_Inverse_is_Odd)).

note that $\mathrm{erf}(\mathrm{erf}^{-1}(x))=x$, taking derivative on both sides leads to
$$ \frac{d\mathrm{erf}(\mathrm{erf}^{-1}(x))}{d\mathrm{erf}^{-1}(x)}\frac{d\mathrm{erf}^{-1}(x)}{dx}=1 \Rightarrow \frac{d\mathrm{erf}^{-1}(x)}{dx}=\frac{1}{\frac{d\mathrm{erf}(\mathrm{erf}^{-1}(x))}{d\mathrm{erf}^{-1}(x)}}=\frac{\sqrt{\pi}}{2}e^{\mathrm{erf}^{-1}(x)^2} $$

if we denote $y=\mathrm{erf}^{-1}(x)$, then 

$$ \frac{dy}{dx} = \frac{\sqrt{\pi}}{2}e^{y^2} $$

and 

$$\frac{d^2y}{dx^2}=\frac{d}{dx}\frac{\sqrt{\pi}}{2}e^{y^2} = \left(\frac{dy}{dx}\right)^22y $$

continuing computation, we can obtain the general formula:

$$ \frac{d^ny}{dx^n}=\left(\frac{dy}{dx}\right)^n P(y)\Rightarrow \frac{d^{n+1}y}{dx^{n+1}}=\left(\frac{dy}{dx}\right)^{n+1}\left(2nyP(y)+\frac{dP}{dy}\right)  $$


Now considers the [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series) of $\mathrm{erf}^{-1}(x)$ at $x=0$, we have

$$ \mathrm{erf}^{-1}(x)=\mathrm{erf}^{-1}(0) + \frac{dy}{dx}\Big\vert_{x=0}\frac{x}{1!}+ \frac{d^3y}{dx^3}\Big\vert_{x=0}\frac{x^3}{3!}+\cdots$$

the first three terms approximation is given by

$$ \mathrm{erf}^{-1}(x)=\frac{\sqrt{\pi}}{2}\left(x + \frac{\pi}{12}x^3+\frac{7\pi^2}{480}x^5\right) $$
the error is bounded by

$$ \frac{127\pi^3}{40320}x^7 $$

## Implementation
Combining thing together, the process is given by
1. Generate a uniform random sample $z\sim\mathrm{uniform}(0,1)$.
2. use Taylor expansion of $\mathrm{erf}^{-1}(x)$ to compute $\mathrm{erf}^{-1}(2z-1)$.
3. Obtain an Gaussian sample $x=\sqrt{2}\mathrm{erf}^{-1}(2z-1)$.
The code is given in [inverse_cdf.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6). The result ($100,000$ samples with $100th$ Taylor expansion) is shown as follows:
![](inverse_cdf.png)

As we can see, the result is very similar to the shape of normal distribution.

## Analysis
Though the method is straightforward, there are something should be notices:
1. The approach always "underestimates" the real normal distribution, as is shown above, the range of generated samples is $[-3,3]$ with $100th$ polynomial approximate, to achieve higher precision, high-order polynomials are required. However, notice that the factorial $n!$ become intractable when $n$ is large, actually when I set `orders=200`, it throws the error: `OverflowError: int too large to convert to float`.
2. The computation of Taylor coefficients is not efficient, though there are some techniques can be used to improve its performance, as $n$ becomes large, numerical problem may arise.

To solve this problems of Taylor expansion, one way is to use the [minimax rational approximation](https://en.wikipedia.org/wiki/Minimax_approximation_algorithm). The core idea is approximating $\mathrm{erf}^{-1}(x)$ universally instead of locally. This approach is adopted by [R language](https://github.com/SurajGupta/r-source/blob/master/src/nmath/qnorm.c).

# Box-Muller Transform
The second approach is called the [Box–Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform). It has two forms. We first introduce the basic form, which is proposed by Box and Muller. Then we introduce the polar form, which simplifies the computation of the basic form.

The idea behind Box-Muller Transform is using two independent uniform samples to generate two independent Gaussian samples.

## Basic form
Let's consider two independent Gaussian random variables $x$ and $y$, it is easy to obtain the joint probability density function:

$$ f(x,y)=f(x)f(y) = \frac{1}{2\pi}e^{-\frac{r^2}{2}} $$

where $r^2:= x^2+y^2$.  We further define $s=\frac{r^2}{2}$, then $r=\sqrt{2s}$, we then have

$$ f(x,y)=\frac{1}{2\pi}e^{-s} $$

Notice that in this form, $f(x,y)$ is simply the product of a uniform random variable $u\sim\mathrm{uniform}(0,2\pi)$ and an exponential random variable $s\sim\mathrm{Exp}(1)$.


Since the CDF of uniform distribution and the exponential distribution are unknown, we can generate two independent Gaussian samples as follows:
1. Generate two uniform samples $U_1,U_2\sim\mathrm{uniform}(0,1)$.
2. Use the inverse CDF of uniform distribution $\theta=2\pi U_1$ to generate a uniform sample $\theta\sim\mathrm{uniform}(0,2\pi)$; similarly, use the inverse CDF of exponential distribution $s=-\ln(U_2)$ (where $1-U_2\sim \mathrm{uniform}(0,1)$).
3. Using the relationship between $x,y$ and $s$ to generate independent Gaussian samples: $x=\sqrt{2s}\cos\theta$, $y=\sqrt{2s}\sin\theta$.

combining these altogether, we have:

$$ U_1,U_2\sim\mathrm{uniform}(0,1)\Rightarrow \begin{cases}x=\sqrt{-2\ln U_1}\cos(2\pi U_2)\sim\mathcal{N}(0,1)\\
y=\sqrt{-2\ln U_1}\sin(2\pi U_2)\sim\mathcal{N}(0,1)
\end{cases} $$

this formula is called the Box-Muller transform. 

The code is given in [box_muller_transform.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6). The result is shown in follows:
![](box_muller.png)

## Polar form
The basic form is straightforward and simple, but its efficiency is not good since it evolves computing $\sin$ and $\cos$ functions.

To improve the performance, the polar form, or the [Marsaglia polar method](https://en.wikipedia.org/wiki/Marsaglia_polar_method) is proposed.

Let $U:=\{(u,v)\mid u^2+v^2\leq1\}$, that is, $U$ is the unite disk, we first generate a uniform sample $(X,Y)\sim \mathrm{uniform}(U)$, this can be done by taking $X,Y\sim\mathrm{uniform}(-1,1)$ and reject the case that $X^2+Y^2>1$.

Now consider $T=x^2+Y^2$, we have

$$ P(T\leq t) = P(X^2+Y^2\leq t)=\frac{\pi*(\sqrt{t})^2}{\pi*1^2}=t $$

which means that $T\sim\mathrm{uniform}(0,1)$, then similar to basic form, we can take $R=\sqrt{-2\ln T}\sim\mathrm{Exp}(1)$. 
Moreover, notice that 

$$\cos\theta=\frac{X}{\sqrt{T}}, \sin\theta=\frac{Y}{\sqrt{T}}$$

we then have that

$$\begin{cases}x=\sqrt{-2\ln T}\frac{X}{\sqrt{T}}\sim\mathcal{N}(0,1)\\
y=\sqrt{-2\ln T}\frac{Y}{\sqrt{T}}\sim\mathcal{N}(0,1)
\end{cases} $$

where $(X,Y)\sim \mathrm{uniform}(U)$ and $X^2+Y^2\leq1$.

The code is given in [box_muller_transform.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6). The result is shown as follows:
![](polar_method.png)


# Central Limit Theorem
According to [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), if $X_1,\dots,X_n$ is a sequence of *i.i.d.* random variables having a distribution with expected value given by $\mu$ and finite variance given by $\sigma^2$, if we define 

$$ \overline{X} = \frac{X_1+\cdots+X_n}{n} $$

then we have

$$ \frac{\overline{X}-\mu}{\sqrt{n}\sigma}\to\mathcal{N}(0,1) $$

if $X_1,\dots,X_n\sim\mathrm{uniform}(0,1)$, then $\mu=\frac12$ and $\sigma^2=1/12$. Thus we can approximate a Gaussian Normal distribution with

$$ \frac{\sqrt{3}(2\overline{X}-1)}{\sqrt{n}} $$

The code is given in [central_limit_theorem.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6#file-central_limit_theorem-py'). The result is shown as follows:
![](central_limit_theorem.png)

The problem of central limit theorem is that it requires to generate a large number of uniform samples to approximate the Gaussian distribution.

# Reject Sampling
Suppose we have two random variables $P\sim f(x)$ and $Q\sim g(x)$, where $f(x)$ and $g(x)$ are probability density function (PDF) of $P$ and $Q$ respectively. 
If $g(x)$ has a simpler form and there exists a constant $k>0$ such that $kg(x)\geq f(x)$ almost everywhere, then 

$$ Q\mid \left(U\leq \frac{f(x)}{kg(x)}\right)\sim f(x) $$

where $U\sim \mathrm{uniform}(0,1)$.  
![](reject_sampling.png)

So the process of reject sampling is:
1. Sample $z_1$ from $g(x)$.
2. Sample $u_1$ from $\mathrm{uniform}(0,1)$.
3. If $u_1\leq\frac{f(z_1)}{kg(z_1)}$, then output (accept) $z_1$; otherwise repeat the above steps.

## Ziggurat algorithm
The problem of reject sampling method is that is relies on the approximation distribution of $g(x)$, if $g(x)$ is far away from $f(x)$, then the sampling process may fail a lot of iterations.

[Ziggurat algorithm](https://en.wikipedia.org/wiki/Ziggurat_algorithm) uses a special kind of $g(x)$ to approximate the Gaussian distribution, where $f(x)=\exp(-x^2/2)$. Notice that $f(x)$ is an even function, we consider its positive part.
![](zigguart_example.png)

We first set $n=8$ points ($n=8$ for clarity, $n=256$ in implementation) $0=x_0<x_1<\cdots<x_{7}$, then we obtain $8$ sets, of which $7$ of them are rectangles and $1$ of them is a bottom strip tailing off to infinity. Now the boundary of rectangles and the  bottom strip tailing of $f(x)$ forms $g(x)$, *i.e.*, 

$$ g(x)=\begin{cases}f(x_i), x\in[x_i,x_{i+1}],i=0,\dots,6\\ f(x), x\geq x_7  \end{cases} $$

The process then goes as follows:
1. Random generate an integer $i\sim\mathrm{uniform}[0,1,\dots,n-1]$ and two uniform samples $U_1,U_2\sim\mathrm{uniform}[0,1]$.
2. If $i\neq n-1$,
	1. If $x=U_1x_i < x_{i-1}$, return $x$.
	2. else  let $y=f(x_i) + U_2[f(x_{i-1})-f(x_i)]$, 
	3. compute $f(x)$, if $y < f(x)$, return $x$.
	4. resampling.
3. Else use fall back strategy.

Step 1 is choosing the rectangle, if the strip trailing is chosen, the fall back strategy is used (step 3); otherwise, we check if the point $(x,y)$ is below  $(x,f(x))$ (step 2.1-2.4).


Now we will answer two questions:
1. How to set $x_i$?
2. What is the fall back strategy when $i=n-1$?

To answer the first question, the strategy is to choose $x_i$ such that the area of each parts are equal.
We first denote $r=x_{n-1}$ ($r=x_7$ in this case), then the area if trailing part is 

$$ v := rf(r) + \int_r^\infty f(x)dx $$

and the area of rectangles are 

$$ x_i[f(x_{i-1})-f(x_i)]=v, i=1,\dots,n-1 $$

notice that $x_0=0$, we can build a set of equations of the variable $r$:

$$ \begin{cases}v = rf(r) + \int_r^\infty f(x)dx\\ x_{i+1}=f^{-1}\left(\frac{v}{x_{i+1}}+f(x_{i+1})\right), i=1,\dots,n-2\\
x_1[1-f(x_1)]=v \end{cases} $$

One way to solve such equations is using the binary search, a reasonable range of $r$ is $(0, 1000)$ (this can be obtained by requiring that $v\geq 0.5/n$ where $0.5$ is the area of the normal distribution when $x\geq0$.). 
The code is given in [zigguart_method.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6#file-zigguart_method-py), the result is given as follows:

| $n$ | $r$ | $v$ | 
| ---|---|---|
|8| 2.2754212368264577| 0.17205641497747212|
|16| 2.628551533516455| 0.08323705176062711|
|32| 2.9249841780865458| 0.04061135850774161|
|64| 3.1846367985555286| 0.019994029894609307|
|128| 3.4187750633592| 0.009906046486301499|
|256| 3.6341135868129495| 0.004927241486946986|

> The result is not accurate, I haven't figured out why, please refer to implementations for details.
{: .prompt-warning }

To answer the second question, note that in this case we have $x>r$, Marsaglia suggests using the follow algorithm:
1. Let $x=-\ln(U_1)/r$ and $y=-\ln(U_2)$
2. If $2y > x^2$ return $r + x$
3. Otherwise, go back to step 1.

This completes the Zigguart sampling strategy.The code of Zigguart sampling is given by [zigguart_sampling.py](https://gist.github.com/MaoSong2022/c9f2e057b0bcc82679cbe067c89decd6#file-zigguart_sampling-py), the result ($10,000$ samples, $n=128$) is shown as follows:
![](zigguart_sampling.png)
It can be seen that the results is very similar to the real normal distribution, which verifies the correctness.

In fact, $99\%$ cases fall into case 1, that is, $x=U_1 x_i < x_{i-1}$, so the computation burden is also acceptable.

# Conclusion
In this blog, we reviewed some classical methods of gene rating Gaussian samples from the uniform samples. It do depends on applications and requirements to choose proper algorithms. 

# References
[How to generate Gaussian samples](https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959)  
[Proof of Marsaglia polar method](https://stats.stackexchange.com/a/146707)
[normaldist-benchmark](https://github.com/miloyip/normaldist-benchmark)  
[numpy ziggurat implementation](https://github.com/numpy/numpy/blob/92b880f64024c0694c69a6397568ce758102f9d8/numpy/random/src/distributions/distributions.c#L137)  
[The Ziggurat Method for Generating Random Variables](https://www.jstatsoft.org/article/view/v005i08)  
[Generating Normally Distributed Values](https://datascience.oneoffcoder.com/generate-gaussian-distributed-values.html)  
