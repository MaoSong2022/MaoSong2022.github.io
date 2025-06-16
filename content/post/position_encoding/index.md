---
title: Position encoding总结
description: 从Absolute position encoding到RoPE
date: 2025-05-19 10:46:39+0800
tags: 
    - position encoding
categories:
    - LLM 
---

> 本文前半部分参考【参考文献1】，推荐大家看博客原文。

# Introduction

在[上一篇blog](https://maosong.website/p/%E5%85%B3%E4%BA%8Eattention-bias%E7%9A%84%E4%B8%80%E4%BA%9B%E6%80%9D%E8%80%83/)中, 我们介绍了Attention的两个性质，也就是在不加position encoding的情况下，Attention对于query是permutation equivariant的，对于key和value是permutation invariant的。
比如说，“我爱你”和“你爱我”这两句话所表示的含义应该是不一样的，但是我们将这两句话作为key和value的时候，我们发现模型的输出是一致的。
这显然是不能接受的，因此，我们就需要加入position encoding，让模型学习到语序讯息，从而明白不同的语序有不同的含义。

下面是测试代码【参考文献1】

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

model_id = "meta-llama/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "The dog chased another dog"
tokens = tok(text, return_tensors="pt")["input_ids"]
embeddings = model.embed_tokens(tokens)
hdim = embeddings.shape[-1]

W_q = nn.Linear(hdim, hdim, bias=False)
W_k = nn.Linear(hdim, hdim, bias=False)
W_v = nn.Linear(hdim, hdim, bias=False)
mha = nn.MultiheadAttention(embed_dim=hdim, num_heads=4, batch_first=True)

with torch.no_grad():
    for param in mha.parameters():
        nn.init.normal_(param, std=0.1) # Initialize weights to be non-negligible

output, _ = mha(W_q(embeddings), W_k(embeddings), W_v(embeddings))

dog1_out = output[0, 2]
dog2_out = output[0, 5]
print(f"Dog output identical?: {torch.allclose(dog1_out, dog2_out, atol=1e-6)}") #True
```

Position encoding可以分为绝对位置编码(absolute position encoding, APE)，相对位置编码(relative position encoding, RPE)以及可学习的位置编码。可学习位置编码主要是BERT类的模型在使用，其训练成本比较高，本文不做讨论。绝对位置编码是transformer里提出的编码模式，但是现在的大多数模型使用的都是相对位置编码。

本文中，我们先介绍位置编码应该具有的性质，然后我们分别介绍绝对位置编码和相对位置编码，我们将着重关注苏剑林老师提出来的RoPE。最后，我们将简单介绍一下LLaMA4使用的NoPE和Qwen系列使用的YARN

# 位置编码

在介绍位置编码之前，我们首先应该关注位置编码的性质，位置编码的目标是为输入的token embedding增加位置信息，那么理想的位置编码应该是怎么样的呢？

我们这里直接引用【参考文献1】中给定的性质：

1. **性质1**: token sequence中每个位置的位置编码都是唯一的。这个很好理解，如果不唯一的话，那么根据前面推导的性质，这两个位置的attention输出就完全一致了
2. **性质2**: 线性相关性。也就是说，如果我们知道了位置$p$处的位置编码，那么理想情况下，我们应该能比较简单地得到$p+k$处的位置编码，理想情况下，我们应该有 $PE(p+k)=W_kPE(p)$.
3. **性质3**: 泛化到长上下文中去。我们希望位置编码不仅在8K的上下文起作用，还希望位置编码能够泛化到32K的上下文
4. **性质4**: 生成模式是固定的。固定的模式有助于模型更好地学习位置相关的信息
5. **性质5**: 可以扩展到多维。我们希望位置编码可以从文本扩展到图片再到视频，也就是从$1D$到$nD$.

# 绝对位置编码

绝对位置编码依照其名称，其思想就是为每个位置的token分配一个固定的位置信息，也就是对于输入的hidden states $\bm{x}=[\bm{x}_1,\dots,\ \bm{x}_m]\in\mathbb{R}^{d\times m}$, 我们有

$$
\bm{x}_i' = \bm{x}_i + p_i, i=1,\dots, m
$$

这里，$p_i\in\mathbb{R}^d$. 我们的attention就变成了

$$
\mathrm{Attn}(X) = (V+P)\mathrm{softmax}\left(\frac{(K+P)^T(Q+P)}{\sqrt{d}}\right)
$$

这里

$$
P = [p_1,\dots,p_m]\in\mathbb{R}^{d\times m}， Q= W_QX, K=W_KX, V=W_VX
$$

## 整数位置编码

一个最简单的想法就是我们使用正整数来标记token所在的位置，也就是
$$
PE(i) = [i, \dots, i]=i\mathbf{1}_{d\times 1}\in\mathbb{R}^d,\ i=1,\dots,m
$$

可以看到，这个简单的设计满足性质1，性质2，性质3，性质4.

但是，注意到attention的输入$X$通常是经过Layer Normalization处理过后的，因此其按列符合正态分布，并且均值和方差一般较小。当我们加上整数位置编码之后，其token本身的信息就会被污染，也就是信噪比非常低(信噪比比较低)。一个解决方法就是我们对$PE(i)$进行normalization，也就是

$$
PE(i)' = \frac{1}{m}PE(i) = \frac{i}{m}\mathbf{1}_{d\times 1}
$$

现在所有的位置编码的值都比较小，但是我们发现新的位置编码不满足性质2了，这是因为现在位置编码还和sequence长度有关，我们从位置$p$到位置$p+k$不仅取决于$k$还取决于sequence长度$m$

## 二进制位置编码

既然整数位置编码的主要问题是对输入影响太大，我们能否找一个不影响输入的整数位置编码方式呢？【参考文献1】提出了二进制位置编码，因为每个token是$d$维的，因此我们可以使用二进制来表示$i$. 比如说，当$d=3$, $m=3$时，我们的位置编码分别为

$$
PE(0) =p_{(000)_2} = [0, 0, 0],\  PE(1) =p_{(001)_2}= [0, 0, 1],\  PE(2) =p_{(010)_2} = [0, 1, 0]
$$

现在，我们二进制位置编码满足性质1，性质2. 对于性质3，由于$d$位二进制的表示范围为 $[0, 2^d-1]$，因此其泛化性受到$d$的影响。

我们还发现，二进制位置编码高位，也就是$PE(i)_{0}$的变化很慢，而低位，也就是$PE(i)_{d}$变化很快，【参考文献1】画出了不同位置的值的变化情况。

二进制位置编码解决了整数位置编码的信噪比过低和线性相关性。但是其问题是其对不同位置的token embedding产生的影响是不一样的。比如位置1和位置2的相同的token embedding之间的区别是：

$$
(\bm{x}_2 + PE(2)) - (\bm{x}_1 + PE(1)) = (\bm{x}_2-\bm{x}_1)+ [0, 1, -1]
$$

一般来说, $\bm{x}_2-\bm{x}_1$比较小，因此使用二进制位置编码的问题是输入位置的微小变化（增加一个token或减少一个token）都会对最终结果产生巨大影响。因此，我们需要想办法解决这个问题

## Sinusoidal

前面提到二进制位置编码的问题是相邻token之间变化太大，不够光滑。因此我们想要增加一个光滑性质，也就是说我们希望：

1. 位置编码值在 $[-1, 1]$之间，防止对token embedding产生影响
2. 相邻token的位置编码尽可能相近，即 $|PE(k+p)-PE(p)| \leq \delta |k|$, 其中 $\delta>0$是一个比较小的数。
3. 与二进制一样，高位的变化比较慢，低位的变化比较快

一个想法就是利用三角函数$\sin$或者$\cos$，三角函数满足前两个性质， 对于第三个性质，我们可以通过控制频率来满足。这样我们得到的位置编码就具有如下形式：

$$
PE(p, i) = \sin\left(\frac{p}{\theta^{i/d}}\right)
$$

其中 $\theta$是我们的超参数。

我们现在来推到一下上面位置编码的线性相关性，我们有

$$
PE(p+k) = \sin\left(\frac{p+k}{\theta^{i/d}}\right)=PE(p)\cos\left(\frac{k}{\theta^{i/d}}\right) + \cos\left(\frac{p}{\theta^{i/d}}\right)\sin\left(\frac{k}{\theta^{i/d}}\right)
$$

我们发现，$\sin$位置编码不满足线性相关性。但是出现的 $\cos$ 给了我们启发，也就是我们可以同时使用 $\sin$ 或者 $\cos$ 来完成位置编码，这也是原始transformer里提出来的Sinusoidal位置编码，其形式为：

$$
\begin{aligned}
    PE(p, 2i) &= \sin\left(\frac{p}{\theta^{2i/d}}\right)\\
    PE(p, 2i+1) &= \cos\left(\frac{p}{\theta^{2i/d}}\right)
\end{aligned}
$$

现在，我们再推导一下线性相关性，我们记 $\omega_i=1/\theta^{2i/d}$ 就得到：

$$
\begin{aligned}
    \begin{bmatrix}
        PE(p+k, 2i)\\
        PE(p+k, 2i+1)\\
    \end{bmatrix}&=\begin{bmatrix}
        \sin \omega_i(p+k)\\
        \cos \omega_i(p+k)
    \end{bmatrix}\\
    &=\begin{bmatrix}
        \sin \omega_i(\omega_ip)\cos(\omega_ik)+\cos w_i(\omega_ip)\sin(\omega_ik)\\
        \cos \omega_i(\omega_ip)\cos(\omega_ik)-\sin w_i(\omega_ip)\sin(\omega_ik)
    \end{bmatrix}\\
    &= \begin{bmatrix}
        \cos(\omega_ik)& \sin(\omega_ik)\\
        -\sin(\omega_ik)& \cos(\omega_ik)
    \end{bmatrix}\begin{bmatrix}
        PE(p, 2i)\\
        PE(p, 2i+1)\\
    \end{bmatrix}
\end{aligned}
$$

也就是说，Sinusoidal位置编码满足线性相关性。

# 相对位置编码

前面介绍了绝对位置编码，每个位置的位置编码是固定的。与之相对的，我们自然会想到，是否存在相对位置编码？与绝对位置编码相比，相对位置编码更能够体现上下文之间的联系。
但是，一个问题就是，绝对位置编码也蕴含了相对信息，我们如何定义相对位置编码？

# RoPE

RoPE由苏剑林老师提出，最早应用于LLaMA架构，后续被大多数模型所采用。

之前的PE大多数关注于加性位置编码，也就是**假设位置编码的形式为 $\bm{x}+\bm{p}$**, 基于这种假设，已有的工作基本都集中于优化下面的Q和K的内积

$$
\langle W_Q\bm{x}_m+\bm{p}_m, W_K\bm{x}_n + \bm{p}_n\rangle
$$

这里 $f_q(\bm{x}_m, m)=W_q\bm{x}_m+\bm{q}_m$, $f_q(\bm{x}_n, n)=W_q\bm{x}_n+\bm{q}_n$

而RoPE里面，作者使用了一个不同的假设，也就是**假设内积应该仅包含两者的相对信息**，也就是

$$
\langle f_q(\bm{x}_m, m), f_q(\bm{x}_n, n)\rangle := g(\bm{x}_m,\bm{x}_n, m-n)
$$

这里的 $f$ 和 $g$都是未知函数。我们的目标就是从这个公式中推导出一个合适的位置编码出来。

不失一般性，我们可以假设

$$
f_q(\bm{x}_m,0) = \bm{x}_m,\quad  f_q(\bm{x}_n, 0) = \bm{x}_n
$$

这个假设代表初始条件下，我们不对输入做任何改变，也就是不增加位置信息。

## 2D推导

RoPE直接使用复平面来进行推导，实际上在实数平面上也是一样的。我们假设 $d=2$, 注意到二维平面上的每个点都可以表示为如下形式

$$
\bm{z} = (x_1,x_2) = (r\cos\theta,r\sin\theta)
$$
其中 ($\mathrm{atan2}$ 定义参考[维基百科](https://en.wikipedia.org/wiki/Polar_coordinate_system))
$$
r = \|\bm{z}\|_2 = \sqrt{x_1^2+x_2^2},\quad  \theta = \mathrm{atan2}(y, x)
$$

现在，对于三个向量 $f_q(\bm{x}_m, m)$, $f_q(\bm{x}_n, n)$, $g(\bm{x}_m,\bm{x}_n, m-n)$ 我们可以写出其极坐标形式：

$$
\begin{aligned}
    f_q(\bm{x}_m,m) &:= (r_1\cos\theta_1, r_1\sin\theta_1)\\
    f_q(\bm{x}_n, n) &:= (r_2\cos\theta_2, r_2\sin\theta_2)\\
    g(\bm{x}_m,\bm{x}_n, m-n) &:= (r_3\cos\theta_3, r_3\sin\theta_3)
\end{aligned}
$$

我们计算内积得到：

$$
\langle f_q(\bm{x}_m, m), f_q(\bm{x}_n, n)\rangle = 
$$

带入初始条件得到

## naive实现

我们接下来看一下如何实现RoPE
$$
\Theta_m\bm{x}=\begin{bmatrix}
    \cos m\theta_0 & -\sin m\theta_0 & &&\cdots &\cdots &\cdots \\
    \sin m\theta_0 & \cos m\theta_0 & &&\cdots &\cdots &\cdots\\
    & & \cos m\theta_1 & -\sin m\theta_1 & \cdots  &\cdots &\cdots\\
    & & \sin m\theta_1 & \cos m\theta_1 & \cdots &\cdots &\cdots \\
    &&&& \ddots &\vdots &\vdots\\
    &&&& & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
    &&&& & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{bmatrix}\begin{bmatrix}
    x_1\\
    x_2\\
    \vdots \\
    x_d
\end{bmatrix}
$$

我们首先将上面的矩阵乘法给算出来，得到：

$$
\Theta_m\bm{x}=\begin{bmatrix}
\cos m\theta_0\ x_1 - \sin m\theta_0\ x_2\\
\sin m\theta_0\ x_1 + \cos m\theta_0\ x_2\\
\vdots\\
\cos m\theta_{d/2}\ x_{d-1} - \sin m\theta_{d/2}\ x_d\\
\sin m\theta_{d/2}\ x_{d-1} + \cos m\theta_{d/2}\ x_d\\
\end{bmatrix}:=\begin{bmatrix}
    o_1\\
    o_2\\
    \vdots\\
    o_{d-1}\\
    ~~o_d~~
\end{bmatrix}
$$

实现的时候，我们通常按照奇偶index来分别计算，然后通过重排序来得到最终的结果，实现代码如下：

```python
def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor.

    This function applies rotary positional embeddings (RoPE) to the input tensor by
    performing a rotation in 2D space for each pair of dimensions.

    Args:
        x (torch.Tensor): Input tensor to apply RoPE to. Shape is (...,seq_len, d_k)
        sin (torch.Tensor): Sine component of rotary embeddings. Shape is (...,seq_len, d_k_half)
        cos (torch.Tensor): Cosine component of rotary embeddings. Shape is (...,seq_len, d_k_half)

    Returns:
        torch.Tensor: Tensor with rotary positional embeddings applied. Shape matches input x.

    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    """
    x_even = x[..., ::2]  # (seq_len, d_k_half)
    x_odd = x[..., 1::2]  # (seq_len, d_k_half)
    odds = cos * x_even - sin * x_odd  # (...,seq_len, d_k_half)
    evens = sin * x_even + cos * x_odd  # (...,seq_len, d_k_half)
    stacked = torch.stack((odds, evens), -2)  # (...,seq_len, 2, d_k_half)
    stacked_trans = rearrange(
        stacked, "... seq_len double d_k_half -> ... seq_len d_k_half double"
    )  # (...,seq_len, d_k_half, 2)
    out = rearrange(
        stacked_trans, "... seq_len d_k_half double -> ... seq_len (d_k_half double)"
    )  # (..., seq_len, d_k)
    return out
```

我们首先按照奇偶分别套用不同的计算公式，接下来，我们将结果stack在一起，也就是
$$
\begin{bmatrix}
o_1 & o_1 & \cdots & o_{d-1} \\
o_2 & o_4 & \cdots & o_d\\
\end{bmatrix}
$$

第一次rearrange实际就是进行转置，我们得到：
$$
\begin{bmatrix}
o_1 & o_2 \\
o_3 & o_4 \\
\vdots& \vdots\\
o_{d-1} & o_d \\
\end{bmatrix}
$$

最后，我们将结果进行展平（第二次rearrange）操作，得到
$$
[
o_1,
o_2,
\dots,
o_{d-1},
o_d ]^T
$$

在实现的时候，我们一般根据$\sin$ 和$\cos$进行分组，也就是

$$
\Theta_m\bm{x}=\begin{bmatrix}
\cos m\theta_0\\
\cos m\theta_0\\
\vdots\\
\cos m\theta_{d/2}\\
\cos m\theta_{d/2}\\
\end{bmatrix}\odot \begin{bmatrix}
x1\\
x2\\
\vdots\\
x_{d-1}\\
x_d\\
\end{bmatrix} + \begin{bmatrix}
\sin m\theta_0\\
\sin m\theta_0\\
\vdots\\
\sin m\theta_{d/2}\\
\sin m\theta_{d/2}\\
\end{bmatrix}\odot
\begin{bmatrix}
-\ x_2\\
x_1\\
\vdots\\
-x_d\\
x_{d-1}\\
\end{bmatrix}
$$

## LLaMA实现

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

## 通用实现

我们可以看到，naive版本的实现与现在大语言模型所采用的实现并不一致，我们先看一下现有的大语言模型的RoPE实现，这里使用了[LLaMA的transformer代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)放在下面，

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
```

我们将上述代码翻译成公式，现在我们的 $\Theta$ 变成了 (对应 `emb = torch.cat((freqs, freqs), dim=-1)`)

$$
\Theta = [\theta_0,\dots,\theta_{d/2},\theta_0,\dots,\theta_{d/2}]^T
$$

实际上$\sin$部分对应的向量现在变成了

$$
[-x_{d/2+1},
-x_{d/2+2},
\dots,
-x_{d},
x_1,
\dots,
x_{d/2}]^T
$$

我们带回到原始公式，可以得到对应的RoPE操作变成了

$$
R_{\theta,m}^d=\begin{bmatrix}
    \cos m\theta_0 &  & &  -\sin m\theta_0 & \cdots &\cdots &\cdots \\
    & & \cos m\theta_1 & &-\sin m\theta_1 & \cdots  &\cdots \\
    & & & \cos m\theta_2 & &-\sin m\theta_2 & \cdots  \\
    \vdots&\vdots&\vdots&\vdots& \vdots &\vdots &\vdots\\
    & & &  &\cos m\theta_{d/2 - 1} & & -\sin m\theta_{d/2 - 1}  \\
    \sin m\theta_0 && & \cos m\theta_0  &&\cdots &\cdots \\
    & & \sin m\theta_1 & &\cos m\theta_1 & \cdots &\cdots  \\
    & & & \sin m\theta_2 & &\cos m\theta_2 & \cdots  \\
    \vdots&\vdots&\vdots&\vdots& \vdots &\vdots &\vdots\\
    &&&& \sin m\theta_{d/2 - 1}&  & \cos m\theta_{d/2 - 1}
\end{bmatrix}
$$
这列每一行的 $\cos$ 和 $\sin$ 都相差了 $d/2$ 列.

因此，这里的区别在于，原始RoPE计算的pair为 $(x_{2i-1}, x_{2i})$, 而LLaMA里的RoPE计算的pair为 $(x_{i}, x_{i+d/2})$.

我们通过验证可以发现，

$$
(R_{\theta,m}^d)^TR_{\theta,n}^d = R_{\theta,n-m}^d
$$

也就是满足RoPE的性质。

总之，transformer library使用这种方式，可以减少计算量，提高整体的计算效率。

为了适应使用原始RoPE的架构，Huggingface对权重进行了转换，使得基于原始RoPE实现的模型也可以获得加速.

假设 $d=8$，原始RoPE的pair为`[(q_0, q_1), (q_2, q_3), (q_4, q_5), (q_6, q_7)]`, 新的pair为 `[(q_0, q_4), (q_1, q_5), (q_2, q_6), (q_3, q_7)]`. 我们希望对index进行remap，我们发现一个满足条件的permutation为 `[0, 2, 4, 6, 1, 3, 5, 7]`, 也就是 `q_0->q_0`, `q_2->q_1`, ..., `q_7->q_7`.

但是，如果我们在推理时这样做，就会降低整体速度，因此Huggingface的做法是改变$W_Q$和 $W_K$的权重，具体来说，就是 $\Pi q=(\Pi W_Q)x$， 左边是在线转换，右侧离线转换好$W_Q$之后，正常计算就可以了。[具体代码](https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py)为

```python
# permute for sliced rotary
def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


state_dict = {
    f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
        loaded[f"layers.{layer_i}.attention.wq.weight"]
    ),
    f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
        loaded[f"layers.{layer_i}.attention.wk.weight"]
    ),
    ...
}
```

# Alibi

除了RoPE之外，我们还有

# 结论

# 参考文献

- [Is LLaMA rotary embedding implementation correct?](https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2)
- [[LLaMA] Rotary positional embedding differs with official implementation](https://github.com/huggingface/transformers/issues/25199)
