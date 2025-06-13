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
这显然是不能接受的，因此，我们就需要加入position encoding，让模型学习到不同的语序有不同的含义。

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

Position encoding可以分为绝对位置编码(absolute position encoding, APE)，相对位置编码(relative position encoding, RPE)，绝对位置编码是transformer里提出的编码模式，但是现在的大多数模型使用的都是相对位置编码。

本文中，我们先介绍位置编码应该具有的性质，然后我们分别介绍绝对位置编码和相对位置编码，我们将着重关注苏剑林老师提出来的RoPE。最后，我们将简单介绍一下LLaMA4使用的NoPE和Qwen系列使用的YARN

# 位置编码

在介绍位置编码之前，我们首先应该关注位置编码的性质，位置编码的目标是为输入的token embedding增加位置信息，那么理想的位置编码应该是怎么样的呢？

我们这里直接引用【参考文献1】中给定的性质：

1. **性质1**: token sequence中每个位置的位置编码都是唯一的。这个很好理解，如果不唯一的话，那么根据前面推导的性质，这两个位置的attention输出就完全一致了
2. **性质2**: 线性相关性。也就是说，如果我们知道了位置$p$处的位置编码，那么理想情况下，我们应该能比较简单地得到$p+k$处的位置编码
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
p_i = [i, \dots, i]=i\mathbf{1}_{d\times 1}\in\mathbb{R}^d,\ i=1,\dots,m
$$

可以看到，这个简单的设计满足性质1，性质2，性质3，性质4.

但是，注意到attention的输入$X$通常是经过Layer Normalization处理过后的，因此其按列符合正态分布，并且均值和方差一般较小。当我们加上整数位置编码之后，其token本身的信息就会被污染，也就是信噪比非常低(信噪比比较低)。一个解决方法就是我们对$p_i$进行normalization，也就是

$$
p_i' = \frac{1}{m}p_i = \frac{i}{m}\mathbf{1}_{d\times 1}
$$

现在所有的位置编码的值都比较小，但是我们发现新的位置编码不满足性质2了，这是因为现在位置编码还和sequence长度有关，我们从位置$p$到位置$p+k$不仅取决于$k$还取决于sequence长度$m$

## 二进制位置编码

既然整数位置编码的主要问题是对输入影响太大，我们能否找一个不影响输入的整数位置编码方式呢？【参考文献1】提出了二进制位置编码，因为每个token是$d$维的，因此我们可以使用二进制来表示$i$. 比如说，当$d=3$, $m=3$时，我们的位置编码分别为

$$
p_0 =p_{(000)_2} = [0, 0, 0],\  p_1 =p_{(001)_2}= [0, 0, 1],\  p_2 =p_{(010)_2} = [0, 1, 0]
$$

现在，我们二进制位置编码满足性质1，性质2. 对于性质3，由于$d$位二进制的表示范围为 $[0, 2^d-1]$，因此其泛化性受到$d$的影响。

我们还发现，二进制位置编码高位，也就是$p_{i,0}$的变化很慢，而低位，也就是$p_{i,d}$变化很快，【参考文献1】画出了不同位置的值的变化情况。

二进制位置编码解决了整数位置编码的信噪比过低和线性相关性。但是其问题是其对不同位置的token embedding产生的影响是不一样的。比如位置1和位置2的相同的token embedding之间的区别是：

$$
(\bm{x}_2 + p_2) - (\bm{x}_1 + p_1) = p_2 - p_1 = [0, 1, 0] - [0, 0, 1] = [0, 1, -1]
$$
这样导致输出的微小变化（增加一个token或减少一个token）都会对最终结果产生巨大影响。因此，我们需要想办法解决这个问题

## Sinusoidal

# 相对位置编码

# RoPE

RoPE由苏剑林老师提出，最早应用于LLaMA架构，后续被大多数模型所采用。

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
\begin{bmatrix}
o_1 \\
o_2\\
\vdots\\
o_{d-1}\\
o_d \\
\end{bmatrix}
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

## 通用实现

我们可以看到，naive版本的实现与现在大语言模型所采用的实现并不一致，我们先把现有的大语言模型的RoPE实现，这里使用了[LLaMA的transformer代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)放在下面，

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
```

我们将上述代码翻译成公式，实际上$\sin$部分对应的向量现在变成了

$$
\begin{bmatrix}
-x_{d/2+1}\\
-x_{d/2+2}\\
\vdots\\
-x_{d}\\
x_1\\
\vdots\\
x_{d/2}\\
\end{bmatrix}
$$

我们带回到原始公式，可以得到对应的rotating matrix现在变成了

$$
\begin{bmatrix}
    \cos m\theta_0 & -\sin m\theta_0 & &&\cdots &\cdots &\cdots \\
    \sin m\theta_0 & \cos m\theta_0 & &&\cdots &\cdots &\cdots\\
    & & \cos m\theta_1 & -\sin m\theta_1 & \cdots  &\cdots &\cdots\\
    & & \sin m\theta_1 & \cos m\theta_1 & \cdots &\cdots &\cdots \\
    &&&& \ddots &\vdots &\vdots\\
    &&&& & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
    &&&& & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{bmatrix}
$$

# 结论

# 参考文献
