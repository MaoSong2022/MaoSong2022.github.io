---
title: tokenizer总结
description: tokenizer总结与BPE的高效实现
date: 2025-05-24 19:56:34+0800
tags: 
    - transformer
categories:
    - LLM 
math: true
---


# Introduction

在自然语言处理中, tokenizer的作用是将一个文本序列通过一个字典转化为一个token id的序列. 我们回顾图片分类任务, 我们在预测的时候, 实际上预测的是类别对应的id, 而不是类别本身. tokenizer做的事情就是提供一个类似于从类别到对应id的字典.

一般来说, 一个tokenizer处理文本序列的过程有两步：

1. pre-tokenize, 也就是预处理, 我们需要将文本序列分割成合适大小的chunks
2. tokenize, 构建chunks到token id的映射

注：实际上, huggingface的tokenizer包括[四个步骤](https://huggingface.co/docs/tokenizers/pipeline), 其中第二第三个步骤与上述一致. 在pre-tokenize之前, 我们有一个normalization过程, 该过程会对文本序列进行处理, 如将文本序列变为小写, 删掉声调符号等, 如下面例子所示：

```python
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])
normalizer.normalize_str("Héllò hôw are ü?")
# "Hello how are u?"
```

其完整流程如下图所示 (图源：[huggingface llm-course](https://huggingface.co/learn/llm-course/chapter6/4?fw=pt))

![tokenization pipeline](tokenization_pipeline.png)

在tokenize之后, 我们会有一个post-processing过程, 比如BERT会在生成的token系列前后加入 `[CLS]` token 和 `[SEP]` token, 例子如下：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token_ids = tokenizer.encode("I love NLP.")
print(token_ids)
# [101, 146, 1567, 23323, 119, 102]
# represents [[CLS], "I", "love", "NLP", ".", [SEP]]
```

构建好tokenizer之后, 我们还要保证tokenizer提供两个功能

1. encoding, 给定文本序列, 将其映射到字典中去得到token id序列
2. decoding, 给定token id序列, 将其解码成文本序列

接下来, 我们将简单介绍一下word tokenizer, character tokenizer以及byte tokenizer, 并分析它们各自的不足.
然后, 我们介绍一些sub-word tokenizer, 最后, 我们介绍现代大语言模型中使用最多的BPE tokenizer

# 无需训练的tokenizer

本节我们将要介绍word tokenizer, character tokenizer以及byte tokenizer，它们的特点就是简单易懂，不需要额外的规则和学习。但是也都有各自的缺点。

## Word tokenizer**

给定一个文本序列,  我们现在需要将其转化为一个token序列. 一个比较自然的想法是, 我们按照空格将序列拆分成若干个单词, 这样每个单词的语义都能比较好的保留. 下面是一个例子

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
indices = tokenizer.encode("hello, world")
# indices = []
```

接下来我们基于一个预定义好的词典, 将其转化为一个token id的序列.
但是, 这种做法的问题就是, 如果出现了预定义好的词典之外的词 (out of vocabulary, OOV) 怎么办？现有的处理方法是使用 `<UNK>` token来表示这些OOV的词.
这显然会丢失语义信息, 因为我们编码成 `<UNK>` token之后, 就没办法再解码回来了.
word tokenizer的缺点为：

1. 单词数量很大，导致很多罕见的单词出现频率很低，降低了tokenizer的利用率
2. 对于不在词典内的单词只能用`<UNK>` token，会损害语义信息

既然基于word的tokenizer有OOV的问题. 我们能否想办法解决这个问题呢？答案是可以的, 我们可以使用 character tokenizer.

## Character tokenizer**

Character tokenizer的基本思想是使用字符而不是单词来编码文本序列. 其实现方式如下：

```python
class CharacterTokenizer:
    def encode(self, s: str) -> list[int]:
        return list(ord(c) for c in s)
    
    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)
```

character tokenizer的词表大小取决于我们的编码方式，UTF-8的编码大概有[110K code points](https://en.wikipedia.org/wiki/UTF-8). character tokenizer的缺点总结如下：

1. character tokenizer会导致我们的词表非常大
2. 和word tokenizer一样，很多character非常罕见，会降低词表的利用率

## Byte tokenizer**

我们发现，character tokenizer和word tokenizer的词表都很大，我们能否想办法降低词表大小，提升每个token的利用率呢？答案是使用Byte tokenizer。

Byte tokenizer的基本思想是, 所有的字符(character)都是由byte组成的, 比如对于UTF-8编码来说, 每个字符由1-4个byte组成.
因此, 所有满足UTF-8编码的文本, 我们都可以将它们转换为基于byte的token序列.
由于现在的大部分文本都是基于UTF-8的, 因此, 我们只讨论UTF-8编码的文本.
Byte tokenizer的实现如下：

```python
class ByteTokenizer:
    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")
```

byte tokenizer的词表很小，其词表大小为 `256`, 这是因为一个byte可以有256中可能的值。

尽管byte tokenizer实现简单，并且词表也很小，可以说byte tokenizer解决了character tokenizer和word tokenizer的问题。
但是，byte tokenizer的问题在于，其encode的到的token序列可能会非常长！我们知道，transformer计算量与token序列的长度是平方级关系的，也就是说token序列长度增加10倍，整体的计算量就会增加100倍，因此我们势必需要考虑token序列的长度。
因此，byte tokenizer的问题为：

1. 产生的token序列过长，增加了transformer的计算量
2. 没有上下文语义信息

## 总结

我们总结一下word tokenizer, character tokenizer以及byte tokenizer三者各自的特点：

|feature |word tokenizer| character tokenizer|byte tokenizer|
| --- | ---|--- | ---|
| granularity| coarse | medium | fine |
| vocabulary | yes | no | no|
| support OOV| bad | good | best|
| #tokens | small | large | very large |
| Chinese | yes | yes | yes|
| support spell error | bad | yes | yes |

因此，这三种tokenizer尽管实现起来很简单，但是其都有各自的问题。为了解决这些问题，我们的做法就是折衷，使用sub-word tokenizer，也就是介于word tokenizer和byte tokenizer之间的方法。

# BPE

## 基本原理与实现

BPE，即byte pair tokenizer的原理非常简单，也就是说对于出现频率比较高的词，我们应该有一个简写的方式，也就是我们使用一个新的token来表示这个词。比如在英语中，我们会使用`plz` 来代替 `please` 以及使用`how r u` 来代替`how are you`.

BPE算法包括以下几个步骤：

1. 对文本序列进行pre-tokenize，分割成不同的单词
2. 当`len(vocab)<vocab_size`时，重复一下步骤：
   1. 对所有单词，统计其相邻character或者byte pair的频率
   2. 计算出现频率最高的pair，使用一个新的token来表示这个pair
   3. 将新的token和其对应的`token_id`加入到`vocab`中

其具体实现如下：

```python
def pre_tokenize(text):
    pass

def get_stats():
    pass

def merge_pair():
    pass

def train(text: str, 
        vocab_size: int, 
        special_tokens: list[str]=['<|endoftext|>']) 
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass
```

## 高效实现

BPE的原理很简单, 我们也实现了其naive版本, 但是naive版本的问题是太慢了. 因此我们将要优化naive版本实现的效率.

首先我们发现, 我们不需要遍历所有的word, 只有含有`best_pair`的word我们才会进行处理, 因此, 我们的第一个改进就是使用 `pair_to_word` 来记录每个pair的来源, 比如

```python
pair_to_word = {(b' ', b't'): [b' the', b' it'], (b't', b'h'): [b'the']}
```

这样, 我们在merge的时候, 直接使用 `pair_to_word[best_pair]` 来获取需要被更新的token序列就可以了.

其次, 注意到每次merge之后, 我们都需要重新计算一次 `pair_freq`, 而实际上, 只有被merge的token序列才需要被重新计数, 其他大部分token序列都是不需要重新计数的.
因此, 一个改进点就是我们在merge的过程中就更新 `pair_freq`, 而不是重新计算. 为了达到这个目标, 我们其实只需要两个操作.
我们用`(b'x', b'a', b'b', b'y')` 和 `best_pair=(b'a', b'b')`来说明, merge之前, 这个序列贡献的`pair_freq`为：

```python
{
    (b'x', b'a'): 1,
    (b'a', b'b'): 1,
    (b'b', b'y'): 1,
}
```

merge之后, token序列变成了`(b'x', b'z', b'y')` (假设新的token为`b'z'`), 这时候的计数为：

```python
{
    (b'x', b'a'): 0,
    (b'a', b'b'): 0,
    (b'b', b'y'): 0,
    (b'x', b'z'): 1,
    (b'z', b'y'): 1,
}
```

也就是说, merge之后, 三个pair的计数减少了1, 分别是`(token_seq[i-1], merge_pair[0])`,`merge_pair` 和 `(merge_pair[1], token_seq[i+2])`. 两个pair的个数增加了1, 分别是 `(token_seq[i-1], new_token)`和`(new_token, token_seq[i+2])` (这里我们假设`merge_pair=(token_seq[i], token_seq[i+1])`) 基于这个结论，我们就可以优化BPE算法了，具体逻辑就是：

1. pretokenize，将text切分为若干个word
2. 计算`word_count`, `pair_freq`, `pair_to_word`
3. 重复挑选频率最高的pair将其merge为一个新的token，然后基于上述更新方式更新`pair_freq`, 直到`vocab`的size达到指定的`vocab_size`

其具体实现如下：

```python

def pre_tokenize(text):
    pass

def get_stats():
    pass

def merge_pair():
    pass

def train(text: str, 
        vocab_size: int, 
        special_tokens: list[str]=['<|endoftext|>']) 
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass
```

# Other subword tokenizers

## WordPiece

WordPiece是Google在预训练BERT时采用的tokenizer

## Unigram

## SentencePiece

# 总结

- Byte-Pair Encoding (BPE): GPT
- Byte Byte-Pair Encoding (BPE): GPT-2
- WordPiece: BERT, DistillBERT, Electra
- Unigram & SentencePiece: ALBERT, XLNet, Marian, T5

sub-word tokenizer的对比 (来自[huggingface llm course](https://huggingface.co/learn/llm-course/chapter6/4?fw=pt))

| Method | BPE | WordPiece | Unigram |
| --- | --- | --- | ---|
| start vocabulary | small | small | large|
| train| merge tokens | merge tokens | remove tokens|
| training step | merge with most frequent pair | merge with best score | remove all tokens minimized the loss |
| learns | merge rules and a vocab | a vocab | a vocab with a score for each token|
| encoding | splits into words and applies merge rules| find the longest subword from the beginning that is in the vocab | finds the most likely split into tokens with learned scores|

# Tokenizer-free

以上都是基于tokenizer

# 实践：Tokenizer

虽然我们已经实现了基于BPE的tokenizer, 但实际上, huggingface已经实现了前面提到的tokenizer pipeline, huggingface的tokenizer包括两种：

1. fast tokenizer, 即[Tokenizer库](https://github.com/huggingface/tokenizers), 这个库是基于Rust开发的
2. slow tokenizer, 这个是transformer库里模型自带的, 比如ChatGLM就有自己开发的tokenizer

huggingface比较了并行处理时两者的区别：

|Setting | Fast tokenizer | Slow tokenizer |
| --- | --- | ---|
| `batched=True` | 10.8s | 4min41s|
| `batched=False` | 59.2s | 5min3s|

huggingface提供的tokenizer库已经非常齐全了, 如果说我们需要开发新的tokenizer的话, 建议直接使用Tokenizer库, 而不是重新造轮子.

# 结论

本文中, 我们介绍了大语言模型中的tokenizer, 我们从byte level, word level到sub-word level, 再到现代大语言模型最常使用的BPE tokenizer, 并给出了其（高效版本）实现. 最后, 我们介绍了一下tokenizer-free的大语言模型和huggingface的tokenizer库. 在未来, 我们将继续深入了解大语言模型的基本原理和实现细节.

# 参考文献

- [cs336 Lecture1](https://stanford-cs336.github.io/spring2025/)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://arxiv.org/pdf/1808.06226)
- [Unigram](https://arxiv.org/pdf/1804.10959)
- [WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
