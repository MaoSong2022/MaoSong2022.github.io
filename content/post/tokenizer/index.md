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

在自然语言处理中, tokenizer的作用是将一个文本序列通过一个字典转化为一个token id的序列.
我们回顾图片分类任务, 模型在预测的时候, 实际上预测的是类别对应的id, 而不是类别本身. tokenizer做的事情就是提供一个类似于从类别到对应id的字典.

一般来说, 一个tokenizer处理文本序列的过程有两步：

1. pre-tokenize, 也就是预处理, 我们需要将文本序列分割成合适大小的chunks (words)
2. tokenize, 构建chunks (words)到token id的映射

注：实际上, huggingface的tokenizer包括[四个步骤](https://huggingface.co/docs/tokenizers/pipeline), 其中第二第三个步骤与上述一致. 在pre-tokenize之前, 我们有一个normalization过程, 该过程会对文本序列进行处理, 如将文本序列变为小写, 删掉声调符号等, 如下面例子所示：

```python
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])
normalizer.normalize_str("Héllò hôw are ü?")
# "Hello how are u?"
```

在tokenize之后, 我们会有一个post-processing过程, 比如BERT会在生成的token系列前后加入 `[CLS]` token 和 `[SEP]` token, 例子如下：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token_ids = tokenizer.encode("I love NLP.")
print(token_ids)
# [101, 146, 1567, 21239, 2101, 119, 102]
# represents [[CLS], "I", "love", "NL", "##P", ".", [SEP]]
```

其完整流程如下图所示 (图源：[huggingface llm-course](https://huggingface.co/learn/llm-course/chapter6/4?fw=pt))

![tokenization pipeline](tokenization_pipeline.png)

构建好tokenizer之后, 我们还要保证tokenizer提供两个接口

1. encoding, 给定文本序列, 将其映射到字典中去得到token id序列
2. decoding, 给定token id序列, 将其解码成文本序列

接下来, 我们将简单介绍一下word tokenizer, character tokenizer以及byte tokenizer, 并分析它们各自的不足.
然后,  我们介绍现代大语言模型中使用最多的BPE tokenizer. 最后, 我们介绍一些sub-word tokenizer.

# Training-free tokenizer

本节我们将要介绍word tokenizer, character tokenizer以及byte tokenizer，它们的特点就是简单易懂，不需要额外的规则和学习。但是它们也都有各自的缺点。

## Word tokenizer

给定一个文本序列,  我们现在需要将其转化为一个token序列. 一个比较自然的想法是, 我们按照空格将序列拆分成若干个单词, 这样每个单词的语义都能比较好地保留. 下面是一个例子

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
indices = tokenizer.encode("hello world")
# indices = [31373, 995]
# decode = ["hello", " world"]
```

接下来我们基于一个预定义好的词典, 将其转化为一个token id的序列.
word tokenizer的问题是不能处理预定义好的词典之外的词 (out of vocabulary, OOV) . 现有的处理方法是使用 `<UNK>` token来表示这些OOV的词.
但这样显然会丢失语义信息, 因为我们编码成 `<UNK>` token之后, 就没办法再解码回原有的语义信息了。
word tokenizer的缺点为：

1. 单词数量很大，很多罕见单词的出现频率很低，降低了tokenizer的利用率
2. 对于不在词典内的单词只能用`<UNK>` token表示，损害了语义信息

既然基于word的tokenizer有OOV的问题. 我们能否想办法解决这个问题呢？答案是可以的, 我们可以使用 character tokenizer.

## Character tokenizer

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
3. token序列的上下文语义信息较差

## Byte tokenizer

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
总之，byte tokenizer的问题为：

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

实际生活中，对于出现频率比较高的词，我们会有一个简写的方式，也就是我们使用一个新的单词来表示这个词。比如在英语中，我们会使用`plz` 来代替 `please` 以及使用`how r u` 来代替`how are you`.
BPE，即byte pair tokenizer的原理也是类似的，对于出现频率比较高的byte pair或者character pair, 我们会使用一个新的token来表示这个pair，这样就压缩了sequence的长度。

BPE算法包括以下几个步骤：

1. 对文本序列进行pre-tokenize，分割成不同的单词
2. 当`len(vocab)<vocab_size`时，重复以下步骤：
   1. 对所有单词，统计其相邻character或者byte pair的频率
   2. 计算出现频率最高的pair，使用一个新的token来表示这个pair
   3. 将新的token和其对应的`token_id`加入到`vocab`中

算法如下图所示

![BPE algorithm](bpe_algorithm.png)

其具体实现见附录A

> 注意：在本文中，我们实际上实现的是BBPE (byte BPE算法)，BBPE与BPE的区别在于我们的最小单元是character还是bytes. 本质上原理是一致的

## 高效实现

BPE的原理很简单, 我们也实现了其naive版本, 但是naive版本的问题是太慢了. 因此我们将要优化naive版本.

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

merge之后, token序列变成了`(b'x', b'z', b'y')` (假设`best_pair`对应的新的token为`b'z'`), 这时候的计数为：

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
2. 计算`word_count`, `pair_freq`, `pair_to_word`, 使用`splits`记录每个word对应的token分布
3. 重复以下过程：
   1. 挑选频率最高的pair将其merge为一个新的token, 基于`pair_to_words`更新对应的`pair_freq`:
   2. 对每个`split`, 按照上述方式更新`pair_freq`和`split`

其具体实现如附录B所示.

# Other subword tokenizers

## WordPiece

WordPiece是Google在预训练BERT时采用的tokenizer，

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
- [Huggingface LLM Course](https://huggingface.co/learn/llm-course/chapter6/1)

# 附录

## 附录A: Naive BPE tokenizer实现

```python
import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    """Pre-tokenize text from a file into chunks of tokens.

    Args:
        file_path: Path to the input text file.

    Returns:
        List of lists containing pre-tokenized text chunks. Each inner list contains
        tokens from one chunk of text split on <|endoftext|> markers.
    """
    with open(file_path) as f:
        data = f.read()

    # Split text into chunks at special token
    chunks = re.split(r"<\|endoftext\|>", data)
    
    # Pattern matches contractions, words, numbers, symbols and whitespace
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(
    bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int]
) -> defaultdict[tuple[bytes, bytes], int]:
    """Calculate frequencies of adjacent byte pairs in the vocabulary.

    Args:
        bytes_tuple_counter: Counter mapping byte tuples to their frequencies.

    Returns:
        Counter mapping byte pairs (bigrams) to their frequencies across all tuples.
    """
    counter = defaultdict(int)
    # Count frequencies of adjacent byte pairs
    for bytes_tuple, count in bytes_tuple_counter.items():
        for i in range(len(bytes_tuple) - 1):
            counter[(bytes_tuple[i], bytes_tuple[i + 1])] += count

    return counter


def merge_pairs(
    bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int], 
    merge_pair: tuple[bytes, bytes]
) -> defaultdict[tuple[bytes, bytes], int]:
    """Merge all occurrences of the specified byte pair in the vocabulary.

    Args:
        bytes_tuple_counter: Counter mapping byte tuples to their frequencies.
        merge_pair: The pair of bytes to merge into a single token.

    Returns:
        Updated counter with the specified pair merged wherever it occurs.
    """
    counter = defaultdict(int)
    for bytes_tuple, count in bytes_tuple_counter.items():
        bytes_list = []
        i = 0
        # Iterate through bytes, merging pairs where found
        while i < len(bytes_tuple):
            if i < len(bytes_tuple) - 1:
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                if pair == merge_pair:
                    bytes_list.append(merge_pair[0] + merge_pair[1])
                    i += 2
                    continue

            bytes_list.append(bytes_tuple[i])
            i += 1
        counter[tuple(bytes_list)] += count

    return counter


def train(file_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on text data.

    Args:
        file_path: Path to the training text file.
        vocab_size: Target size for the vocabulary.
        special_tokens: List of special tokens to include in vocabulary.

    Returns:
        Tuple containing:
        - Dictionary mapping token IDs to byte sequences
        - List of merge rules as (bytes, bytes) pairs
    """
    # Initialize vocabulary with special tokens
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}

    # Add initial byte-level tokens (0-255)
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # Pre-tokenize input text
    pre_tokenized_chunks = pre_tokenize(file_path)
    
    # Count word frequencies
    word_counts = defaultdict(int)
    for chunk in pre_tokenized_chunks:
        for word in chunk:
            word_counts[word] += 1

    # Convert words to byte tuples and count frequencies
    bytes_tuple_counter = defaultdict(int)
    for word, count in word_counts.items():
        bytes_tuple = tuple(bytes([c]) for c in word.encode("utf-8"))
        bytes_tuple_counter[bytes_tuple] += count

    merges = []
    # Learn merge rules until target vocab size is reached
    while len(vocab) < vocab_size:
        new_token_id = len(vocab)
        stats = get_stats(bytes_tuple_counter)
        
        # Find the most frequent byte pair
        best_score = (0, bytes([0]), bytes([0]))
        for pair, count in stats.items():
            current_score = (count, pair[0], pair[1])
            if current_score > best_score:
                best_score = current_score

        freq, merge_pair = best_score[0], (best_score[1], best_score[2])

        # Apply the merge and update vocabulary
        bytes_tuple_counter = merge_pairs(bytes_tuple_counter, merge_pair)
        new_bytes = merge_pair[0] + merge_pair[1]
        vocab[new_token_id] = new_bytes
        merges.append(merge_pair)

    return vocab, merges
```

## 附录A: Efficient BPE tokenizer实现

```python
"""Byte Pair Encoding (BPE) tokenizer implementation.

This module implements a BPE tokenizer that learns subword units from text data.
It includes functions for pre-tokenization, pair frequency counting, merging tokens,
and training the BPE vocabulary.
"""

import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    """Pre-tokenize text into chunks of words and subwords.

    Args:
        file_path: Path to the text file to tokenize.

    Returns:
        List of lists containing pre-tokenized strings. The outer list represents text chunks
        separated by <|endoftext|> tokens, while inner lists contain the pre-tokenized strings.
    """
    with open(file_path) as f:
        data = f.read()

    # Split text into chunks at <|endoftext|> tokens
    chunks = re.split(r"<\|endoftext\|>", data)
    
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(
    splits: dict[bytes, list[bytes]], word_freqs: dict[bytes, int]
) -> tuple[defaultdict[tuple[bytes, bytes], int], defaultdict[tuple[bytes, bytes], set[bytes]]]:
    """Calculate statistics for adjacent token pairs in the vocabulary.

    Args:
        splits: Dictionary mapping words to their current tokenization.
        word_freqs: Dictionary mapping words to their frequencies in the corpus.

    Returns:
        A tuple containing:
        - pair_freqs: Dictionary mapping token pairs to their frequencies
        - pair_to_word: Dictionary mapping token pairs to the set of words containing them
    """
    pair_freqs = defaultdict(int)
    pair_to_word = defaultdict(set)
    
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue

        # Count frequencies of adjacent pairs and track which words contain each pair
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
            pair_to_word[pair].add(word)
            
    return pair_freqs, pair_to_word


def get_merge_pair(pair_freqs: defaultdict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    """Find the most frequent pair of adjacent tokens to merge.

    Args:
        pair_freqs: Dictionary mapping token pairs to their frequencies.

    Returns:
        A tuple containing:
        - The best pair to merge (as a tuple of two byte sequences)
        - The frequency of that pair
        
    Note:
        If multiple pairs have the same frequency, the lexicographically larger pair is chosen.
    """
    best_pair = None
    max_freq = -1
    
    for pair, freq in pair_freqs.items():
        if freq > max_freq or (freq == max_freq and pair > best_pair):
            max_freq = freq
            best_pair = pair

    return best_pair, max_freq


def merge_pairs(
    splits: dict[bytes, list[bytes]],
    merge_pair: tuple[bytes, bytes],
    pair_freqs: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]],
    word_freqs: dict[bytes, int],
) -> dict[bytes, list[bytes]]:
    """Merge all occurrences of the selected token pair and update statistics.

    Args:
        splits: Dictionary mapping words to their current tokenization.
        merge_pair: The pair of tokens to merge.
        pair_freqs: Dictionary mapping token pairs to their frequencies.
        pair_to_words: Dictionary mapping token pairs to words containing them.
        word_freqs: Dictionary mapping words to their frequencies.

    Returns:
        Updated splits dictionary with the merged tokens.
    """
    token1, token2 = merge_pair
    new_token = token1 + token2
    words_to_update = list(pair_to_words[merge_pair])

    # Remove the merged pair from tracking dictionaries
    if merge_pair in pair_freqs:
        del pair_freqs[merge_pair]
    if merge_pair in pair_to_words:
        del pair_to_words[merge_pair]

    # Process each word containing the merge pair
    for word in words_to_update:
        split = splits[word]
        freq_of_word = word_freqs[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            # Check if current position contains the merge pair
            if split[i] == token1 and split[i + 1] == token2:
                # Update frequencies for pairs involving tokens before the merge
                if i > 0:
                    prev_token = split[i - 1]
                    old_left_pair = (prev_token, token1)
                    pair_freqs[old_left_pair] -= freq_of_word
                    new_left_pair = (prev_token, new_token)
                    pair_freqs[new_left_pair] += freq_of_word
                    pair_to_words[new_left_pair].add(word)

                # Update frequencies for pairs involving tokens after the merge
                if i < len(split) - 2:
                    next_token = split[i + 2]
                    old_right_pair = (token2, next_token)
                    pair_freqs[old_right_pair] -= freq_of_word
                    new_right_pair = (new_token, next_token)
                    pair_freqs[new_right_pair] += freq_of_word
                    pair_to_words[new_right_pair].add(word)

                # Replace the pair with the merged token
                split = split[:i] + [new_token] + split[i + 2 :]
            else:
                i += 1

        # Update the tokenization for this word
        splits[word] = split

    return splits


def train_bpe(
    vocab_size: int, special_tokens: list[str], file_path: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the input text.

    Args:
        vocab_size: Target vocabulary size.
        special_tokens: List of special tokens to include in vocabulary.
        file_path: Path to training text file.

    Returns:
        A tuple containing:
        - vocab: Dictionary mapping token IDs to byte sequences
        - merges: List of merge operations (as tuples of byte sequences)
    """
    # Initialize vocabulary with special tokens and base characters
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    merges = []

    # Add all possible bytes (0-255) to the vocabulary
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # Pre-tokenize input text and count word frequencies
    pre_tokenized_chunks = pre_tokenize(file_path)
    word_freqs = defaultdict(int)
    for chunk in pre_tokenized_chunks:
        for word in chunk:
            word_freqs[word.encode("utf-8")] += 1

    # Initialize each word as sequence of bytes
    splits = {word: [bytes([i]) for i in word] for word in word_freqs}

    # Get initial statistics
    pair_freqs, pair_to_words = get_stats(splits, word_freqs)

    # Main training loop: merge pairs until desired vocabulary size is reached
    while len(vocab) < vocab_size:
        merge_pair, freq = get_merge_pair(pair_freqs)
        splits = merge_pairs(splits, merge_pair, pair_freqs, pair_to_words, word_freqs)
        
        # Add merged token to vocabulary
        new_token_id = len(vocab)
        vocab[new_token_id] = merge_pair[0] + merge_pair[1]
        merges.append((merge_pair[0], merge_pair[1]))

    return vocab, merges

```
