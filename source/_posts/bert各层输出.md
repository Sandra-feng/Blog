---
title: bert各层输出
categories:
  - 代码修改笔记
tags:
  - 学习记录
toc: true
date: 2023-10-30 14:04:21
updated: 2023-10-30 14:04:21
comments: true
---

# bert各层输出

在最新的transformers接口中，我们获取bert的各个层输出，需要这样：

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state  

last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output
hidden_states = outputs.hidden_states
attentions = outputs.attentions
```

**last_hidden_state**：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。（通常用于命名实体识别）
**pooler_output**：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
**hidden_states**：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
**attentions**：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。



我们知道bert由12个transformer组成，需要用到这12个transformer块的时候就用hidden_states。使用前的实例化需要配置。

```python
config = BertConfig.from_pretrained( 'bert-base-uncased', output_hidden_states=True, output_attentions=True)

bert = BertModel.from_pretrained('bert-base-uncased',config = config)
```

