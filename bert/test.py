import torch
import random
from transformers import BertTokenizer
from IPython.display import clear_output

print(torch.__version__)
# 指定繁簡中文 BERT-BASE 預訓練模型
pretrainedModelName = "bert-base-chinese"

# 取得此預訓練模型所使用的 tokenizer
# tokenizer是一个将纯文本转换为编码的过程，该过程不涉及将词转换成为词向量，仅仅是对纯文本进行分词，并且添加[MASK]、[SEP]、[CLS]标记，然后将这些词转换为字典索引
# from_pretrained(model)：载入预训练词汇表
tokenizer = BertTokenizer.from_pretrained(pretrainedModelName)
clear_output()
vocab = tokenizer.vocab
print(len(vocab))

# 在vocab裏面抽10個vocab
random_tokens = random.sample(list(vocab), 10)
random_ids = []
for t in random_tokens:
    random_ids.append(vocab[t])
for t, id in zip(random_tokens, random_ids):
    print(t, id)

# 利用中文 BERT 的 tokenizer 將一個中文句子斷詞
text = "你好，世界"
tokens = tokenizer.tokenize(text)
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print(text)
print(tokens)
print(tokens_ids)

text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(tokens_ids)
