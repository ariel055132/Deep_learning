import torch
import random
from transformers import BertTokenizer
from IPython.display import clear_output
from transformers import BertForMaskedLM


"""
這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
"""

# 指定繁簡中文 BERT-BASE 預訓練模型
pretrainedModelName = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(pretrainedModelName)
clear_output()

text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

# 除了 tokens 以外我們還需要辨別句子的 segment ids
tokens_tensor = torch.tensor([tokens_ids])
segment_tensors = torch.zeros_like(tokens_tensor)
maskedLM_model = BertForMaskedLM.from_pretrained(pretrainedModelName)
clear_output()

# 使用 masked LM 估計 [MASK] 位置所代表的實際 token
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segment_tensors)
    predictions = outputs[0]
del maskedLM_model

# 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
masked_index = 5
k = 3
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
print("輸入 tokens ：", tokens[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:10]), '...')