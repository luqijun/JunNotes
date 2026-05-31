import numpy as np
from text2vec import SentenceModel
from test_data import test_text_pairs

def semantic_similarity(v1, v2):
    """标准余弦相似度公式实现"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 1. 指定你的本地模型文件夹路径
local_model_path = './models/shibing624/text2vec-base-chinese'

# 2. 从本地路径初始化模型（此时不会再触发联网下载）
model = SentenceModel(local_model_path)

# 3. 测试文本对的相似度
for t1, t2 in test_text_pairs:
    e1 = model.encode(t1)
    e2 = model.encode(t2)
    score = semantic_similarity(e1, e2)
    print(f"手动计算相似度: {score:.4f} | '{t1}' vs '{t2}'")
