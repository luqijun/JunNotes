from test_data import test_text_pairs
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("./models/BAAI/bge-base-zh-v1___5")

def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1, normalize_embeddings=True)
    emb2 = model.encode(text2, normalize_embeddings=True)
    return float(util.cos_sim(emb1, emb2))

# 3. 测试文本对的相似度
for t1, t2 in test_text_pairs:
    score = semantic_similarity(t1, t2)
    print(f"手动计算相似度: {score:.4f} | '{t1}' vs '{t2}'")
