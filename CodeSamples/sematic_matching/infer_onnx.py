import argparse
import time

import numpy as np
import onnxruntime as ort
from test_data import test_text_pairs
from transformers import AutoTokenizer


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Text embedding inference with ONNX')
    parser.add_argument('--model_dir', type=str, default="./models/BAAI/bge-base-zh-v1___5/onnx",
                        help='Path to the ONNX model directory')
    parser.add_argument('--model_name', type=str,
                        default="model.onnx", help='Name of the ONNX model file')
    parser.add_argument('--providers', type=str, default='CPUExecutionProvider',
                        help='Comma-separated list of execution providers')
    args = parser.parse_args()
    return args


def main():

    # 1. 加载 Tokenizer 和 ONNX 会话
    args = parse_args()
    model_dir = args.model_dir
    model_name = args.model_name
    providers = [p.strip() for p in args.providers.split(',')]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(
        f"{model_dir}/{model_name}", providers=providers)

    def get_embeddings(texts):
        # 2. 文本分词
        inputs = tokenizer(texts, padding=True,
                           truncation=True, return_tensors="np")

        # 3. ONNX 推理
        # 输入必须匹配 ONNX 模型的 input_ids, attention_mask 等
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        if 'token_type_ids' in ort_inputs and "bge" in model_dir:
            ort_inputs.pop('token_type_ids')
        ort_outs = session.run(None, ort_inputs)

        # 4. 获取 Last Hidden State (通常是第一个输出)
        # 维度: [batch_size, seq_len, 768]
        last_hidden_state = ort_outs[0]

        # 5. 平均池化 (Mean Pooling) 得到句子向量
        # 注意：为了准确，应结合 attention_mask 排除 padding 部分
        mask = inputs['attention_mask']
        mask_expanded = np.expand_dims(mask, axis=-1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    # 测试
    for t1, t2 in test_text_pairs:
        vecs = get_embeddings([t1, t2])
        similarity = np.dot(vecs[0], vecs[1]) / \
            (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        print(f"ONNX 推理相似度: {similarity:.4f} | '{t1}' vs '{t2}'")

    # 测试时间
    test_count = 100
    start_time = time.time()
    for i in range(test_count):
        for t1, t2 in test_text_pairs:
            vecs = get_embeddings([t1, t2])
            similarity = np.dot(
                vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))

    end_time = time.time()
    print(
        f"测试 {test_count} 次 耗时: {end_time - start_time:.2f} 秒，\n"
        f"平均每次: {(end_time - start_time) / test_count:.4f} 秒, \n"
        f"平均每对文本: {(end_time - start_time) / test_count/ len(test_text_pairs):.4f} 秒"
    )


if __name__ == "__main__":
    # --model_dir ./models/shibing624/text2vec-base-chinese/onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_int8 --model_name model_quantized.onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8 --model_name model_quantized.onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel --model_name model_quantized.onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_calibrated --model_name model_quantized.onnx
    main()

# ==================================(py38 1060ti onnxruntime-gpu==1.17.1)==================================
# ./models/shibing624/text2vec-base-chinese/onnx
# ONNX 推理相似度: 0.4578 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.6518 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9362 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.7840 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8095 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9874 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9744 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9962 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.9429 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.9191 | '侧围左上铰链' vs '左上侧合页铰链'

# ./models/BAAI/bge-base-zh-v1___5/onnx
# ONNX 推理相似度: 0.5845 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5929 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9356 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8547 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8118 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9484 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9817 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9493 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8239 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.7982 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 18.27 秒，
# 平均每次: 0.1827 秒,
# 平均每对文本: 0.0166 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_int8
# ONNX 推理相似度: 0.6979 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.7032 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9454 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8412 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8938 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9737 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9719 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9745 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.9093 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.9211 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 12.33 秒，
# 平均每次: 0.1233 秒,
# 平均每对文本: 0.0112 秒


# ./models/BAAI/bge-base-zh-v1___5/onnx_O1
# ONNX 推理相似度: 0.5845 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5929 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9356 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8547 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8118 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9484 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9817 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9493 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8239 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.7982 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 19.62 秒，
# 平均每次: 0.1962 秒,
# 平均每对文本: 0.0178 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8
# ONNX 推理相似度: 0.6979 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.7032 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9454 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8412 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8938 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9737 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9719 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9745 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.9093 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.9211 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 11.98 秒，
# 平均每次: 0.1198 秒,
# 平均每对文本: 0.0109 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel
# ONNX 推理相似度: 0.4493 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.4295 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9409 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.7140 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.6004 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9749 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9587 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9485 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8805 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.8997 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 12.80 秒，
# 平均每次: 0.1280 秒,
# 平均每对文本: 0.0116 秒

# ==================================(py311 3090 onnxruntime-gpu==1.17.1)==================================
# --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1
# ONNX 推理相似度: 0.5845 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5929 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9356 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8547 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8118 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9484 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9817 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9493 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8239 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.7982 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 19.69 秒，
# 平均每次: 0.1969 秒,
# 平均每对文本: 0.0179 秒

# --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8
# ONNX 推理相似度: 0.7157 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.7011 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9508 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8525 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8965 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9737 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9744 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9739 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.9201 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.9198 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 8.20 秒，
# 平均每次: 0.0820 秒,
# 平均每对文本: 0.0075 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel
# ONNX 推理相似度: 0.5655 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5083 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9268 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8328 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8124 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9379 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9735 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9275 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8058 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.8032 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 7.61 秒，
# 平均每次: 0.0761 秒,
# 平均每对文本: 0.0069 秒
# 平均每对文本: 0.0069 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_calibrated --model_name model_quantized.onnx
# ONNX 推理相似度: 0.5958 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5847 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.8818 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.7925 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8246 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9166 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9009 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9056 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8295 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.8537 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 8.62 秒，
# 平均每次: 0.0862 秒,
# 平均每对文本: 0.0078 秒

# ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_manual (手动动态量化与./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel一致)
# ONNX 推理相似度: 0.5655 | '苹果' vs '香蕉'
# ONNX 推理相似度: 0.5083 | '香蕉' vs '橘子'
# ONNX 推理相似度: 0.9268 | '像人的猴子' vs '像猴子的人'
# ONNX 推理相似度: 0.8328 | '买苹果手机' vs '买iphone'
# ONNX 推理相似度: 0.8124 | '如何更换手机卡' vs '怎么换SIM卡'
# ONNX 推理相似度: 1.0000 | '侧围左上铰链' vs '侧围左上铰链'
# ONNX 推理相似度: 0.9379 | '侧围左上铰链1' vs '侧围左上铰链2'
# ONNX 推理相似度: 0.9735 | '侧围左上铰链' vs '左上侧围铰链'
# ONNX 推理相似度: 0.9275 | '侧围左上铰链' vs '侧围左下铰链'
# ONNX 推理相似度: 0.8058 | '侧围上铰链' vs '侧合页上铰链'
# ONNX 推理相似度: 0.8032 | '侧围左上铰链' vs '左上侧合页铰链'
# 测试 100 次 耗时: 7.35 秒，
# 平均每次: 0.0735 秒,
# 平均每对文本: 0.0067 秒
