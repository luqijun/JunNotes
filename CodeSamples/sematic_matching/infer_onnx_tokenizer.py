import time
import onnxruntime as ort
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import numpy as np
import argparse
from test_data import test_text_pairs


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


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """均池化，结合 attention_mask 排除 padding 部分

    Args:
        last_hidden_state (np.ndarray): shape=(batch_size, seq_length, hidden_size)
        attention_mask (np.ndarray): shape=(batch_size, seq_length)

    Returns:
        np.ndarray: 平均池化结果
    """
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def main():
    args = parse_args()
    model_dir = args.model_dir
    model_name = args.model_name
    providers = [p.strip() for p in args.providers.split(',')]

    # ──────────────────────────────────────────────
    # 1. 加载两种 Tokenizer
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("加载 Tokenizer...")

    # transformers.AutoTokenizer（原有）
    auto_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("auto_tokenizer.model_max_length = ",
          auto_tokenizer.model_max_length)

    # tokenizers.Tokenizer（HuggingFace tokenizers 库，速度更快）
    # tokenizer.json 通常与模型文件存放在同一目录
    hf_tokenizer: Tokenizer = Tokenizer.from_file(
        f"{model_dir}/tokenizer.json")
    # 启用 padding 与截断，以匹配 AutoTokenizer 的默认行为
    hf_tokenizer.enable_padding(pad_id=auto_tokenizer.pad_token_id,
                                pad_token=auto_tokenizer.pad_token)
    hf_tokenizer.enable_truncation(max_length=auto_tokenizer.model_max_length)

    # ──────────────────────────────────────────────
    # 2. 加载 ONNX 会话
    # ──────────────────────────────────────────────
    session = ort.InferenceSession(
        f"{model_dir}/{model_name}", providers=providers)

    def _ort_run(input_ids, attention_mask, token_type_ids=None):
        """执行 ONNX 推理并返回句向量"""
        ort_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        # bge 系列模型不使用 token_type_ids
        if token_type_ids is not None and "bge" not in model_dir:
            ort_inputs["token_type_ids"] = token_type_ids.astype(np.int64)

        ort_outs = session.run(None, ort_inputs)
        last_hidden_state = ort_outs[0]  # [batch, seq_len, hidden]
        return mean_pooling(last_hidden_state, attention_mask)

    # ──────────────────────────────────────────────
    # 3. 封装两种推理函数
    # ──────────────────────────────────────────────
    def get_embeddings_auto(texts):
        """使用 transformers.AutoTokenizer 分词 → ONNX 推理"""
        enc = auto_tokenizer(texts, padding=True,
                             truncation=True, return_tensors="np")
        return _ort_run(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids"),
        )

    def get_embeddings_hf(texts):
        """使用 tokenizers.Tokenizer 分词 → ONNX 推理"""
        encodings = hf_tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array(
            [e.type_ids for e in encodings], dtype=np.int64)

        return _ort_run(input_ids, attention_mask, token_type_ids)

    # ──────────────────────────────────────────────
    # 4. 功能对比：相似度结果
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【功能对比】AutoTokenizer vs tokenizers.Tokenizer 推理结果")
    print("=" * 60)

    max_diff = 0.0
    for t1, t2 in test_text_pairs:
        vecs_auto = get_embeddings_auto([t1, t2])
        vecs_hf = get_embeddings_hf([t1, t2])

        sim_auto = cosine_similarity(vecs_auto[0], vecs_auto[1])
        sim_hf = cosine_similarity(vecs_hf[0], vecs_hf[1])
        diff = abs(sim_auto - sim_hf)
        max_diff = max(max_diff, diff)

        # 向量间最大绝对误差（逐元素）
        vec_diff = np.max(np.abs(vecs_auto - vecs_hf))

        print(f"\n文本对: '{t1}' vs '{t2}'")
        print(f"  AutoTokenizer 相似度  : {sim_auto:.6f}")
        print(f"  HF Tokenizer  相似度  : {sim_hf:.6f}")
        print(f"  相似度差值            : {diff:.2e}")
        print(f"  向量最大逐元素误差    : {vec_diff:.2e}")

    print(f"\n所有文本对最大相似度差值: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("✅ 两种 Tokenizer 结果高度一致（差异 < 1e-4）")
    else:
        print("⚠️  两种 Tokenizer 存在较明显差异，请检查配置")

    # ──────────────────────────────────────────────
    # 5. 性能对比
    # ──────────────────────────────────────────────
    test_count = 100
    print("\n" + "=" * 60)
    print(f"【性能对比】各运行 {test_count} 轮")
    print("=" * 60)

    # AutoTokenizer 耗时
    start = time.time()
    for _ in range(test_count):
        for t1, t2 in test_text_pairs:
            get_embeddings_auto([t1, t2])
    auto_elapsed = time.time() - start

    # HF Tokenizer 耗时
    start = time.time()
    for _ in range(test_count):
        for t1, t2 in test_text_pairs:
            get_embeddings_hf([t1, t2])
    hf_elapsed = time.time() - start

    n_pairs = len(test_text_pairs)

    def _report(label, elapsed):
        print(f"\n  [{label}]")
        print(f"    总耗时          : {elapsed:.2f} 秒")
        print(f"    平均每轮        : {elapsed / test_count:.4f} 秒")
        print(f"    平均每对文本    : {elapsed / test_count / n_pairs:.4f} 秒")

    _report("transformers.AutoTokenizer", auto_elapsed)
    _report("tokenizers.Tokenizer      ", hf_elapsed)

    speedup = auto_elapsed / hf_elapsed if hf_elapsed > 0 else float('inf')
    print(f"\n  🚀 tokenizers.Tokenizer 相对加速比: {speedup:.2f}x")


if __name__ == "__main__":
    # --model_dir ./models/shibing624/text2vec-base-chinese/onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_int8 --model_name model_quantized.onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8 --model_name model_quantized.onnx
    # --model_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel --model_name model_quantized.onnx
    main()
