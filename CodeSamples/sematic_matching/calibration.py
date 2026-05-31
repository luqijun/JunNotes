import argparse
from operator import is_

from datasets import Dataset
from gen_calibra_data import read_and_split
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantFormat, QuantizationConfig, QuantizationMode
from transformers import AutoTokenizer


def main(args):
    model_id = args.model_id
    save_dir = args.save_dir
    tokenizer_path = args.tokenizer_path
    calibration_file = args.calibration_file
    max_length = args.max_length
    use_gpu = args.use_gpu
    quant_format = args.quant_format
    quant_mode = args.quant_mode
    per_channel = args.per_channel
    quant_type = args.quant_type
    operators_to_quantize = args.operators_to_quantize.split(',') if args.operators_to_quantize else None

    # 初始化量化器
    quantizer = ORTQuantizer.from_pretrained(model_id)

    ranges = None
    is_static_quant = (quant_type == "static")
    if is_static_quant:
        # 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 准备校准数据集（文本列表）
        calibration_texts = read_and_split(calibration_file)
        raw_dataset = Dataset.from_dict({"text": calibration_texts})

        # 分词预处理
        def preprocess_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np"  # 返回 numpy 数组，ONNX Runtime 需要
            )

        tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

        # 只保留模型需要的输入列（input_ids, attention_mask 等）
        keep_cols = ['input_ids', 'attention_mask']
        # BGE 模型不使用 token_type_ids，确保不包含它
        if 'bge' not in model_id and 'token_type_ids' in tokenized_dataset.column_names:
            keep_cols.append('token_type_ids')
        tokenized_dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if c not in keep_cols]
        )

        # 4. 创建校准配置（数据集已经是模型输入格式）
        calibration_config = AutoCalibrationConfig.minmax(tokenized_dataset)

        # 5. 执行校准
        ranges = quantizer.fit(
            dataset=tokenized_dataset,
            calibration_config=calibration_config,
            use_gpu=use_gpu,
        )

    # 6. 量化配置与导出
    # 映射字符串到枚举值
    format_map = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator
    }
    mode_map = {
        "qlinearops": QuantizationMode.QLinearOps,  # 默认的线性量化模式，适用于大多数操作符 (静态量化)
        "integerops": QuantizationMode.IntegerOps  # 整数量化模式，适用于注意力操作符
    }

    if not is_static_quant:
        print("正在执行动态量化，校准数据将被忽略，且量化范围将根据每次推理动态计算。")
        quant_mode = 'integerops'

    if operators_to_quantize:
        print(f"将仅量化以下操作符: {operators_to_quantize}")
        q_config = QuantizationConfig(
            is_static=is_static_quant,
            format=format_map.get(quant_format.lower(), QuantFormat.QDQ),
            mode=mode_map.get(quant_mode.lower(), QuantizationMode.QLinearOps),
            per_channel=per_channel,
            operators_to_quantize=operators_to_quantize
        )
    else:
        print("将量化所有支持的操作符")
        q_config = QuantizationConfig(
            is_static=is_static_quant,
            format=format_map.get(quant_format.lower(), QuantFormat.QDQ),
            mode=mode_map.get(quant_mode.lower(), QuantizationMode.QLinearOps),
            per_channel=per_channel
        )

    quantizer.quantize(
        save_dir=save_dir,
        quantization_config=q_config,
        calibration_tensors_range=ranges,
    )

    print(f"量化完成！模型已保存至: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型量化校准脚本")
    parser.add_argument("--model_id", type=str, default="./models/BAAI/bge-base-zh-v1___5/onnx_O1", help="ONNX模型路径")
    parser.add_argument("--save_dir", type=str, default="./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_calibrated", help="量化后模型保存路径")
    parser.add_argument("--tokenizer_path", type=str, default="./models/BAAI/bge-base-zh-v1___5", help="分词器路径")
    parser.add_argument("--calibration_file", type=str, default="calibration_samples.txt", help="校准数据文件")
    parser.add_argument("--max_length", type=int, default=64, help="最大序列长度")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="是否使用GPU")
    parser.add_argument("--quant_format", type=str, default="QDQ", choices=["QDQ", "QOperator"], help="量化格式")
    parser.add_argument("--quant_mode", type=str, default="QLinearOps", choices=["QLinearOps", "IntegerOps"], help="量化模式")
    parser.add_argument("--per_channel", action="store_true", default=False, help="是否按通道量化")
    parser.add_argument("--operators_to_quantize", type=str, default="", help="要量化的操作符，逗号分隔. 例如: 'MatMul,Attention'，如果为空则量化所有支持的操作符")
    parser.add_argument("--quant_type", type=str, default="static", choices=["static", "dynamic"], help="量化类型: static (静态量化) 或 dynamic (动态量化)")

    args = parser.parse_args()
    main(args)
    # --save_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_per_channel_manual --quant_type dynamic --per_channel
    # --save_dir ./models/BAAI/bge-base-zh-v1___5/onnx_O1_int8_calibrated2 --calibration_file calibration_samples2.txt --per_channel
