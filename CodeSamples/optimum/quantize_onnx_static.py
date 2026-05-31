import argparse
import os

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static


class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_size=(640, 480)):
        self.image_dir = image_dir
        self.input_size = input_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.current_index = 0

    def get_next(self):
        if self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
            self.current_index += 1

            # 读取并预处理图片
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = img[np.newaxis, ...]  # 添加batch维度
            img = img.astype(np.float32) / 255.0  # 归一化

            return {"input": img}  # 替换 "input" 为你的模型输入节点名
        return {"input": None}  # 结束标志


def main():
    parser = argparse.ArgumentParser(description="ONNX模型静态量化")
    parser.add_argument("--model-input", type=str, default="model.onnx", help="输入ONNX模型路径")
    parser.add_argument("--model-output", type=str, default="model_quantized_static.onnx", help="输出量化模型路径")
    parser.add_argument("--image-dir", type=str, required=True, help="校准图片目录路径")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 480], help="模型输入尺寸 (宽度 高度)")
    parser.add_argument("--quant-type", type=str, default="QInt8", choices=["QInt8", "QUInt8"], help="量化类型")

    args = parser.parse_args()

    # 准备校准数据读取器
    reader = MyCalibrationDataReader(args.image_dir, tuple(args.input_size))

    # 确定量化类型
    quant_type = QuantType.QInt8 if args.quant_type == "QInt8" else QuantType.QUInt8

    # 执行静态量化
    quantize_static(
        model_input=args.model_input,
        model_output=args.model_output,
        calibration_data_reader=reader,
        weight_type=quant_type
    )

    print(f"模型量化完成，输出路径: {args.model_output}")


if __name__ == "__main__":
    main()
