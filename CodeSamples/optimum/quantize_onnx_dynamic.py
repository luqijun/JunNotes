import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(model_input, model_output, weight_type):
    """
    Quantize an ONNX model dynamically

    Args:
        model_input: Path to the input ONNX model
        model_output: Path to save the quantized model
        weight_type: Quantization type (QInt8 or QUInt8)
    """
    quant_type = QuantType.QInt8 if weight_type.lower() == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        model_input=model_input,
        model_output=model_output,
        weight_type=quant_type
    )
    print(f"Model quantized successfully! Input: {model_input}, Output: {model_output}, Quantization type: {weight_type}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model dynamically")
    parser.add_argument("--input", default="model.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", default="model_quantized.onnx", help="Path to save quantized model")
    parser.add_argument("--quant-type", default="QInt8", choices=["QInt8", "QUInt8"], help="Quantization type")

    args = parser.parse_args()
    quantize_model(args.input, args.output, args.quant_type)


if __name__ == "__main__":
    main()
