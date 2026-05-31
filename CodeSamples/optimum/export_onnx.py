from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "./models/shibing624/text2vec-base-chinese"
save_dir = "./onnx_models/text2vec-base-chinese"

# 加载并自动转换成 ONNX
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 保存到本地
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("ONNX 模型已成功转换并保存！")