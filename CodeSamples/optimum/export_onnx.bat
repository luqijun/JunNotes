optimum-cli export onnx --model ./models/BAAI/bge-base-zh-v1___5 --task feature-extraction --optimize O1 ./models/BAAI/bge-base-zh-v1___5/onnx_O1
@REM optimum-cli export onnx --model ./models/BAAI/bge-base-zh-v1___5 --task feature-extraction --optimize O3 ./models/BAAI/bge-base-zh-v1___5/onnx_O3
@REM optimum-cli export onnx --model ./models/BAAI/bge-base-zh-v1___5 --task feature-extraction  ./models/BAAI/bge-base-zh-v1___5/onnx_optimiezed
@REM optimum-cli export onnx --model ./models/Jerry0/text2vec-base-chinese --task feature-extraction ./jerry@text2vec-base-chinese/
@REM optimum-cli export onnx --model ./models/Jerry0/text2vec-base-chinese --task feature-extraction ./jerry@text2vec-base-chinese/ --device cpu
pause