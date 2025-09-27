docker run --name vLLM-DeepSeek-R1-0528-Qwen3-8B \
    --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "all_proxy=http://host.docker.internal:10808" \
    --env "http_proxy=http://host.docker.internal:10808" \
    --env "https_proxy=http://host.docker.internal:10808" \
    --env "no_proxy=localhost,127.0.0.1,::1" \
    -p 6200:8000 \
    vllm/vllm-openai:latest \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    # --model Qwen/Qwen3-0.6B


    # --ipc=host \