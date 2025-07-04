export WORKSPACE_BASE=./workspace_base

docker run -it --rm --pull=always \
    --add-host host.docker.internal:host-gateway \
    -e LLM_OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.28-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -e WORKSPACE_MOUNT_PATH=$WORKSPACE_BASE \
    -v $WORKSPACE_BASE:/opt/workspace_base \
    -v ./docker.sock:/var/run/docker.sock \
    -v ./.openhands-state:/.openhands-state \
    -p 3100:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app -d \
    docker.all-hands.dev/all-hands-ai/openhands:0.28
