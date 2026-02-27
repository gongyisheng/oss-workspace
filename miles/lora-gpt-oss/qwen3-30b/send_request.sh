#!/bin/bash

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
LORA_PATH=${LORA_PATH:-my-lora-adapter}

curl -X POST "http://${HOST}:${PORT}/generate" \
    -H "Content-Type: application/json" \
    -d "{
    \"text\": [
        \"Reverse the following security code: 0HRUP0A ->\"
    ],
    \"sampling_params\": {
        \"max_new_tokens\": 32,
        \"temperature\": 0
    },
    \"lora_path\": [\"${LORA_PATH}\"]
    }"
