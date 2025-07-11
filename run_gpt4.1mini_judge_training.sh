#!/bin/bash

echo "🚀 Starting GPT-4.1-mini Judge Training..."
echo "============================================"

echo "📝 Note: OpenAI API Key will be loaded from .env file automatically"

# GPU 메모리 확인
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "🏃 Starting training with GPT-4.1-mini as judge model..."
echo "Configuration: 7b-gpt4.1mini-judge.yaml"
echo "Expected training time: Several hours (depends on OpenAI API rate limits)"
echo ""

# 훈련 실행
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b-gpt4.1mini-judge.yaml

echo ""
echo "🎉 Training completed!"
echo "Check logs and checkpoints in: checkpoints/7b-gpt4.1mini-judge/" 