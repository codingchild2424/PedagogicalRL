#!/bin/bash

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up..."
    
    # Kill training processes
    pkill -f "train_rl" 2>/dev/null || true
    pkill -f "accelerate.*launch" 2>/dev/null || true
    pkill -f "deepspeed" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    
    # Kill port processes
    kill $(lsof -ti:8005) 2>/dev/null || true
    
    echo "🧹 Cleanup completed. Exiting..."
    exit 1
}

# Set trap for Ctrl+C
trap cleanup SIGINT SIGTERM

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

export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error

# 훈련 실행
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b-gpt4.1mini-judge.yaml

echo ""
echo "🎉 Training completed!"
echo "Check logs and checkpoints in: checkpoints/7b-gpt4.1mini-judge/" 