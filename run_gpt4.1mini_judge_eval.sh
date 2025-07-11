#!/bin/bash

echo "🔍 Starting GPT-4.1-mini Judge Evaluation..."
echo "============================================="

echo "📝 Note: OpenAI API Key will be loaded from .env file automatically"

# 모델 선택 (기본값 설정)
MODEL_CONFIG=${1:-"GPT-4.1mini-Judge"}

# 훈련된 모델 경로 설정
TRAINED_MODEL_PATH="checkpoints/7b-gpt4.1mini-judge/model"

echo ""
echo "🎯 Evaluating with configuration: $MODEL_CONFIG.yaml"
echo "📁 Using trained model: $TRAINED_MODEL_PATH"
echo "Judge Model: GPT-4.1-mini (via OpenAI API)"
echo "Expected evaluation time: ~30 minutes (depends on OpenAI API rate limits)"
echo ""

# 평가 실행 (훈련된 모델 경로로 override)
python eval.py --config-name $MODEL_CONFIG.yaml \
  teacher_model.model_name_or_path=$TRAINED_MODEL_PATH

echo ""
echo "📊 Evaluation completed!"
echo "Check results in wandb or logs." 