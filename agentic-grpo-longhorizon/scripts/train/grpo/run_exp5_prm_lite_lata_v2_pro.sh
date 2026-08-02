#!/bin/bash
# ============================================================================
# Exp 5: PRM-Lite v5-Pro + LATA v2-Pro - Optimized Joint Training
#
# System optimizations:
#   - grpo_lata_v2 advantage estimator with adaptive length normalization
#   - prm_lite_v5 reward with curriculum process weighting
#   - Larger group size (12 vs 8) for lower variance
#   - Cosine LR schedule with warmup
#   - Entropy bonus for exploration
#   - Advantage clipping to prevent gradient explosion
#   - Early stopping based on validation pass rate
# ============================================================================
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR

cd /workspace/agentic-grpo-longhorizon
mkdir -p experiments/prm_lite_lata_v2_pro

echo "=== Starting PRM-Lite v5 + LATA v2-Pro Training ==="
echo "Expected outcome: ~0.28-0.32 overall pass (15-30% improvement over v4 Joint)"
echo ""

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=prm_lite_lata_v2_pro 2>&1 | tee experiments/prm_lite_lata_v2_pro/training.log
