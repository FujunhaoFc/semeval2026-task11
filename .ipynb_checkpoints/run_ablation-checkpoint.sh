#!/bin/bash
# =============================================================================
# SemEval 2026 Task 11 - Pipeline Ablation Experiments
#
# 在 autodl 4090 服务器上跑，结果提交到 Codabench post-competition phase
#
# 使用前修改下面的路径配置，然后:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh          # 跑全部
#   ./run_ablation.sh st1      # 只跑 subtask 1
#   ./run_ablation.sh st2 st3  # 只跑 subtask 2 和 3
# =============================================================================

set -e

# ========================== 路径配置（按你的实际情况修改）==========================

# 源代码目录（你的脚本所在位置）
SRC_DIR="/root/semeval2026_task11/final_submission/src"

# 测试数据
TEST_ST1="${SRC_DIR}/../data/official/test_data/test_data_subtask_1.json"
TEST_ST2="${SRC_DIR}/../data/official/test_data/test_data_subtask_2.json"
TEST_ST3="${SRC_DIR}/../data/official/test_data/test_data_subtask_3.json"
TEST_ST4="${SRC_DIR}/../data/official/test_data/test_data_subtask_4.json"

# 模型 checkpoint
DEBERTA_BASE="/root/autodl-tmp/models/balanced"
XLMR_BASE="/root/autodl-tmp/models/subtask3_xlmr_seeds"
SEEDS="13,21,42,87"

# 输出目录
OUT_DIR="${SRC_DIR}/ablation_pipeline_results"

# DeepSeek API
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-d097894d57fc429a80ac6ad3b1e56c4b}"

# ========================== 辅助函数 ==========================

mkdir -p "${OUT_DIR}"

# 为每个 prediction 文件生成单独的 zip（Codabench 提交用）
make_zip() {
    local json_path="$1"
    local zip_path="${json_path%.json}.zip"
    local dir=$(dirname "$json_path")
    local name=$(basename "$json_path")
    cd "$dir" && zip -q "$zip_path" "$name" && cd - > /dev/null
    echo "  → ZIP: ${zip_path}"
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# ========================== Subtask 1 消融 ==========================
run_st1() {
    echo ""
    echo "============================================================"
    echo "  Subtask 1 消融实验  $(timestamp)"
    echo "============================================================"

    cd "${SRC_DIR}"

    # --- Config 1: Full pipeline (rules + deepseek + deberta ensemble) ---
    echo ""
    echo "[ST1] Config: full_pipeline"
    python hybrid_pipeline_ensemble.py \
        --mode predict \
        --test-data "${TEST_ST1}" \
        --output "${OUT_DIR}/st1_full_pipeline.json" \
        --deberta-model "${DEBERTA_BASE}" \
        --seeds "${SEEDS}" \
        --ensemble prob \
        --deepseek-model deepseek-chat \
        --device cuda
    make_zip "${OUT_DIR}/st1_full_pipeline.json"

    # --- Config 2: DeepSeek + Neural (no rules) ---
    echo ""
    echo "[ST1] Config: deepseek_neural (no rules)"
    python hybrid_pipeline_ensemble.py \
        --mode predict \
        --test-data "${TEST_ST1}" \
        --output "${OUT_DIR}/st1_deepseek_neural.json" \
        --deberta-model "${DEBERTA_BASE}" \
        --seeds "${SEEDS}" \
        --ensemble prob \
        --deepseek-model deepseek-chat \
        --device cuda \
        --no-rules
    make_zip "${OUT_DIR}/st1_deepseek_neural.json"

    # --- Config 3: Neural only (no rules, no deepseek) ---
    echo ""
    echo "[ST1] Config: neural_only"
    python hybrid_pipeline_ensemble.py \
        --mode predict \
        --test-data "${TEST_ST1}" \
        --output "${OUT_DIR}/st1_neural_only.json" \
        --deberta-model "${DEBERTA_BASE}" \
        --seeds "${SEEDS}" \
        --ensemble prob \
        --device cuda \
        --no-rules --no-deepseek
    make_zip "${OUT_DIR}/st1_neural_only.json"

    # --- Config 4: DeepSeek only (no rules, no deberta) ---
    echo ""
    echo "[ST1] Config: deepseek_only"
    python hybrid_pipeline_ensemble.py \
        --mode predict \
        --test-data "${TEST_ST1}" \
        --output "${OUT_DIR}/st1_deepseek_only.json" \
        --deepseek-model deepseek-chat \
        --device cuda \
        --no-rules --no-deberta
    make_zip "${OUT_DIR}/st1_deepseek_only.json"

    # --- Config 5: Rules only (no deepseek, no deberta) ---
    echo ""
    echo "[ST1] Config: rules_only"
    python hybrid_pipeline_ensemble.py \
        --mode predict \
        --test-data "${TEST_ST1}" \
        --output "${OUT_DIR}/st1_rules_only.json" \
        --device cuda \
        --no-deepseek --no-deberta
    make_zip "${OUT_DIR}/st1_rules_only.json"

    echo ""
    echo "[ST1] Done! 共5个配置"
}


# ========================== Subtask 2 消融 ==========================
run_st2() {
    echo ""
    echo "============================================================"
    echo "  Subtask 2 消融实验  $(timestamp)"
    echo "============================================================"

    cd "${SRC_DIR}"

    # --- Config 1: Full pipeline (deepseek retrieval + deepseek symbolic + deberta) ---
    echo ""
    echo "[ST2] Config: full_pipeline"
    python predict_subtask2_hybrid.py \
        --test_path "${TEST_ST2}" \
        --output_path "${OUT_DIR}/st2_full_pipeline.json" \
        --deberta_path "${DEBERTA_BASE}" \
        --seeds "${SEEDS}" \
        --deepseek_model deepseek-chat \
        --device cuda
    make_zip "${OUT_DIR}/st2_full_pipeline.json"

    # --- Config 2: Neural only (no deepseek) ---
    echo ""
    echo "[ST2] Config: neural_only"
    python predict_subtask2_hybrid.py \
        --test_path "${TEST_ST2}" \
        --output_path "${OUT_DIR}/st2_neural_only.json" \
        --deberta_path "${DEBERTA_BASE}" \
        --seeds "${SEEDS}" \
        --device cuda \
        --no_deepseek
    make_zip "${OUT_DIR}/st2_neural_only.json"

    echo ""
    echo "[ST2] Done! 共2个配置"
}


# ========================== Subtask 3 消融 ==========================
run_st3() {
    echo ""
    echo "============================================================"
    echo "  Subtask 3 消融实验  $(timestamp)"
    echo "============================================================"

    cd "${SRC_DIR}"

    # --- Config 1: Full pipeline (deepseek + xlm-roberta ensemble) ---
    echo ""
    echo "[ST3] Config: full_pipeline"
    python predict_subtask3_hybrid.py \
        --model_path "${XLMR_BASE}" \
        --seeds "${SEEDS}" \
        --test_path "${TEST_ST3}" \
        --output_path "${OUT_DIR}/st3_full_pipeline.json" \
        --device cuda \
        --use_deepseek \
        --deepseek_model deepseek-chat
    make_zip "${OUT_DIR}/st3_full_pipeline.json"

    # --- Config 2: Neural only (no deepseek) ---
    echo ""
    echo "[ST3] Config: neural_only"
    python predict_subtask3_hybrid.py \
        --model_path "${XLMR_BASE}" \
        --seeds "${SEEDS}" \
        --test_path "${TEST_ST3}" \
        --output_path "${OUT_DIR}/st3_neural_only.json" \
        --device cuda
    make_zip "${OUT_DIR}/st3_neural_only.json"

    echo ""
    echo "[ST3] Done! 共2个配置"
}


# ========================== Subtask 4 消融 ==========================
run_st4() {
    echo ""
    echo "============================================================"
    echo "  Subtask 4 消融实验  $(timestamp)"
    echo "============================================================"

    cd "${SRC_DIR}"

    # --- Config 1: Full pipeline (deepseek + xlm-roberta ensemble) ---
    echo ""
    echo "[ST4] Config: full_pipeline"
    python predict_subtask4_hybrid.py \
        --test_path "${TEST_ST4}" \
        --output_path "${OUT_DIR}/st4_full_pipeline.json" \
        --xlmr_path "${XLMR_BASE}" \
        --seeds "${SEEDS}" \
        --deepseek_model deepseek-chat \
        --device cuda
    make_zip "${OUT_DIR}/st4_full_pipeline.json"

    # --- Config 2: Neural only (no deepseek) ---
    echo ""
    echo "[ST4] Config: neural_only"
    python predict_subtask4_hybrid.py \
        --test_path "${TEST_ST4}" \
        --output_path "${OUT_DIR}/st4_neural_only.json" \
        --xlmr_path "${XLMR_BASE}" \
        --seeds "${SEEDS}" \
        --device cuda \
        --no_deepseek
    make_zip "${OUT_DIR}/st4_neural_only.json"

    echo ""
    echo "[ST4] Done! 共2个配置"
}


# ========================== 主入口 ==========================

echo "============================================================"
echo "  SemEval 2026 Task 11 - Pipeline Ablation"
echo "  Output: ${OUT_DIR}"
echo "============================================================"

# 如果指定了参数，只跑对应的 subtask
if [ $# -gt 0 ]; then
    for arg in "$@"; do
        case "$arg" in
            st1|ST1|subtask1) run_st1 ;;
            st2|ST2|subtask2) run_st2 ;;
            st3|ST3|subtask3) run_st3 ;;
            st4|ST4|subtask4) run_st4 ;;
            *) echo "未知参数: $arg (可用: st1 st2 st3 st4)" ;;
        esac
    done
else
    # 默认跑全部
    run_st1
    run_st2
    run_st3
    run_st4
fi

echo ""
echo "============================================================"
echo "  全部完成!  $(timestamp)"
echo "============================================================"
echo ""
echo "输出文件:"
ls -la "${OUT_DIR}"/*.json 2>/dev/null || echo "  (无)"
echo ""
echo "ZIP 文件（直接上传 Codabench）:"
ls -la "${OUT_DIR}"/*.zip 2>/dev/null || echo "  (无)"
echo ""
echo "提交说明:"
echo "  ST1: 5个zip分别提交到 Subtask 1 post-competition"
echo "  ST2: 2个zip分别提交到 Subtask 2 post-competition"
echo "  ST3: 2个zip分别提交到 Subtask 3 post-competition"
echo "  ST4: 2个zip分别提交到 Subtask 4 post-competition"
echo ""
echo "记录每个 zip 提交后 Codabench 返回的 Combined Score，填入论文表格。"
