# junhaofu at SemEval-2026 Task 11: A Neuro-Symbolic Hybrid Pipeline for Content-Independent Syllogistic Reasoning

This repository contains the code for our submission to [SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in LLMs](https://sites.google.com/view/semeval-2026-task-11).

## Results

| Subtask | Description | Rank | Combined Score |
|---------|-------------|------|----------------|
| ST1 | English Binary Classification | #11 (tied 1st) | 100.0 |
| ST2 | English Retrieval + Classification | #4 | 54.43 |
| ST3 | Multilingual Classification | #5 | 82.59 |
| ST4 | Multilingual Retrieval + Classification | #8 | 30.31 |

## System Architecture

Our system uses a three-layer neuro-symbolic hybrid pipeline:

```
Input Syllogism
    │
    ▼
Layer 1: Regex Rule System (ST1 only)
    │ ── success ──▶ Valid/Invalid
    ▼ fail
Layer 2: LLM Symbolic Parsing (DeepSeek-V3)
    │   Parse → Mood + Figure → Lookup 24 valid forms
    │ ── success ──▶ Valid/Invalid
    ▼ fail
Layer 3: Neural Fallback
    │   DeBERTa-v3-large (ST1/2) or XLM-RoBERTa-large (ST3/4)
    │   Multi-seed ensemble with LoRA
    └──────────────▶ Valid/Invalid
```

For ST2/ST4 (retrieval subtasks), a premise retrieval step is added before classification, using DeepSeek for semantic retrieval with keyword-based fallback.

## Key Findings

- **Near-zero content effect**: Our symbolic approach achieves TCE < 1% by reasoning over logical structure rather than semantic content
- **Symbolic accuracy**: When parsing succeeds, the lookup table achieves 96%+ accuracy across all tested LLMs
- **Parsing is the bottleneck**: Errors predominantly come from parsing failures, not reasoning errors
- **Reasoning models ≠ better parsers**: DeepSeek-R1 has the highest symbolic accuracy (97.9%) but lowest parsing rate (20.6%), making general-purpose chat models more suitable

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── train_data.json                # Training data (960 samples)
│   └── test_data/
│       ├── test_data_subtask_1.json
│       ├── test_data_subtask_2.json
│       ├── test_data_subtask_3.json
│       └── test_data_subtask_4.json
├── logic_lm.py                        # Regex-based rule system (24 valid syllogistic forms)
├── hybrid_pipeline_ensemble.py        # ST1 prediction pipeline
├── predict_subtask2_hybrid.py         # ST2 prediction pipeline
├── predict_subtask3_hybrid.py         # ST3 prediction pipeline
├── predict_subtask4_hybrid.py         # ST4 prediction pipeline
├── trainer_debiased.py                # DeBERTa trainer (ST1/ST2)
├── trainer_multilingual.py            # XLM-RoBERTa trainer (ST3/ST4)
├── ablation_llm_comparison.py         # Ablation: LLM parser comparison
├── analyze_content_effect.py          # Ablation: content effect analysis
└── run_ablation.sh                    # Ablation: pipeline layer experiments
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"  # Required for symbolic parsing layer
```

## Usage

### Training

**DeBERTa (Subtask 1/2):**
```bash
python trainer_debiased.py \
    --data_path data/train_data.json \
    --model_name microsoft/deberta-v3-large \
    --output_dir models/deberta \
    --seeds "13,21,42,87" \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_label_smoothing \
    --label_smoothing 0.1
```

**XLM-RoBERTa (Subtask 3/4):**
```bash
python trainer_multilingual.py \
    --data_path data/train_data.json \
    --model_name xlm-roberta-large \
    --output_dir models/xlmr \
    --seeds "13,21,42,87" \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-5 \
    --lora_r 16 \
    --lora_alpha 32
```

### Prediction

**Subtask 1 (Full Pipeline):**
```bash
python hybrid_pipeline_ensemble.py \
    --mode predict \
    --test-data data/test_data/test_data_subtask_1.json \
    --output predictions_st1.json \
    --deberta-model models/deberta \
    --seeds "13,21,42,87" \
    --deepseek-model deepseek-chat
```

**Subtask 2:**
```bash
python predict_subtask2_hybrid.py \
    --test_path data/test_data/test_data_subtask_2.json \
    --output_path predictions_st2.json \
    --deberta_path models/deberta \
    --seeds "13,21,42,87"
```

**Subtask 3:**
```bash
python predict_subtask3_hybrid.py \
    --model_path models/xlmr \
    --seeds "13,21,42,87" \
    --test_path data/test_data/test_data_subtask_3.json \
    --output_path predictions_st3.json \
    --use_deepseek
```

**Subtask 4:**
```bash
python predict_subtask4_hybrid.py \
    --test_path data/test_data/test_data_subtask_4.json \
    --output_path predictions_st4.json \
    --xlmr_path models/xlmr \
    --seeds "13,21,42,87"
```

### Ablation Experiments

**LLM Parser Comparison (on training data):**
```bash
python ablation_llm_comparison.py --data data/train_data.json
```

**Pipeline Layer Ablation (generates predictions for Codabench):**
```bash
bash run_ablation.sh
```

**Content Effect Analysis:**
```bash
python analyze_content_effect.py \
    --data data/train_data.json \
    --results-dir ablation_results \
    --output-dir analysis_results
```

## Model Weights

Model weights are not included due to size (~17GB total). Use the training scripts above to reproduce. Expected training time: ~30 min per seed on a single NVIDIA 4090.

| Model | Base | Seeds | Usage |
|-------|------|-------|-------|
| DeBERTa-v3-large + LoRA | `microsoft/deberta-v3-large` | 13, 21, 42, 87 | ST1, ST2 |
| XLM-RoBERTa-large + LoRA | `xlm-roberta-large` | 13, 21, 42, 87 | ST3, ST4 |

## Citation

```bibtex
@inproceedings{fu2026semeval,
    title={junhaofu at SemEval-2026 Task 11: A Neuro-Symbolic Hybrid Pipeline for Content-Independent Syllogistic Reasoning},
    author={Fu, Junhao},
    booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
    year={2026}
}
```
