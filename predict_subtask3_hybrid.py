#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Subtask 3 Hybrid Inference Script

支持:
1. Multi-seed ensemble (prob/logit/vote)
2. DeepSeek symbolic layer (复用Subtask1的符号推理)
3. XLM-RoBERTa / mDeBERTa / DeBERTa

架构:
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: DeepSeek API + 符号验证 ────成功───→ Valid/Invalid │
│      ↓ 失败                                                 │
│  Layer 2: XLM-RoBERTa ensemble ──────────────→ Valid/Invalid │
└─────────────────────────────────────────────────────────────┘

Usage:
    # 单模型
    python predict_subtask3_hybrid.py \
        --model_path models/subtask3/seed42/best_model.pt \
        --test_path data/test_data_subtask_3.json \
        --output_path predictions_subtask3.json

    # Ensemble
    python predict_subtask3_hybrid.py \
        --model_path models/subtask3 \
        --seeds "13,21,42" \
        --test_path data/test_data_subtask_3.json \
        --output_path predictions_subtask3.json

    # 带DeepSeek层
    python predict_subtask3_hybrid.py \
        --model_path models/subtask3 \
        --seeds "13,21,42" \
        --test_path data/test_data_subtask_3.json \
        --output_path predictions_subtask3.json \
        --use_deepseek
"""

import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import math
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# DeepSeek API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai库未安装，DeepSeek不可用")


# =============================================================================
# DeepSeek Symbolic Layer (复用Subtask1)
# =============================================================================
VALID_FORMS = {
    ("AAA", 1), ("EAE", 1), ("AII", 1), ("EIO", 1), ("AAI", 1), ("EAO", 1),
    ("EAE", 2), ("AEE", 2), ("EIO", 2), ("AOO", 2), ("EAO", 2), ("AEO", 2),
    ("IAI", 3), ("AII", 3), ("OAO", 3), ("EIO", 3), ("EAO", 3), ("AAI", 3),
    ("AEE", 4), ("IAI", 4), ("EIO", 4), ("EAO", 4), ("AAI", 4), ("AEO", 4),
}

PARSE_PROMPT = """You are a logic expert. Parse the syllogism into structured JSON format.

## Proposition Types
- A (Universal Affirmative): "All X are Y", "Every X is Y"
- E (Universal Negative): "No X are Y", "No X is Y"
- I (Particular Affirmative): "Some X are Y", "There are X that are Y", "A few X are Y"
- O (Particular Negative): "Some X are not Y", "Not all X are Y"

## Important: Distinguish I vs A
- "some", "a few", "certain", "a portion of" → I (Particular)
- "all", "every", "any", "each" → A (Universal)

## Output Format
Output ONLY JSON, no other content:
{
  "p1_type": "A/E/I/O",
  "p1_subj": "subject of premise 1",
  "p1_pred": "predicate of premise 1",
  "p2_type": "A/E/I/O", 
  "p2_subj": "subject of premise 2",
  "p2_pred": "predicate of premise 2",
  "c_type": "A/E/I/O",
  "c_subj": "subject of conclusion",
  "c_pred": "predicate of conclusion"
}

## Examples

Input: All dogs are mammals. All mammals are animals. Therefore, all dogs are animals.
Output: {"p1_type": "A", "p1_subj": "dogs", "p1_pred": "mammals", "p2_type": "A", "p2_subj": "mammals", "p2_pred": "animals", "c_type": "A", "c_subj": "dogs", "c_pred": "animals"}

Input: No reptiles are mammals. Some pets are reptiles. Therefore, some pets are not mammals.
Output: {"p1_type": "E", "p1_subj": "reptiles", "p1_pred": "mammals", "p2_type": "I", "p2_subj": "pets", "p2_pred": "reptiles", "c_type": "O", "c_subj": "pets", "c_pred": "mammals"}

Input: All hammers are tools. Some hammers are heavy. Therefore, some tools are heavy.
Output: {"p1_type": "A", "p1_subj": "hammers", "p1_pred": "tools", "p2_type": "I", "p2_subj": "hammers", "p2_pred": "heavy", "c_type": "I", "c_subj": "tools", "c_pred": "heavy"}

## Notes
1. Extract core noun phrases, remove articles and modifiers
2. Keep terms consistent across premises and conclusion
3. The input may be in various languages - extract the logical structure regardless of language

Now parse this syllogism:
"""


def get_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def parse_with_deepseek(client, syllogism: str, model: str = "deepseek-chat") -> Optional[Dict]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logic expert. Output only JSON."},
                {"role": "user", "content": PARSE_PROMPT + syllogism}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{.*\}']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1) if '```' in pattern else match.group(0)
                break
        
        return json.loads(content)
    except Exception as e:
        return None


def validate_deepseek_output(output: Dict) -> bool:
    if not output:
        return False
    
    required = ['p1_type', 'p1_subj', 'p1_pred', 'p2_type', 'p2_subj', 'p2_pred', 'c_type', 'c_subj', 'c_pred']
    for key in required:
        if key not in output:
            return False
    
    for key in ['p1_type', 'p2_type', 'c_type']:
        val = output.get(key, '').upper()
        if val not in ['A', 'E', 'I', 'O']:
            return False
        output[key] = val
    
    for key in ['p1_subj', 'p1_pred', 'p2_subj', 'p2_pred', 'c_subj', 'c_pred']:
        if not output.get(key, '').strip():
            return False
    
    return True


def normalize_term(term: str) -> str:
    term = term.lower().strip()
    for prefix in ['a ', 'an ', 'the ', 'some ', 'all ', 'every ', 'any ']:
        if term.startswith(prefix):
            term = term[len(prefix):]
    
    if term.endswith('ies') and len(term) > 4:
        term = term[:-3] + 'y'
    elif term.endswith('es') and len(term) > 3:
        term = term[:-1]
    elif term.endswith('s') and not term.endswith('ss') and len(term) > 2:
        term = term[:-1]
    
    return term.strip()


def terms_match(t1: str, t2: str) -> bool:
    n1 = normalize_term(t1)
    n2 = normalize_term(t2)
    if n1 == n2:
        return True
    if len(n1) >= 3 and len(n2) >= 3:
        if n1 in n2 or n2 in n1:
            return True
    return False


def calculate_mood(output: Dict) -> str:
    return (output['p1_type'] + output['p2_type'] + output['c_type']).upper()


def calculate_figure(output: Dict) -> Optional[int]:
    S = normalize_term(output['c_subj'])
    P = normalize_term(output['c_pred'])
    
    p1_s = normalize_term(output['p1_subj'])
    p1_p = normalize_term(output['p1_pred'])
    p2_s = normalize_term(output['p2_subj'])
    p2_p = normalize_term(output['p2_pred'])
    
    # Find middle term M
    all_terms = [(p1_s, 'p1_s'), (p1_p, 'p1_p'), (p2_s, 'p2_s'), (p2_p, 'p2_p')]
    
    M = None
    m_positions = []
    
    for term, pos in all_terms:
        if not terms_match(term, S) and not terms_match(term, P):
            if M is None:
                M = term
            if terms_match(term, M):
                m_positions.append(pos)
    
    if not M or len(m_positions) != 2:
        return None
    
    m_in_p1 = 'p1_s' in m_positions or 'p1_p' in m_positions
    m_in_p2 = 'p2_s' in m_positions or 'p2_p' in m_positions
    
    if not (m_in_p1 and m_in_p2):
        return None
    
    p1_pos = 'p1_s' if 'p1_s' in m_positions else 'p1_p'
    p2_pos = 'p2_s' if 'p2_s' in m_positions else 'p2_p'
    
    # Determine figure
    if p1_pos == 'p1_s' and p2_pos == 'p2_p':
        return 1
    elif p1_pos == 'p1_p' and p2_pos == 'p2_p':
        return 2
    elif p1_pos == 'p1_s' and p2_pos == 'p2_s':
        return 3
    elif p1_pos == 'p1_p' and p2_pos == 'p2_s':
        return 4
    
    return None


def check_validity_by_table(mood: str, figure: int) -> bool:
    return (mood, figure) in VALID_FORMS


# =============================================================================
# Model Architecture
# =============================================================================
def get_lora_target_modules(model_name: str) -> List[str]:
    model_name_lower = model_name.lower()
    if "deberta" in model_name_lower:
        return ["query_proj", "key_proj", "value_proj", "dense"]
    elif "roberta" in model_name_lower or "xlm" in model_name_lower:
        return ["query", "key", "value", "dense"]
    else:
        return ["query", "key", "value", "dense"]


class TestDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["syllogism"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "id": s.get("id", str(idx)),
            "language": s.get("language", "unknown"),
            "syllogism": s["syllogism"],
        }


class MultilingualClassifier(nn.Module):
    def __init__(
        self,
        model_name="xlm-roberta-large",
        num_labels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.15,
        lora_targets=None,
        use_scl=False,
        scl_proj_dim=256,
        use_adversarial=False,
        **kwargs,
    ):
        super().__init__()
        
        if lora_targets is None:
            lora_targets = get_lora_target_modules(model_name)
        
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, peft_cfg)
        
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, num_labels)
        
        self.use_scl = use_scl
        if use_scl:
            self.projection = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, scl_proj_dim),
            )
        
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.grl = nn.Identity()
            self.plausibility_classifier = nn.Sequential(
                nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h // 2, 2),
            )
    
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# =============================================================================
# Ensemble Predictor
# =============================================================================
class EnsemblePredictor:
    def __init__(
        self,
        model_paths: List[str],
        model_name: str,
        device: str = "cuda",
        ensemble_method: str = "prob",
        use_fp16: bool = True,
        **model_kwargs,
    ):
        self.device = device
        self.ensemble_method = ensemble_method
        self.models = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        for path in model_paths:
            # Load config
            config_path = os.path.join(os.path.dirname(path), "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            else:
                cfg = model_kwargs
            
            model = MultilingualClassifier(
                model_name=cfg.get("model_name", model_name),
                lora_r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.15),
                use_scl=cfg.get("use_scl", False),
                scl_proj_dim=cfg.get("scl_proj_dim", 256),
                use_adversarial=cfg.get("use_adversarial", False),
            )
            
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            
            if use_fp16 and device != "cpu":
                model.half()
            
            self.models.append(model)
        
        print(f"✓ Ensemble已加载: {len(self.models)} models")
    
    @torch.no_grad()
    def predict_batch(self, input_ids, attention_mask) -> torch.Tensor:
        all_logits = []
        
        for model in self.models:
            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
        
        stacked = torch.stack(all_logits, dim=0)
        
        if self.ensemble_method == "logit":
            return stacked.mean(dim=0)
        elif self.ensemble_method == "prob":
            probs = F.softmax(stacked, dim=-1)
            return probs.mean(dim=0).log()
        elif self.ensemble_method == "vote":
            votes = stacked.argmax(dim=-1)
            vote_sum = votes.sum(dim=0)
            threshold = len(self.models) / 2
            preds = (vote_sum > threshold).long()
            # Return pseudo-logits
            return torch.stack([1 - preds, preds], dim=-1).float()
        
        return stacked.mean(dim=0)
    
    def predict_single(self, syllogism: str) -> Dict:
        inputs = self.tokenizer(
            syllogism,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)
        
        logits = self.predict_batch(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        
        return {
            "prediction": bool(pred),
            "confidence": probs[0].max().item(),
            "prob_valid": probs[0][1].item(),
        }


# =============================================================================
# Hybrid Pipeline
# =============================================================================
class HybridPipelineSubtask3:
    def __init__(
        self,
        model_paths,
        model_name: str,
        device: str = "cuda",
        ensemble_method: str = "prob",
        use_deepseek: bool = False,
        deepseek_model: str = "deepseek-chat",
        use_fp16: bool = True,
    ):
        self.device = device
        self.use_deepseek = use_deepseek and OPENAI_AVAILABLE
        
        # DeepSeek client
        if self.use_deepseek:
            self.deepseek_client = get_deepseek_client()
            self.deepseek_model = deepseek_model
            if self.deepseek_client:
                print(f"✓ DeepSeek API已连接 (模型: {deepseek_model})")
            else:
                print("✗ DeepSeek API未配置 (缺少DEEPSEEK_API_KEY)")
                self.use_deepseek = False
        
        # Ensemble predictor
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        
        self.ensemble = EnsemblePredictor(
            model_paths=model_paths,
            model_name=model_name,
            device=device,
            ensemble_method=ensemble_method,
            use_fp16=use_fp16,
        )
        
        # Stats
        self.stats = {
            "total": 0,
            "deepseek_success": 0,
            "ensemble_used": 0,
        }
        
        print(f"\nPipeline状态:")
        print(f"  DeepSeek层: {'✓ 启用' if self.use_deepseek else '✗ 禁用'}")
        print(f"  Ensemble层: ✓ 启用 ({len(self.ensemble.models)} models)")
    
    def predict_one(self, syllogism: str, language: str = "unknown") -> Dict:
        result = {
            "syllogism": syllogism,
            "language": language,
            "prediction": None,
            "source": None,
            "confidence": None,
            "details": {},
        }
        
        self.stats["total"] += 1
        
        # Layer 1: DeepSeek symbolic
        if self.use_deepseek and self.deepseek_client:
            parsed = parse_with_deepseek(self.deepseek_client, syllogism, self.deepseek_model)
            
            if validate_deepseek_output(parsed):
                mood = calculate_mood(parsed)
                figure = calculate_figure(parsed)
                
                if figure is not None:
                    is_valid = check_validity_by_table(mood, figure)
                    result["prediction"] = is_valid
                    result["source"] = "deepseek_symbolic"
                    result["confidence"] = 1.0
                    result["details"] = {"parsed": parsed, "mood": mood, "figure": figure}
                    self.stats["deepseek_success"] += 1
                    time.sleep(0.1)
                    return result
            
            time.sleep(0.1)
        
        # Layer 2: Ensemble
        ensemble_result = self.ensemble.predict_single(syllogism)
        result["prediction"] = ensemble_result["prediction"]
        result["source"] = "ensemble"
        result["confidence"] = ensemble_result["confidence"]
        result["details"] = {"prob_valid": ensemble_result["prob_valid"]}
        self.stats["ensemble_used"] += 1
        
        return result
    
    def print_stats(self):
        print(f"\n{'='*50}")
        print("Pipeline统计")
        print(f"{'='*50}")
        print(f"总样本数: {self.stats['total']}")
        if self.use_deepseek:
            print(f"DeepSeek成功: {self.stats['deepseek_success']} ({100*self.stats['deepseek_success']/max(1,self.stats['total']):.1f}%)")
        print(f"Ensemble处理: {self.stats['ensemble_used']} ({100*self.stats['ensemble_used']/max(1,self.stats['total']):.1f}%)")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Subtask 3 Hybrid Inference")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. '13,21,42'")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions_subtask3.json")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Ensemble settings
    parser.add_argument("--ensemble", type=str, default="prob", choices=["prob", "logit", "vote"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    
    # DeepSeek settings
    parser.add_argument("--use_deepseek", action="store_true")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    
    args = parser.parse_args()
    
    # Resolve model paths
    if args.seeds:
        seed_list = [s.strip() for s in args.seeds.replace(",", " ").split() if s.strip()]
        model_paths = []
        for seed in seed_list:
            path = os.path.join(args.model_path, f"seed{seed}", "best_model.pt")
            if os.path.exists(path):
                model_paths.append(path)
            else:
                print(f"警告: 未找到 {path}")
        if not model_paths:
            raise ValueError("没有找到任何模型文件！")
    else:
        if os.path.isfile(args.model_path):
            model_paths = [args.model_path]
        else:
            model_paths = [os.path.join(args.model_path, "best_model.pt")]
    
    print(f"加载模型: {model_paths}")
    
    # Load test data
    print(f"\n加载测试数据: {args.test_path}")
    with open(args.test_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"测试样本数: {len(test_samples)}")
    
    # Language distribution
    lang_counts = Counter(s.get("language", "unknown") for s in test_samples)
    print("\n语言分布:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} ({100*count/len(test_samples):.1f}%)")
    
    # Create pipeline
    use_fp16 = (args.device.startswith("cuda") and not args.no_fp16) or args.fp16
    
    pipeline = HybridPipelineSubtask3(
        model_paths=model_paths,
        model_name=args.model_name,
        device=args.device,
        ensemble_method=args.ensemble,
        use_deepseek=args.use_deepseek,
        deepseek_model=args.deepseek_model,
        use_fp16=use_fp16,
    )
    
    # Run predictions
    print("\n开始预测...")
    predictions = []
    detailed_predictions = []
    
    source_counts = defaultdict(int)
    source_valid = defaultdict(int)
    lang_preds = defaultdict(lambda: {"valid": 0, "invalid": 0})
    
    for sample in tqdm(test_samples, desc="预测中"):
        result = pipeline.predict_one(sample["syllogism"], sample.get("language", "unknown"))
        
        source = result["source"]
        source_counts[source] += 1
        if result["prediction"]:
            source_valid[source] += 1
        
        lang = sample.get("language", "unknown")
        if result["prediction"]:
            lang_preds[lang]["valid"] += 1
        else:
            lang_preds[lang]["invalid"] += 1
        
        predictions.append({
            "id": sample.get("id", ""),
            "validity": result["prediction"],
        })
        
        detailed_predictions.append({
            "id": sample.get("id", ""),
            "validity": result["prediction"],
            "language": lang,
            "source": result["source"],
            "confidence": result["confidence"],
            "details": result["details"],
        })
    
    pipeline.print_stats()
    
    # Source distribution
    print(f"\n按来源统计:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        valid = source_valid[source]
        print(f"  {source}: {count} (Valid={valid}, Invalid={count-valid})")
    
    # Per-language distribution
    print(f"\n按语言统计:")
    for lang in sorted(lang_preds.keys()):
        v = lang_preds[lang]["valid"]
        iv = lang_preds[lang]["invalid"]
        print(f"  {lang}: Valid={v}, Invalid={iv}")
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n提交文件已保存: {args.output_path}")
    
    detailed_path = args.output_path.replace(".json", "_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
    print(f"详细文件已保存: {detailed_path}")
    
    # Summary
    valid_count = sum(1 for p in predictions if p["validity"])
    print(f"\n预测分布: Valid={valid_count}, Invalid={len(predictions)-valid_count}")


if __name__ == "__main__":
    main()