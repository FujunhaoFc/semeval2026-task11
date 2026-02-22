#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Subtask 4: Multilingual Retrieval + Classification

Subtask 4 = Subtask 2 的多语言版本

架构:
┌─────────────────────────────────────────────────────────────┐
│  Step 1: DeepSeek 分割+检索（一步完成）                      │
│    - 输入：完整多语言三段论文本                              │
│    - 输出：结论 + 2个相关前提索引                            │
│    - Fallback: 简单启发式分割 + 关键词检索                   │
├─────────────────────────────────────────────────────────────┤
│  Step 2: DeepSeek 符号推理                                   │
│    - 解析 Mood + Figure                                      │
│    - 查表判断 validity                                       │
│      ↓ 失败                                                 │
│  Fallback: XLM-RoBERTa ensemble                              │
└─────────────────────────────────────────────────────────────┘

Usage:
    python predict_subtask4_hybrid.py \
        --test_path ../data/official/test_data/test_data_subtask_4.json \
        --output_path predictions_subtask4.json \
        --xlmr_path /root/autodl-tmp/models/subtask3_xlmr_seeds \
        --seeds "13,21,42,87"
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import time
import argparse
import string
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 24种有效三段论
# =============================================================================
VALID_FORMS = {
    ("AAA", 1), ("EAE", 1), ("AII", 1), ("EIO", 1), ("AAI", 1), ("EAO", 1),
    ("EAE", 2), ("AEE", 2), ("EIO", 2), ("AOO", 2), ("EAO", 2), ("AEO", 2),
    ("IAI", 3), ("AII", 3), ("OAO", 3), ("EIO", 3), ("EAO", 3), ("AAI", 3),
    ("AEE", 4), ("IAI", 4), ("EIO", 4), ("EAO", 4), ("AAI", 4), ("AEO", 4),
}


# =============================================================================
# DeepSeek 分割+检索 (一步完成)
# =============================================================================
PARSE_AND_RETRIEVE_PROMPT = """You are a logic expert. Analyze this syllogism text (may contain distractor sentences).

Tasks:
1. Split the text into individual sentences
2. Identify the CONCLUSION (usually the last sentence, often marked by words like "therefore", "thus", "hence", "it follows", etc. in various languages)
3. Find the TWO premises that are logically relevant to the conclusion

A valid syllogism has:
- Two premises sharing a "middle term" (appears in both premises but NOT in conclusion)
- The subject of conclusion (S) appears in one premise
- The predicate of conclusion (P) appears in the other premise

Input text:
{text}

Output ONLY a JSON object:
{{
  "sentences": ["sentence1", "sentence2", ...],
  "conclusion_index": <index of conclusion in sentences>,
  "conclusion": "the conclusion sentence",
  "relevant_premise_indices": [idx1, idx2]
}}

Answer:"""


def get_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def parse_and_retrieve_with_deepseek(
    client, 
    text: str,
    model: str = "deepseek-chat"
) -> Optional[Dict]:
    """用DeepSeek一步完成分割和检索"""
    try:
        prompt = PARSE_AND_RETRIEVE_PROMPT.format(text=text)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logic expert. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # 提取JSON
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{.*\}']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1) if '```' in pattern else match.group(0)
                break
        
        result = json.loads(content)
        
        # 验证结果
        if not isinstance(result.get("sentences"), list) or len(result["sentences"]) < 3:
            return None
        if not isinstance(result.get("relevant_premise_indices"), list) or len(result["relevant_premise_indices"]) != 2:
            return None
        
        idx1, idx2 = result["relevant_premise_indices"]
        n = len(result["sentences"])
        if not (0 <= idx1 < n and 0 <= idx2 < n and idx1 != idx2):
            return None
        
        return result
    except Exception as e:
        return None


# =============================================================================
# 简单启发式分割 (Fallback)
# =============================================================================
def simple_split_sentences(text: str) -> List[str]:
    """简单的句子分割（多语言）"""
    # 按常见句号分割
    # 中文句号：。
    # 英文句号：.
    # 阿拉伯语句号：。
    
    # 先替换中文句号为英文句号
    text = text.replace('。', '. ')
    
    # 按句号分割
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?':
            sentence = current.strip()
            if sentence and len(sentence) > 5:  # 过滤太短的
                sentences.append(sentence)
            current = ""
    
    if current.strip() and len(current.strip()) > 5:
        sentences.append(current.strip())
    
    return sentences


def simple_retrieve(sentences: List[str]) -> Tuple[int, List[int]]:
    """简单的检索：假设最后一句是结论，返回前两句作为前提"""
    if len(sentences) < 3:
        return len(sentences) - 1, [0, 1] if len(sentences) >= 2 else [0, 0]
    
    conclusion_idx = len(sentences) - 1
    # 简单策略：返回第一句和倒数第二句
    relevant = [0, len(sentences) - 2]
    
    return conclusion_idx, relevant


# =============================================================================
# DeepSeek 符号推理
# =============================================================================
SYMBOLIC_PARSE_PROMPT = """You are a logic expert. Parse this syllogism into structured JSON.

Proposition Types:
- A (Universal Affirmative): "All X are Y", "Every X is Y"
- E (Universal Negative): "No X are Y"
- I (Particular Affirmative): "Some X are Y"
- O (Particular Negative): "Some X are not Y"

Important: Distinguish I vs A carefully:
- "some", "a few", "certain" → I
- "all", "every", "any", "each" → A

Syllogism:
{syllogism}

Output ONLY JSON:
{{
  "p1_type": "A/E/I/O",
  "p1_subj": "subject of premise 1",
  "p1_pred": "predicate of premise 1",
  "p2_type": "A/E/I/O", 
  "p2_subj": "subject of premise 2",
  "p2_pred": "predicate of premise 2",
  "c_type": "A/E/I/O",
  "c_subj": "subject of conclusion",
  "c_pred": "predicate of conclusion"
}}

Answer:"""


def parse_syllogism_with_deepseek(
    client, 
    syllogism: str,
    model: str = "deepseek-chat"
) -> Optional[Dict]:
    """用DeepSeek解析三段论"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logic expert. Output only JSON."},
                {"role": "user", "content": SYMBOLIC_PARSE_PROMPT.format(syllogism=syllogism)}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{.*\}']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1) if '```' in pattern else match.group(0)
                break
        
        return json.loads(content)
    except:
        return None


def validate_parsed_output(output: Dict) -> bool:
    """验证解析输出"""
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
    
    if p1_pos == 'p1_s' and p2_pos == 'p2_p':
        return 1
    elif p1_pos == 'p1_p' and p2_pos == 'p2_p':
        return 2
    elif p1_pos == 'p1_s' and p2_pos == 'p2_s':
        return 3
    elif p1_pos == 'p1_p' and p2_pos == 'p2_s':
        return 4
    
    return None


def check_validity(mood: str, figure: int) -> bool:
    return (mood, figure) in VALID_FORMS


# =============================================================================
# XLM-RoBERTa Ensemble
# =============================================================================
def get_lora_target_modules(model_name: str) -> List[str]:
    model_name_lower = model_name.lower()
    if "deberta" in model_name_lower:
        return ["query_proj", "key_proj", "value_proj", "dense"]
    elif "roberta" in model_name_lower or "xlm" in model_name_lower:
        return ["query", "key", "value", "dense"]
    else:
        return ["query", "key", "value", "dense"]


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


class XLMRobertaEnsemble:
    def __init__(self, model_paths: List[str], model_name: str, device: str = "cuda", use_fp16: bool = True):
        self.device = device
        self.models = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        for path in model_paths:
            config_path = os.path.join(path, "config.json")
            ckpt_path = os.path.join(path, "best_model.pt")
            
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            else:
                cfg = {}
            
            model = MultilingualClassifier(
                model_name=cfg.get("model_name", model_name),
                lora_r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.15),
                use_scl=cfg.get("use_scl", False),
                scl_proj_dim=cfg.get("scl_proj_dim", 256),
                use_adversarial=cfg.get("use_adversarial", False),
            )
            
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            
            if use_fp16 and device != "cpu":
                model.half()
            
            self.models.append(model)
        
        print(f"✓ XLM-RoBERTa ensemble已加载: {len(self.models)} models")
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to(self.device)
        
        all_probs = []
        for model in self.models:
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = F.softmax(logits.float(), dim=-1)
            all_probs.append(probs)
        
        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred = avg_probs.argmax(dim=-1).item()
        conf = avg_probs.max().item()
        
        return {
            "validity": pred == 1,
            "confidence": conf,
        }


# =============================================================================
# Subtask 4 Hybrid Pipeline
# =============================================================================
class Subtask4HybridPipeline:
    def __init__(
        self,
        xlmr_paths: List[str],
        xlmr_model_name: str,
        device: str = "cuda",
        use_deepseek: bool = True,
        deepseek_model: str = "deepseek-chat",
        use_fp16: bool = True,
    ):
        self.device = device
        
        # DeepSeek
        self.use_deepseek = use_deepseek and OPENAI_AVAILABLE
        self.deepseek_client = None
        self.deepseek_model = deepseek_model
        
        if self.use_deepseek:
            self.deepseek_client = get_deepseek_client()
            if self.deepseek_client:
                print(f"✓ DeepSeek API已连接 (模型: {deepseek_model})")
            else:
                print("✗ DeepSeek API未配置")
                self.use_deepseek = False
        
        # XLM-RoBERTa
        self.xlmr = XLMRobertaEnsemble(xlmr_paths, xlmr_model_name, device, use_fp16)
        
        # Stats
        self.stats = {
            "total": 0,
            "deepseek_parse_retrieve_success": 0,
            "fallback_parse_retrieve_used": 0,
            "deepseek_validity_success": 0,
            "xlmr_validity_used": 0,
        }
        
        print(f"\nPipeline状态:")
        print(f"  DeepSeek: {'✓ 启用' if self.use_deepseek else '✗ 禁用'}")
        print(f"  XLM-RoBERTa: ✓ 启用 ({len(self.xlmr.models)} models)")
    
    def predict_one(self, item: Dict) -> Dict:
        """预测单个样本"""
        item_id = item.get("id", "")
        syllogism_text = item.get("syllogism", "")
        
        self.stats["total"] += 1
        
        # Step 1: 分割 + 检索
        sentences = None
        conclusion = None
        relevant_indices = None
        
        if self.use_deepseek and self.deepseek_client:
            result = parse_and_retrieve_with_deepseek(
                self.deepseek_client, syllogism_text, self.deepseek_model
            )
            
            if result:
                sentences = result["sentences"]
                conclusion_idx = result.get("conclusion_index", len(sentences) - 1)
                conclusion = result.get("conclusion", sentences[conclusion_idx])
                relevant_indices = sorted(result["relevant_premise_indices"])
                self.stats["deepseek_parse_retrieve_success"] += 1
            
            time.sleep(0.1)
        
        # Fallback: 简单分割
        if sentences is None or relevant_indices is None:
            sentences = simple_split_sentences(syllogism_text)
            if len(sentences) >= 2:
                conclusion_idx, relevant_indices = simple_retrieve(sentences)
                conclusion = sentences[conclusion_idx]
            else:
                # 无法分割，直接用XLM-RoBERTa
                result = self.xlmr.predict(syllogism_text)
                return {
                    "id": item_id,
                    "validity": result["validity"],
                    "relevant_premises": [0, 1] if result["validity"] else [],
                    "_source": "xlmr_no_parse",
                }
            self.stats["fallback_parse_retrieve_used"] += 1
        
        # Step 2: 构建三段论
        idx1, idx2 = relevant_indices[0], relevant_indices[1]
        
        # 确保索引有效
        if idx1 >= len(sentences) or idx2 >= len(sentences):
            idx1, idx2 = 0, min(1, len(sentences) - 1)
        
        p1 = sentences[idx1]
        p2 = sentences[idx2]
        
        # 构建三段论文本
        syllogism = f"{p1} {p2} Therefore, {conclusion}"
        
        # Step 3: 判断Validity
        validity = None
        source = None
        
        if self.use_deepseek and self.deepseek_client:
            parsed_syl = parse_syllogism_with_deepseek(
                self.deepseek_client, syllogism, self.deepseek_model
            )
            
            if validate_parsed_output(parsed_syl):
                mood = calculate_mood(parsed_syl)
                figure = calculate_figure(parsed_syl)
                
                if figure is not None:
                    validity = check_validity(mood, figure)
                    source = "deepseek_symbolic"
                    self.stats["deepseek_validity_success"] += 1
            
            time.sleep(0.1)
        
        if validity is None:
            result = self.xlmr.predict(syllogism)
            validity = result["validity"]
            source = "xlmr"
            self.stats["xlmr_validity_used"] += 1
        
        return {
            "id": item_id,
            "validity": validity,
            "relevant_premises": relevant_indices if validity else [],
            "_source": source,
            "_retrieved": relevant_indices,
        }
    
    def predict_batch(self, data: List[Dict], show_progress: bool = True) -> List[Dict]:
        results = []
        iterator = tqdm(data, desc="预测中") if show_progress else data
        
        for item in iterator:
            result = self.predict_one(item)
            results.append(result)
        
        return results
    
    def print_stats(self):
        print(f"\n{'='*50}")
        print("Pipeline统计")
        print(f"{'='*50}")
        print(f"总样本数: {self.stats['total']}")
        print(f"DeepSeek分割+检索成功: {self.stats['deepseek_parse_retrieve_success']} ({100*self.stats['deepseek_parse_retrieve_success']/max(1,self.stats['total']):.1f}%)")
        print(f"Fallback分割+检索: {self.stats['fallback_parse_retrieve_used']} ({100*self.stats['fallback_parse_retrieve_used']/max(1,self.stats['total']):.1f}%)")
        print(f"DeepSeek符号推理成功: {self.stats['deepseek_validity_success']} ({100*self.stats['deepseek_validity_success']/max(1,self.stats['total']):.1f}%)")
        print(f"XLM-RoBERTa判断: {self.stats['xlmr_validity_used']} ({100*self.stats['xlmr_validity_used']/max(1,self.stats['total']):.1f}%)")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Subtask 4 Multilingual Hybrid Predictor")
    
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions_subtask4.json")
    parser.add_argument("--xlmr_path", type=str, required=True)
    parser.add_argument("--xlmr_model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--seeds", type=str, default="13,21,42,87")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_deepseek", action="store_true")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    
    args = parser.parse_args()
    
    # 解析seeds
    seed_list = [s.strip() for s in args.seeds.replace(",", " ").split() if s.strip()]
    xlmr_paths = []
    for seed in seed_list:
        path = os.path.join(args.xlmr_path, f"seed{seed}")
        if os.path.exists(path):
            xlmr_paths.append(path)
        else:
            print(f"警告: 未找到 {path}")
    
    if not xlmr_paths:
        raise ValueError("没有找到任何XLM-RoBERTa模型！")
    
    # 加载数据
    print(f"加载测试数据: {args.test_path}")
    with open(args.test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"测试样本数: {len(data)}")
    
    # 创建pipeline
    use_fp16 = (args.device.startswith("cuda") and not args.no_fp16) or args.fp16
    
    pipeline = Subtask4HybridPipeline(
        xlmr_paths=xlmr_paths,
        xlmr_model_name=args.xlmr_model_name,
        device=args.device,
        use_deepseek=not args.no_deepseek,
        deepseek_model=args.deepseek_model,
        use_fp16=use_fp16,
    )
    
    # 预测
    predictions = pipeline.predict_batch(data)
    
    pipeline.print_stats()
    
    # 保存结果
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 提交格式
    submission = []
    for pred in predictions:
        submission.append({
            "id": pred["id"],
            "validity": pred["validity"],
            "relevant_premises": pred["relevant_premises"],
        })
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    print(f"\n提交文件已保存: {args.output_path}")
    
    # 详细输出
    detailed_path = args.output_path.replace(".json", "_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"详细文件已保存: {detailed_path}")
    
    # 统计
    valid_count = sum(1 for p in predictions if p["validity"])
    print(f"\n预测分布: Valid={valid_count}, Invalid={len(predictions)-valid_count}")
    
    # 来源统计
    sources = defaultdict(int)
    for p in predictions:
        sources[p.get("_source", "unknown")] += 1
    print(f"\n来源分布:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({100*cnt/len(predictions):.1f}%)")


if __name__ == "__main__":
    main()