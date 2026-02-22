#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - 三层推理Pipeline

架构:
┌─────────────────────────────────────────────────────────────┐
│  三段论文本                                                  │
│      ↓                                                      │
│  Layer 1: 规则系统 (logic_lm_v3) ─────成功───→ Valid/Invalid  │
│      ↓ 失败                                                 │
│  Layer 2: DeepSeek API + 符号验证 ────成功───→ Valid/Invalid  │
│      ↓ 失败                                                 │
│  Layer 3: DeBERTa 直接分类 ──────────────────→ Valid/Invalid |
└─────────────────────────────────────────────────────────────┘

运行前设置:
export DEEPSEEK_API_KEY="your-api-key"

使用:
python hybrid_pipeline.py --test-data ../data/official/test_data/test_data_subtask_1.json --output predictions.json
"""

import json
import os
import re
import time
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# ============== 导入规则系统 ==============
# 支持 logic_lm.py 和 logic_lm_v3.py 两种文件名
try:
    from logic_lm import LogicLMPredictor, VALID_SYLLOGISMS
    RULE_SYSTEM_AVAILABLE = True
except ImportError:
    # try:
    #     from logic_lm_v3 import LogicLMPredictor, VALID_SYLLOGISMS
    #     RULE_SYSTEM_AVAILABLE = True
    # except ImportError:
    print("警告: logic_lm.py / logic_lm_v3.py 未找到，规则系统不可用")
    RULE_SYSTEM_AVAILABLE = False

# ============== DeepSeek API ==============
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("警告: openai 库未安装，DeepSeek API不可用")
    OPENAI_AVAILABLE = False

# ============== DeBERTa ==============
try:
    import torch
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("警告: transformers 库未安装，DeBERTa不可用")
    TRANSFORMERS_AVAILABLE = False


# ============== 24种有效三段论（用于DeepSeek验证） ==============
VALID_FORMS = {
    # Figure 1
    ("AAA", 1), ("EAE", 1), ("AII", 1), ("EIO", 1), ("AAI", 1), ("EAO", 1),
    # Figure 2
    ("EAE", 2), ("AEE", 2), ("EIO", 2), ("AOO", 2), ("EAO", 2), ("AEO", 2),
    # Figure 3
    ("IAI", 3), ("AII", 3), ("OAO", 3), ("EIO", 3), ("EAO", 3), ("AAI", 3),
    # Figure 4
    ("AEE", 4), ("IAI", 4), ("EIO", 4), ("EAO", 4), ("AAI", 4), ("AEO", 4),
}


# ============== DeepSeek API 相关函数 ==============
PARSE_PROMPT = """你是一个逻辑学专家。请将三段论解析为结构化的JSON格式。

## 命题类型说明
- A (全称肯定): "All X are Y", "Every X is Y", "Any X is Y"
- E (全称否定): "No X are Y", "No X is Y" 
- I (特称肯定): "Some X are Y", "There are X that are Y"
- O (特称否定): "Some X are not Y", "Not all X are Y"

## 输出格式
严格输出JSON，不要有其他内容：
{
  "p1_type": "A/E/I/O",
  "p1_subj": "第一前提的主项",
  "p1_pred": "第一前提的谓项",
  "p2_type": "A/E/I/O", 
  "p2_subj": "第二前提的主项",
  "p2_pred": "第二前提的谓项",
  "c_type": "A/E/I/O",
  "c_subj": "结论的主项",
  "c_pred": "结论的谓项"
}

## 示例

输入: All dogs are mammals. All mammals are animals. Therefore, all dogs are animals.
输出: {"p1_type": "A", "p1_subj": "dogs", "p1_pred": "mammals", "p2_type": "A", "p2_subj": "mammals", "p2_pred": "animals", "c_type": "A", "c_subj": "dogs", "c_pred": "animals"}

输入: No reptiles are mammals. Some pets are reptiles. Therefore, some pets are not mammals.
输出: {"p1_type": "E", "p1_subj": "reptiles", "p1_pred": "mammals", "p2_type": "I", "p2_subj": "pets", "p2_pred": "reptiles", "c_type": "O", "c_subj": "pets", "c_pred": "mammals"}

输入: Some flowers are not roses. All roses are plants. Therefore, some plants are not flowers.
输出: {"p1_type": "O", "p1_subj": "flowers", "p1_pred": "roses", "p2_type": "A", "p2_subj": "roses", "p2_pred": "plants", "c_type": "O", "c_subj": "plants", "c_pred": "flowers"}

## 注意事项
1. 主项(subj)是命题的主语，谓项(pred)是命题的谓语
2. 提取核心名词/名词短语，去掉冠词和修饰语
3. 保持术语的原始形式（单复数一致）
4. 如果有干扰句，只关注构成三段论的三个核心句子（两个前提+一个结论）

现在请解析以下三段论：
"""


def get_deepseek_client():
    """获取DeepSeek API客户端"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def parse_with_deepseek(client, syllogism: str, model: str = "deepseek-chat") -> Optional[Dict]:
    """用DeepSeek API解析三段论"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个逻辑学专家，专门分析三段论的结构。只输出JSON，不要有任何其他内容。"},
                {"role": "user", "content": PARSE_PROMPT + syllogism}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        return json.loads(content)
        
    except Exception as e:
        return None


def validate_deepseek_output(output: Dict) -> bool:
    """验证DeepSeek输出格式"""
    if not output:
        return False
    
    required_keys = ['p1_type', 'p1_subj', 'p1_pred', 
                     'p2_type', 'p2_subj', 'p2_pred',
                     'c_type', 'c_subj', 'c_pred']
    
    for key in required_keys:
        if key not in output:
            return False
    
    for key in ['p1_type', 'p2_type', 'c_type']:
        val = output.get(key, '').upper()
        if val not in ['A', 'E', 'I', 'O']:
            return False
        # 标准化为大写
        output[key] = val
    
    for key in ['p1_subj', 'p1_pred', 'p2_subj', 'p2_pred', 'c_subj', 'c_pred']:
        if not output.get(key, '').strip():
            return False
    
    return True


def normalize_term(term: str) -> str:
    """标准化术语（小写，去除多余空格，单复数归一化）"""
    term = term.lower().strip()
    
    # 去除常见前缀
    prefixes = ['a ', 'an ', 'the ', 'some ', 'all ', 'every ', 'any ']
    for prefix in prefixes:
        if term.startswith(prefix):
            term = term[len(prefix):]
    
    # 单复数归一化（简单规则）
    if term.endswith('ies') and len(term) > 4:
        term = term[:-3] + 'y'  # categories -> category
    elif term.endswith('es') and len(term) > 3:
        if term.endswith('sses') or term.endswith('xes') or term.endswith('ches') or term.endswith('shes'):
            term = term[:-2]  # boxes -> box, classes -> class
        else:
            term = term[:-1]  # vehicles -> vehicle (keep the e)
    elif term.endswith('s') and not term.endswith('ss') and len(term) > 2:
        term = term[:-1]  # cars -> car
    
    return term.strip()


def terms_match(t1: str, t2: str) -> bool:
    """检查两个术语是否匹配（考虑单复数和子串）"""
    n1 = normalize_term(t1)
    n2 = normalize_term(t2)
    
    # 完全匹配
    if n1 == n2:
        return True
    
    # 子串匹配（较短的是较长的子串，且长度至少3）
    if len(n1) >= 3 and len(n2) >= 3:
        if n1 in n2 or n2 in n1:
            # 确保不是太短的匹配
            shorter = min(len(n1), len(n2))
            if shorter >= 3:
                return True
    
    return False


def calculate_mood(output: Dict) -> str:
    """计算Mood"""
    return (output['p1_type'] + output['p2_type'] + output['c_type']).upper()


def calculate_figure(output: Dict) -> Optional[int]:
    """计算Figure（使用模糊匹配）"""
    # 提取并标准化术语
    p1_subj = normalize_term(output['p1_subj'])
    p1_pred = normalize_term(output['p1_pred'])
    p2_subj = normalize_term(output['p2_subj'])
    p2_pred = normalize_term(output['p2_pred'])
    c_subj = normalize_term(output['c_subj'])
    c_pred = normalize_term(output['c_pred'])
    
    # S = 结论主项, P = 结论谓项
    S = c_subj
    P = c_pred
    
    # 判断一个术语是否匹配结论中的S或P
    def matches_conclusion(term):
        return terms_match(term, S) or terms_match(term, P)
    
    # 找出中项M（出现在前提中但不匹配结论中任何项的术语）
    premise_terms = [
        (p1_subj, 'p1_subj'),
        (p1_pred, 'p1_pred'),
        (p2_subj, 'p2_subj'),
        (p2_pred, 'p2_pred')
    ]
    
    # 找出不匹配结论的术语作为中项候选
    middle_candidates = []
    for term, pos in premise_terms:
        if not matches_conclusion(term):
            middle_candidates.append((term, pos))
    
    # 找出在两个前提中都出现的中项
    M = None
    m_positions = []
    
    if len(middle_candidates) >= 2:
        # 检查哪些候选在两个前提中都出现
        for i, (t1, pos1) in enumerate(middle_candidates):
            for t2, pos2 in middle_candidates[i+1:]:
                if terms_match(t1, t2):
                    M = t1
                    m_positions = [pos1, pos2]
                    break
            if M:
                break
    
    if not M:
        # 如果没找到，尝试只用一个中项
        if len(middle_candidates) == 1:
            M, pos = middle_candidates[0]
            if pos.startswith('p1'):
                if terms_match(M, p2_subj):
                    m_positions = [pos, 'p2_subj']
                elif terms_match(M, p2_pred):
                    m_positions = [pos, 'p2_pred']
            else:
                if terms_match(M, p1_subj):
                    m_positions = ['p1_subj', pos]
                elif terms_match(M, p1_pred):
                    m_positions = ['p1_pred', pos]
    
    if not M or len(m_positions) < 2:
        return None
    
    # 判断Figure
    has_p1_subj = 'p1_subj' in m_positions
    has_p1_pred = 'p1_pred' in m_positions
    has_p2_subj = 'p2_subj' in m_positions
    has_p2_pred = 'p2_pred' in m_positions
    
    if has_p1_subj and has_p2_pred:
        return 1
    elif has_p1_pred and has_p2_pred:
        return 2
    elif has_p1_subj and has_p2_subj:
        return 3
    elif has_p1_pred and has_p2_subj:
        return 4
    else:
        return None


def check_validity_by_table(mood: str, figure: int) -> bool:
    """查表判断有效性"""
    return (mood, figure) in VALID_FORMS


# ============== DeBERTa 相关函数 ==============
# 导入自定义模型类
try:
    from trainer_debiased import DebiasedDeBERTaClassifier
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    try:
        from trainer_subtask1 import DebiasedDeBERTaClassifier
        CUSTOM_MODEL_AVAILABLE = True
    except ImportError:
        CUSTOM_MODEL_AVAILABLE = False
        print("警告: DebiasedDeBERTaClassifier 未找到")



def _parse_seed_list(seeds_str: str):
    """Parse comma/space-separated seed list into unique ints (preserve order)."""
    if seeds_str is None:
        return None
    s = str(seeds_str).strip()
    if not s:
        return []
    parts = re.split(r"[\s,]+", s)
    seeds = []
    seen = set()
    for p in parts:
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            raise ValueError(f"Invalid seed value: {p!r}")
        if v not in seen:
            seeds.append(v)
            seen.add(v)
    return seeds


def _is_valid_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(os.path.join(path, "best_model.pt"))


def _resolve_seed_model_paths(base_dir: str, seeds):
    """Resolve per-seed model directories under base_dir."""
    if not seeds:
        raise ValueError("Empty seed list for DeBERTa ensemble.")
    base_dir = os.path.abspath(base_dir)
    resolved = []
    missing = []
    for seed in seeds:
        candidates = [
            os.path.join(base_dir, f"seed{seed}"),
            os.path.join(base_dir, f"seed_{seed}"),
            os.path.join(base_dir, f"split_seed{seed}"),
            os.path.join(base_dir, str(seed)),
            os.path.join(base_dir, f"split_{seed}"),
        ]
        found = None
        for c in candidates:
            if _is_valid_model_dir(c):
                found = c
                break
        if found is None and os.path.isdir(base_dir):
            try:
                for name in os.listdir(base_dir):
                    p = os.path.join(base_dir, name)
                    if not os.path.isdir(p):
                        continue
                    if not _is_valid_model_dir(p):
                        continue
                    if re.search(rf"(?<!\d){seed}(?!\d)", name):
                        found = p
                        break
            except FileNotFoundError:
                pass
        if found is None:
            missing.append(seed)
        else:
            resolved.append(found)
    if missing:
        raise ValueError(
            f"Cannot resolve model dirs for seeds {missing} under base dir: {base_dir}. "
            f"Expected subdirs like seed13/seed21/... each containing config.json and best_model.pt."
        )
    return resolved


class DeBERTaPredictor:
    """Single DeBERTa (DebiasedDeBERTaClassifier) predictor wrapper."""
    def __init__(self, model_path: str, device: str = "cuda", use_fp16: bool = True):
        self.device = device
        self.use_fp16 = bool(use_fp16) and str(device).startswith("cuda")

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        model_name = cfg.get("model_name", "microsoft/deberta-v3-large")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = DebiasedDeBERTaClassifier(
            model_name=model_name,
            lora_r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            lora_targets=tuple([
                x.strip() for x in cfg.get("lora_targets", "query_proj,key_proj,value_proj,dense").split(",")
                if x.strip()
            ]),
            use_template_fusion=cfg.get("use_template_fusion", False),
            template_fusion_type=cfg.get("template_fusion_type", "concat"),
            use_scl=cfg.get("use_scl", False),
            scl_temp=cfg.get("scl_temp", 0.07),
            scl_weight=cfg.get("scl_weight", 0.5),
            use_adversarial=cfg.get("use_adversarial", False),
            adv_weight=cfg.get("adv_weight", 0.2),
            grl_alpha=cfg.get("grl_alpha", 0.5),
            use_label_smoothing=cfg.get("use_label_smoothing", True),
            label_smoothing=cfg.get("label_smoothing", 0.1),
            use_focal_loss=cfg.get("use_focal_loss", False),
            focal_gamma=cfg.get("focal_gamma", 2.0),
        )

        ckpt_path = os.path.join(model_path, "best_model.pt")
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, syllogism: str) -> float:
        """Return P(valid=1)."""
        inputs = self.tokenizer(
            syllogism,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs["logits"]  # [1, 2]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[0, 1].item())

    def predict(self, syllogism: str) -> bool:
        return self.predict_proba(syllogism) >= 0.5


class DeBERTaEnsemblePredictor:
    """Multi-seed ensemble predictor.

    method:
      - prob: average softmax probabilities P(valid)
      - logit: average logits then softmax
      - vote: majority vote over predicted labels (ties resolved by prob-average)
    """
    def __init__(self, model_paths, device: str = "cuda", method: str = "prob", use_fp16: bool = True):
        if not model_paths or len(model_paths) < 2:
            raise ValueError("DeBERTaEnsemblePredictor requires at least 2 model paths.")
        self.device = device
        self.method = method
        self.use_fp16 = bool(use_fp16) and str(device).startswith("cuda")

        self.models = []
        self.tokenizer = None

        for mp in model_paths:
            config_path = os.path.join(mp, "config.json")
            with open(config_path, "r") as f:
                cfg = json.load(f)

            model_name = cfg.get("model_name", "microsoft/deberta-v3-large")
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = DebiasedDeBERTaClassifier(
                model_name=model_name,
                lora_r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                lora_targets=tuple([
                    x.strip() for x in cfg.get("lora_targets", "query_proj,key_proj,value_proj,dense").split(",")
                    if x.strip()
                ]),
                use_template_fusion=cfg.get("use_template_fusion", False),
                template_fusion_type=cfg.get("template_fusion_type", "concat"),
                use_scl=cfg.get("use_scl", False),
                scl_temp=cfg.get("scl_temp", 0.07),
                scl_weight=cfg.get("scl_weight", 0.5),
                use_adversarial=cfg.get("use_adversarial", False),
                adv_weight=cfg.get("adv_weight", 0.2),
                grl_alpha=cfg.get("grl_alpha", 0.5),
                use_label_smoothing=cfg.get("use_label_smoothing", True),
                label_smoothing=cfg.get("label_smoothing", 0.1),
                use_focal_loss=cfg.get("use_focal_loss", False),
                focal_gamma=cfg.get("focal_gamma", 2.0),
            )

            ckpt_path = os.path.join(mp, "best_model.pt")
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            if self.use_fp16:
                model.half()
            model.eval()
            self.models.append(model)

        if self.tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer for DeBERTa ensemble.")

    @torch.inference_mode()
    def predict_proba(self, syllogism: str) -> float:
        inputs = self.tokenizer(
            syllogism,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        logits_list = []
        votes = []
        for model in self.models:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            logits = outputs["logits"]  # [1,2]
            logits_list.append(logits.float())
            votes.append(int(torch.argmax(logits, dim=-1).item()))

        if self.method == "vote":
            ones = sum(votes)
            zeros = len(votes) - ones
            if ones > zeros:
                return 1.0
            if zeros > ones:
                return 0.0
            # tie -> prob-average

        if self.method == "logit":
            avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)  # [1,2]
            probs = torch.softmax(avg_logits, dim=-1)
            return float(probs[0, 1].item())

        # default: prob-average
        probs_list = [torch.softmax(lg, dim=-1) for lg in logits_list]
        avg_prob = torch.mean(torch.stack([p[0, 1] for p in probs_list]), dim=0)
        return float(avg_prob.item())

    def predict(self, syllogism: str) -> bool:
        return self.predict_proba(syllogism) >= 0.5


# ============== 主Pipeline ==============

class HybridPipeline:
    def __init__(self, 
                 deberta_model_path: str = None,
                 deepseek_model: str = "deepseek-chat",
                 device: str = "cuda",
                 use_rules: bool = True,
                 use_deepseek: bool = True,
                 use_deberta: bool = True,
                 deberta_ensemble_method: str = "prob",
                 deberta_use_fp16: bool = True):
        
        self.use_rules = use_rules and RULE_SYSTEM_AVAILABLE
        self.use_deepseek = use_deepseek and OPENAI_AVAILABLE
        self.use_deberta = use_deberta and TRANSFORMERS_AVAILABLE and CUSTOM_MODEL_AVAILABLE and deberta_model_path
        self.deberta_ensemble_method = deberta_ensemble_method
        self.deberta_use_fp16 = deberta_use_fp16
        
        # 初始化规则系统
        if self.use_rules:
            self.rule_predictor = LogicLMPredictor()
            print("✓ 规则系统已加载")
        
        # 初始化DeepSeek
        if self.use_deepseek:
            self.deepseek_client = get_deepseek_client()
            self.deepseek_model = deepseek_model
            if self.deepseek_client:
                print(f"✓ DeepSeek API已连接 (模型: {deepseek_model})")
            else:
                print("✗ DeepSeek API未配置 (缺少DEEPSEEK_API_KEY)")
                self.use_deepseek = False
        
        # 初始化DeBERTa
        if self.use_deberta:
            try:
                if isinstance(deberta_model_path, (list, tuple)):
                    self.deberta_predictor = DeBERTaEnsemblePredictor(
                        list(deberta_model_path),
                        device=device,
                        method=self.deberta_ensemble_method,
                        use_fp16=self.deberta_use_fp16,
                    )
                    print(f"✓ DeBERTa ensemble已加载: {len(deberta_model_path)} models")
                else:
                    self.deberta_predictor = DeBERTaPredictor(
                        deberta_model_path,
                        device=device,
                        use_fp16=self.deberta_use_fp16,
                    )
                    print(f"✓ DeBERTa模型已加载: {deberta_model_path}")

            except Exception as e:
                print(f"✗ DeBERTa加载失败: {e}")
                self.use_deberta = False
        
        # 统计
        self.stats = {
            'total': 0,
            'rule_success': 0,
            'deepseek_success': 0,
            'deberta_used': 0,
            'no_prediction': 0
        }
        
        # 打印Pipeline状态总结
        print("\n" + "=" * 50)
        print("Pipeline 状态总结")
        print("=" * 50)
        print(f"Layer 1 (规则系统): {'✓ 启用' if self.use_rules else '✗ 禁用'}")
        print(f"Layer 2 (DeepSeek):  {'✓ 启用' if self.use_deepseek else '✗ 禁用'}")
        print(f"Layer 3 (DeBERTa):   {'✓ 启用' if self.use_deberta else '✗ 禁用'}")
        if not self.use_deberta and not self.use_deepseek:
            print("\n⚠️ 警告: DeepSeek 和 DeBERTa 都未启用，规则系统失败时将使用默认值 False！")
        print("=" * 50)
    
    def predict_one(self, syllogism: str) -> Dict:
        """预测单个样本"""
        result = {
            'syllogism': syllogism,
            'prediction': None,
            'source': None,
            'details': {}
        }
        
        self.stats['total'] += 1

        # Layer 0: 两个特称前提快速检查（III/IOI/OII等一定无效）
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', syllogism) if s.strip()]
        if len(sentences) >= 2:
            some_count = sum(1 for s in sentences[:-1]
                            if re.match(r'^(Some|There exist some|There are some|A few|A portion of|At least one|At least some|Among)\b', s, re.I))
            if some_count >= 2:
                result['prediction'] = False
                result['source'] = 'rule_particular'
                result['details'] = {'reason': 'Two particular premises'}
                self.stats['rule_success'] += 1
                return result

        # Layer 0.5: 双重否定预处理 "There are no X that are not Y" → "All X are Y"
        syllogism = re.sub(
            r"There are no ([^.]+?) that are not ([^.]+?)\.",
            r"All \1 are \2.",
            syllogism,
            flags=re.IGNORECASE
        )
        
        # Layer 0.6: "a number of X are not Y" → "Some X are not Y" (确保识别为O类)
        syllogism = re.sub(
            r"[Aa] number of ([^.]+?) are not ([^.]+?)\.",
            r"Some \1 are not \2.",
            syllogism,
            flags=re.IGNORECASE
        )
        
        # Layer 1: 规则系统
        if self.use_rules:
            rule_result = self.rule_predictor.predict(syllogism)
            
            # 检查规则系统是否成功
            if rule_result.get('source') == 'symbolic' and rule_result.get('figure') is not None:
                result['prediction'] = rule_result['prediction'] == 'valid'
                result['source'] = 'rule_system'
                result['details'] = {
                    'mood': rule_result.get('mood'),
                    'figure': rule_result.get('figure'),
                    'form_name': rule_result.get('form_name')
                }
                self.stats['rule_success'] += 1
                return result
        
        # Layer 2: DeepSeek + 符号验证
        if self.use_deepseek and self.deepseek_client:
            parsed = parse_with_deepseek(self.deepseek_client, syllogism, self.deepseek_model)
            
            if validate_deepseek_output(parsed):
                mood = calculate_mood(parsed)
                figure = calculate_figure(parsed)
                
                if figure is not None:
                    is_valid = check_validity_by_table(mood, figure)
                    result['prediction'] = is_valid
                    result['source'] = 'deepseek_symbolic'
                    result['details'] = {
                        'parsed': parsed,
                        'mood': mood,
                        'figure': figure
                    }
                    self.stats['deepseek_success'] += 1
                    time.sleep(0.1)  # 避免API限流
                    return result
            
            time.sleep(0.1)
        
        # Layer 3: DeBERTa
        if self.use_deberta:
            is_valid = self.deberta_predictor.predict(syllogism)
            result['prediction'] = is_valid
            result['source'] = 'deberta'
            self.stats['deberta_used'] += 1
            return result
        
        # 所有方法都失败
        result['prediction'] = False  # 默认invalid
        result['source'] = 'default'
        self.stats['no_prediction'] += 1
        return result
    
    def predict_batch(self, syllogisms: List[str], show_progress: bool = True) -> List[Dict]:
        """批量预测"""
        results = []
        iterator = tqdm(syllogisms, desc="预测中") if show_progress else syllogisms
        
        for syllogism in iterator:
            result = self.predict_one(syllogism)
            results.append(result)
        
        return results
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("Pipeline统计")
        print("=" * 50)
        print(f"总样本数: {self.stats['total']}")
        print(f"规则系统成功: {self.stats['rule_success']} ({100*self.stats['rule_success']/max(1,self.stats['total']):.1f}%)")
        print(f"DeepSeek成功: {self.stats['deepseek_success']} ({100*self.stats['deepseek_success']/max(1,self.stats['total']):.1f}%)")
        print(f"DeBERTa兜底: {self.stats['deberta_used']} ({100*self.stats['deberta_used']/max(1,self.stats['total']):.1f}%)")
        if self.stats['no_prediction'] > 0:
            print(f"无法预测: {self.stats['no_prediction']}")


def evaluate_on_train(pipeline: HybridPipeline, data_path: str, num_samples: int = None):
    """在训练集上评估（有标签）"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    print(f"\n评估数据集: {data_path}")
    print(f"样本数量: {len(data)}")
    
    correct = 0
    plausible_correct, plausible_total = 0, 0
    implausible_correct, implausible_total = 0, 0
    
    results = []
    
    for item in tqdm(data, desc="评估中"):
        syllogism = item['syllogism']
        true_validity = item['validity']
        plausibility = item.get('plausibility')
        
        result = pipeline.predict_one(syllogism)
        result['true_validity'] = true_validity
        result['plausibility'] = plausibility
        
        is_correct = (result['prediction'] == true_validity)
        result['correct'] = is_correct
        
        if is_correct:
            correct += 1
        
        if plausibility is not None:
            if plausibility:
                plausible_total += 1
                if is_correct:
                    plausible_correct += 1
            else:
                implausible_total += 1
                if is_correct:
                    implausible_correct += 1
        
        results.append(result)
    
    # 打印结果
    pipeline.print_stats()
    
    accuracy = 100 * correct / len(data)
    print(f"\n准确率: {accuracy:.2f}% ({correct}/{len(data)})")
    
    if plausible_total > 0 and implausible_total > 0:
        plausible_acc = 100 * plausible_correct / plausible_total
        implausible_acc = 100 * implausible_correct / implausible_total
        tce = abs(plausible_acc - implausible_acc)
        
        print(f"\nPlausible准确率: {plausible_acc:.2f}% ({plausible_correct}/{plausible_total})")
        print(f"Implausible准确率: {implausible_acc:.2f}% ({implausible_correct}/{implausible_total})")
        print(f"TCE: {tce:.2f}%")
        
        import math
        score = accuracy / (1 + math.log(1 + tce))
        print(f"Score: {score:.2f}")
    
    return results


def predict_test(pipeline: HybridPipeline, data_path: str, output_path: str):
    """预测测试集并保存结果"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n预测测试集: {data_path}")
    print(f"样本数量: {len(data)}")
    
    predictions = []
    detailed_predictions = []  # 包含完整信息的版本
    
    # 按来源统计
    source_counts = {'rule_system': 0, 'deepseek_symbolic': 0, 'deberta': 0, 'default': 0}
    source_valid = {'rule_system': 0, 'deepseek_symbolic': 0, 'deberta': 0, 'default': 0}
    
    for item in tqdm(data, desc="预测中"):
        syllogism = item['syllogism']
        item_id = item.get('id', '')
        
        result = pipeline.predict_one(syllogism)
        
        # 记录来源统计
        source = result.get('source', 'default')
        source_counts[source] = source_counts.get(source, 0) + 1
        if result['prediction']:
            source_valid[source] = source_valid.get(source, 0) + 1
        
        # 提交格式（只有id和validity）
        predictions.append({
            'id': item_id,
            'validity': result['prediction']
        })
        
        # 详细格式（用于调试）
        detailed_predictions.append({
            'id': item_id,
            'validity': result['prediction'],
            'source': result['source'],
            'details': result.get('details', {})
        })
    
    pipeline.print_stats()
    
    # 打印详细的来源统计
    print("\n" + "=" * 50)
    print("按来源统计预测结果")
    print("=" * 50)
    for source, count in source_counts.items():
        if count > 0:
            valid = source_valid.get(source, 0)
            invalid = count - valid
            print(f"{source}: {count} 样本 (Valid={valid}, Invalid={invalid})")
    
    # 保存提交格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n提交文件已保存到: {output_path}")
    
    # 保存详细格式
    detailed_output = output_path.replace('.json', '_detailed.json')
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(detailed_predictions, f, ensure_ascii=False, indent=2)
    print(f"详细文件已保存到: {detailed_output}")
    
    # 统计预测分布
    valid_count = sum(1 for p in predictions if p['validity'])
    invalid_count = len(predictions) - valid_count
    print(f"\n最终预测分布: Valid={valid_count}, Invalid={invalid_count}")
    
    # 警告：如果全是一个类别
    if valid_count == 0:
        print("\n⚠️ 警告: 所有预测都是 Invalid！请检查模型是否正确加载。")
    elif invalid_count == 0:
        print("\n⚠️ 警告: 所有预测都是 Valid！请检查模型是否正确加载。")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="SemEval 2026 Task 11 - 三层推理Pipeline")
    
    parser.add_argument("--mode", type=str, choices=["evaluate", "predict"], default="evaluate",
                        help="模式: evaluate(在训练集评估) 或 predict(预测测试集)")
    parser.add_argument("--train-data", type=str, default="../data/official/train_data.json",
                        help="训练数据路径")
    parser.add_argument("--test-data", type=str, default=None,
                        help="测试数据路径")
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="预测输出路径")
    parser.add_argument("--deberta-model", type=str, default="/root/autodl-tmp/models/balanced",
                        help="DeBERTa模型路径")
    parser.add_argument("--seeds", type=str, default=None,
                        help="DeBERTa ensemble seeds (comma/space-separated), e.g. 13,21,42,87. If provided, --deberta-model is treated as a base directory containing per-seed subdirs.")
    parser.add_argument("--ensemble", type=str, default="prob", choices=["prob", "logit", "vote"],
                        help="Aggregation for multi-seed DeBERTa: prob(average probs), logit(average logits), vote(majority vote).")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for DeBERTa inference on CUDA (reduces memory).")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 even on CUDA.")
    parser.add_argument("--deepseek-model", type=str, default="deepseek-chat",
                        choices=["deepseek-chat", "deepseek-reasoner"],
                        help="DeepSeek模型")
    parser.add_argument("--num", type=int, default=None,
                        help="评估样本数量（用于快速测试）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--no-rules", action="store_true",
                        help="禁用规则系统")
    parser.add_argument("--no-deepseek", action="store_true",
                        help="禁用DeepSeek")
    parser.add_argument("--no-deberta", action="store_true",
                        help="禁用DeBERTa")
    
    args = parser.parse_args()
    
    # 初始化Pipeline
    # Resolve DeBERTa model paths (single model or multi-seed ensemble)
    deberta_use_fp16 = (str(args.device).startswith("cuda") and not args.no_fp16) or bool(args.fp16)
    if args.seeds:
        seed_list = _parse_seed_list(args.seeds)
        deberta_paths = _resolve_seed_model_paths(args.deberta_model, seed_list)
    else:
        deberta_paths = args.deberta_model

    pipeline = HybridPipeline(
        deberta_model_path=deberta_paths,
        deepseek_model=args.deepseek_model,
        device=args.device,
        deberta_ensemble_method=args.ensemble,
        deberta_use_fp16=deberta_use_fp16,
        use_rules=not args.no_rules,
        use_deepseek=not args.no_deepseek,
        use_deberta=not args.no_deberta
    )
    
    if args.mode == "evaluate":
        evaluate_on_train(pipeline, args.train_data, args.num)
    else:
        if not args.test_data:
            print("错误: predict模式需要指定--test-data")
            return
        predict_test(pipeline, args.test_data, args.output)


if __name__ == "__main__":
    main()