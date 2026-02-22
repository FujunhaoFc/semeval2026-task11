#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Subtask 2: Hybrid Predictor

架构:
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Parser (分割句子，识别结论)                         │
├─────────────────────────────────────────────────────────────┤
│  Step 2: DeepSeek检索 (选出2个相关前提)                      │
│      ↓ 失败                                                 │
│  Fallback: 关键词检索                                        │
├─────────────────────────────────────────────────────────────┤
│  Step 3: DeepSeek符号推理 (判断validity)                     │
│      ↓ 失败                                                 │
│  Fallback: DeBERTa ensemble                                  │
└─────────────────────────────────────────────────────────────┘

Usage:
    python predict_subtask2_hybrid.py \
        --test_path ../data/official/test_data/test_data_subtask_2.json \
        --output_path predictions_subtask2.json \
        --deberta_path /root/autodl-tmp/models/balanced \
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
# Parser: 分割句子，识别结论
# =============================================================================
class Subtask2Parser:
    """解析带干扰句的三段论"""
    
    def __init__(self):
        self.conclusion_markers = sorted([
            "therefore", "thus", "hence", "consequently", "as a result",
            "it follows that", "it follows", "from this",
            "this proves that", "this proves",
            "we must conclude that", "we must conclude",
            "it can be concluded that", "it can be concluded",
            "this means that", "this means",
            "this leads to the conclusion that", "this leads to",
            "this implies that", "this implies",
            "it must be that", "it must be the case that",
            "it logically follows that", "it logically follows",
            "from this, it follows that",
            "this has led to the conclusion that", "this has led to",
            "as such", "so,", "so ",
            "consequently, it is concluded that",
            "therefore, it is concluded that",
        ], key=len, reverse=True)
        
        self.sentence_end_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def _split_into_sentences(self, text: str) -> List[str]:
        protected = text
        abbreviations = [
            ("i.e.", "<<<IE>>>"), ("e.g.", "<<<EG>>>"),
            ("etc.", "<<<ETC>>>"), ("Dr.", "<<<DR>>>"),
            ("Mr.", "<<<MR>>>"), ("Mrs.", "<<<MRS>>>"),
        ]
        for abbr, placeholder in abbreviations:
            protected = protected.replace(abbr, placeholder)
        
        sentences = self.sentence_end_pattern.split(protected)
        
        result = []
        for sent in sentences:
            for abbr, placeholder in abbreviations:
                sent = sent.replace(placeholder, abbr)
            sent = sent.strip()
            if sent:
                result.append(sent)
        
        return result
    
    def _find_conclusion_marker(self, sentence: str) -> Tuple[Optional[str], int]:
        sentence_lower = sentence.lower()
        for marker in self.conclusion_markers:
            pos = sentence_lower.find(marker)
            if pos != -1:
                if pos == 0 or sentence_lower[pos-1] in ' \t.,;:':
                    return marker, pos
        return None, -1
    
    def _clean_conclusion(self, text: str, marker: str) -> str:
        pattern = re.compile(re.escape(marker) + r'[,\s]*', re.IGNORECASE)
        result = pattern.sub('', text, count=1).strip()
        if result.lower().startswith('that '):
            result = result[5:]
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        return result
    
    def parse(self, syllogism_text: str) -> Dict:
        sentences = self._split_into_sentences(syllogism_text)
        
        if not sentences:
            return {
                "premises": [], "premise_count": 0,
                "conclusion": "", "conclusion_raw": "",
                "conclusion_marker": None, "full_text": syllogism_text,
                "parse_success": False, "error": "No sentences found"
            }
        
        conclusion_idx = -1
        conclusion_marker = None
        
        for i in range(len(sentences) - 1, -1, -1):
            marker, pos = self._find_conclusion_marker(sentences[i])
            if marker is not None:
                conclusion_idx = i
                conclusion_marker = marker
                break
        
        if conclusion_idx == -1:
            conclusion_idx = len(sentences) - 1
            conclusion_marker = None
        
        premises = sentences[:conclusion_idx]
        conclusion_raw = sentences[conclusion_idx]
        
        if conclusion_marker:
            conclusion = self._clean_conclusion(conclusion_raw, conclusion_marker)
        else:
            conclusion = conclusion_raw
        
        return {
            "premises": premises,
            "premise_count": len(premises),
            "conclusion": conclusion,
            "conclusion_raw": conclusion_raw,
            "conclusion_marker": conclusion_marker,
            "full_text": syllogism_text,
            "parse_success": True,
            "error": None
        }


# =============================================================================
# 关键词检索器 (Fallback)
# =============================================================================
class KeywordRetriever:
    """基于关键词覆盖的前提检索器"""
    
    def __init__(self):
        self.stopwords = {
            'all', 'every', 'some', 'no', 'any', 'each', 'none', 'few', 'many',
            'a', 'an', 'the', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did',
            'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as',
            'and', 'or', 'but', 'if', 'then', 'than',
            'it', 'its', 'they', 'them', 'their', 'there', 'here',
            'not', 'also', 'only', 'just', 'even',
            'thing', 'things', 'type', 'types', 'kind', 'kinds',
            'therefore', 'thus', 'hence', 'consequently',
            'known', 'fact', 'true', 'case', 'way', 'without',
            'classified', 'considered', 'called', 'single', 'entire',
        }
    
    def normalize_word(self, word: str) -> str:
        word = word.lower().strip(string.punctuation)
        if len(word) > 3:
            if word.endswith('ies'):
                return word[:-3] + 'y'
            elif word.endswith('es') and word[-3] in 'sxzh':
                return word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
        return word
    
    def extract_keywords(self, text: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = set()
        for word in words:
            if word not in self.stopwords and len(word) > 2:
                normalized = self.normalize_word(word)
                if normalized not in self.stopwords:
                    keywords.add(normalized)
        return keywords
    
    def retrieve(self, premises: List[str], conclusion: str) -> List[int]:
        if len(premises) < 2:
            return list(range(len(premises)))
        
        conclusion_kw = self.extract_keywords(conclusion)
        premise_kw_list = [self.extract_keywords(p) for p in premises]
        
        best_pair = (0, 1)
        best_score = -1
        
        for i, j in combinations(range(len(premises)), 2):
            combined = premise_kw_list[i] | premise_kw_list[j]
            coverage = len(combined & conclusion_kw)
            
            # 中项检测
            p1_middle = premise_kw_list[i] - conclusion_kw
            p2_middle = premise_kw_list[j] - conclusion_kw
            shared_middle = p1_middle & p2_middle
            
            score = coverage * 2 + len(shared_middle)
            
            if score > best_score:
                best_score = score
                best_pair = (i, j)
        
        return sorted(list(best_pair))


# =============================================================================
# DeepSeek 检索
# =============================================================================
RETRIEVAL_PROMPT = """你是逻辑学专家。给定多个前提句和一个结论，找出与结论直接相关的两个前提。

三段论的结构：
- 两个前提共享一个"中项"（中项出现在两个前提中，但不出现在结论中）
- 结论的主项(S)出现在一个前提中
- 结论的谓项(P)出现在另一个前提中
- 这两个前提组合起来可以推导出结论

前提列表：
{premises_list}

结论：{conclusion}

请仔细分析每个前提与结论的关系，找出能够组成有效三段论的两个前提。
只输出两个相关前提的索引（从0开始），格式：[idx1, idx2]
例如：[0, 3] 表示第0个和第3个前提是相关的。

答案："""


def get_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def retrieve_with_deepseek(
    client, 
    premises: List[str], 
    conclusion: str,
    model: str = "deepseek-chat"
) -> Optional[List[int]]:
    """用DeepSeek检索相关前提"""
    try:
        premises_list = "\n".join([f"[{i}] {p}" for i, p in enumerate(premises)])
        prompt = RETRIEVAL_PROMPT.format(
            premises_list=premises_list,
            conclusion=conclusion
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是逻辑学专家，擅长分析三段论结构。只输出索引列表，不要解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        content = response.choices[0].message.content.strip()
        
        # 解析 [idx1, idx2] 格式
        match = re.search(r'\[(\d+)\s*,\s*(\d+)\]', content)
        if match:
            idx1, idx2 = int(match.group(1)), int(match.group(2))
            if 0 <= idx1 < len(premises) and 0 <= idx2 < len(premises) and idx1 != idx2:
                return sorted([idx1, idx2])
        
        return None
    except Exception as e:
        return None


# =============================================================================
# DeepSeek 符号推理 (复用Subtask1)
# =============================================================================
PARSE_PROMPT = """你是一个逻辑学专家。请将三段论解析为结构化的JSON格式。

## 命题类型说明
- A (全称肯定): "All X are Y", "Every X is Y", "Any X is Y"
- E (全称否定): "No X are Y", "No X is Y"
- I (特称肯定): "Some X are Y", "There are X that are Y", "A few X are Y"
- O (特称否定): "Some X are not Y", "Not all X are Y"

## 特别注意 I 和 A 的区分
- "some", "a few", "certain", "a portion of" → I (特称)
- "all", "every", "any", "each" → A (全称)

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

现在请解析以下三段论：
"""


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
                {"role": "system", "content": "你是逻辑学专家。只输出JSON，不要有任何其他内容。"},
                {"role": "user", "content": PARSE_PROMPT + syllogism}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # 提取JSON
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
# DeBERTa Ensemble (复用)
# =============================================================================
class DeBERTaClassifier(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_labels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.15,
        lora_targets=("query_proj", "key_proj", "value_proj", "dense"),
        use_scl=False,
        scl_proj_dim=256,
        use_adversarial=False,
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.deberta = get_peft_model(self.deberta, peft_cfg)
        
        h = self.deberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, num_labels)
        
        self.use_scl = use_scl
        if use_scl:
            self.projection = nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, scl_proj_dim),
            )
        
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.plausibility_classifier = nn.Sequential(
                nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h // 2, 2),
            )
    
    def forward(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class DeBERTaEnsemble:
    def __init__(self, model_paths: List[str], device: str = "cuda", use_fp16: bool = True):
        self.device = device
        self.models = []
        self.tokenizer = None
        
        for path in model_paths:
            config_path = os.path.join(path, "config.json")
            ckpt_path = os.path.join(path, "best_model.pt")
            
            with open(config_path) as f:
                cfg = json.load(f)
            
            model_name = cfg.get("model_name", "microsoft/deberta-v3-large")
            
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model = DeBERTaClassifier(
                model_name=model_name,
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
        
        print(f"✓ DeBERTa ensemble已加载: {len(self.models)} models")
    
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
# Subtask 2 Hybrid Pipeline
# =============================================================================
class Subtask2HybridPipeline:
    def __init__(
        self,
        deberta_paths: List[str],
        device: str = "cuda",
        use_deepseek: bool = True,
        deepseek_model: str = "deepseek-chat",
        use_fp16: bool = True,
    ):
        self.device = device
        self.parser = Subtask2Parser()
        self.keyword_retriever = KeywordRetriever()
        
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
        
        # DeBERTa
        self.deberta = DeBERTaEnsemble(deberta_paths, device, use_fp16)
        
        # Stats
        self.stats = {
            "total": 0,
            "deepseek_retrieval_success": 0,
            "keyword_retrieval_used": 0,
            "deepseek_validity_success": 0,
            "deberta_validity_used": 0,
        }
        
        print(f"\nPipeline状态:")
        print(f"  DeepSeek: {'✓ 启用' if self.use_deepseek else '✗ 禁用'}")
        print(f"  DeBERTa: ✓ 启用 ({len(self.deberta.models)} models)")
    
    def _construct_syllogism(self, p1: str, p2: str, conclusion: str) -> str:
        """构建标准三段论格式"""
        # 确保句子以句号结尾
        if not p1.endswith('.'):
            p1 += '.'
        if not p2.endswith('.'):
            p2 += '.'
        if not conclusion.endswith('.'):
            conclusion += '.'
        
        return f"{p1} {p2} Therefore, {conclusion}"
    
    def predict_one(self, item: Dict) -> Dict:
        """预测单个样本"""
        item_id = item.get("id", "")
        syllogism_text = item.get("syllogism", "")
        
        self.stats["total"] += 1
        
        # Step 1: 解析
        parsed = self.parser.parse(syllogism_text)
        
        if not parsed["parse_success"] or len(parsed["premises"]) < 2:
            # 解析失败，直接用DeBERTa
            result = self.deberta.predict(syllogism_text)
            return {
                "id": item_id,
                "validity": result["validity"],
                "relevant_premises": [0, 1] if result["validity"] else [],
                "_source": "deberta_parse_failed",
            }
        
        premises = parsed["premises"]
        conclusion = parsed["conclusion"]
        
        # Step 2: 检索相关前提
        relevant_indices = None
        
        if self.use_deepseek and self.deepseek_client:
            relevant_indices = retrieve_with_deepseek(
                self.deepseek_client, premises, conclusion, self.deepseek_model
            )
            if relevant_indices:
                self.stats["deepseek_retrieval_success"] += 1
            time.sleep(0.1)
        
        if relevant_indices is None:
            relevant_indices = self.keyword_retriever.retrieve(premises, conclusion)
            self.stats["keyword_retrieval_used"] += 1
        
        # Step 3: 构建三段论
        idx1, idx2 = relevant_indices[0], relevant_indices[1]
        p1 = premises[idx1]
        p2 = premises[idx2]
        syllogism = self._construct_syllogism(p1, p2, conclusion)
        
        # Step 4: 判断Validity
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
            result = self.deberta.predict(syllogism)
            validity = result["validity"]
            source = "deberta"
            self.stats["deberta_validity_used"] += 1
        
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
        print(f"DeepSeek检索成功: {self.stats['deepseek_retrieval_success']} ({100*self.stats['deepseek_retrieval_success']/max(1,self.stats['total']):.1f}%)")
        print(f"关键词检索使用: {self.stats['keyword_retrieval_used']} ({100*self.stats['keyword_retrieval_used']/max(1,self.stats['total']):.1f}%)")
        print(f"DeepSeek符号推理成功: {self.stats['deepseek_validity_success']} ({100*self.stats['deepseek_validity_success']/max(1,self.stats['total']):.1f}%)")
        print(f"DeBERTa判断使用: {self.stats['deberta_validity_used']} ({100*self.stats['deberta_validity_used']/max(1,self.stats['total']):.1f}%)")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Subtask 2 Hybrid Predictor")
    
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions_subtask2.json")
    parser.add_argument("--deberta_path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="13,21,42,87")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_deepseek", action="store_true")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    
    args = parser.parse_args()
    
    # 解析seeds
    seed_list = [s.strip() for s in args.seeds.replace(",", " ").split() if s.strip()]
    deberta_paths = []
    for seed in seed_list:
        path = os.path.join(args.deberta_path, f"seed{seed}")
        if os.path.exists(path):
            deberta_paths.append(path)
        else:
            print(f"警告: 未找到 {path}")
    
    if not deberta_paths:
        raise ValueError("没有找到任何DeBERTa模型！")
    
    # 加载数据
    print(f"加载测试数据: {args.test_path}")
    with open(args.test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"测试样本数: {len(data)}")
    
    # 创建pipeline
    use_fp16 = (args.device.startswith("cuda") and not args.no_fp16) or args.fp16
    
    pipeline = Subtask2HybridPipeline(
        deberta_paths=deberta_paths,
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