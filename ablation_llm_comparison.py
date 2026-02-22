#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Ablation: LLM Comparison for Symbolic Parsing

测试不同 LLM 作为三段论 symbolic parser 的效果。
流程: LLM parsing → mood+figure → 查24种有效三段论表 → validity

Usage:
    # 跑所有模型
    python ablation_llm_comparison.py --data train_data.json

    # 只跑指定模型
    python ablation_llm_comparison.py --data train_data.json --models deepseek-v3 qwen3-max

    # 从断点续跑
    python ablation_llm_comparison.py --data train_data.json --models glm-4.7 --resume

    # 查看已有结果
    python ablation_llm_comparison.py --data train_data.json --report-only

环境变量:
    export DEEPSEEK_API_KEY="..."
    export QWEN_API_KEY="..."
    export SILICONFLOW_API_KEY="..."
    export OPENAI_API_KEY="..."
"""

import os
import json
import re
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai: pip install openai")
    exit(1)

from tqdm import tqdm


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
# Parsing Prompt (与 hybrid_pipeline_ensemble.py 完全一致)
# =============================================================================
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

SYSTEM_PROMPT = "你是一个逻辑学专家，专门分析三段论的结构。只输出JSON，不要有任何其他内容。"


# =============================================================================
# 模型配置
# =============================================================================
MODEL_CONFIGS = {
    "deepseek-v3": {
        "display_name": "DeepSeek-V3",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "is_reasoning": False,
        "rate_limit_delay": 0.1,
    },
    "deepseek-r1": {
        "display_name": "DeepSeek-R1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-reasoner",
        "is_reasoning": True,
        "rate_limit_delay": 0.5,  # R1 较慢，间隔大一点
    },
    "qwen3-max": {
        "display_name": "Qwen3-Max",
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "is_reasoning": False,
        "rate_limit_delay": 0.1,
    },
    "glm-5": {
        "display_name": "GLM-5",
        "api_key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5",
        "is_reasoning": False,
        "rate_limit_delay": 0.1,
    },
    "minimax-m2.5": {
        "display_name": "MiniMax-M2.5",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "model": "MiniMax-M2.5",
        "is_reasoning": False,
        "rate_limit_delay": 0.1,
    },
    "kimi-k2.5": {
        "display_name": "Kimi-K2.5",
        "api_key_env": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5",
        "is_reasoning": False,
        "rate_limit_delay": 0.1,
    },
    "gpt-5.2": {
        "display_name": "GPT-5.2",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.2",
        "is_reasoning": False,
        "rate_limit_delay": 0.2,
    },
}


# =============================================================================
# 核心逻辑函数 (复用自 hybrid_pipeline_ensemble.py)
# =============================================================================

def validate_output(output: Dict) -> bool:
    """验证LLM输出格式"""
    if not output:
        return False
    required_keys = [
        'p1_type', 'p1_subj', 'p1_pred',
        'p2_type', 'p2_subj', 'p2_pred',
        'c_type', 'c_subj', 'c_pred',
    ]
    for key in required_keys:
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
    """标准化术语"""
    term = term.lower().strip()
    prefixes = ['a ', 'an ', 'the ', 'some ', 'all ', 'every ', 'any ']
    for prefix in prefixes:
        if term.startswith(prefix):
            term = term[len(prefix):]
    if term.endswith('ies') and len(term) > 4:
        term = term[:-3] + 'y'
    elif term.endswith('es') and len(term) > 3:
        if term.endswith('sses') or term.endswith('xes') or term.endswith('ches') or term.endswith('shes'):
            term = term[:-2]
        else:
            term = term[:-1]
    elif term.endswith('s') and not term.endswith('ss') and len(term) > 2:
        term = term[:-1]
    return term.strip()


def terms_match(t1: str, t2: str) -> bool:
    """检查两个术语是否匹配"""
    n1 = normalize_term(t1)
    n2 = normalize_term(t2)
    if n1 == n2:
        return True
    if len(n1) >= 3 and len(n2) >= 3:
        if n1 in n2 or n2 in n1:
            shorter = min(len(n1), len(n2))
            if shorter >= 3:
                return True
    return False


def calculate_mood(output: Dict) -> str:
    return (output['p1_type'] + output['p2_type'] + output['c_type']).upper()


def calculate_figure(output: Dict) -> Optional[int]:
    """计算Figure（使用模糊匹配）"""
    p1_subj = normalize_term(output['p1_subj'])
    p1_pred = normalize_term(output['p1_pred'])
    p2_subj = normalize_term(output['p2_subj'])
    p2_pred = normalize_term(output['p2_pred'])
    c_subj = normalize_term(output['c_subj'])
    c_pred = normalize_term(output['c_pred'])

    S = c_subj
    P = c_pred

    def matches_conclusion(term):
        return terms_match(term, S) or terms_match(term, P)

    premise_terms = [
        (p1_subj, 'p1_subj'),
        (p1_pred, 'p1_pred'),
        (p2_subj, 'p2_subj'),
        (p2_pred, 'p2_pred'),
    ]

    middle_candidates = []
    for term, pos in premise_terms:
        if not matches_conclusion(term):
            middle_candidates.append((term, pos))

    M = None
    m_positions = []

    if len(middle_candidates) >= 2:
        for i, (t1, pos1) in enumerate(middle_candidates):
            for t2, pos2 in middle_candidates[i + 1:]:
                if terms_match(t1, t2):
                    M = t1
                    m_positions = [pos1, pos2]
                    break
            if M:
                break

    if not M:
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
    return (mood, figure) in VALID_FORMS


# =============================================================================
# LLM API 调用
# =============================================================================

def extract_json_from_text(text: str) -> Optional[Dict]:
    """从LLM返回的文本中提取JSON"""
    if not text:
        return None
    text = text.strip()

    # 尝试提取 ```json ... ``` 块
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)

    # 提取第一个 { ... }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def call_llm(client: OpenAI, model: str, syllogism: str,
             is_reasoning: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], str]:
    """
    调用LLM解析三段论。
    返回: (parsed_dict or None, raw_response_text)
    """
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PARSE_PROMPT + syllogism},
                ],
                max_tokens=800 if is_reasoning else 500,
            )
            # reasoning 模型不支持 temperature 参数
            # kimi-k2.5 只允许 temperature=1
            if not is_reasoning and "kimi" not in model.lower():
                kwargs["temperature"] = 0

            response = client.chat.completions.create(**kwargs)

            # 提取内容 (R1 的 reasoning_content 在 message 里，但最终回答还是 content)
            content = ""
            msg = response.choices[0].message

            # 有些 reasoning 模型把 reasoning 放在 reasoning_content 字段
            # 但最终的结构化输出在 content 里
            if hasattr(msg, 'content') and msg.content:
                content = msg.content.strip()

            # 如果 content 为空，尝试从 reasoning_content 提取
            if not content and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                content = msg.reasoning_content.strip()

            parsed = extract_json_from_text(content)
            return parsed, content

        except Exception as e:
            err_str = str(e)
            if "rate" in err_str.lower() or "429" in err_str:
                wait = 5 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"\n  API error after {max_retries} attempts: {e}")
                return None, f"ERROR: {e}"

    return None, "ERROR: max retries exceeded"


# =============================================================================
# 评估计算
# =============================================================================

def compute_metrics(results: List[Dict], data: List[Dict]) -> Dict:
    """计算所有指标"""
    total = len(data)
    assert len(results) == total

    # 基础统计
    parse_success = 0       # JSON 解析成功
    figure_success = 0      # mood + figure 计算成功 (= 覆盖率)
    symbolic_correct = 0    # 符号推理覆盖的样本中正确的
    overall_correct = 0     # 所有样本的正确数（未覆盖的默认 False）

    # 按 plausibility 分组
    plausible_correct, plausible_total = 0, 0
    implausible_correct, implausible_total = 0, 0

    # 四象限统计
    quadrants = {
        "VP": {"correct": 0, "total": 0},
        "VI": {"correct": 0, "total": 0},
        "IP": {"correct": 0, "total": 0},
        "II": {"correct": 0, "total": 0},
    }

    for i, (res, item) in enumerate(zip(results, data)):
        true_validity = item['validity']
        plausibility = item.get('plausibility')

        # 四象限 key
        if true_validity and plausibility:
            qk = "VP"
        elif true_validity and not plausibility:
            qk = "VI"
        elif not true_validity and plausibility:
            qk = "IP"
        else:
            qk = "II"
        quadrants[qk]["total"] += 1

        # 解析是否成功
        if res.get("parse_success"):
            parse_success += 1

        # figure 是否成功
        predicted = res.get("predicted_validity")
        if res.get("figure_success"):
            figure_success += 1
            is_correct = (predicted == true_validity)
            if is_correct:
                symbolic_correct += 1
                overall_correct += 1
                quadrants[qk]["correct"] += 1
            # plausibility 分组
            if plausibility is not None:
                if plausibility:
                    plausible_total += 1
                    if is_correct:
                        plausible_correct += 1
                else:
                    implausible_total += 1
                    if is_correct:
                        implausible_correct += 1
        else:
            # 未覆盖的默认 False
            is_correct = (False == true_validity)
            if is_correct:
                overall_correct += 1
                quadrants[qk]["correct"] += 1
            if plausibility is not None:
                if plausibility:
                    plausible_total += 1
                    if is_correct:
                        plausible_correct += 1
                else:
                    implausible_total += 1
                    if is_correct:
                        implausible_correct += 1

    # 计算指标
    parse_rate = 100 * parse_success / total
    coverage = 100 * figure_success / total
    symbolic_acc = 100 * symbolic_correct / figure_success if figure_success > 0 else 0
    overall_acc = 100 * overall_correct / total

    # TCE
    plausible_acc = 100 * plausible_correct / plausible_total if plausible_total > 0 else 0
    implausible_acc = 100 * implausible_correct / implausible_total if implausible_total > 0 else 0
    tce = abs(plausible_acc - implausible_acc)

    # Combined Score
    combined_score = overall_acc / (1 + math.log(1 + tce))

    # 四象限 accuracy
    quad_acc = {}
    for qk, qv in quadrants.items():
        quad_acc[qk] = 100 * qv["correct"] / qv["total"] if qv["total"] > 0 else 0

    return {
        "total": total,
        "parse_success": parse_success,
        "parse_rate": round(parse_rate, 2),
        "figure_success": figure_success,
        "coverage": round(coverage, 2),
        "symbolic_acc": round(symbolic_acc, 2),
        "overall_correct": overall_correct,
        "overall_acc": round(overall_acc, 2),
        "plausible_acc": round(plausible_acc, 2),
        "implausible_acc": round(implausible_acc, 2),
        "tce": round(tce, 2),
        "combined_score": round(combined_score, 2),
        "quadrant_acc": {k: round(v, 2) for k, v in quad_acc.items()},
    }


# =============================================================================
# 单个模型的完整实验流程
# =============================================================================

def run_single_model(model_key: str, data: List[Dict], output_dir: str,
                     resume: bool = False) -> Optional[Dict]:
    """运行单个模型的实验"""
    config = MODEL_CONFIGS[model_key]
    display = config["display_name"]

    # 检查 API key
    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        print(f"\n✗ {display}: 环境变量 {config['api_key_env']} 未设置，跳过")
        return None

    # 断点续跑: 加载已有结果
    cache_path = os.path.join(output_dir, f"cache_{model_key}.json")
    results = {}
    if resume and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"\n  从缓存加载了 {len(results)} 条已有结果")

    # 初始化 client
    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    delay = config["rate_limit_delay"]
    is_reasoning = config.get("is_reasoning", False)

    print(f"\n{'='*60}")
    print(f"  模型: {display}")
    print(f"  API:  {config['base_url']}")
    print(f"  Model: {config['model']}")
    print(f"  Reasoning: {is_reasoning}")
    print(f"{'='*60}")

    # 逐条处理
    skipped = 0
    for item in tqdm(data, desc=f"{display}"):
        item_id = item['id']

        # 已有结果则跳过
        if item_id in results:
            skipped += 1
            continue

        syllogism = item['syllogism']

        # 调用 LLM
        parsed, raw = call_llm(
            client, config["model"], syllogism,
            is_reasoning=is_reasoning,
        )

        # 验证和计算
        res = {
            "id": item_id,
            "parse_success": False,
            "figure_success": False,
            "predicted_validity": None,
            "mood": None,
            "figure": None,
            "raw_response_len": len(raw) if raw else 0,
        }

        if parsed and validate_output(parsed):
            res["parse_success"] = True
            mood = calculate_mood(parsed)
            figure = calculate_figure(parsed)
            res["mood"] = mood
            res["figure"] = figure

            if figure is not None:
                res["figure_success"] = True
                res["predicted_validity"] = check_validity_by_table(mood, figure)

        results[item_id] = res

        # 定期保存缓存 (每50条)
        if (len(results) - skipped) % 50 == 0:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)

        time.sleep(delay)

    # 最终保存缓存
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # 按原始数据顺序排列结果
    ordered_results = []
    for item in data:
        r = results.get(item['id'])
        if r is None:
            r = {"id": item['id'], "parse_success": False,
                 "figure_success": False, "predicted_validity": None}
        ordered_results.append(r)

    # 计算指标
    metrics = compute_metrics(ordered_results, data)
    metrics["model_key"] = model_key
    metrics["display_name"] = display

    # 保存详细结果
    detail_path = os.path.join(output_dir, f"detail_{model_key}.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": config,
            "metrics": metrics,
            "results": ordered_results,
        }, f, ensure_ascii=False, indent=2)

    # 打印本模型结果
    print(f"\n  --- {display} 结果 ---")
    print(f"  解析成功率:  {metrics['parse_rate']}% ({metrics['parse_success']}/{metrics['total']})")
    print(f"  覆盖率:     {metrics['coverage']}% ({metrics['figure_success']}/{metrics['total']})")
    print(f"  符号准确率:  {metrics['symbolic_acc']}% (覆盖样本中)")
    print(f"  整体准确率:  {metrics['overall_acc']}% ({metrics['overall_correct']}/{metrics['total']})")
    print(f"  Plausible:  {metrics['plausible_acc']}%")
    print(f"  Implausible: {metrics['implausible_acc']}%")
    print(f"  TCE:        {metrics['tce']}%")
    print(f"  Combined:   {metrics['combined_score']}")
    print(f"  四象限: VP={metrics['quadrant_acc']['VP']}% VI={metrics['quadrant_acc']['VI']}% "
          f"IP={metrics['quadrant_acc']['IP']}% II={metrics['quadrant_acc']['II']}%")

    return metrics


# =============================================================================
# 汇总报告
# =============================================================================

def print_comparison_table(all_metrics: List[Dict]):
    """打印对比表格"""
    print("\n")
    print("=" * 120)
    print("  LLM Symbolic Parser 对比实验结果")
    print("=" * 120)

    # 表头
    header = (
        f"{'Model':<16s} | {'Parse%':>7s} | {'Cover%':>7s} | "
        f"{'SymAcc%':>7s} | {'OvAcc%':>7s} | "
        f"{'Plaus%':>7s} | {'Implau%':>7s} | {'TCE':>6s} | {'Score':>7s} | "
        f"{'VP':>6s} | {'VI':>6s} | {'IP':>6s} | {'II':>6s}"
    )
    print(header)
    print("-" * 120)

    # 按 Combined Score 降序
    for m in sorted(all_metrics, key=lambda x: x['combined_score'], reverse=True):
        q = m['quadrant_acc']
        row = (
            f"{m['display_name']:<16s} | {m['parse_rate']:>6.1f}% | {m['coverage']:>6.1f}% | "
            f"{m['symbolic_acc']:>6.1f}% | {m['overall_acc']:>6.1f}% | "
            f"{m['plausible_acc']:>6.1f}% | {m['implausible_acc']:>6.1f}% | {m['tce']:>5.1f}% | "
            f"{m['combined_score']:>7.2f} | "
            f"{q['VP']:>5.1f}% | {q['VI']:>5.1f}% | {q['IP']:>5.1f}% | {q['II']:>5.1f}%"
        )
        print(row)

    print("=" * 120)
    print("\n指标说明:")
    print("  Parse%   = JSON解析成功率")
    print("  Cover%   = Mood+Figure计算成功的覆盖率")
    print("  SymAcc%  = 覆盖样本中的准确率")
    print("  OvAcc%   = 整体准确率 (未覆盖的默认 Invalid)")
    print("  TCE      = |acc_plausible - acc_implausible|")
    print("  Score    = Accuracy / (1 + ln(1 + TCE))")
    print("  VP/VI/IP/II = Valid+Plausible / Valid+Implausible / Invalid+Plausible / Invalid+Implausible 的准确率")


def generate_latex_table(all_metrics: List[Dict]) -> str:
    """生成 LaTeX 表格代码"""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|ccc|cc|cc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{LLM Parser} & \textbf{Parse\%} & \textbf{Cover\%} & \textbf{Sym.Acc} & "
                 r"\textbf{Acc} & \textbf{TCE} & \textbf{Score} \\")
    lines.append(r"\hline")

    for m in sorted(all_metrics, key=lambda x: x['combined_score'], reverse=True):
        name = m['display_name'].replace("_", r"\_")
        lines.append(
            f"{name} & {m['parse_rate']:.1f} & {m['coverage']:.1f} & {m['symbolic_acc']:.1f} & "
            f"{m['overall_acc']:.1f} & {m['tce']:.1f} & {m['combined_score']:.2f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Comparison of different LLMs as symbolic parsers on the training set (960 samples). "
                 r"Parse\%: JSON parsing success rate. Cover\%: percentage where mood and figure are "
                 r"successfully computed. Sym.Acc: accuracy on covered samples. "
                 r"Uncovered samples default to Invalid.}")
    lines.append(r"\label{tab:llm_comparison}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Comparison for Symbolic Parsing")
    parser.add_argument("--data", type=str, required=True, help="训练数据路径 (train_data.json)")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"指定模型 (可选: {', '.join(MODEL_CONFIGS.keys())}). 默认跑全部.")
    parser.add_argument("--resume", action="store_true", help="从缓存断点续跑")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                        help="输出目录 (default: ablation_results)")
    parser.add_argument("--report-only", action="store_true",
                        help="只从已有结果生成报告，不跑实验")
    args = parser.parse_args()

    # 加载数据
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载数据: {len(data)} 条样本")

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要跑的模型
    if args.models:
        model_keys = []
        for m in args.models:
            m = m.lower().strip()
            if m in MODEL_CONFIGS:
                model_keys.append(m)
            else:
                print(f"警告: 未知模型 '{m}', 可选: {', '.join(MODEL_CONFIGS.keys())}")
        if not model_keys:
            print("没有有效的模型，退出")
            return
    else:
        model_keys = list(MODEL_CONFIGS.keys())

    # report-only 模式
    if args.report_only:
        all_metrics = []
        for mk in MODEL_CONFIGS:
            detail_path = os.path.join(args.output_dir, f"detail_{mk}.json")
            if os.path.exists(detail_path):
                with open(detail_path, 'r', encoding='utf-8') as f:
                    detail = json.load(f)
                all_metrics.append(detail["metrics"])
        if all_metrics:
            print_comparison_table(all_metrics)
            latex = generate_latex_table(all_metrics)
            latex_path = os.path.join(args.output_dir, "table_llm_comparison.tex")
            with open(latex_path, 'w') as f:
                f.write(latex)
            print(f"\nLaTeX 表格已保存到: {latex_path}")
        else:
            print("未找到任何已有结果")
        return

    # 运行实验
    all_metrics = []
    for mk in model_keys:
        metrics = run_single_model(mk, data, args.output_dir, resume=args.resume)
        if metrics:
            all_metrics.append(metrics)

    # 尝试加载其他已有结果以生成完整表格
    for mk in MODEL_CONFIGS:
        if mk not in model_keys:
            detail_path = os.path.join(args.output_dir, f"detail_{mk}.json")
            if os.path.exists(detail_path):
                with open(detail_path, 'r', encoding='utf-8') as f:
                    detail = json.load(f)
                all_metrics.append(detail["metrics"])

    # 输出对比表
    if all_metrics:
        print_comparison_table(all_metrics)

        # 保存汇总
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n汇总结果已保存到: {summary_path}")

        # 生成 LaTeX 表格
        latex = generate_latex_table(all_metrics)
        latex_path = os.path.join(args.output_dir, "table_llm_comparison.tex")
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX 表格已保存到: {latex_path}")


if __name__ == "__main__":
    main()