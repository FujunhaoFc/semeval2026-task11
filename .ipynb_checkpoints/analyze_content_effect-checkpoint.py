#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Content Effect & Quadrant Analysis (维度三)

分析 LLM symbolic parsing 在 validity × plausibility 四象限上的表现。

输入:
  - train_data.json (960条, 带 validity + plausibility 标签)
  - ablation_results/detail_{model}.json (维度二的详细结果)

输出:
  - 四象限 accuracy 热力图 (PDF)
  - Content effect 对比表格
  - 错误分类统计
  - LaTeX 表格

Usage:
    python analyze_content_effect.py \
        --data train_data.json \
        --results-dir ablation_results \
        --output-dir analysis_results
"""

import os
import json
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# 数据加载
# =============================================================================

def load_data(data_path: str) -> List[Dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_detail(detail_path: str) -> Dict:
    with open(detail_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_models(results_dir: str) -> Dict[str, str]:
    """发现所有 detail_{model}.json 文件"""
    models = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("detail_") and fname.endswith(".json"):
            model_key = fname[len("detail_"):-len(".json")]
            models[model_key] = os.path.join(results_dir, fname)
    return models


# =============================================================================
# 四象限分析
# =============================================================================

def get_quadrant(validity: bool, plausibility: bool) -> str:
    if validity and plausibility:
        return "VP"
    elif validity and not plausibility:
        return "VI"
    elif not validity and plausibility:
        return "IP"
    else:
        return "II"


def quadrant_analysis(data: List[Dict], results: List[Dict]) -> Dict:
    """四象限 + content effect 分析"""
    assert len(data) == len(results)

    # 四象限统计
    quads = {q: {"correct": 0, "total": 0} for q in ["VP", "VI", "IP", "II"]}

    # 按 plausibility 分组
    plaus = {"plausible": {"correct": 0, "total": 0},
             "implausible": {"correct": 0, "total": 0}}

    # 按 source 分组的四象限
    source_quads = defaultdict(lambda: {q: {"correct": 0, "total": 0} for q in ["VP", "VI", "IP", "II"]})

    # 错误分类
    errors = {
        "parse_fail": 0,        # JSON 解析失败
        "figure_fail": 0,       # mood/figure 计算失败
        "symbolic_wrong": 0,    # 符号推理判断错误
        "default_wrong": 0,     # 默认 False 判断错误（即 true validity = True 的未覆盖样本）
    }

    for item, res in zip(data, results):
        true_v = item['validity']
        true_p = item.get('plausibility')
        qk = get_quadrant(true_v, true_p)

        quads[qk]["total"] += 1

        # 预测结果
        if res.get("figure_success"):
            predicted = res["predicted_validity"]
            is_correct = (predicted == true_v)
            source = "symbolic"
        else:
            predicted = False  # 默认 Invalid
            is_correct = (False == true_v)
            source = "default"

        if is_correct:
            quads[qk]["correct"] += 1

        # plausibility 分组
        if true_p is not None:
            pk = "plausible" if true_p else "implausible"
            plaus[pk]["total"] += 1
            if is_correct:
                plaus[pk]["correct"] += 1

        # 按 source 分组
        if res.get("figure_success"):
            src = "symbolic"
        elif res.get("parse_success"):
            src = "parse_ok_no_figure"
        else:
            src = "parse_fail"
        source_quads[src][qk]["total"] += 1
        if is_correct:
            source_quads[src][qk]["correct"] += 1

        # 错误分类
        if not is_correct:
            if not res.get("parse_success"):
                errors["parse_fail"] += 1
            elif not res.get("figure_success"):
                errors["figure_fail"] += 1
            elif res.get("figure_success"):
                errors["symbolic_wrong"] += 1
            else:
                errors["default_wrong"] += 1

    # 计算各项指标
    quad_acc = {}
    for qk in ["VP", "VI", "IP", "II"]:
        t = quads[qk]["total"]
        c = quads[qk]["correct"]
        quad_acc[qk] = round(100 * c / t, 2) if t > 0 else 0

    plaus_acc = {}
    for pk in ["plausible", "implausible"]:
        t = plaus[pk]["total"]
        c = plaus[pk]["correct"]
        plaus_acc[pk] = round(100 * c / t, 2) if t > 0 else 0

    tce = round(abs(plaus_acc["plausible"] - plaus_acc["implausible"]), 2)

    overall_correct = sum(q["correct"] for q in quads.values())
    overall_total = sum(q["total"] for q in quads.values())
    overall_acc = round(100 * overall_correct / overall_total, 2) if overall_total > 0 else 0
    combined_score = round(overall_acc / (1 + math.log(1 + tce)), 2)

    # Source 分组 accuracy
    source_acc = {}
    for src, sq in source_quads.items():
        src_total = sum(sq[q]["total"] for q in sq)
        src_correct = sum(sq[q]["correct"] for q in sq)
        source_acc[src] = {
            "total": src_total,
            "correct": src_correct,
            "acc": round(100 * src_correct / src_total, 2) if src_total > 0 else 0,
            "quad_acc": {q: round(100 * sq[q]["correct"] / sq[q]["total"], 2) if sq[q]["total"] > 0 else 0
                        for q in ["VP", "VI", "IP", "II"]},
        }

    return {
        "quad_acc": quad_acc,
        "plaus_acc": plaus_acc,
        "tce": tce,
        "overall_acc": overall_acc,
        "combined_score": combined_score,
        "source_acc": source_acc,
        "errors": errors,
        "quads_raw": {q: {"correct": quads[q]["correct"], "total": quads[q]["total"]} for q in quads},
    }


# =============================================================================
# 可视化
# =============================================================================

def plot_quadrant_heatmap(all_analyses: Dict[str, Dict], output_path: str):
    """生成四象限 accuracy 热力图，每个子图是一个模型"""
    models = list(all_analyses.keys())
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)

    quad_labels = [["VP", "VI"], ["IP", "II"]]
    display_labels = [
        ["Valid\nPlausible", "Valid\nImplausible"],
        ["Invalid\nPlausible", "Invalid\nImplausible"],
    ]

    for idx, model_name in enumerate(models):
        ax = axes[0, idx]
        analysis = all_analyses[model_name]
        qa = analysis["quad_acc"]

        matrix = np.array([
            [qa["VP"], qa["VI"]],
            [qa["IP"], qa["II"]],
        ])

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect='equal')

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val < 40 or val > 85 else "black"
                ax.text(j, i, f"{val:.1f}%",
                        ha="center", va="center", fontsize=14, fontweight="bold", color=color)
                ax.text(j, i + 0.35, display_labels[i][j],
                        ha="center", va="center", fontsize=7, color=color, alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{model_name}\nAcc={analysis['overall_acc']}% TCE={analysis['tce']}%",
                      fontsize=10, fontweight="bold")

    fig.suptitle("Accuracy by Validity × Plausibility Quadrant", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  热力图已保存: {output_path}")


def plot_content_effect_comparison(all_analyses: Dict[str, Dict], output_path: str):
    """Content effect 柱状对比图"""
    models = list(all_analyses.keys())
    n = len(models)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n), 5))

    x = np.arange(n)
    width = 0.3

    plaus_vals = [all_analyses[m]["plaus_acc"]["plausible"] for m in models]
    implaus_vals = [all_analyses[m]["plaus_acc"]["implausible"] for m in models]
    tce_vals = [all_analyses[m]["tce"] for m in models]

    bars1 = ax.bar(x - width / 2, plaus_vals, width, label='Plausible', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x + width / 2, implaus_vals, width, label='Implausible', color='#FF9800', alpha=0.85)

    # 在柱子上方标注 TCE
    for i, tce in enumerate(tce_vals):
        max_h = max(plaus_vals[i], implaus_vals[i])
        ax.text(i, max_h + 1.5, f"TCE={tce:.1f}%",
                ha='center', fontsize=9, fontweight='bold', color='#D32F2F')

    ax.set_xlabel('LLM Parser', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Content Effect: Plausible vs Implausible Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Content effect 图已保存: {output_path}")


def plot_error_breakdown(all_analyses: Dict[str, Dict], output_path: str):
    """错误分类堆叠柱状图"""
    models = list(all_analyses.keys())
    n = len(models)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n), 5))

    error_types = ["parse_fail", "figure_fail", "symbolic_wrong"]
    labels = ["Parse Failure", "Figure Failure", "Symbolic Wrong"]
    colors = ["#E53935", "#FB8C00", "#FDD835"]

    x = np.arange(n)
    bottoms = np.zeros(n)

    for etype, label, color in zip(error_types, labels, colors):
        vals = [all_analyses[m]["errors"].get(etype, 0) for m in models]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += np.array(vals)

    ax.set_xlabel('LLM Parser', fontsize=11)
    ax.set_ylabel('Error Count', fontsize=11)
    ax.set_title('Error Classification by Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  错误分类图已保存: {output_path}")


# =============================================================================
# LaTeX 表格生成
# =============================================================================

def generate_quadrant_latex(all_analyses: Dict[str, Dict]) -> str:
    """生成四象限分析 LaTeX 表格"""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|cccc|cc|c}")
    lines.append(r"\hline")
    lines.append(r"\textbf{LLM} & \textbf{VP} & \textbf{VI} & \textbf{IP} & \textbf{II} "
                 r"& \textbf{Plaus.} & \textbf{Implaus.} & \textbf{TCE$\downarrow$} \\")
    lines.append(r"\hline")

    for model_name, analysis in sorted(all_analyses.items(), key=lambda x: x[1]['tce']):
        qa = analysis["quad_acc"]
        pa = analysis["plaus_acc"]
        name = model_name.replace("_", r"\_")
        lines.append(
            f"{name} & {qa['VP']:.1f} & {qa['VI']:.1f} & {qa['IP']:.1f} & {qa['II']:.1f} "
            f"& {pa['plausible']:.1f} & {pa['implausible']:.1f} & {analysis['tce']:.1f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Accuracy (\%) by validity$\times$plausibility quadrant for each LLM parser. "
                 r"VP=Valid+Plausible, VI=Valid+Implausible, IP=Invalid+Plausible, II=Invalid+Implausible. "
                 r"TCE=Total Content Effect $|$Acc$_\text{plaus}$-Acc$_\text{implaus}|$.}")
    lines.append(r"\label{tab:quadrant_analysis}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_error_latex(all_analyses: Dict[str, Dict]) -> str:
    """生成错误分析 LaTeX 表格"""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|ccc|c}")
    lines.append(r"\hline")
    lines.append(r"\textbf{LLM} & \textbf{Parse Fail} & \textbf{Figure Fail} "
                 r"& \textbf{Sym. Wrong} & \textbf{Total Err.} \\")
    lines.append(r"\hline")

    for model_name, analysis in all_analyses.items():
        e = analysis["errors"]
        total_err = e["parse_fail"] + e["figure_fail"] + e["symbolic_wrong"] + e.get("default_wrong", 0)
        name = model_name.replace("_", r"\_")
        lines.append(
            f"{name} & {e['parse_fail']} & {e['figure_fail']} & {e['symbolic_wrong']} & {total_err} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Error classification for each LLM parser on training data (960 samples).}")
    lines.append(r"\label{tab:error_analysis}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# 打印报告
# =============================================================================

def print_full_report(all_analyses: Dict[str, Dict]):
    """打印完整分析报告"""
    print("\n" + "=" * 90)
    print("  维度三: Content Effect & Quadrant Analysis")
    print("=" * 90)

    # 四象限对比表
    print("\n--- 四象限 Accuracy (%) ---")
    header = f"{'Model':<16s} | {'VP':>6s} | {'VI':>6s} | {'IP':>6s} | {'II':>6s} | {'Plaus':>6s} | {'Implau':>6s} | {'TCE':>5s} | {'Score':>6s}"
    print(header)
    print("-" * 90)
    for model_name, analysis in sorted(all_analyses.items(), key=lambda x: x[1]['combined_score'], reverse=True):
        qa = analysis["quad_acc"]
        pa = analysis["plaus_acc"]
        print(f"{model_name:<16s} | {qa['VP']:>5.1f}% | {qa['VI']:>5.1f}% | {qa['IP']:>5.1f}% | "
              f"{qa['II']:>5.1f}% | {pa['plausible']:>5.1f}% | {pa['implausible']:>5.1f}% | "
              f"{analysis['tce']:>4.1f}% | {analysis['combined_score']:>6.2f}")

    # 错误分类
    print("\n--- 错误分类 ---")
    header2 = f"{'Model':<16s} | {'ParseFail':>10s} | {'FigureFail':>10s} | {'SymWrong':>10s} | {'TotalErr':>10s}"
    print(header2)
    print("-" * 70)
    for model_name, analysis in all_analyses.items():
        e = analysis["errors"]
        total = e["parse_fail"] + e["figure_fail"] + e["symbolic_wrong"] + e.get("default_wrong", 0)
        print(f"{model_name:<16s} | {e['parse_fail']:>10d} | {e['figure_fail']:>10d} | "
              f"{e['symbolic_wrong']:>10d} | {total:>10d}")

    # Source 分组
    print("\n--- 按 Source 分组的 Accuracy ---")
    for model_name, analysis in all_analyses.items():
        print(f"\n  [{model_name}]")
        for src, info in analysis["source_acc"].items():
            sq = info["quad_acc"]
            print(f"    {src:<20s}: {info['total']:>4d} samples, acc={info['acc']:.1f}%  "
                  f"(VP={sq['VP']:.0f}% VI={sq['VI']:.0f}% IP={sq['IP']:.0f}% II={sq['II']:.0f}%)")

    # 关键发现总结
    print("\n--- 关键发现 ---")
    tce_sorted = sorted(all_analyses.items(), key=lambda x: x[1]['tce'])
    print(f"  最低 TCE: {tce_sorted[0][0]} ({tce_sorted[0][1]['tce']}%)")
    print(f"  最高 TCE: {tce_sorted[-1][0]} ({tce_sorted[-1][1]['tce']}%)")

    # 所有模型的 IP 象限（Invalid + Plausible，最容易被内容误导的）
    print(f"\n  Invalid+Plausible (最难) 各模型 accuracy:")
    for m, a in sorted(all_analyses.items(), key=lambda x: x[1]['quad_acc']['IP']):
        print(f"    {m:<16s}: {a['quad_acc']['IP']:.1f}%")

    print(f"\n  Valid+Implausible (反直觉) 各模型 accuracy:")
    for m, a in sorted(all_analyses.items(), key=lambda x: x[1]['quad_acc']['VI']):
        print(f"    {m:<16s}: {a['quad_acc']['VI']:.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Content Effect & Quadrant Analysis")
    parser.add_argument("--data", type=str, required=True, help="训练数据路径 (train_data.json)")
    parser.add_argument("--results-dir", type=str, default="ablation_results",
                        help="维度二的结果目录 (包含 detail_*.json)")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                        help="输出目录")
    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data)
    print(f"加载数据: {len(data)} 条样本")
    data_by_id = {item['id']: item for item in data}

    # 发现模型
    model_files = discover_models(args.results_dir)
    if not model_files:
        print(f"错误: 在 {args.results_dir} 中未找到 detail_*.json 文件")
        return
    print(f"发现 {len(model_files)} 个模型: {', '.join(model_files.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 对每个模型做分析
    all_analyses = {}
    for model_key, detail_path in sorted(model_files.items()):
        detail = load_detail(detail_path)
        results = detail.get("results", [])
        display_name = detail.get("model", {}).get("display_name", model_key)

        # 确保结果和数据对齐
        aligned_results = []
        for item in data:
            found = None
            for r in results:
                if r.get("id") == item["id"]:
                    found = r
                    break
            if found is None:
                found = {"id": item["id"], "parse_success": False,
                         "figure_success": False, "predicted_validity": None}
            aligned_results.append(found)

        analysis = quadrant_analysis(data, aligned_results)
        all_analyses[display_name] = analysis
        print(f"  ✓ {display_name}: acc={analysis['overall_acc']}%, TCE={analysis['tce']}%")

    # 打印完整报告
    print_full_report(all_analyses)

    # 生成图表
    print("\n--- 生成图表 ---")
    plot_quadrant_heatmap(
        all_analyses,
        os.path.join(args.output_dir, "quadrant_heatmap.pdf")
    )
    plot_content_effect_comparison(
        all_analyses,
        os.path.join(args.output_dir, "content_effect.pdf")
    )
    plot_error_breakdown(
        all_analyses,
        os.path.join(args.output_dir, "error_breakdown.pdf")
    )

    # 生成 LaTeX
    print("\n--- 生成 LaTeX 表格 ---")
    quad_latex = generate_quadrant_latex(all_analyses)
    quad_path = os.path.join(args.output_dir, "table_quadrant.tex")
    with open(quad_path, 'w') as f:
        f.write(quad_latex)
    print(f"  四象限表格: {quad_path}")

    error_latex = generate_error_latex(all_analyses)
    error_path = os.path.join(args.output_dir, "table_errors.tex")
    with open(error_path, 'w') as f:
        f.write(error_latex)
    print(f"  错误分析表格: {error_path}")

    # 保存完整分析 JSON
    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, ensure_ascii=False, indent=2)
    print(f"  分析汇总: {summary_path}")

    print("\n✓ 维度三分析完成!")


if __name__ == "__main__":
    main()