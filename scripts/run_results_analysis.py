#!/usr/bin/env python3
"""
Results analysis script for hallucination classification experiments.

Generates LaTeX tables, metrics summaries, and audio vs text comparison analysis.

Usage:
    python scripts/run_results_analysis.py --results-dir ./hallucination_results
    python scripts/run_results_analysis.py --results-dir ./hallucination_results --output-dir ./tables
    python scripts/run_results_analysis.py --results-dir ./hallucination_results --models Qwen2.5-Omni-3B Qwen2-Audio-7B-Instruct
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Tuple

# ========================== CONFIGURATION ==========================

EXPERIMENT_TYPES = ['audio', 'text']
LANGUAGES = ['english', 'kazakh', 'russian']
APPROACHES = ['direct', 'cot']

ALL_MODELS = [
    "Qwen2.5-Omni-3B",
    "Qwen2-Audio-7B-Instruct",
    "gemma-3n-E4B-it",
    "LFM2-Audio-1.5B",
    "Step-Audio-2-mini",
]

TYPE_MAPPING = {
    'factual_contradiction': 'factual_contradiction',
    'factual_fabrication': 'factual_fabrication',
    'contextual_inconsistency': 'contextual_inconsistency',
    'contradiction': 'factual_contradiction',
    'fabrication': 'factual_fabrication',
    'none': 'none',
}


# ========================== METRICS ==========================

class MetricsCalculator:
    def __init__(self, results_dir: str, models: List[str] = None):
        self.results_dir = Path(results_dir)
        self.models = models or ALL_MODELS
        self.results = {}

    def load_all_results(self) -> Dict:
        all_results = {}
        for model in self.models:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                print(f"Model directory not found: {model_dir}")
                continue
            all_results[model] = {}
            for exp_type in EXPERIMENT_TYPES:
                exp_dir = model_dir / exp_type
                if not exp_dir.exists():
                    continue
                all_results[model][exp_type] = {}
                for lang in LANGUAGES:
                    result_file = exp_dir / f"{lang}_results.csv"
                    if result_file.exists():
                        try:
                            df = pd.read_csv(result_file, encoding='utf-8')
                            all_results[model][exp_type][lang] = df
                            print(f"Loaded {model}/{exp_type}/{lang}: {len(df)} samples")
                        except Exception as e:
                            print(f"Failed to load {result_file}: {e}")
        return all_results

    def normalize_types(self, types: List[str]) -> List[str]:
        return [TYPE_MAPPING.get(str(t).lower().strip(), 'none') for t in types]

    def calculate_binary_metrics(self, df: pd.DataFrame, approach: str) -> Tuple[float, float]:
        y_true = ['yes'] * len(df)
        y_pred = df[f'pred_{approach}_binary'].fillna('no').tolist()
        y_true_bin = [1 if str(x).lower() == 'yes' else 0 for x in y_true]
        y_pred_bin = [1 if str(x).lower() == 'yes' else 0 for x in y_pred]
        f1 = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        return f1, acc

    def calculate_multiclass_metrics(self, df: pd.DataFrame, approach: str,
                                     task: str) -> Tuple[float, float]:
        if task == 'type':
            y_true = self.normalize_types(df['hallucination_type'].fillna('none').tolist())
            y_pred = self.normalize_types(df[f'pred_{approach}_type'].fillna('none').tolist())
        elif task == 'level':
            y_true = [str(x).lower().strip() for x in df['hallucination_level'].fillna('none').tolist()]
            y_pred = [str(x).lower().strip() for x in df[f'pred_{approach}_degree'].fillna('none').tolist()]
        else:
            raise ValueError(f"Unknown task: {task}")
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def calculate_all_metrics(self) -> Dict:
        all_results = self.load_all_results()
        metrics = {}
        for model, model_data in all_results.items():
            metrics[model] = {}
            for exp_type, exp_data in model_data.items():
                metrics[model][exp_type] = {}
                for lang, df in exp_data.items():
                    if df.empty:
                        continue
                    metrics[model][exp_type][lang] = {}
                    for approach in APPROACHES:
                        if f'pred_{approach}_binary' not in df.columns:
                            continue
                        f1_bin, acc_bin = self.calculate_binary_metrics(df, approach)
                        f1_type, acc_type = self.calculate_multiclass_metrics(df, approach, 'type')
                        f1_level, acc_level = self.calculate_multiclass_metrics(df, approach, 'level')
                        metrics[model][exp_type][lang][approach] = {
                            'binary': {'f1': f1_bin, 'acc': acc_bin},
                            'type': {'f1': f1_type, 'acc': acc_type},
                            'level': {'f1': f1_level, 'acc': acc_level},
                        }
        self.results = metrics
        return metrics

    def print_metrics_summary(self):
        if not self.results:
            self.calculate_all_metrics()
        print("CLASSIFICATION METRICS SUMMARY")
        print("=" * 80)
        for model, model_data in self.results.items():
            print(f"\n{model}")
            print("-" * 60)
            for exp_type, exp_data in model_data.items():
                print(f"\n  {exp_type.upper()}")
                for lang, lang_data in exp_data.items():
                    print(f"\n  {lang.upper()}")
                    print(f"  {'Task':<18} | {'Approach':<8} | {'F1':>5} | {'Acc':>5}")
                    print(f"  {'-'*45}")
                    for approach in APPROACHES:
                        if approach not in lang_data:
                            continue
                        d = lang_data[approach]
                        for task_name, key in [('Binary', 'binary'), ('Type', 'type'), ('Level', 'level')]:
                            print(f"  {task_name:<18} | {approach:<8} | "
                                  f"{d[key]['f1']:.3f} | {d[key]['acc']:.3f}")


# ========================== LATEX TABLES ==========================

class LaTeXTableGenerator:
    def __init__(self, metrics: Dict, tables_dir: Path):
        self.metrics = metrics
        self.tables_dir = tables_dir
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(self, val) -> str:
        return f"{val:.3f}" if isinstance(val, float) else "---"

    def _get_metric(self, model, exp_type, lang, approach, task, metric):
        try:
            return self.metrics[model][exp_type][lang][approach][task][metric]
        except KeyError:
            return None

    def generate_experiment_type_table(self, model: str, experiment_type: str) -> str:
        if model not in self.metrics or experiment_type not in self.metrics[model]:
            return f"% No data for {model} - {experiment_type}"
        exp_data = self.metrics[model][experiment_type]
        latex = f"""\\begin{{table*}}[htbp]
\\centering
\\scriptsize
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{ll|cc|cc}}
\\hline
\\shortstack{{\\textbf{{Classification}} \\\\ \\textbf{{Task}}}} & \\textbf{{Language}}
& \\multicolumn{{2}}{{c|}}{{\\textbf{{Direct}}}}
& \\multicolumn{{2}}{{c}}{{\\textbf{{CoT}}}} \\\\
& & \\textbf{{F1}} & \\textbf{{Acc}} & \\textbf{{F1}} & \\textbf{{Acc}} \\\\
\\hline\n"""
        for section, task_key in [("Binary classification", "binary"),
                                   ("Hallucination type", "type"),
                                   ("Hallucination level", "level")]:
            latex += f"\\multirow{{3}}{{*}}{{\\textbf{{{section}}}}} \n"
            for lang in LANGUAGES:
                vals = []
                for approach in ['direct', 'cot']:
                    for metric in ['f1', 'acc']:
                        v = self._get_metric(model, experiment_type, lang, approach, task_key, metric)
                        vals.append(self._fmt(v))
                latex += f"& {lang.capitalize()} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\\n"
            latex += "\\hline\n"

        model_safe = model.replace('/', '_').replace('.', '_')
        exp_display = experiment_type.capitalize()
        latex += f"""\\end{{tabular}}
\\caption{{{exp_display} classification performance for {model} (Direct vs CoT)}}
\\label{{tab:{model_safe}_{experiment_type}}}
\\end{{table*}}"""
        return latex

    def generate_comparison_table(self, model: str, approach: str) -> str:
        if model not in self.metrics:
            return f"% No data for {model}"
        latex = f"""\\begin{{table*}}[htbp]
\\centering
\\scriptsize
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{ll|cc|cc}}
\\hline
\\shortstack{{\\textbf{{Classification}} \\\\ \\textbf{{Task}}}} & \\textbf{{Language}}
& \\multicolumn{{2}}{{c|}}{{\\textbf{{Audio Input}}}}
& \\multicolumn{{2}}{{c}}{{\\textbf{{Text Input}}}} \\\\
& & \\textbf{{F1}} & \\textbf{{Acc}} & \\textbf{{F1}} & \\textbf{{Acc}} \\\\
\\hline\n"""
        for section, task_key in [("Binary classification", "binary"),
                                   ("Hallucination type", "type"),
                                   ("Hallucination level", "level")]:
            latex += f"\\multirow{{3}}{{*}}{{\\textbf{{{section}}}}} \n"
            for lang in LANGUAGES:
                vals = []
                for exp_type in ['audio', 'text']:
                    for metric in ['f1', 'acc']:
                        v = self._get_metric(model, exp_type, lang, approach, task_key, metric)
                        vals.append(self._fmt(v))
                latex += f"& {lang.capitalize()} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\\n"
            latex += "\\hline\n"

        model_safe = model.replace('/', '_').replace('.', '_')
        latex += f"""\\end{{tabular}}
\\caption{{Audio vs Text comparison for {model} ({approach.upper()} approach)}}
\\label{{tab:cmp_{model_safe}_{approach}}}
\\end{{table*}}"""
        return latex

    def save_all_tables(self):
        for model in self.metrics:
            model_safe = model.replace('/', '_').replace('.', '_')
            for exp_type in EXPERIMENT_TYPES:
                if exp_type in self.metrics[model]:
                    table = self.generate_experiment_type_table(model, exp_type)
                    path = self.tables_dir / f"table_{model_safe}_{exp_type}.tex"
                    path.write_text(table, encoding='utf-8')
                    print(f"Saved: {path}")
            for approach in APPROACHES:
                table = self.generate_comparison_table(model, approach)
                path = self.tables_dir / f"comparison_{model_safe}_{approach}.tex"
                path.write_text(table, encoding='utf-8')
                print(f"Saved: {path}")

    def print_all_tables(self):
        for model in self.metrics:
            for exp_type in EXPERIMENT_TYPES:
                if exp_type in self.metrics[model]:
                    print(f"\n{'='*80}")
                    print(f"{model} - {exp_type.upper()}")
                    print('=' * 80)
                    print(self.generate_experiment_type_table(model, exp_type))
            for approach in APPROACHES:
                print(f"\n{'='*80}")
                print(f"{model} - {approach.upper()} COMPARISON")
                print('=' * 80)
                print(self.generate_comparison_table(model, approach))


# ========================== COMPARISON ANALYSIS ==========================

def generate_audio_text_comparison(metrics: Dict, output_dir: Path):
    comparison_dir = output_dir / "audio_text_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison = {}
    for model, model_data in metrics.items():
        if 'audio' not in model_data or 'text' not in model_data:
            continue
        comparison[model] = {}
        for lang in LANGUAGES:
            if lang not in model_data.get('audio', {}) or lang not in model_data.get('text', {}):
                continue
            comparison[model][lang] = {}
            for approach in APPROACHES:
                a = model_data['audio'].get(lang, {}).get(approach)
                t = model_data['text'].get(lang, {}).get(approach)
                if not a or not t:
                    continue
                comparison[model][lang][approach] = {}
                for task in ['binary', 'type', 'level']:
                    comparison[model][lang][approach][task] = {
                        'audio': a[task], 'text': t[task],
                        'f1_diff': t[task]['f1'] - a[task]['f1'],
                        'acc_diff': t[task]['acc'] - a[task]['acc'],
                    }

    # Save JSON
    out_file = comparison_dir / "audio_text_comparison.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")

    # Print summary
    print("\nAUDIO vs TEXT COMPARISON")
    print("=" * 50)
    for model, model_data in comparison.items():
        print(f"\n{model}")
        for lang, lang_data in model_data.items():
            print(f"  {lang.upper()}")
            print(f"  {'Task':<8} | {'Approach':<8} | {'F1 Diff':>8} | {'Better':<6}")
            print(f"  {'-'*40}")
            for approach, tasks in lang_data.items():
                for task, vals in tasks.items():
                    diff = vals['f1_diff']
                    better = "Text" if diff > 0 else "Audio" if diff < 0 else "Same"
                    print(f"  {task:<8} | {approach:<8} | {diff:+8.3f} | {better}")


def generate_detailed_analysis(results_dir: Path, models: List[str], output_dir: Path):
    analysis_dir = output_dir / "detailed_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    calc = MetricsCalculator(str(results_dir), models)
    all_results = calc.load_all_results()

    for model, model_data in all_results.items():
        analysis = {}
        for exp_type, exp_data in model_data.items():
            analysis[exp_type] = {}
            for lang, df in exp_data.items():
                if df.empty:
                    continue
                lang_analysis = {}
                for approach in APPROACHES:
                    if f'pred_{approach}_binary' not in df.columns:
                        continue
                    y_true_bin = [1] * len(df)
                    y_pred_bin = [1 if str(x).lower() == 'yes' else 0
                                 for x in df[f'pred_{approach}_binary'].fillna('no')]
                    y_true_type = calc.normalize_types(df['hallucination_type'].fillna('none').tolist())
                    y_pred_type = calc.normalize_types(df[f'pred_{approach}_type'].fillna('none').tolist())
                    y_true_level = [str(x).lower().strip()
                                    for x in df['hallucination_level'].fillna('none')]
                    y_pred_level = [str(x).lower().strip()
                                    for x in df[f'pred_{approach}_degree'].fillna('none')]

                    lang_analysis[approach] = {
                        'binary_confusion': {
                            'tp': sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1),
                            'fp': sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1),
                            'tn': sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0),
                            'fn': sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0),
                        },
                        'type_distribution': {
                            'ground_truth': pd.Series(y_true_type).value_counts().to_dict(),
                            'predictions': pd.Series(y_pred_type).value_counts().to_dict(),
                        },
                        'level_distribution': {
                            'ground_truth': pd.Series(y_true_level).value_counts().to_dict(),
                            'predictions': pd.Series(y_pred_level).value_counts().to_dict(),
                        },
                    }
                analysis[exp_type][lang] = lang_analysis

        model_safe = model.replace('/', '_').replace('.', '_')
        out_file = analysis_dir / f"detailed_{model_safe}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_file}")


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(description="Hallucination Classification Results Analysis")
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to hallucination_results directory')
    parser.add_argument('--output-dir', type=str, default='tables',
                        help='Output directory for tables and analysis (default: tables)')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models to analyze (default: all found). Known models: {ALL_MODELS}')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect models if not specified
    models = args.models
    if models is None:
        models = [d.name for d in results_dir.iterdir()
                  if d.is_dir() and not d.name.startswith('.')]
        print(f"Auto-detected models: {models}")

    if not models:
        print("No model results found.")
        return

    print("GENERATING CLASSIFICATION RESULTS")
    print("=" * 60)

    # Calculate metrics
    calc = MetricsCalculator(str(results_dir), models)
    metrics = calc.calculate_all_metrics()

    if not metrics:
        print("No metrics calculated. Check results directory.")
        return

    calc.print_metrics_summary()

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    gen = LaTeXTableGenerator(metrics, output_dir)
    gen.save_all_tables()
    gen.print_all_tables()

    # Detailed analysis
    print("\nGenerating detailed analysis...")
    generate_detailed_analysis(results_dir, models, output_dir)

    # Audio-text comparison
    print("\nGenerating audio-text comparison...")
    generate_audio_text_comparison(metrics, output_dir)

    # Save all metrics as JSON
    metrics_file = output_dir / "all_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved: {metrics_file}")

    print(f"\nAll outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
