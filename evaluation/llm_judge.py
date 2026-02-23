#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation

Evaluates model outputs using LLM as a judge with literature-based criteria.
Compares multiple model predictions and provides detailed scoring.

Usage:
    python llm_judge.py --input comparison.csv --output judge_results.csv
"""

import os
import sys
import time
import atexit
import shutil

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./results/comparison.csv"
DEFAULT_OUTPUT_CSV = "./results/judge_results.csv"
DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_TORCH_CACHE_DIR = "./cache/torch"

def setup_cache(cache_dir, torch_cache_dir):
    """Setup cache directories"""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(torch_cache_dir, exist_ok=True)

    print(f"HuggingFace cache: {cache_dir}")
    print(f"Torch cache: {torch_cache_dir}")

    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = torch_cache_dir
    os.environ['TMPDIR'] = torch_cache_dir
    os.environ['TEMP'] = torch_cache_dir
    os.environ['TMP'] = torch_cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        subprocess.run(['pkill', '-9', '-f', 'VLLM::Worker'],
                      capture_output=True, check=False)
        
        time.sleep(2)
        
        if os.path.exists(TORCH_CACHE_DIR):
            shutil.rmtree(TORCH_CACHE_DIR, ignore_errors=True)
            print(f"Removed cache: {TORCH_CACHE_DIR}")
    except Exception as e:
        print(f"Cleanup warning: {e}")

atexit.register(cleanup_on_exit)

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ============================================================
# CONFIG
# ============================================================
JUDGE_MODELS = {
    'llama405b': {
        'name': "meta-llama/Meta-Llama-3.1-405B-Instruct",
        'tensor_parallel': 8,
        'batch_size': 16
    },
    'qwen72b': {
        'name': "Qwen/Qwen2.5-72B-Instruct",
        'tensor_parallel': 8,
        'batch_size': 32
    }
}

MAX_MODEL_LEN = 4096

# ============================================================
# ENHANCED TCCC-SPECIFIC JUDGE PROMPT
# Based on medical LLM evaluation literature (Med-HALT, clinical QA frameworks)
# ============================================================
JUDGE_SYSTEM_PROMPT = """**LENGTH CALIBRATION**: Reference answers are typically 1-3 sentences (30-50 words). \
A concise, accurate answer matching this length should score 5 on completeness. \
Do NOT reward verbosity — extra length without new clinical information should be penalized.

You are an expert evaluator for Tactical Combat Casualty Care (TCCC) medical responses. \
Evaluate answers using established medical AI assessment criteria adapted for combat medicine.

**EVALUATION CRITERIA (1-5 scale):**

1. **Medical Accuracy** (1-5)
   - Are medical facts, terminology, and procedures correct per current TCCC guidelines?
   - Are drug names, dosages, routes, and timing accurate?
   - Are there any factually incorrect or outdated information?
   - 1 = Contains dangerous errors or misinformation
   - 3 = Mostly accurate with minor inaccuracies
   - 5 = Completely accurate per TCCC standards

2. **Protocol Adherence** (1-5)
   - Does it follow TCCC/MARCH protocol (Massive hemorrhage → Airway → Respiration → Circulation → Hypothermia)?
   - Are treatment priorities in correct sequence?
   - Does it respect care phase constraints (Care Under Fire vs Tactical Field Care vs TACEVAC)?
   - 1 = Violates core TCCC protocols
   - 3 = Generally follows protocols with minor deviations
   - 5 = Perfect adherence to TCCC doctrine

3. **Completeness** (1-5)
   - Are all CRITICAL elements present for safe execution?
   - A concise answer covering key points scores higher than a verbose answer with redundant information.
   - 1 = Missing essential information that could lead to harm
   - 3 = Contains key information but lacks some important details  
   - 5 = Covers all critical clinical elements efficiently — length is irrelevant

4. **Actionability** (1-5)
   - Can a combat medic realistically execute this in the field?
   - Is it practical given tactical constraints (time pressure, limited resources, hostile environment)?
   - Are instructions clear and specific enough to follow under stress?
   - 1 = Impractical or impossible in combat conditions
   - 3 = Feasible but may be challenging in field conditions
   - 5 = Highly practical and field-ready

5. **Safety** (1-5)
   - Are contraindications, warnings, and risks appropriately mentioned?
   - Could following this answer cause harm to the casualty?
   - Are dose limits, monitoring requirements, and danger signs addressed?
   - 1 = Dangerous advice that could cause serious harm
   - 3 = Generally safe but missing some important warnings
   - 5 = Fully addresses safety concerns and risk mitigation

**OUTPUT FORMAT:**
Respond with ONLY valid JSON (no markdown, no extra text):
{
  "medical_accuracy": <1-5>,
  "protocol_adherence": <1-5>,
  "completeness": <1-5>,
  "actionability": <1-5>,
  "safety": <1-5>,
  "overall": <average of above 5 scores>,
  "rationale": "<1-2 sentences explaining the key strengths or weaknesses>"
}"""

JUDGE_USER_TEMPLATE = """Question: {question}

Reference Answer (ground truth, {ref_word_count} words):
{reference}

Model Answer (to evaluate, {pred_word_count} words):
{prediction}

IMPORTANT: Evaluate based on factual correctness and clinical accuracy ONLY.
- Do NOT penalize shorter answers if they are clinically correct
- Do NOT reward longer answers for being more detailed
- A concise correct answer should score the same as a detailed correct answer
- Only penalize length if information is MISSING or WRONG"""
# ============================================================
# LOAD JUDGE MODEL
# ============================================================
def load_judge_model(judge_key: str):
    """Load judge model with vLLM"""
    config = JUDGE_MODELS[judge_key]
    
    print("=" * 70)
    print(f"LOADING JUDGE MODEL: {judge_key.upper()}")
    print("=" * 70)
    print(f"Model: {config['name']}")
    print(f"Tensor Parallel: {config['tensor_parallel']} GPUs")
    print(f"Batch Size: {config['batch_size']}")
    
    if judge_key == 'llama405b':
        print("Note: 405B model will take 5-10 minutes to load...")
    
    llm = LLM(
        model=config['name'],
        tensor_parallel_size=config['tensor_parallel'],
        max_model_len=MAX_MODEL_LEN,
        download_dir=CACHE_DIR,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_num_seqs=config['batch_size'],
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['name'],
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    print(f"{judge_key.upper()} loaded successfully\n")
    return llm, tokenizer, config['batch_size']


# ============================================================
# PROMPT & PARSING
# ============================================================
def create_judge_prompt(question: str, reference: str, prediction: str, tokenizer) -> str:
    ref_wc = len(reference.split())
    pred_wc = len(prediction.split())
    
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
        ref_word_count=ref_wc,
        pred_word_count=pred_wc
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_judge_response(response: str) -> Dict:
    """Parse JSON from judge response"""
    try:
        response = response.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Updated criteria names
            for key in ['medical_accuracy', 'protocol_adherence', 'completeness', 
                       'actionability', 'safety']:
                if key in result:
                    result[key] = max(1, min(5, result[key]))
            
            if 'overall' not in result:
                scores = [result.get(k, 3) for k in ['medical_accuracy', 'protocol_adherence',
                         'completeness', 'actionability', 'safety']]
                result['overall'] = sum(scores) / len(scores)
            
            return result
        
    except Exception as e:
        print(f"Parse error: {e}")
    
    return {
        'medical_accuracy': 3,
        'protocol_adherence': 3,
        'completeness': 3,
        'actionability': 3,
        'safety': 3,
        'overall': 3,
        'rationale': 'Parsing failed'
    }


# ============================================================
# EVALUATION
# ============================================================
def evaluate_models_with_judge(llm, tokenizer, batch_size: int, 
                               question_df: pd.DataFrame,
                               model_names: List[str],
                               judge_name: str,
                               output_dir: Path) -> pd.DataFrame:
    """
    Evaluate multiple models with judge
    
    Args:
        llm: Judge LLM
        tokenizer: Judge tokenizer
        batch_size: Batch size for inference
        question_df: DataFrame with questions, references, and model predictions
        model_names: List of model column names to evaluate (e.g., 'finetuned', 'baseline')
        judge_name: Judge identifier (llama405b or qwen72b)
        output_dir: Output directory
    """
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATING WITH {judge_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Samples: {len(question_df)}")
    
    result_df = question_df.copy()
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"]
    )
    
    # Evaluate each model
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        
        if pred_col not in result_df.columns:
            print(f"  Warning: Skipping {model_name}: Column '{pred_col}' not found")
            continue
        
        print(f"\n[{model_names.index(model_name)+1}/{len(model_names)}] Evaluating {model_name}...")
        
        # Create prompts
        prompts = []
        for _, row in result_df.iterrows():
            prompt = create_judge_prompt(
                row['question'],
                row['reference_answer'],
                row[pred_col],
                tokenizer
            )
            prompts.append(prompt)
        
        # Generate
        outputs = llm.generate(prompts, sampling_params)
        
        # Parse results
        results = []
        for output in tqdm(outputs, desc=f"Parsing {model_name}"):
            response = output.outputs[0].text
            parsed = parse_judge_response(response)
            results.append(parsed)
        
        # Add scores to dataframe (updated criteria names)
        for key in ['medical_accuracy', 'protocol_adherence', 'completeness', 
                    'actionability', 'safety', 'overall', 'rationale']:
            result_df[f'{judge_name}_{model_name}_{key}'] = [r[key] for r in results]
    
    # Save individual judge results
    output_path = output_dir / f"{judge_name}_scores.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"{judge_name.upper()} EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    
    criteria = ['medical_accuracy', 'protocol_adherence', 'completeness', 
                'actionability', 'safety', 'overall']
    
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        if pred_col not in result_df.columns:
            continue
        
        print(f"\n{model_name.upper()}:")
        for criterion in criteria:
            col = f'{judge_name}_{model_name}_{criterion}'
            if col in result_df.columns:
                mean_score = result_df[col].mean()
                print(f"  {criterion:25s}: {mean_score:.2f}")
    
    return result_df


# ============================================================
# INTER-JUDGE AGREEMENT
# ============================================================
def analyze_inter_judge_agreement(results_df: pd.DataFrame, 
                                  model_names: List[str],
                                  output_dir: Path):
    """Analyze agreement between two judges"""
    
    print("\n" + "=" * 70)
    print("INTER-JUDGE AGREEMENT ANALYSIS")
    print("=" * 70)
    
    criteria = ['medical_accuracy', 'protocol_adherence', 'completeness', 
                'actionability', 'safety', 'overall']
    
    agreement_results = {}
    
    for model in model_names:
        # Check if both judges evaluated this model
        llama_cols = [f'llama405b_{model}_{c}' for c in criteria]
        qwen_cols = [f'qwen72b_{model}_{c}' for c in criteria]
        
        if not all(col in results_df.columns for col in llama_cols):
            print(f"Skipping {model}: Missing llama405b scores")
            continue
        if not all(col in results_df.columns for col in qwen_cols):
            print(f"Skipping {model}: Missing qwen72b scores")
            continue
        
        print(f"\n{model.upper()}:")
        model_agreement = {}
        
        for criterion in criteria:
            judge1 = results_df[f'llama405b_{model}_{criterion}'].values
            judge2 = results_df[f'qwen72b_{model}_{criterion}'].values
            
            pearson_r, pearson_p = pearsonr(judge1, judge2)
            spearman_r, spearman_p = spearmanr(judge1, judge2)
            mae = np.abs(judge1 - judge2).mean()
            agreement_within_1 = (np.abs(judge1 - judge2) <= 1).mean()
            
            model_agreement[criterion] = {
                'pearson_r': pearson_r,
                'spearman_r': spearman_r,
                'mae': mae,
                'agreement_within_1': agreement_within_1
            }
            
            print(f"  {criterion:25s}: r={pearson_r:.3f}, MAE={mae:.2f}, Within-1={agreement_within_1:.1%}")
        
        agreement_results[model] = model_agreement
    
    # Save
    output_path = output_dir / "inter_judge_agreement.json"
    with open(output_path, 'w') as f:
        json.dump(agreement_results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    return agreement_results


# ============================================================
# CONSENSUS SCORING
# ============================================================
def calculate_consensus_scores(results_df: pd.DataFrame,
                               model_names: List[str],
                               output_dir: Path):
    """Calculate consensus scores (average of two judges)"""
    
    print("\n" + "=" * 70)
    print("CALCULATING CONSENSUS SCORES")
    print("=" * 70)
    
    criteria = ['medical_accuracy', 'protocol_adherence', 'completeness', 
                'actionability', 'safety', 'overall']
    
    consensus_df = results_df.copy()
    consensus_summary = []
    
    for model in model_names:
        # Check if both judges evaluated this model
        has_llama = all(f'llama405b_{model}_{c}' in results_df.columns for c in criteria)
        has_qwen = all(f'qwen72b_{model}_{c}' in results_df.columns for c in criteria)
        
        if not (has_llama and has_qwen):
            print(f"Skipping {model}: Missing scores from one or both judges")
            continue
        
        row = {'model': model}
        
        for criterion in criteria:
            llama_col = f'llama405b_{model}_{criterion}'
            qwen_col = f'qwen72b_{model}_{criterion}'
            consensus_col = f'consensus_{model}_{criterion}'
            
            # Calculate consensus (average)
            consensus_df[consensus_col] = (
                consensus_df[llama_col] + consensus_df[qwen_col]
            ) / 2
            
            # Summary stats
            row[f'{criterion}_llama'] = consensus_df[llama_col].mean()
            row[f'{criterion}_qwen'] = consensus_df[qwen_col].mean()
            row[f'{criterion}_consensus'] = consensus_df[consensus_col].mean()
        
        consensus_summary.append(row)
    
    # Save full results
    output_path = output_dir / "consensus_scores.csv"
    consensus_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Save summary
    summary_df = pd.DataFrame(consensus_summary)
    output_path = output_dir / "consensus_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Display
    print("\nConsensus Scores (1-5 scale):")
    for _, row in summary_df.iterrows():
        print(f"\n{row['model'].upper()}:")
        for criterion in criteria:
            consensus = row[f'{criterion}_consensus']
            print(f"  {criterion:25s}: {consensus:.2f}")
    
    return summary_df


# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_judge_comparison(results_df: pd.DataFrame,
                         model_names: List[str],
                         output_dir: Path):
    """Visualize judge comparisons"""
    
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    criteria = ['medical_accuracy', 'protocol_adherence', 'completeness', 
                'actionability', 'safety', 'overall']
    
    for model in model_names:
        # Check if model has both judge scores
        has_llama = all(f'llama405b_{model}_{c}' in results_df.columns for c in criteria)
        has_qwen = all(f'qwen72b_{model}_{c}' in results_df.columns for c in criteria)
        
        if not (has_llama and has_qwen):
            print(f"  Warning: Skipping {model}: Missing scores")
            continue
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, criterion in enumerate(criteria):
            ax = axes[idx]
            
            judge1 = results_df[f'llama405b_{model}_{criterion}'].values
            judge2 = results_df[f'qwen72b_{model}_{criterion}'].values
            
            ax.scatter(judge1, judge2, alpha=0.5, s=20)
            ax.plot([1, 5], [1, 5], 'r--', alpha=0.5, label='Perfect agreement')
            
            r, _ = pearsonr(judge1, judge2)
            
            ax.set_xlabel('Llama 405B Score', fontsize=12)
            ax.set_ylabel('Qwen 72B Score', fontsize=12)
            ax.set_title(f'{criterion.replace("_", " ").title()}\n(r={r:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        output_path = output_dir / f"{model}_judge_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# MAIN - MODIFIED FOR PHASE 8.1 CSV
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 8.3: Enhanced LLM Judge Evaluation for TCCC"
    )
    
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Phase 8.1 comparison CSV (e.g., comparison_5qa_combined.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for judge results"
    )
    parser.add_argument(
        "--judges",
        type=str,
        nargs='+',
        default=['llama405b', 'qwen72b'],
        choices=['llama405b', 'qwen72b'],
        help="Which judges to use (default: both)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 8.3: ENHANCED LLM JUDGE EVALUATION")
    print("=" * 70)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Judges: {', '.join(args.judges)}")
    print("=" * 70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 8.1 comparison CSV
    if not args.input_csv.exists():
        print(f"ERROR: Input CSV not found: {args.input_csv}")
        return
    
    question_df = pd.read_csv(args.input_csv)
    print(f"\nLoaded {len(question_df)} questions")
    print(f"Columns: {', '.join(question_df.columns.tolist())}")
    
    # Auto-detect model columns (looking for *_prediction columns)
    pred_cols = [col for col in question_df.columns if col.endswith('_prediction')]
    model_names = [col.replace('_prediction', '') for col in pred_cols]
    
    print(f"Detected {len(model_names)} models: {', '.join(model_names)}")
    
    if len(model_names) == 0:
        print("ERROR: No model predictions found in CSV")
        print("Expected columns ending with '_prediction' (e.g., finetuned_prediction, baseline_prediction)")
        return
    
    # Check required columns
    if 'question' not in question_df.columns:
        print("ERROR: 'question' column not found in CSV")
        return
    if 'reference_answer' not in question_df.columns:
        print("ERROR: 'reference_answer' column not found in CSV")
        return
    
    # Evaluate with each judge
    results_df = question_df.copy()
    
    for i, judge_key in enumerate(args.judges):
        is_last_judge = (i == len(args.judges) - 1)  # Check if last judge
        
        print(f"\n{'=' * 70}")
        print(f"STARTING EVALUATION WITH {judge_key.upper()}")
        print(f"{'=' * 70}")
        
        # Load judge model
        llm, tokenizer, batch_size = load_judge_model(judge_key)
        
        try:
            # Evaluate all models
            results_df = evaluate_models_with_judge(
                llm, tokenizer, batch_size,
                results_df, model_names, judge_key,
                args.output_dir
            )
        
        finally:
            # Cleanup
            print(f"\nCleaning up {judge_key.upper()}...")
            del llm
            del tokenizer
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            if is_last_judge:
                import subprocess
                subprocess.run(['pkill', '-9', '-f', 'VLLM::Worker'],
                              capture_output=True, check=False)
                print("  Waiting for GPU cleanup (5 seconds)...")
                time.sleep(5)
            else:
                print(f"  Loading next judge immediately...")
                time.sleep(2)
            
            print(f"{judge_key.upper()} cleanup complete\n")
    
    # Inter-judge agreement (if both judges used)
    if len(args.judges) == 2:
        agreement_results = analyze_inter_judge_agreement(
            results_df, model_names, args.output_dir
        )
        
        # Visualizations
        plot_judge_comparison(results_df, model_names, args.output_dir)
        
        # Consensus scores
        consensus_df = calculate_consensus_scores(
            results_df, model_names, args.output_dir
        )
    
    print("\n" + "=" * 70)
    print("PHASE 8.3 COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - llama405b_scores.csv (if used)")
    print("  - qwen72b_scores.csv (if used)")
    if len(args.judges) == 2:
        print("  - consensus_scores.csv")
        print("  - consensus_summary.csv")
        print("  - inter_judge_agreement.json")
        print("  - *_judge_comparison.png (scatter plots)")


if __name__ == "__main__":
    main()