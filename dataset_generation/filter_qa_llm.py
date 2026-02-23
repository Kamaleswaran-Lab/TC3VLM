#!/usr/bin/env python3
"""
Q/A Quality Filtering with LLM

Filters low-quality question-answer pairs using LLM as a judge.
Evaluates relevance, accuracy, and educational value of generated Q/A pairs.

Usage:
    python filter_qa_llm.py --input qa_pairs.csv --output filtered_qa.csv
"""

import os
import sys
import time
import atexit
import shutil

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/qa_pairs.csv"
DEFAULT_OUTPUT_CSV = "./data/filtered_qa.csv"
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-Instruct"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_TORCH_CACHE_DIR = "./cache/torch"

# ============================================================
# Set up cache directories
# ============================================================
def setup_cache(cache_dir, torch_cache_dir):
    """Setup cache directories and environment variables"""
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(torch_cache_dir, exist_ok=True)

    print(f"HuggingFace cache: {cache_dir}")
    print(f"Torch cache: {torch_cache_dir}")

    # Environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir

    # Torch Inductor cache
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = torch_cache_dir
    os.environ['TMPDIR'] = torch_cache_dir
    os.environ['TEMP'] = torch_cache_dir
    os.environ['TMP'] = torch_cache_dir

    # CUDA settings
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

# Now import everything else
import json
import torch
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
import gc
import argparse

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("/hpc/home/jkim1/workspace/TCCC/data")

# Judge model - Llama 405B for highest accuracy
JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"
TENSOR_PARALLEL = 8
BATCH_SIZE = 16
MAX_MODEL_LEN = 4096

# Filtering thresholds
OVERALL_THRESHOLD = 3.0  # Remove if overall < 2.0
SAFETY_THRESHOLD = 3.0   # Remove if safety < 3.0

# ============================================================
# JUDGE PROMPT
# ============================================================
JUDGE_SYSTEM_PROMPT = """You are an expert in Tactical Combat Casualty Care (TCCC) education. Your task is to evaluate the quality of educational question-answer pairs for training purposes.

Evaluate the QA pair on these 5 criteria (1-5 scale):

1. **Factual Correctness** (1-5)
   - Are all medical facts accurate?
   - Are there any dangerous errors?
   - 1 = Major errors, 5 = Perfectly accurate

2. **Completeness** (1-5)
   - Are all critical details included (drugs, dosages, procedures)?
   - Are important steps or information missing?
   - 1 = Missing essential info, 5 = Comprehensive

3. **Clinical Relevance** (1-5)
   - Is this useful for field medics in combat?
   - Is it actionable and practical?
   - 1 = Not useful, 5 = Highly actionable

4. **Specificity** (1-5)
   - Does it include specific details (drug names, exact dosages, protocols)?
   - Or is it too vague/general?
   - 1 = Too vague, 5 = Appropriately specific

5. **Safety** (1-5)
   - Could learning from this QA cause harm?
   - Are there any dangerous omissions?
   - 1 = Potentially dangerous, 5 = Safe to learn from

Respond with ONLY valid JSON, no other text:
{
  "factual_correctness": <1-5>,
  "completeness": <1-5>,
  "clinical_relevance": <1-5>,
  "specificity": <1-5>,
  "safety": <1-5>,
  "overall": <average of above>,
  "rationale": "<brief 1-2 sentence explanation>"
}"""

JUDGE_USER_TEMPLATE = """Question: {question}

Answer: {answer}

Evaluate this QA pair for training quality. Focus on medical accuracy and educational value."""

# ============================================================
# LOAD JUDGE MODEL
# ============================================================
def load_judge_model():
    """Load Llama 405B judge model with vLLM"""
    
    print("=" * 70)
    print("LOADING JUDGE MODEL: LLAMA 3.1 405B")
    print("=" * 70)
    print(f"Model: {JUDGE_MODEL}")
    print(f"Tensor Parallel: {TENSOR_PARALLEL} GPUs")
    print(f"Batch Size: {BATCH_SIZE}")
    print("Note: 405B model will take 5-10 minutes to load...")
    
    llm = LLM(
        model=JUDGE_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL,
        max_model_len=MAX_MODEL_LEN,
        download_dir=CACHE_DIR,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_num_seqs=BATCH_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    print(f"Llama 405B judge model loaded successfully\n")
    return llm, tokenizer

# ============================================================
# EVALUATION
# ============================================================
def create_judge_prompt(question: str, answer: str, tokenizer) -> str:
    """Create prompt for judge model"""
    
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        answer=answer
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def parse_judge_response(response: str) -> Dict:
    """Parse JSON from judge response"""
    try:
        response = response.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Validate scores are 1-5
            for key in ['factual_correctness', 'completeness', 'clinical_relevance', 
                       'specificity', 'safety']:
                if key in result:
                    result[key] = max(1, min(5, result[key]))
            
            # Calculate overall if not present
            if 'overall' not in result:
                scores = [result.get(k, 3) for k in ['factual_correctness', 'completeness', 
                         'clinical_relevance', 'specificity', 'safety']]
                result['overall'] = sum(scores) / len(scores)
            
            return result
        
    except Exception as e:
        print(f"Parse error: {e}")
    
    # Fallback - give neutral score
    return {
        'factual_correctness': 3,
        'completeness': 3,
        'clinical_relevance': 3,
        'specificity': 3,
        'safety': 3,
        'overall': 3,
        'rationale': 'Parsing failed - neutral score'
    }


def evaluate_qa_pairs(llm, tokenizer, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Evaluate all QA pairs with judge model"""
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {source_type.upper()} QA PAIRS")
    print(f"{'=' * 70}")
    print(f"Total samples: {len(df)}")
    
    # Create prompts
    prompts = []
    for _, row in df.iterrows():
        prompt = create_judge_prompt(
            row['question'],
            row['answer'],
            tokenizer
        )
        prompts.append(prompt)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"]
    )
    
    # Batch inference
    print("\nRunning inference with Llama 405B...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Parse results
    print("Parsing results...")
    results = []
    for output in tqdm(outputs, desc="Processing"):
        response = output.outputs[0].text
        parsed = parse_judge_response(response)
        results.append(parsed)
    
    # Add scores to dataframe
    result_df = df.copy()
    for key in ['factual_correctness', 'completeness', 'clinical_relevance', 
                'specificity', 'safety', 'overall', 'rationale']:
        result_df[f'judge_{key}'] = [r[key] for r in results]
    
    # Summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    
    criteria = ['factual_correctness', 'completeness', 'clinical_relevance', 
                'specificity', 'safety', 'overall']
    
    for criterion in criteria:
        mean_score = result_df[f'judge_{criterion}'].mean()
        print(f"  {criterion:25s}: {mean_score:.2f} / 5.00")
    
    return result_df


def filter_qa_pairs(df: pd.DataFrame, source_type: str) -> tuple:
    """Filter QA pairs based on judge scores"""
    
    print(f"\n{'=' * 70}")
    print(f"FILTERING: {source_type.upper()}")
    print(f"{'=' * 70}")
    
    # Filter conditions
    passed = df[
        (df['judge_overall'] >= OVERALL_THRESHOLD) & 
        (df['judge_safety'] >= SAFETY_THRESHOLD)
    ].copy()
    
    removed = df[
        (df['judge_overall'] < OVERALL_THRESHOLD) | 
        (df['judge_safety'] < SAFETY_THRESHOLD)
    ].copy()
    
    # Statistics
    print(f"\nTotal samples: {len(df)}")
    print(f"  Passed: {len(passed)} ({len(passed)/len(df)*100:.1f}%)")
    print(f"  Removed: {len(removed)} ({len(removed)/len(df)*100:.1f}%)")
    
    # Removal reasons
    overall_fail = (df['judge_overall'] < OVERALL_THRESHOLD).sum()
    safety_fail = (df['judge_safety'] < SAFETY_THRESHOLD).sum()
    both_fail = ((df['judge_overall'] < OVERALL_THRESHOLD) & 
                 (df['judge_safety'] < SAFETY_THRESHOLD)).sum()
    
    print(f"\nRemoval reasons:")
    print(f"  Overall < {OVERALL_THRESHOLD}: {overall_fail}")
    print(f"  Safety < {SAFETY_THRESHOLD}: {safety_fail}")
    print(f"  Both: {both_fail}")
    
    # Score distribution
    print(f"\nScore distribution (passed):")
    for criterion in ['overall', 'safety', 'factual_correctness', 'completeness', 'specificity']:
        mean = passed[f'judge_{criterion}'].mean()
        std = passed[f'judge_{criterion}'].std()
        print(f"  {criterion:20s}: {mean:.2f} ± {std:.2f}")
    
    return passed, removed

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.3: Filter QA pairs using Llama 405B judge"
    )
    parser.add_argument(
        "--input-file", type=str, required=True,
        help="Direct input CSV path (e.g. phase5.1_qa_pairs_5qa_32B.csv)"
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output CSV path (optional, auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    stem = input_path.stem  # e.g. "phase5.1_qa_pairs_5qa_32B"
    
    output_path  = Path(args.output_file) if args.output_file else DATA_DIR / f"phase5.3_filtered_{stem}.csv"
    removed_path = DATA_DIR / f"phase5.3_removed_{stem}.csv"
    metadata_path = DATA_DIR / f"phase5.3_metadata_{stem}.json"
    
    print("=" * 70)
    print("PHASE 5.3: LLM-BASED QA QUALITY FILTERING")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} QA pairs")
    
    llm, tokenizer = load_judge_model()
    
    try:
        evaluated_df = evaluate_qa_pairs(llm, tokenizer, df, stem)
        passed, removed = filter_qa_pairs(evaluated_df, stem)
        
        passed.to_csv(output_path, index=False)
        removed.to_csv(removed_path, index=False)
        print(f"\nSaved filtered: {output_path}")
        print(f"Saved removed:  {removed_path}")
        
        metadata = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'judge_model': JUDGE_MODEL,
            'overall_threshold': OVERALL_THRESHOLD,
            'safety_threshold': SAFETY_THRESHOLD,
            'total': len(df),
            'passed': len(passed),
            'removed': len(removed),
            'retention_rate': len(passed) / len(df),
            'mean_scores': {k: float(passed[f'judge_{k}'].mean())
                           for k in ['overall', 'safety', 'factual_correctness',
                                     'completeness', 'clinical_relevance', 'specificity']}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
    finally:
        del llm, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*70}")
    print(f"DONE: {len(passed)}/{len(df)} passed ({len(passed)/len(df)*100:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()