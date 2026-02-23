#!/usr/bin/env python3
"""
Model Evaluation

Evaluates fine-tuned models on test set with multiple metrics.
Optimized with batch inference and parallel video loading.

Usage:
    python evaluate_model.py --test-data test.jsonl --model ./models/lora --output-dir ./results
"""

import os
import sys

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_OUTPUT_DIR = "./results/evaluation"
DEFAULT_VIDEO_DIR = "./data/videos/raw"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_FRAMES = 32

def setup_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
from tqdm import tqdm
import logging

from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import cv2
import av

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# For optimization
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_FRAMES = 32
TARGET_SIZE = 448
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4
NUM_WORKERS = 8

# ============================================================
# FRAME EXTRACTION: PyAV preferred -> OpenCV fallback
# ============================================================
def extract_frames_opencv(video_path: str, start_time: float, end_time: float,
                          max_frames: int, target_size: int) -> List[np.ndarray]:

    # PyAV preferred
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"

        seek_pts = int(start_time / float(stream.time_base))
        container.seek(seek_pts, stream=stream)

        raw_frames = []
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_time:
                continue
            if t > end_time:
                break
            raw_frames.append(frame)
        container.close()

        if raw_frames:
            indices = np.linspace(0, len(raw_frames) - 1,
                                  min(max_frames, len(raw_frames)), dtype=int)
            frames = []
            for i in indices:
                img = raw_frames[i].to_ndarray(format='rgb24')
                frames.append(cv2.resize(img, (target_size, target_size)))
            return frames if frames else None
    except Exception:
        pass

    # ── OpenCV fallback ────────────────────────────────────────────────────
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return None

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame

        if total_frames <= max_frames:
            indices = list(range(start_frame, end_frame))
        else:
            indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (target_size, target_size))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames if len(frames) > 0 else None

    except Exception:
        return None


# ============================================================
# PARALLEL VIDEO LOADING
# ============================================================
def extract_frames_worker(args):
    video_path, start_time, end_time, max_frames, target_size, sample_id = args
    frames = extract_frames_opencv(video_path, start_time, end_time, max_frames, target_size)
    return sample_id, frames


def preload_all_videos(test_samples: List[Dict], max_workers: int = NUM_WORKERS) -> Dict:
    logger.info(f"Pre-loading {len(test_samples)} videos with {max_workers} workers...")

    video_cache = {}
    tasks = [
        (sample['video'], sample['start_time'], sample['end_time'],
         MAX_FRAMES, TARGET_SIZE, idx)
        for idx, sample in enumerate(test_samples)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_frames_worker, task): task[5] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading videos"):
            sample_id, frames = future.result()
            if frames is not None:
                video_cache[sample_id] = frames

    logger.info(f"Loaded {len(video_cache)}/{len(test_samples)} videos")
    return video_cache


# ============================================================
# DATASET & DATALOADER
# ============================================================
class TCCCDataset(Dataset):
    def __init__(self, test_samples: List[Dict], video_cache: Dict):
        self.test_samples = test_samples
        self.video_cache = video_cache
        self.valid_indices = [i for i in range(len(test_samples)) if i in video_cache]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx = self.valid_indices[idx]
        sample = self.test_samples[sample_idx]
        return {
            'sample_idx': sample_idx,
            'frames':     self.video_cache[sample_idx],
            'question':   sample['conversations'][0]['value'].replace("<video>\n", "").replace("<video>", ""),
            'reference':  sample['conversations'][1]['value'],
            'metadata':   sample.get('metadata', {})
        }


def collate_fn(batch):
    return {
        'sample_indices': [item['sample_idx'] for item in batch],
        'frames':         [item['frames']      for item in batch],
        'questions':      [item['question']    for item in batch],
        'references':     [item['reference']   for item in batch],
        'metadata':       [item['metadata']    for item in batch],
    }


# ============================================================
# MODEL LOADING
# ============================================================
def _get_model_class(base_model_name: str):
    """Auto-select Qwen3 / Qwen2"""
    if "Qwen3" in base_model_name:
        return Qwen3VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def load_baseline_model(base_model_name: str):
    logger.info(f"Loading baseline model: {base_model_name}")
    ModelClass = _get_model_class(base_model_name)

    model = ModelClass.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
        cache_dir=CACHE_DIR
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_model_name, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    processor.tokenizer.padding_side = 'left'

    used_mem = torch.cuda.memory_allocated(0) / (1024**3)
    logger.info(f"Baseline model loaded | GPU 0: {used_mem:.1f} GB used")
    return model, processor


def load_finetuned_model(base_model_name: str, lora_path: str):
    logger.info(f"Loading fine-tuned model: {base_model_name} + {lora_path}")
    ModelClass = _get_model_class(base_model_name)

    base_model = ModelClass.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
        cache_dir=CACHE_DIR
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_model_name, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    processor.tokenizer.padding_side = 'left'

    used_mem = torch.cuda.memory_allocated(0) / (1024**3)
    logger.info(f"Fine-tuned model loaded (LoRA merged) | GPU 0: {used_mem:.1f} GB used")
    return model, processor


# ============================================================
# BATCH INFERENCE
# ============================================================
def extract_answer_from_generation(generated_text: str) -> str:
    if not generated_text or generated_text.strip() == "":
        return ""
    patterns = [
        "\nassistant\n", "assistant\n", "\nassistant",
        "<|im_start|>assistant\n", "<|im_start|>assistant"
    ]
    for pattern in patterns:
        if pattern in generated_text:
            answer = generated_text.split(pattern)[-1].strip()
            return answer.replace("<|im_end|>", "").strip()
    return generated_text.strip()


def batch_generate(model, processor, frames_batch: List, questions_batch: List,
                   max_new_tokens: int) -> List[str]:
    try:
        prompts = [
            f"<|im_start|>user\n"
            f"<|vision_start|><|video_pad|><|vision_end|>{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for q in questions_batch
        ]

        inputs = processor(
            text=prompts, videos=frames_batch,
            padding=True, return_tensors="pt"
        ).to("cuda:0")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        answers = []
        for i, output in enumerate(outputs):
            generated_ids = output[inputs['input_ids'][i].shape[0]:]
            generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            answers.append(extract_answer_from_generation(generated_text))

        del inputs, outputs
        torch.cuda.empty_cache()
        return answers

    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM during batch generation")
        torch.cuda.empty_cache()
        return [""] * len(questions_batch)
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return [""] * len(questions_batch)


# ============================================================
# BATCH EVALUATION
# ============================================================
def evaluate_model_batched(model, processor, dataloader, model_name: str,
                            max_new_tokens: int) -> Tuple[List[str], List[Dict]]:
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {model_name} (Batch size: {BATCH_SIZE})")
    logger.info(f"{'='*70}")

    all_predictions = [None] * len(dataloader.dataset.test_samples)
    all_results = []

    for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        predictions = batch_generate(
            model, processor, batch['frames'], batch['questions'], max_new_tokens
        )
        for i, sample_idx in enumerate(batch['sample_indices']):
            all_predictions[sample_idx] = predictions[i]
            all_results.append({
                'sample_id':        sample_idx,
                'chunk_id':         dataloader.dataset.test_samples[sample_idx].get('id', f'sample_{sample_idx}'),
                'question':         batch['questions'][i],
                'reference_answer': batch['references'][i],
                'predicted_answer': predictions[i],
                'march_category':   batch['metadata'][i].get('march_category', 'N/A'),
                'care_phase':       batch['metadata'][i].get('care_phase', 'N/A'),
                'question_type':    batch['metadata'][i].get('question_type', 'N/A'),
                'source':           batch['metadata'][i].get('source', 'unknown'),
            })

    all_predictions = [p if p is not None else "" for p in all_predictions]
    return all_predictions, all_results


# ============================================================
# METRICS
# ============================================================
def calculate_metrics(predictions: List[str], references: List[str]) -> Dict:
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p]
    if not valid_pairs:
        return {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0,
                'rougeL': 0.0, 'bertscore_f1': 0.0,
                'num_valid': 0, 'num_total': len(predictions)}

    valid_preds, valid_refs = zip(*valid_pairs)

    smoothing = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([r.split()], p.split(), smoothing_function=smoothing)
                   for p, r in zip(valid_preds, valid_refs)]

    scorer_r = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for p, r in zip(valid_preds, valid_refs):
        s = scorer_r.score(r, p)
        rouge_scores['rouge1'].append(s['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(s['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(s['rougeL'].fmeasure)

    _, _, F1 = bert_score(list(valid_preds), list(valid_refs), lang='en', verbose=False)

    return {
        'bleu':          np.mean(bleu_scores),
        'rouge1':        np.mean(rouge_scores['rouge1']),
        'rouge2':        np.mean(rouge_scores['rouge2']),
        'rougeL':        np.mean(rouge_scores['rougeL']),
        'bertscore_f1':  F1.mean().item(),
        'num_valid':     len(valid_pairs),
        'num_total':     len(predictions),
    }


# ============================================================
# SAVE PER-MODEL RESULTS
# ============================================================
def save_results(model_name, output_dir, test_samples, references,
                 finetuned_predictions, finetuned_metrics,
                 baseline_predictions, baseline_metrics, args):

    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calc_rougeL(pred, ref):
        if not pred:
            return 0.0
        return rscorer.score(ref, pred)['rougeL'].fmeasure

    comparison_df = pd.DataFrame({
        'sample_id':           range(len(test_samples)),
        'chunk_id':            [s.get('id', f'sample_{i}') for i, s in enumerate(test_samples)],
        'question':            [s['conversations'][0]['value'] for s in test_samples],
        'reference_answer':    references,
        'finetuned_prediction': finetuned_predictions,
        'march_category':      [s.get('metadata', {}).get('march_category', 'N/A') for s in test_samples],
        'care_phase':          [s.get('metadata', {}).get('care_phase', 'N/A') for s in test_samples],
        'question_type':       [s.get('metadata', {}).get('question_type', 'N/A') for s in test_samples],
        'source':              [s.get('metadata', {}).get('source', 'unknown') for s in test_samples],
    })

    if baseline_predictions is not None:
        comparison_df['baseline_prediction'] = baseline_predictions

    # Deduplication
    before = len(comparison_df)
    comparison_df = comparison_df.drop_duplicates(subset=['chunk_id', 'question'], keep='first')
    if len(comparison_df) < before:
        logger.warning(f"Removed {before - len(comparison_df)} duplicate rows")

    # Per-sample ROUGE-L
    comparison_df['finetuned_rougeL'] = comparison_df.apply(
        lambda row: calc_rougeL(row['finetuned_prediction'], row['reference_answer']), axis=1
    )
    if baseline_predictions is not None:
        comparison_df['baseline_rougeL'] = comparison_df.apply(
            lambda row: calc_rougeL(row['baseline_prediction'], row['reference_answer']), axis=1
        )
        comparison_df['improvement'] = comparison_df['finetuned_rougeL'] - comparison_df['baseline_rougeL']

    # Print summary
    print(f"\nFINE-TUNED ({model_name})")
    print(f"  Valid: {finetuned_metrics['num_valid']}/{finetuned_metrics['num_total']}")
    print(f"  BLEU:      {finetuned_metrics['bleu']:.4f}")
    print(f"  ROUGE-1:   {finetuned_metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:   {finetuned_metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:   {finetuned_metrics['rougeL']:.4f}")
    print(f"  BERTScore: {finetuned_metrics['bertscore_f1']:.4f}")

    if baseline_metrics is not None:
        print(f"\n📈 IMPROVEMENT ({model_name} - Baseline)")
        for key, label in [('bleu','BLEU'), ('rouge1','ROUGE-1'), ('rouge2','ROUGE-2'),
                            ('rougeL','ROUGE-L'), ('bertscore_f1','BERTScore')]:
            imp = finetuned_metrics[key] - baseline_metrics[key]
            pct = 100 * imp / max(baseline_metrics[key], 1e-9)
            print(f"  {label:12s} {imp:+.4f}  ({pct:+.1f}%)")

    # Save CSV & JSON
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison_csv = out / f"comparison_{model_name}.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    metrics_dict = {
        'finetuned':  finetuned_metrics,
        'model_name': model_name,
        'test_file':  str(args.test_file),
        'n_samples':  len(test_samples),
        'batch_size': args.batch_size,
    }
    if baseline_metrics is not None:
        metrics_dict['baseline'] = baseline_metrics
        metrics_dict['improvement'] = {
            k: finetuned_metrics[k] - baseline_metrics[k]
            for k in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
        }

    metrics_json = out / f"metrics_{model_name}.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Saved: {comparison_csv}")
    print(f"Saved: {metrics_json}")


# ============================================================
# MAIN
# ============================================================
def evaluate(args):
    print("=" * 70)
    print("MODEL EVALUATION (OPTIMIZED)")
    print("=" * 70)
    print(f"Test file:     {args.test_file}")
    print(f"Base model:    {args.base_model}")
    print(f"Models:        {args.model_names}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Skip baseline: {args.skip_baseline}")
    print("=" * 70)

    # Validate model_paths / model_names count
    if len(args.model_paths) != len(args.model_names):
        raise ValueError("--model-paths and --model-names must have the same number of entries")

    # Load test data
    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_samples = []
    with open(test_file, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))

    if args.n_samples:
        test_samples = test_samples[:args.n_samples]

    logger.info(f"Loaded {len(test_samples)} test samples")
    references = [s['conversations'][1]['value'] for s in test_samples]

    # Load videos once
    video_cache = preload_all_videos(test_samples, max_workers=args.num_workers)

    dataset = TCCCDataset(test_samples, video_cache)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    logger.info(f"Dataloader ready: {len(dataset)} valid samples")

    # Baseline (once only)
    baseline_predictions = None
    baseline_metrics = None

    if not args.skip_baseline:
        baseline_model, baseline_processor = load_baseline_model(args.base_model)

        baseline_predictions, _ = evaluate_model_batched(
            baseline_model, baseline_processor,
            dataloader, f"Baseline ({args.base_model})", args.max_new_tokens
        )
        baseline_metrics = calculate_metrics(baseline_predictions, references)

        print(f"\nBASELINE ({args.base_model})")
        print(f"  Valid: {baseline_metrics['num_valid']}/{baseline_metrics['num_total']}")
        print(f"  BLEU:      {baseline_metrics['bleu']:.4f}")
        print(f"  ROUGE-1:   {baseline_metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:   {baseline_metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:   {baseline_metrics['rougeL']:.4f}")
        print(f"  BERTScore: {baseline_metrics['bertscore_f1']:.4f}")

        del baseline_model, baseline_processor
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info("Skipping baseline evaluation")

    # Sequential model evaluation (no video reload)
    print("\n" + "=" * 70)
    print(f"Evaluating {len(args.model_paths)} fine-tuned models")
    print("=" * 70)

    for model_path, model_name in zip(args.model_paths, args.model_names):
        logger.info(f"\n>>> {model_name} ({model_path})")

        finetuned_model, finetuned_processor = load_finetuned_model(
            args.base_model, model_path
        )

        finetuned_predictions, _ = evaluate_model_batched(
            finetuned_model, finetuned_processor,
            dataloader, model_name, args.max_new_tokens
        )
        finetuned_metrics = calculate_metrics(finetuned_predictions, references)

        # Model-specific output_dir: args.output_dir / model_name
        model_output_dir = Path(args.output_dir) / model_name / "eval"
        save_results(
            model_name, model_output_dir, test_samples, references,
            finetuned_predictions, finetuned_metrics,
            baseline_predictions, baseline_metrics, args
        )

        del finetuned_model, finetuned_processor
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned models (video loaded once)"
    )

    parser.add_argument("--test-file",    type=str, required=True)
    parser.add_argument("--output-dir",   type=str, required=True,
                        help="Base output dir. Per-model results saved under output_dir/model_name/eval/")
    parser.add_argument("--video-dir",    type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--base-model",   type=str, default=BASE_MODEL)
    parser.add_argument("--cache-dir",    type=str, default=DEFAULT_CACHE_DIR)

    # Support multiple models
    parser.add_argument("--model-paths",  type=str, nargs='+', required=True,
                        help="LoRA adapter dirs (space-separated)")
    parser.add_argument("--model-names",  type=str, nargs='+', required=True,
                        help="Model names for output files (space-separated, same order as --model-paths)")

    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--n-samples",    type=int, default=None)
    parser.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers",  type=int, default=8)
    parser.add_argument("--max-frames",   type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--target-size",  type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()

    # Assign globals from args
    CACHE_DIR      = args.cache_dir
    VIDEO_DIR      = Path(args.video_dir)
    BATCH_SIZE     = args.batch_size
    NUM_WORKERS    = args.num_workers
    MAX_FRAMES     = args.max_frames
    TARGET_SIZE    = args.target_size
    MAX_NEW_TOKENS = args.max_new_tokens

    # Setup cache
    setup_cache(CACHE_DIR)

    evaluate(args)