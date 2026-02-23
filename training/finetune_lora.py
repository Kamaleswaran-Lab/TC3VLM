#!/usr/bin/env python3
"""
Fine-tuning with LoRA

Fine-tunes vision-language models using LoRA (Low-Rank Adaptation).
Optimizations:
- Frame pre-caching for faster training
- PyAV with OpenCV fallback for video decoding
- Empty batch handling
- Reproducible with fixed seed

Usage:
    python finetune_lora.py --train-data train.jsonl --val-data val.jsonl --output-dir ./models
"""

import os
import sys

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_OUTPUT_DIR = "./models/lora"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_FRAME_CACHE_DIR = "./cache/frames"
DEFAULT_VIDEO_DIR = "./data/videos/raw"

# Setup cache directories
def setup_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import torch
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import cv2
import numpy as np
import logging
import av
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
BASE_OUTPUT_DIR = Path("/work/jkim1/TCCC/models")

# Frame cache directory
FRAME_CACHE_DIR = Path("/work/jkim1/TCCC/frame_cache")

# Training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 5
SAVE_STEPS = 50
EVAL_STEPS = 50
SEED = 42

# Video parameters
MAX_FRAMES = 28
TARGET_SIZE = 448
MAX_SEQ_LENGTH = 12288

# LoRA parameters
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training mode for ablation study
TRAINING_MODE = "combined"  # Options: caption, visual, combined


# ============================================================
# FRAME EXTRACTION (PyAV preferred)
# ============================================================
def extract_frames_pyav(video_path: str, start_time: float, end_time: float,
                        max_frames: int, target_size: int) -> Optional[List[np.ndarray]]:
    """Extract frames using PyAV - handles AV1, VP9, H264"""
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

        if not raw_frames:
            return None

        if len(raw_frames) <= max_frames:
            selected = raw_frames
        else:
            indices = np.linspace(0, len(raw_frames) - 1, max_frames, dtype=int)
            selected = [raw_frames[i] for i in indices]

        frames = []
        for frame in selected:
            img = frame.to_ndarray(format='rgb24')
            img = cv2.resize(img, (target_size, target_size))
            frames.append(img)

        return frames if frames else None

    except Exception as e:
        logger.debug(f"PyAV failed {Path(video_path).name}: {e}")
        return None


def extract_frames_opencv(video_path: str, start_time: float, end_time: float,
                          max_frames: int, target_size: int) -> Optional[List[np.ndarray]]:
    """Extract frames using OpenCV (fallback)"""
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
        return frames if frames else None

    except Exception as e:
        logger.debug(f"OpenCV failed {Path(video_path).name}: {e}")
        return None


def extract_frames(video_path: str, start_time: float, end_time: float,
                   max_frames: int, target_size: int) -> Optional[List[np.ndarray]]:
    """PyAV 우선, 실패 시 OpenCV fallback"""
    frames = extract_frames_pyav(video_path, start_time, end_time, max_frames, target_size)
    if frames is None:
        frames = extract_frames_opencv(video_path, start_time, end_time, max_frames, target_size)
    return frames


# ============================================================
# FRAME PRE-CACHE
# ============================================================
def get_cache_key(video_path: str, start_time: float, end_time: float,
                  max_frames: int, target_size: int) -> str:
    """샘플별 고유 캐시 키 생성"""
    key_str = f"{video_path}_{start_time:.3f}_{end_time:.3f}_{max_frames}_{target_size}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(cache_key: str) -> Path:
    return FRAME_CACHE_DIR / f"{cache_key}.npy"


def precache_frames(samples, max_frames, target_size, local_rank=0, num_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    to_process = []
    seen_keys = set()  # Prevent duplicates

    for sample in samples:
        cache_key = get_cache_key(
            sample['video'], sample['start_time'], sample['end_time'],
            max_frames, target_size
        )
        if cache_key in seen_keys:  # Key fix
            continue
        seen_keys.add(cache_key)
        
        cache_path = get_cache_path(cache_key)
        if cache_path.exists():
            results[cache_key] = True
        else:
            to_process.append((sample, cache_key))

    if local_rank == 0:
        logger.info(f"Frame cache: {len(results)} already cached, {len(to_process)} to process")

    if not to_process:
        return results

    def process_one(args):
        sample, cache_key = args
        frames = extract_frames(
            sample['video'], sample['start_time'], sample['end_time'],
            max_frames, target_size
        )
        if frames is not None and len(frames) > 0:
            np.save(get_cache_path(cache_key), np.stack(frames))
            return cache_key, True
        return cache_key, False

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, arg): arg for arg in to_process}
        iterator = tqdm(as_completed(futures), total=len(to_process), desc="Pre-caching frames") \
                   if local_rank == 0 else as_completed(futures)
        for future in iterator:
            cache_key, ok = future.result()
            results[cache_key] = ok
            if ok: success += 1
            else: failed += 1

    if local_rank == 0:
        logger.info(f"Pre-cache complete: {success} success, {failed} failed")

    return results


def load_cached_frames(cache_key: str) -> Optional[List[np.ndarray]]:
    """캐시에서 프레임 로드"""
    cache_path = get_cache_path(cache_key)
    if not cache_path.exists():
        return None
    try:
        frames_array = np.load(str(cache_path))
        return [frames_array[i] for i in range(len(frames_array))]
    except Exception as e:
        logger.error(f"Cache load error {cache_key}: {e}")
        return None


# ============================================================
# DATASET
# ============================================================
class TCCCVideoQADataset(Dataset):
    """Dataset for TCCC Video-QA pairs (with frame cache support)"""

    def __init__(self, jsonl_path: str, processor, max_frames: int, target_size: int,
                 max_seq_length: int, training_mode: str = "combined",
                 cache_results: Optional[Dict[str, bool]] = None):
        self.processor = processor
        self.max_frames = max_frames
        self.target_size = target_size
        self.max_seq_length = max_seq_length
        self.training_mode = training_mode
        self.samples = []
        self.cache_keys = []

        # Load JSONL
        raw_samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line)

                if training_mode == "caption":
                    if sample.get('metadata', {}).get('source') == 'visual_grounded':
                        continue
                elif training_mode == "visual":
                    if sample.get('metadata', {}).get('source') != 'visual_grounded':
                        continue

                raw_samples.append(sample)

        # Remove uncacheable samples
        for sample in raw_samples:
            cache_key = get_cache_key(
                sample['video'], sample['start_time'], sample['end_time'],
                max_frames, target_size
            )
            if cache_results is not None and not cache_results.get(cache_key, False):
                logger.debug(f"Skipping uncacheable sample: {sample.get('video', '')}")
                continue
            self.samples.append(sample)
            self.cache_keys.append(cache_key)

        logger.info(f"Loaded {len(self.samples)} samples (mode: {training_mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        sample = self.samples[idx]
        cache_key = self.cache_keys[idx]

        # Load frames from cache (disk read only, no decoding)
        frames = load_cached_frames(cache_key)

        # Fallback to real-time extraction on cache miss
        if frames is None:
            frames = extract_frames(
                sample['video'], sample['start_time'], sample['end_time'],
                self.max_frames, self.target_size
            )

        if frames is None or len(frames) == 0:
            logger.warning(f"Skipping sample {idx}: No frames")
            return None

        # Prepare text
        conversation = sample['conversations']
        question = conversation[0]['value'].replace("<video>\n", "")
        answer = conversation[1]['value']

        # Truncate answer to prevent OOM
        answer_words = answer.split()
        if len(answer_words) > 150:
            answer = ' '.join(answer_words[:150]) + '...'

        # Construct prompt
        prompt = (
            f"<|im_start|>user\n"
            f"<|vision_start|><|video_pad|><|vision_end|>{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )

        try:
            inputs = self.processor(
                text=[prompt],
                videos=[frames],
                padding=True,
                return_tensors="pt"
            )

            if inputs['input_ids'].shape[1] > self.max_seq_length:
                logger.warning(f"Sample {idx} exceeds max_seq_length, skipping")
                return None

            # Labels masking
            labels = inputs['input_ids'].clone()
            assistant_tokens = self.processor.tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            assistant_len = len(assistant_tokens)

            ids = labels[0].tolist()
            masked = False
            for i in range(len(ids) - assistant_len):
                if ids[i:i + assistant_len] == assistant_tokens:
                    labels[0, :i + assistant_len] = -100
                    masked = True
                    break

            if not masked:
                logger.warning(f"Sample {idx}: assistant token not found, masking all")
                labels[0, :] = -100

            result = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)
            }

            if 'pixel_values' in inputs:
                result['pixel_values'] = inputs['pixel_values'].squeeze(0)
            if 'image_grid_thw' in inputs:
                result['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)

            return result

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return None


# ============================================================
# DATA COLLATOR
# ============================================================
@dataclass
class DataCollatorForVideoQA:
    """Custom data collator for video Q/A"""

    processor: AutoProcessor

    def __call__(self, features: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f is not None]

        if len(features) == 0:
            # Return dummy batch instead of crashing
            logger.warning("Empty batch - returning dummy batch")
            return {
                'input_ids': torch.zeros((1, 1), dtype=torch.long),
                'attention_mask': torch.zeros((1, 1), dtype=torch.long),
                'labels': torch.full((1, 1), -100, dtype=torch.long),
            }

        batch = {}

        for key in ['input_ids', 'attention_mask', 'labels']:
            tensors = [f[key] for f in features]
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors,
                batch_first=True,
                padding_value=0 if key != 'labels' else -100
            )


        if 'pixel_values' in features[0]:
            batch['pixel_values'] = torch.cat([f['pixel_values'] for f in features], dim=0)

        if 'image_grid_thw' in features[0]:
            # Check shape then stack
            thw_tensors = [f['image_grid_thw'] for f in features]
            if thw_tensors[0].dim() == 1:
                thw_tensors = [t.unsqueeze(0) for t in thw_tensors]
            batch['image_grid_thw'] = torch.cat(thw_tensors, dim=0)

        return batch


# ============================================================
# MODEL SETUP
# ============================================================
def setup_model_and_processor(base_model: str):
    """Load model with LoRA configuration"""

    logger.info(f"Loading model: {base_model}")

    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
        cache_dir=CACHE_DIR
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    return model, processor


# ============================================================
# LOAD SAMPLES (for pre-cache)
# ============================================================
def load_all_samples(jsonl_path: str) -> List[Dict]:
    """JSONL에서 video/start_time/end_time만 추출"""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                'video': sample['video'],
                'start_time': sample['start_time'],
                'end_time': sample['end_time'],
            })
    return samples


# ============================================================
# MAIN
# ============================================================
def finetune(train_data_path, val_data_path, output_dir, training_mode,
             base_model, resume_checkpoint=None):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if local_rank == 0:
        print("=" * 70)
        print("PHASE 7: FINE-TUNING WITH LORA (Optimized)")
        print("=" * 70)
        print(f"\nTraining data:  {train_data_path}")
        print(f"Validation data:{val_data_path}")
        print(f"Training mode:  {training_mode}")
        print(f"Frame cache:    {FRAME_CACHE_DIR}")
        print(f"LORA r={LORA_R}, alpha={LORA_ALPHA}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # STEP 1: Frame Pre-cache (only rank 0, others wait)
    # --------------------------------------------------------
    if local_rank == 0:
        logger.info("\n[Step 1] Pre-caching frames...")
        train_samples = load_all_samples(train_data_path)
        val_samples = load_all_samples(val_data_path)
        all_samples = train_samples + val_samples

        cache_results = precache_frames(all_samples, MAX_FRAMES, TARGET_SIZE, local_rank=0, num_workers=8)
        
        logger.info(f"Cache ready: {sum(cache_results.values())}/{len(cache_results)} samples")
    else:
        cache_results = None

    torch.distributed.barrier()

    # Build cache_results for non-rank-0 processes (cache files are in shared storage)
    if local_rank != 0:
        train_samples = load_all_samples(train_data_path)
        val_samples = load_all_samples(val_data_path)
        all_samples = train_samples + val_samples
        cache_results = {
            get_cache_key(s['video'], s['start_time'], s['end_time'], MAX_FRAMES, TARGET_SIZE):
            get_cache_path(
                get_cache_key(s['video'], s['start_time'], s['end_time'], MAX_FRAMES, TARGET_SIZE)
            ).exists()
            for s in all_samples
        }

    # --------------------------------------------------------
    # STEP 2: Model load
    # --------------------------------------------------------
    if local_rank == 0:
        logger.info("\n[Step 2] Loading model...")
    model, processor = setup_model_and_processor(base_model)

    # --------------------------------------------------------
    # STEP 3: Dataset
    # --------------------------------------------------------
    if local_rank == 0:
        logger.info("\n[Step 3] Loading datasets...")

    train_dataset = TCCCVideoQADataset(
        train_data_path, processor, MAX_FRAMES, TARGET_SIZE,
        MAX_SEQ_LENGTH, training_mode, cache_results
    )
    val_dataset = TCCCVideoQADataset(
        val_data_path, processor, MAX_FRAMES, TARGET_SIZE,
        MAX_SEQ_LENGTH, "combined", cache_results
    )

    data_collator = DataCollatorForVideoQA(processor=processor)

    # --------------------------------------------------------
    # STEP 4: Training
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer
    )

    if local_rank == 0:
        logger.info("\n[Step 4] Starting training...")

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # --------------------------------------------------------
    # STEP 5: Save
    # --------------------------------------------------------
    if local_rank == 0:
        final_output_dir = output_dir / "final_model"
        trainer.save_model(str(final_output_dir))
        processor.save_pretrained(str(final_output_dir))

        print("\n" + "=" * 70)
        print("PHASE 7 COMPLETE")
        print("=" * 70)
        print(f"Model saved: {final_output_dir}")
        print(f"  Training mode:     {training_mode}")
        print(f"  Epochs:            {NUM_EPOCHS}")
        print(f"  Training samples:  {len(train_dataset)}")
        print(f"  Validation samples:{len(val_dataset)}")
        print(f"  LORA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")


# ============================================================
# ARGPARSE
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Fine-tune Qwen2-VL with LoRA (Optimized)")

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--training-mode", type=str, default="combined",
                        choices=["caption", "visual", "combined"])
    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    # Training hyperparameters
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)

    # Video parameters
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--frame-cache-dir", type=str, default=str(FRAME_CACHE_DIR))

    # LoRA parameters
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)

    args = parser.parse_args()

    # Update globals
    FRAME_CACHE_DIR = Path(args.frame_cache_dir)
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    LEARNING_RATE = args.learning_rate
    WARMUP_STEPS = args.warmup_steps
    SAVE_STEPS = args.save_steps
    EVAL_STEPS = args.eval_steps
    MAX_FRAMES = args.max_frames
    TARGET_SIZE = args.target_size
    MAX_SEQ_LENGTH = args.max_seq_length
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout

    finetune(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        training_mode=args.training_mode,
        base_model=args.base_model,
        resume_checkpoint=args.resume_from_checkpoint
    )