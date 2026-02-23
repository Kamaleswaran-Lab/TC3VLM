#!/usr/bin/env python3
"""
Large Model Baseline Comparison

Compares fine-tuned models against larger baseline models.
Generates predictions from large models and merges with fine-tuned results.

Usage:
    python compare_large_models.py --large-model Qwen/Qwen2.5-VL-72B-Instruct \
        --finetuned-csv results.csv --test-data test.jsonl --output-dir ./comparison
"""

import os
import sys

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_OUTPUT_DIR = "./results/comparison"
DEFAULT_VIDEO_DIR = "./data/videos/raw"
DEFAULT_MAX_FRAMES = 32
DEFAULT_BATCH_SIZE = 4

def setup_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
TARGET_SIZE  = 448
MAX_NEW_TOKENS = 256
BATCH_SIZE   = 2   # 72B models use more memory
NUM_WORKERS  = 8


# ============================================================
# MODEL CLASS AUTO-SELECTION
# ============================================================
def get_model_class(model_name: str):
    if "Qwen2.5" in model_name or "Qwen2_5" in model_name:
        return Qwen2_5_VLForConditionalGeneration
    elif "Qwen3" in model_name:
        return Qwen3VLForConditionalGeneration
    else:
        return Qwen2VLForConditionalGeneration


# ============================================================
# FRAME EXTRACTION
# ============================================================
def extract_frames(video_path, start_time, end_time, max_frames, target_size):
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"
        seek_pts = int(start_time / float(stream.time_base))
        container.seek(seek_pts, stream=stream)
        raw_frames = []
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_time: continue
            if t > end_time: break
            raw_frames.append(frame)
        container.close()
        if raw_frames:
            indices = np.linspace(0, len(raw_frames)-1, min(max_frames, len(raw_frames)), dtype=int)
            return [cv2.resize(raw_frames[i].to_ndarray(format='rgb24'), (target_size, target_size))
                    for i in indices]
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: return None
        sf, ef = int(start_time*fps), int(end_time*fps)
        indices = np.linspace(sf, ef-1, min(max_frames, ef-sf), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(cv2.resize(frame, (target_size, target_size)), cv2.COLOR_BGR2RGB))
        cap.release()
        return frames if frames else None
    except Exception:
        return None


def extract_worker(args):
    video_path, start_time, end_time, max_frames, target_size, idx = args
    return idx, extract_frames(video_path, start_time, end_time, max_frames, target_size)


def preload_videos(test_samples):
    logger.info(f"Pre-loading {len(test_samples)} videos...")
    tasks = [(s['video'], s['start_time'], s['end_time'], MAX_FRAMES, TARGET_SIZE, i)
             for i, s in enumerate(test_samples)]
    video_cache = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(extract_worker, t): t[5] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading videos"):
            idx, frames = future.result()
            if frames is not None:
                video_cache[idx] = frames
    logger.info(f"Loaded {len(video_cache)}/{len(test_samples)} videos")
    return video_cache


# ============================================================
# DATASET
# ============================================================
class EvalDataset(Dataset):
    def __init__(self, test_samples, video_cache):
        self.samples = test_samples
        self.cache = video_cache
        self.valid = [i for i in range(len(test_samples)) if i in video_cache]

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        s = self.samples[i]
        return {
            'idx':       i,
            'frames':    self.cache[i],
            'question':  s['conversations'][0]['value'].replace("<video>\n","").replace("<video>",""),
            'reference': s['conversations'][1]['value'],
        }


def collate_fn(batch):
    return {
        'indices':    [b['idx']       for b in batch],
        'frames':     [b['frames']    for b in batch],
        'questions':  [b['question']  for b in batch],
        'references': [b['reference'] for b in batch],
    }


# ============================================================
# INFERENCE
# ============================================================
def run_inference(model, processor, dataloader, model_label) -> Dict[int, str]:
    predictions = {}

    for batch in tqdm(dataloader, desc=f"Inference: {model_label}"):
        prompts = [
            f"<|im_start|>user\n"
            f"<|vision_start|><|video_pad|><|vision_end|>{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for q in batch['questions']
        ]
        try:
            inputs = processor(
                text=prompts, videos=batch['frames'],
                padding=True, return_tensors="pt"
            ).to("cuda:0")

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            for i, (output, idx) in enumerate(zip(outputs, batch['indices'])):
                gen_ids = output[inputs['input_ids'][i].shape[0]:]
                text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                predictions[idx] = text

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Batch error: {e}")
            for idx in batch['indices']:
                predictions[idx] = ""

    return predictions


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Large model comparison")
    parser.add_argument("--large-model",    type=str, required=True,
                        help="Large model to compare (e.g. Qwen/Qwen2.5-VL-72B-Instruct)")
    parser.add_argument("--large-model-label", type=str, default=None,
                        help="Short label for large model (default: auto from model name)")
    parser.add_argument("--finetuned-csvs", type=str, nargs='+', required=True,
                        help="Existing finetuned comparison CSVs (can pass multiple)")
    parser.add_argument("--finetuned-labels", type=str, nargs='+', required=True,
                        help="Labels for each finetuned CSV (e.g. qwen3_r32 qwen2_r32)")
    parser.add_argument("--test-file",      type=str, required=True)
    parser.add_argument("--output-dir",     type=str, required=True)
    parser.add_argument("--video-dir",      type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--cache-dir",      type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size",     type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-frames",     type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--n-samples",      type=int, default=None)
    parser.add_argument("--tensor-parallel",type=int, default=1,
                        help="Number of GPUs for tensor parallelism (use vLLM for 72B)")
    args = parser.parse_args()
    
    # Assign globals from args
    global CACHE_DIR, VIDEO_DIR, BATCH_SIZE, MAX_FRAMES
    CACHE_DIR = args.cache_dir
    VIDEO_DIR = Path(args.video_dir)
    BATCH_SIZE = args.batch_size
    MAX_FRAMES = args.max_frames
    
    # Setup cache
    setup_cache(CACHE_DIR)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto label
    model_label = args.large_model_label or args.large_model.split("/")[-1].replace("-Instruct", "")

    print("=" * 65)
    print("LARGE MODEL COMPARISON")
    print("=" * 65)
    print(f"Large model  : {args.large_model} ({model_label})")
    print(f"Finetuned    : {args.finetuned_labels}")
    print(f"Output       : {output_dir}")
    print("=" * 65)

    # Load test data
    test_samples = []
    with open(args.test_file, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
    if args.n_samples:
        test_samples = test_samples[:args.n_samples]
    logger.info(f"{len(test_samples)} test samples")

    # Load videos
    video_cache = preload_videos(test_samples)
    dataset = EvalDataset(test_samples, video_cache)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Load large model & run inference
    logger.info(f"\nLoading {model_label}...")
    ModelClass = get_model_class(args.large_model)

    model = ModelClass.from_pretrained(
        args.large_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # 72B uses multi-GPU auto
        trust_remote_code=True,
        attn_implementation="eager",
        cache_dir=CACHE_DIR,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.large_model, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    processor.tokenizer.padding_side = 'left'

    large_preds = run_inference(model, processor, dataloader, model_label)

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Merge with each finetuned CSV
    for csv_path, ft_label in zip(args.finetuned_csvs, args.finetuned_labels):
        logger.info(f"\nMerging: {ft_label}")
        ft_df = pd.read_csv(csv_path)

        # Add large model predictions
        large_col = f"{model_label}_prediction"
        ft_df[large_col] = ft_df['sample_id'].map(
            lambda i: large_preds.get(i, "")
        )

        # Save CSV for LLM judge (finetuned vs large model)
        judge_df = ft_df[['question', 'reference_answer', 'finetuned_prediction']].copy()
        judge_df[large_col] = ft_df[large_col]

        out_path = output_dir / f"comparison_{ft_label}_vs_{model_label}.csv"
        judge_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        # Statistics
        ft_len = ft_df['finetuned_prediction'].str.len().mean()
        lg_len = ft_df[large_col].str.len().mean()
        print(f"  Avg response length → Finetuned: {ft_len:.0f}, {model_label}: {lg_len:.0f}")

    print("\n" + "=" * 65)
    print("COMPARISON COMPLETE")
    print("=" * 65)
    print(f"\nNext step - LLM Judge:")
    for ft_label in args.finetuned_labels:
        csv_name = f"comparison_{ft_label}_vs_{model_label}.csv"
        print(f"\n  python evaluation/llm_judge.py \\")
        print(f"      --input-csv {output_dir}/{csv_name} \\")
        print(f"      --output-dir {output_dir}/{ft_label}_vs_{model_label}_judge \\")
        print(f"      --judges llama405b qwen72b")


if __name__ == "__main__":
    main()