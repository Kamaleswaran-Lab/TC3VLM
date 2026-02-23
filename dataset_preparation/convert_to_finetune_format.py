#!/usr/bin/env python3
"""
Convert to Fine-tuning Format

Converts Q/A pairs to vision-language model training format (JSONL).
Prepares data with video timing information and proper structure.

Usage:
    python convert_to_finetune_format.py --train train.csv --val val.csv --output-dir ./finetune_data
"""

import os
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_OUTPUT_DIR = "./data/finetune"
DEFAULT_VIDEO_DIR = "./data/videos/raw"
DEFAULT_CHUNKS_CSV = "./data/segmented_videos.csv"

# ============================================================
# CHUNK METADATA LOADING
# ============================================================
def load_chunk_metadata(chunks_csv):
    """Load chunk metadata for video timing information"""

    print("Loading chunk metadata...")
    chunks_df = pd.read_csv(chunks_csv)
    print(f"Loaded {len(chunks_df)} chunks")

    chunk_dict = {}
    for _, row in chunks_df.iterrows():
        chunk_dict[row['chunk_id']] = {
            'video_file': row['video_file'],
            'start_time': row['start_time'],
            'end_time':   row['end_time'],
        }

    return chunk_dict


# ============================================================
# CONVERSION
# ============================================================
def create_training_sample(row, chunk_dict):
    """Convert a Q/A pair to Qwen2-VL training format"""

    chunk_id = row['chunk_id']

    if chunk_id not in chunk_dict:
        return None

    chunk_info = chunk_dict[chunk_id]
    video_path = VIDEO_DIR / chunk_info['video_file']

    if not video_path.exists():
        return None

    sample = {
        "id": chunk_id,
        "video": str(video_path),
        "start_time": float(chunk_info['start_time']),
        "end_time":   float(chunk_info['end_time']),
        "conversations": [
            {
                "from":  "human",
                "value": f"<video>\n{row['question']}",
            },
            {
                "from":  "gpt",
                "value": row['answer'],
            },
        ],
        "metadata": {
            "video_id":      row.get('video_id',      'unknown'),
            "march_category": row.get('march_category', 'N/A'),
            "care_phase":    row.get('care_phase',    'N/A'),
            "skill_level":   row.get('skill_level',   'N/A'),
            "content_type":  row.get('content_type',  'N/A'),
            "question_type": row.get('question_type', 'N/A'),
            "source":        row.get('source',        'unknown'),
        },
    }

    return sample


def process_split(split_name, input_csv, chunk_dict, output_dir):
    """Process train/val/test split"""

    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*70}")

    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"ERROR: {split_name} CSV not found: {input_path}")
        return 0

    df = pd.read_csv(input_path)
    print(f"Total Q/A pairs: {len(df)}")

    if 'source' in df.columns:
        print(f"\nSource distribution:")
        print(df['source'].value_counts())

    samples = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split_name}"):
        sample = create_training_sample(row, chunk_dict)
        if sample is not None:
            samples.append(sample)
        else:
            skipped += 1

    print(f"\nConverted: {len(samples)}")
    print(f"Skipped:   {skipped}")

    output_file = output_dir / f"{split_name}.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_file}")
    return len(samples)


# ============================================================
# MAIN
# ============================================================
def convert_to_finetune_format(input_dir, output_dir):
    print("=" * 70)
    print("CONVERT TO FINE-TUNING FORMAT")
    print("Converts Q/A pairs to Qwen2-VL JSONL format")
    print("=" * 70)

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    # Validate input files
    splits = [
        ('train', input_dir / "train.csv"),
        ('val',   input_dir / "val.csv"),
        ('test',  input_dir / "test.csv"),
    ]
    for split_name, path in splits:
        if not path.exists():
            raise FileNotFoundError(
                f"{split_name}.csv not found in {input_dir}\n"
                f"Please run Phase 6.1 first to generate the split CSVs."
            )

    chunk_dict = load_chunk_metadata()

    stats = {}
    for split_name, input_csv in splits:
        stats[split_name] = process_split(split_name, input_csv, chunk_dict, output_dir)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print("\nConversion Summary:")
    for split, count in stats.items():
        print(f"  {split.capitalize()}: {count} samples")

    print(f"\nOutput directory: {output_dir}")
    print("  Files:")
    for split in ['train', 'val', 'test']:
        jsonl_file = output_dir / f"{split}.jsonl"
        if jsonl_file.exists():
            print(f"    - {jsonl_file.name}")

    print("\n For Ablation Study:")
    print("  To filter training data by source:")
    print("  - Load train.jsonl")
    print("  - Filter by metadata['source']:")
    print("    • 'caption':          Caption-based Q/A only")
    print("    • 'visual_grounded':  Visual Q/A only")
    print("    • Both:               Combined training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6.2: Convert Q/A pairs to fine-tuning format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Phase 6.1 output directory containing train.csv / val.csv / test.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for train.jsonl / val.jsonl / test.jsonl"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print the first entry of train.jsonl after conversion"
    )

    args = parser.parse_args()

    convert_to_finetune_format(args.input_dir, args.output_dir)

    if args.show_sample:
        train_jsonl = Path(args.output_dir) / "train.jsonl"
        if train_jsonl.exists():
            print("\n" + "=" * 70)
            print("Sample JSONL Entry:")
            print("=" * 70)
            with open(train_jsonl, 'r') as f:
                sample = json.loads(f.readline())
            print(json.dumps(sample, indent=2, ensure_ascii=False))