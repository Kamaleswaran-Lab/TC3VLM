#!/usr/bin/env python3
"""
Train/Val/Test Split

Splits Q/A pairs at video level to prevent data leakage.
Combines caption and visual Q/A for comprehensive training.

Usage:
    python split_dataset.py --caption-qa qa_caption.csv --visual-qa qa_visual.csv --output-dir ./splits
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_DIR = "./data/splits"

# ============================================================
# DATA LOADING
# ============================================================
def load_qa_data(caption_csv, visual_csv):
    """
    Load and combine caption and visual Q/A pairs
    Both will be included in train/val/test for ablation study
    """
    dfs = []

    caption_path = Path(caption_csv)
    if caption_path.exists():
        df_caption = pd.read_csv(caption_path)
        if 'source_caption' in df_caption.columns and 'source' not in df_caption.columns:
            df_caption['source'] = 'caption'
        print(f"Loaded caption Q/A: {len(df_caption)} pairs")
        dfs.append(df_caption)
    else:
        print(f"Caption Q/A not found: {caption_path}")

    visual_path = Path(visual_csv)
    if visual_path.exists():
        df_visual = pd.read_csv(visual_path)
        print(f"Loaded visual Q/A: {len(df_visual)} pairs")
        dfs.append(df_visual)
    else:
        print(f"Visual Q/A not found: {visual_path}")

    if len(dfs) == 0:
        raise ValueError("No Q/A data found!")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(df)} Q/A pairs")

    if 'source' not in df.columns:
        print("Warning: 'source' column not found. Ablation study may be difficult.")
    else:
        print(f"\nSource distribution:")
        print(df['source'].value_counts())

    return df


# ============================================================
# VIDEO-LEVEL ANALYSIS
# ============================================================
def analyze_video_distribution(df):
    print("\n" + "=" * 70)
    print("Video-Level Statistics")
    print("=" * 70)

    video_stats = df.groupby('video_id').agg(
        n_chunks=('chunk_id', 'nunique'),
        n_qa=('question', 'count'),
        march_category=('march_category', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'),
        care_phase=('care_phase', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'),
    )

    print(f"\nTotal unique videos: {len(video_stats)}")
    print(f"Total Q/A pairs: {len(df)}")
    print(f"Avg Q/A per video: {len(df) / len(video_stats):.2f}")
    print(f"\nQ/A count per video:")
    print(video_stats['n_qa'].describe())
    print(f"\nMARCH category distribution:")
    print(video_stats['march_category'].value_counts())

    return video_stats


# ============================================================
# RANDOM VIDEO SPLIT
# ============================================================
def split_videos(video_stats, train_ratio, val_ratio, test_ratio):
    print("\n" + "=" * 70)
    print("Video-Level Split (Random)")
    print("=" * 70)
    print("Note: Using random split to ensure video-level separation")
    print("All splits contain both caption and visual Q/A for ablation study")

    video_ids = video_stats.index.tolist()

    train_videos, temp_videos = train_test_split(
        video_ids,
        test_size=(val_ratio + test_ratio),
        random_state=RANDOM_SEED,
    )
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=RANDOM_SEED,
    )

    total = len(video_ids)
    print(f"\nVideo split:")
    print(f"  Train: {len(train_videos)} videos ({len(train_videos)/total*100:.1f}%)")
    print(f"  Val:   {len(val_videos)} videos ({len(val_videos)/total*100:.1f}%)")
    print(f"  Test:  {len(test_videos)} videos ({len(test_videos)/total*100:.1f}%)")

    return train_videos, val_videos, test_videos


# ============================================================
# CREATE SPLITS
# ============================================================
def create_splits(df, train_videos, val_videos, test_videos):
    train_df = df[df['video_id'].isin(train_videos)].copy()
    val_df   = df[df['video_id'].isin(val_videos)].copy()
    test_df  = df[df['video_id'].isin(test_videos)].copy()

    print("\n" + "=" * 70)
    print("Q/A Pair Split")
    print("=" * 70)
    print(f"Train: {len(train_df)} Q/A pairs ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} Q/A pairs ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} Q/A pairs ({len(test_df)/len(df)*100:.1f}%)")

    if 'source' in df.columns:
        print("\nSource Distribution by Split:")
        for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n{name}:")
            for src, count in split['source'].value_counts().items():
                print(f"  {src}: {count} ({count/len(split)*100:.1f}%)")

    return train_df, val_df, test_df


# ============================================================
# ANALYZE SPLITS
# ============================================================
def analyze_splits(train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print("Distribution Analysis")
    print("=" * 70)

    print("\nMARCH Category Distribution (%):")
    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name}:")
        for cat, pct in split['march_category'].value_counts(normalize=True).mul(100).round(1).head(10).items():
            print(f"  {cat}: {pct}%")

    print("\nCare Phase Distribution (%):")
    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name}:")
        for phase, pct in split['care_phase'].value_counts(normalize=True).mul(100).round(1).items():
            print(f"  {phase}: {pct}%")

    if 'question_type' in train_df.columns:
        print("\nQuestion Type Distribution (%):")
        for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n{name}:")
            for qtype, pct in split['question_type'].value_counts(normalize=True).mul(100).round(1).items():
                print(f"  {qtype}: {pct}%")


# ============================================================
# SAVE
# ============================================================
def save_splits(train_df, val_df, test_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        'train': output_dir / "train.csv",
        'val':   output_dir / "val.csv",
        'test':  output_dir / "test.csv",
    }
    train_df.to_csv(paths['train'], index=False)
    val_df.to_csv(paths['val'],   index=False)
    test_df.to_csv(paths['test'], index=False)

    print(f"\nTrain saved: {paths['train']}")
    print(f"Val saved:   {paths['val']}")
    print(f"Test saved:  {paths['test']}")
    return paths


def save_statistics(df, train_df, val_df, test_df,
                    train_videos, val_videos, test_videos,
                    caption_csv, visual_csv, output_dir):
    stats_path = Path(output_dir) / "stats.txt"
    total_videos = len(train_videos) + len(val_videos) + len(test_videos)

    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 6.1: Train/Val/Test Split Statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write("Input Files:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Caption Q/A: {caption_csv}\n")
        f.write(f"Visual Q/A:  {visual_csv}\n\n")

        f.write("Ablation Study Design:\n")
        f.write("-" * 40 + "\n")
        f.write("All splits contain both caption and visual Q/A pairs.\n")
        f.write("For training, filter by 'source' column:\n")
        f.write("  - Condition 1: source != 'visual_grounded'  (caption only)\n")
        f.write("  - Condition 2: source == 'visual_grounded'  (visual only)\n")
        f.write("  - Condition 3: all sources                  (caption + visual)\n")
        f.write("Val/Test remain unchanged for all conditions.\n\n")

        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Q/A pairs:      {len(df)}\n")
        f.write(f"Total unique videos:  {df['video_id'].nunique()}\n")
        f.write(f"Total unique chunks:  {df['chunk_id'].nunique()}\n\n")

        if 'source' in df.columns:
            f.write("Source Distribution:\n")
            f.write("-" * 40 + "\n")
            for src, count in df['source'].value_counts().items():
                f.write(f"{src}: {count} ({count/len(df)*100:.1f}%)\n")
            f.write("\n")

        f.write("Video Split:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train videos: {len(train_videos)} ({len(train_videos)/total_videos*100:.1f}%)\n")
        f.write(f"Val videos:   {len(val_videos)} ({len(val_videos)/total_videos*100:.1f}%)\n")
        f.write(f"Test videos:  {len(test_videos)} ({len(test_videos)/total_videos*100:.1f}%)\n\n")

        f.write("Q/A Pair Split:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train Q/A: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"Val Q/A:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)\n")
        f.write(f"Test Q/A:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)\n\n")

        if 'source' in df.columns:
            f.write("Source Distribution by Split:\n")
            f.write("-" * 40 + "\n")
            for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                f.write(f"\n{name}:\n")
                for src, count in split['source'].value_counts().items():
                    f.write(f"  {src}: {count} ({count/len(split)*100:.1f}%)\n")
            f.write("\n")

        f.write("MARCH Category Distribution (%):\n")
        f.write("-" * 40 + "\n")
        for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            f.write(f"\n{name}:\n")
            for cat, pct in split['march_category'].value_counts(normalize=True).mul(100).round(1).items():
                f.write(f"  {cat}: {pct}%\n")

    print(f"Statistics saved: {stats_path}")


# ============================================================
# MAIN
# ============================================================
def split_dataset(caption_csv, visual_csv, output_dir, train_ratio, val_ratio, test_ratio):
    print("=" * 70)
    print("TRAIN/VAL/TEST SPLIT")
    print("Video-level split for ablation study")
    print("=" * 70)
    print(f"\nCaption CSV: {caption_csv}")
    print(f"Visual CSV:  {visual_csv}")
    print(f"Output dir:  {output_dir}")
    print(f"Split ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    print(f"Random seed: {RANDOM_SEED}")
    print("\n  Ablation Study Design:")
    print("  - All splits contain caption + visual Q/A")
    print("  - Filter by 'source' column during training:")
    print("    • Condition 1: Caption only")
    print("    • Condition 2: Visual only")
    print("    • Condition 3: Caption + Visual")
    print("  - Val/Test remain unchanged for all conditions")

    df = load_qa_data(caption_csv, visual_csv)

    if 'video_id' not in df.columns:
        raise ValueError("'video_id' column not found in combined dataframe!")

    video_stats = analyze_video_distribution(df)
    train_videos, val_videos, test_videos = split_videos(video_stats, train_ratio, val_ratio, test_ratio)
    train_df, val_df, test_df = create_splits(df, train_videos, val_videos, test_videos)
    analyze_splits(train_df, val_df, test_df)
    save_splits(train_df, val_df, test_df, output_dir)
    save_statistics(df, train_df, val_df, test_df,
                    train_videos, val_videos, test_videos,
                    caption_csv, visual_csv, output_dir)

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)
    print("\n  Next Steps for Ablation Study:")
    print("  In your fine-tuning script, load train set and filter:")
    print("  - train_df[train_df['source'] != 'visual_grounded']  # Caption only")
    print("  - train_df[train_df['source'] == 'visual_grounded']  # Visual only")
    print("  - train_df  # All (caption + visual)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6.1: Split Q/A pairs into train/val/test at video level"
    )
    parser.add_argument(
        "--caption-csv",
        type=str,
        required=True,
        help="Path to caption-based Q/A CSV (e.g. filtered_caption_qa_pairs.csv)"
    )
    parser.add_argument(
        "--visual-csv",
        type=str,
        required=True,
        help="Path to visual-grounded Q/A CSV (e.g. filtered_visual_qa_pairs.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for train/val/test CSVs and stats"
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio",  type=float, default=TEST_RATIO)

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Split ratios must sum to 1.0")

    split_dataset(
        caption_csv=args.caption_csv,
        visual_csv=args.visual_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )