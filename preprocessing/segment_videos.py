#!/usr/bin/env python3
"""
Video Segmentation

Splits long videos into manageable chunks for processing. Short videos are kept
whole, while long videos are split into overlapping segments to ensure complete
coverage of content.

Usage:
    python segment_videos.py --input downloaded.csv --output segments.csv
    python segment_videos.py --input downloaded.csv --output segments.csv --chunk-duration 300
"""

import pandas as pd
from pathlib import Path
import sys
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/downloaded_videos.csv"
DEFAULT_OUTPUT_CSV = "./data/segmented_videos.csv"
DEFAULT_SHORT_THRESHOLD = 480  # 8 minutes - videos shorter than this are kept whole
DEFAULT_CHUNK_DURATION = 360   # 6 minutes - target chunk size for long videos
DEFAULT_OVERLAP_DURATION = 120 # 2 minutes - overlap between consecutive chunks

# ============================================================
# MAIN
# ============================================================
def segment_videos(input_csv, output_csv, short_threshold=DEFAULT_SHORT_THRESHOLD,
                  chunk_duration=DEFAULT_CHUNK_DURATION, overlap_duration=DEFAULT_OVERLAP_DURATION):
    """
    Segment videos into manageable chunks.
    
    Args:
        input_csv: Path to CSV with downloaded videos
        output_csv: Path to save segmented video info
        short_threshold: Videos shorter than this (seconds) are kept whole
        chunk_duration: Target duration for each chunk (seconds)
        overlap_duration: Overlap between consecutive chunks (seconds)
    """
    print("=" * 70)
    print("VIDEO SEGMENTATION")
    print("Splits long videos into chunks for processing")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Short video threshold: {short_threshold}s ({short_threshold/60:.1f} min)")
    print(f"  Chunk duration: {chunk_duration}s ({chunk_duration/60:.1f} min)")
    print(f"  Overlap duration: {overlap_duration}s ({overlap_duration/60:.1f} min)")
    
    # Load downloaded videos
    df = pd.read_csv(input_csv)
    df = df[df['download_status'] == 'success']
    print(f"\nLoaded {len(df)} successfully downloaded videos")
    
    chunks = []
    stats = {
        'single_chunk': 0,
        'multiple_chunks': 0,
        'skipped': 0
    }
    
    # Process each video
    for idx, row in df.iterrows():
        video_id = row['video_id']
        video_file = row['video_file']
        duration = row['duration_seconds']
        
        # Skip if no duration info
        if pd.isna(duration) or duration <= 0:
            print(f"Warning: Skipping {video_file}: No duration info")
            stats['skipped'] += 1
            continue
        
        # Case 1: Short video - keep whole
        if duration <= short_threshold:
            chunks.append({
                'chunk_id': f"{video_id}_chunk_0",
                'video_id': video_id,
                'video_file': video_file,
                'video_url': row['video_url'],
                'video_title': row['video_title'],
                'start_time': 0.0,
                'end_time': duration,
                'chunk_duration': duration,
                'chunk_index': 0,
                'total_chunks': 1
            })
            stats['single_chunk'] += 1
        
        # Case 2: Long video - split into chunks with overlap
        else:
            chunk_index = 0
            start_time = 0.0
            
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                
                chunks.append({
                    'chunk_id': f"{video_id}_chunk_{chunk_index}",
                    'video_id': video_id,
                    'video_file': video_file,
                    'video_url': row['video_url'],
                    'video_title': row['video_title'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'chunk_duration': end_time - start_time,
                    'chunk_index': chunk_index,
                    'total_chunks': -1  # Will update later
                })
                
                # Move to next chunk with overlap
                start_time = end_time - overlap_duration
                chunk_index += 1
                
                if end_time >= duration:
                    break
            
            # Update total_chunks for this video
            video_chunk_count = chunk_index
            for i in range(len(chunks) - video_chunk_count, len(chunks)):
                chunks[i]['total_chunks'] = video_chunk_count
            
            stats['multiple_chunks'] += 1
        
        # Progress update every 50 videos
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{len(df)} videos processed, {len(chunks)} chunks created")
            sys.stdout.flush()
    
    # Create output dataframe
    chunks_df = pd.DataFrame(chunks)
    
    # Save to CSV
    chunks_df.to_csv(output_csv, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SEGMENTATION COMPLETE")
    print("=" * 70)
    
    print(f"\nStatistics:")
    print(f"  Videos processed: {len(df)}")
    print(f"  Single chunk (<={short_threshold/60:.1f} min): {stats['single_chunk']}")
    print(f"  Multiple chunks (>{short_threshold/60:.1f} min): {stats['multiple_chunks']}")
    print(f"  Skipped (no duration): {stats['skipped']}")
    print(f"  Total chunks created: {len(chunks_df)}")
    
    print(f"\nChunk duration distribution:")
    print(chunks_df['chunk_duration'].describe())
    
    print(f"\nSample chunks:")
    print(chunks_df.head(3)[['chunk_id', 'video_title', 'start_time', 'end_time', 'chunk_duration']])
    
    print(f"\nOutput saved: {output_csv}")
    print(f"  Columns: {list(chunks_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment videos into manageable chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python segment_videos.py --input downloaded.csv --output segments.csv
  python segment_videos.py --input downloaded.csv --output segments.csv --chunk-duration 300
        """)
    
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV,
                        help=f"Input CSV file with downloaded videos (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV file with segmented videos (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--short-threshold", type=int, default=DEFAULT_SHORT_THRESHOLD,
                        help=f"Short video threshold in seconds (default: {DEFAULT_SHORT_THRESHOLD})")
    parser.add_argument("--chunk-duration", type=int, default=DEFAULT_CHUNK_DURATION,
                        help=f"Chunk duration in seconds for long videos (default: {DEFAULT_CHUNK_DURATION})")
    parser.add_argument("--overlap-duration", type=int, default=DEFAULT_OVERLAP_DURATION,
                        help=f"Overlap duration in seconds between chunks (default: {DEFAULT_OVERLAP_DURATION})")
    
    args = parser.parse_args()
    
    segment_videos(
        input_csv=args.input,
        output_csv=args.output,
        short_threshold=args.short_threshold,
        chunk_duration=args.chunk_duration,
        overlap_duration=args.overlap_duration
    )