#!/usr/bin/env python3
"""
Caption Extraction from Videos

Extracts captions from video chunks using YouTube subtitles or Whisper transcription.
YouTube subtitles are preferred when available, with Whisper as fallback.

Usage:
    python extract_captions.py --input segments.csv --output captions.csv --video-dir ./videos
    python extract_captions.py --input segments.csv --output captions.csv --video-dir ./videos --whisper-model small
"""

import pandas as pd
from pathlib import Path
import subprocess
import re
from tqdm import tqdm
import whisper
import tempfile
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/segmented_videos.csv"
DEFAULT_OUTPUT_CSV = "./data/captions.csv"
DEFAULT_VIDEO_DIR = "./data/videos/raw"
DEFAULT_WHISPER_MODEL = "base"

# ============================================================
# YOUTUBE SUBTITLE EXTRACTION
# ============================================================
def download_youtube_subtitles(video_url, video_id):
    """
    Download YouTube subtitles with timestamps
    Returns: list of (start_time, end_time, text) tuples or None
    """
    try:
        temp_dir = Path("/tmp/tccc_subtitles")
        temp_dir.mkdir(exist_ok=True)
        
        cmd = [
            'yt-dlp',
            '--skip-download',
            '--write-auto-sub',
            '--sub-lang', 'en',
            '--sub-format', 'vtt',
            '--output', str(temp_dir / video_id),
            video_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        subtitle_file = temp_dir / f'{video_id}.en.vtt'
        if not subtitle_file.exists():
            return None
        
        subtitles = parse_vtt_file(subtitle_file)
        subtitle_file.unlink()
        
        return subtitles
        
    except Exception:
        return None


def parse_vtt_file(vtt_file):
    """
    Parse VTT file and extract timestamps
    Returns: list of (start_seconds, end_seconds, text)
    """
    with open(vtt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    subtitles = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp line (e.g., 00:00:05.000 --> 00:00:08.000)
        if '-->' in line:
            match = re.match(
                r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})',
                line
            )
            if match:
                # Convert to seconds
                start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
                
                start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                
                # Get text from next non-empty lines
                i += 1
                text_parts = []
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text = lines[i].strip()
                    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                    if text:
                        text_parts.append(text)
                    i += 1
                
                if text_parts:
                    text = ' '.join(text_parts)
                    subtitles.append((start_seconds, end_seconds, text))
        
        i += 1
    
    return subtitles


def extract_caption_for_chunk(subtitles, start_time, end_time):
    """Extract caption text for a specific time range"""
    if not subtitles:
        return None
    
    chunk_texts = []
    
    for sub_start, sub_end, text in subtitles:
        # Check if subtitle overlaps with chunk time range
        if sub_end >= start_time and sub_start <= end_time:
            chunk_texts.append(text)
    
    if not chunk_texts:
        return None
    
    # Join and clean
    caption = ' '.join(chunk_texts)
    caption = re.sub(r'\s+', ' ', caption).strip()
    caption = re.sub(r'\[.*?\]', '', caption)  # Remove [music], [applause], etc.
    caption = re.sub(r'\(.*?\)', '', caption)  # Remove (inaudible), etc.
    
    return caption if len(caption) > 50 else None


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================
def extract_audio_segment(video_path, start_time, end_time, output_audio_path):
    """Extract audio segment using ffmpeg"""
    try:
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz for Whisper
            '-ac', '1',  # Mono
            '-loglevel', 'error',
            str(output_audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return False
        
        # Check if file was created
        if Path(output_audio_path).exists() and Path(output_audio_path).stat().st_size > 1000:
            return True
        
        return False
        
    except Exception:
        return False


def transcribe_chunk_with_whisper(video_path, start_time, end_time, model_name="base"):
    """Transcribe a specific chunk using Whisper"""
    audio_path = None
    
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name
        
        # Extract audio segment
        success = extract_audio_segment(video_path, start_time, end_time, audio_path)
        
        if not success:
            return None
        
        # Load Whisper model (cached)
        if not hasattr(transcribe_chunk_with_whisper, 'model'):
            transcribe_chunk_with_whisper.model = whisper.load_model(model_name)
        
        model = transcribe_chunk_with_whisper.model
        
        # Transcribe
        result = model.transcribe(audio_path, fp16=False)
        caption = result["text"].strip()
        
        return caption if len(caption) > 50 else None
        
    except Exception:
        return None
        
    finally:
        if audio_path and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except:
                pass


# ============================================================
# MAIN
# ============================================================
def extract_captions(input_csv, output_csv, video_dir, use_whisper=True, whisper_model="base"):
    """
    Extract captions from video chunks.
    
    Args:
        input_csv: Path to CSV with segmented videos
        output_csv: Path to save caption results
        video_dir: Directory containing video files
        use_whisper: Whether to use Whisper for transcription fallback
        whisper_model: Whisper model size (tiny, base, small, medium, large)
    """
    video_dir = Path(video_dir)
    
    print("=" * 70)
    print("CAPTION EXTRACTION")
    print("Extracts captions using YouTube subtitles or Whisper transcription")
    print("=" * 70)
    
    # Load segmented videos
    if not Path(input_csv).exists():
        print(f"ERROR: Input file not found: {input_csv}")
        print("Please run video segmentation first")
        return
    
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} video chunks")
    
    # Add caption columns
    df['caption_text'] = None
    df['caption_source'] = None
    df['caption_length'] = 0
    
    # Statistics
    stats = {
        'youtube': 0,
        'whisper': 0,
        'failed': 0
    }
    
    # Cache for YouTube subtitles (one download per video)
    subtitle_cache = {}
    
    print("\nExtracting captions...")
    
    # Process each chunk
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        video_id = row['video_id']
        video_url = row['video_url']
        video_file = row['video_file']
        video_path = video_dir / video_file
        start_time = row['start_time']
        end_time = row['end_time']
        
        if not video_path.exists():
            stats['failed'] += 1
            continue
        
        caption = None
        source = None
        
        # Try YouTube subtitles first (cached per video)
        if video_id not in subtitle_cache:
            subtitle_cache[video_id] = download_youtube_subtitles(video_url, video_id)
        
        subtitles = subtitle_cache[video_id]
        
        if subtitles:
            caption = extract_caption_for_chunk(subtitles, start_time, end_time)
            if caption:
                source = 'youtube'
                stats['youtube'] += 1
        
        # Fallback to Whisper
        if not caption and use_whisper:
            caption = transcribe_chunk_with_whisper(video_path, start_time, end_time, whisper_model)
            if caption:
                source = 'whisper'
                stats['whisper'] += 1
        
        if not caption:
            stats['failed'] += 1
            continue
        
        # Store results
        df.at[idx, 'caption_text'] = caption
        df.at[idx, 'caption_source'] = source
        df.at[idx, 'caption_length'] = len(caption)
    
    # Save
    df.to_csv(output_csv, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("CAPTION EXTRACTION COMPLETE")
    print("=" * 70)
    
    total_success = stats['youtube'] + stats['whisper']
    success_rate = (total_success / len(df)) * 100 if len(df) > 0 else 0
    
    print(f"\nTotal chunks processed: {len(df)}")
    print(f"  YouTube subtitles: {stats['youtube']}")
    print(f"  Whisper transcription: {stats['whisper']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Show examples
    print(f"\nCaption Examples:")
    examples = df[df['caption_text'].notna()].head(3)
    for _, row in examples.iterrows():
        print(f"\n{row['chunk_id']} ({row['caption_source']}):")
        print(f"  {row['caption_text'][:120]}...")
    
    print(f"\nOutput: {output_csv}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract captions from video chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python extract_captions.py --input segments.csv --output captions.csv --video-dir ./videos
  python extract_captions.py --input segments.csv --output captions.csv --video-dir ./videos --whisper-model small
        """)
    
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV,
                        help=f"Input CSV file with segmented videos (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV file with captions (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR,
                        help=f"Directory containing video files (default: {DEFAULT_VIDEO_DIR})")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Skip Whisper transcription (YouTube subtitles only)")
    
    args = parser.parse_args()
    
    extract_captions(
        input_csv=args.input,
        output_csv=args.output,
        video_dir=args.video_dir,
        use_whisper=not args.no_whisper,
        whisper_model=args.whisper_model
    )