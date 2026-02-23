#!/usr/bin/env python3
"""
Download and Trim Test Set Videos

Reads from a CSV file containing video URLs with timestamp ranges,
downloads full videos, then trims to specified segments.

CSV format: rows with empty Video URL/Title inherit from the previous row
(e.g., same video with multiple timestamp segments).

Usage:
    python download_testset.py --input test_set.csv --output-dir ./test_videos
"""

import csv
import re
import sys
import subprocess
import argparse
from pathlib import Path

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_OUTPUT_DIR = "./data/videos/test_set"
DEFAULT_RAW_SUBDIR = "raw"
DEFAULT_CLIP_SUBDIR = "clips"
DEFAULT_COOKIE_FILE = "./cookies.txt"
DEFAULT_APPTAINER_IMAGE = "./containers/yt-dlp-runtime.sif"

# ============================================================
# UTIL
# ============================================================
def clean_url(url: str) -> str:
    """Strip playlist/offset params; ensure https:// prefix."""
    url = url.strip()
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    url = re.sub(r"&list=[^&]*",   "", url)
    url = re.sub(r"\?list=[^&]*&", "?", url)
    url = re.sub(r"\?list=[^&]*$", "", url)
    url = re.sub(r"[&?]t=[^&]*",   "", url)
    url = re.sub(r"&rco=[^&]*",    "", url)
    return url

def extract_video_id(url: str) -> str:
    """Extract 11-char YouTube video ID from URL."""
    url = clean_url(url)
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")

def ts_to_seconds(ts: str) -> float:
    """'H:MM:SS' or 'M:SS' -> float seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]

def raw_path_for(video_id: str, raw_dir: Path):
    """Return existing raw file path for video_id, or None."""
    matches = list(raw_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None

# ============================================================
# LOAD CSV  (handle multi-row / continuation rows)
# ============================================================
def load_clips(csv_path: str) -> list[dict]:
    """
    Parse CSV into a flat list of clip dicts.

    Rows with an empty 'Video URL' column are treated as additional
    timestamp segments for the previous video (e.g., Kherson clip1/clip2).
    clip_index is incremented per video_id.
    """
    clips = []
    prev  = {}   # last seen full-row data

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title   = row.get("Video Title",    "").strip()
            url_raw = row.get("Video URL",       "").strip()
            start   = row.get("Start timestamp", "").strip()
            end     = row.get("End Timestamp",   "").strip()
            cat     = row.get("Category",        "").strip()
            typ     = row.get("Type",            "").strip()

            # Skip rows with no timestamp info at all
            if not start or not end:
                continue

            # Continuation row: empty URL -> inherit previous video info
            if not url_raw:
                if not prev:
                    print(f"  Warning: Skipping continuation row (no previous row): start={start}")
                    continue
                url_raw = prev["url_raw"]
                title   = prev["title"]
                cat     = prev["category"]
                typ     = prev["type"]
            else:
                # New video row -> update prev
                prev = {"url_raw": url_raw, "title": title,
                        "category": cat, "type": typ}

            try:
                video_id = extract_video_id(url_raw)
            except ValueError as e:
                print(f"  Warning: {e}")
                continue

            # Auto-increment clip index per video_id
            clip_index = sum(1 for c in clips if c["video_id"] == video_id) + 1
            safe_title = re.sub(r"[^\w\- ]", "", title).strip().replace(" ", "_")[:60]

            clips.append({
                "video_id"  : video_id,
                "url"       : clean_url(url_raw),
                "title"     : safe_title,
                "category"  : cat,
                "type"      : typ,
                "clip_index": clip_index,
                "start"     : start,
                "end"       : end,
            })

    return clips

# ============================================================
# DOWNLOAD
# ============================================================
def download_video(video_id: str, url: str, raw_dir: Path, 
                  cookie_file: str = None, apptainer_image: str = None) -> Path:
    """Download full video to raw_dir. Skip if already present."""
    existing = raw_path_for(video_id, raw_dir)
    if existing:
        print(f"  Already downloaded: {existing.name}")
        return existing

    use_apptainer = apptainer_image and Path(apptainer_image).exists()
    use_cookies   = cookie_file and Path(cookie_file).exists()

    if use_apptainer:
        cmd = [
            "apptainer", "exec",
            "--bind", f"{raw_dir.parent.parent}:{raw_dir.parent.parent}:rw",
            apptainer_image, "yt-dlp",
        ]
    else:
        cmd = ["yt-dlp"]

    cmd += [
        "--no-playlist",
        "--merge-output-format", "mp4",
        "-o", str(raw_dir / "%(id)s.%(ext)s"),
    ]
    if use_cookies:
        cmd += ["--cookies", cookie_file]
    cmd.append(url)

    print(f"  Downloading {video_id} ...")
    result = subprocess.run(cmd, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (code {result.returncode}) for {url}")

    downloaded = raw_path_for(video_id, raw_dir)
    if not downloaded:
        raise RuntimeError(f"File not found after download for {video_id}")
    print(f"  Saved: {downloaded.name}")
    return downloaded

# ============================================================
# TRIM
# ============================================================
def trim_clip(src: Path, start: str, end: str, out: Path) -> None:
    """ffmpeg: re-encode src segment [start, end] -> out."""
    if out.exists():
        print(f"  Clip already exists: {out.name}")
        return

    duration = ts_to_seconds(end) - ts_to_seconds(start)
    if duration <= 0:
        raise ValueError(f"Invalid timestamps: start={start} end={end}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", str(src),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    print(f"  Trimming {start} -> {end} ({duration:.0f}s) -> {out.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise RuntimeError(f"ffmpeg failed for {out.name}")
    print(f"  Clip saved: {out.name}")

# ============================================================
# MAIN
# ============================================================
def main(input_csv, output_dir, cookie_file=None, apptainer_image=None):
    """
    Process test set: download full videos and trim to specified segments.
    
    Args:
        input_csv: Path to CSV with video URLs and timestamps
        output_dir: Directory for output (will create raw/ and clips/ subdirs)
        cookie_file: Optional path to cookies.txt
        apptainer_image: Optional path to Apptainer image with yt-dlp
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    clip_dir = output_dir / "clips"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOAD AND TRIM TEST SET VIDEOS")
    print(f"Input CSV : {input_csv}")
    print(f"Raw dir   : {raw_dir}")
    print(f"Clip dir  : {clip_dir}")
    print("=" * 70)

    clips = load_clips(input_csv)
    unique_videos = {c["video_id"] for c in clips}
    print(f"\nLoaded {len(clips)} clips from {len(unique_videos)} unique video(s)\n")
    for c in clips:
        print(f"  [{c['video_id']}] clip{c['clip_index']}  "
              f"{c['start']} -> {c['end']}  |  {c['title'][:50]}")
    print()

    results = []
    for i, clip in enumerate(clips, 1):
        vid = clip["video_id"]
        print(f"\n[{i}/{len(clips)}] {clip['title']} | clip{clip['clip_index']}")
        try:
            raw = download_video(vid, clip["url"], raw_dir, cookie_file, apptainer_image)
            out = clip_dir / f"{vid}_clip{clip['clip_index']}.mp4"
            trim_clip(raw, clip["start"], clip["end"], out)
            results.append({**clip, "status": "success", "clip_path": str(out)})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({**clip, "status": "failed", "error": str(e)})

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok   = [r for r in results if r["status"] == "success"]
    fail = [r for r in results if r["status"] == "failed"]
    print(f"Success : {len(ok)}/{len(results)}")
    if fail:
        print(f"Failed  : {len(fail)}")
        for r in fail:
            print(f"   - {r['video_id']}_clip{r['clip_index']}: {r.get('error', '')}")
    print(f"\nClips saved to: {clip_dir}")

    # Write manifest
    manifest_path = clip_dir / "manifest.csv"
    fieldnames = ["video_id", "clip_index", "title", "category", "type",
                  "start", "end", "status", "clip_path"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and trim test set videos from YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python download_testset.py --input test_set.csv --output-dir ./test_videos
  python download_testset.py --input test_set.csv --output-dir ./test_videos --cookies cookies.txt
        """)
    
    parser.add_argument("--input", required=True,
                        help="Input CSV file with video URLs and timestamps")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--cookies", default=None,
                        help="Path to cookies.txt file for age-restricted videos")
    parser.add_argument("--apptainer", default=None,
                        help="Path to Apptainer/Singularity image with yt-dlp")
    
    args = parser.parse_args()
    
    try:
        main(
            input_csv=args.input,
            output_dir=args.output_dir,
            cookie_file=args.cookies,
            apptainer_image=args.apptainer
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)