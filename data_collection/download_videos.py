#!/usr/bin/env python3
"""
Video Download from YouTube

This script downloads videos from YouTube using yt-dlp, with optional Apptainer 
container support for improved compatibility. It handles age-restricted content
via cookies and maintains a log of download status.

Usage:
    python download_videos.py --input videos.csv --output-dir ./videos
    python download_videos.py --input videos.csv --output-dir ./videos --no-resume --retry-failed
"""

import pandas as pd
from pathlib import Path
import time
import re
import argparse
import sys
import random
import subprocess
from datetime import datetime

# ============================================================
# DEFAULT CONFIG (can be overridden via command-line arguments)
# ============================================================
DEFAULT_OUTPUT_DIR = "./data/videos/raw"
DEFAULT_OUTPUT_CSV = "./data/downloaded_videos.csv"
DEFAULT_SKIPPED_LOG = "./data/skipped_videos.csv"
DEFAULT_COOKIE_FILE = "./cookies.txt"
DEFAULT_APPTAINER_IMAGE = "./containers/yt-dlp-runtime.sif"

# Columns to keep (in specific order)
DOWNLOAD_COLUMNS = [
    'video_id',
    'video_url',
    'video_title',
    'channel_name',
    'video_file',
    'duration_seconds',
    'download_status'
]

# ============================================================
# UTIL
# ============================================================
def clean_url(url: str) -> str:
    """Remove playlist parameters from URL"""
    if pd.isna(url) or url is None or not isinstance(url, str) or url.strip() == "":
        return ""
    
    url = re.sub(r"&list=[^&]*", "", url)
    url = re.sub(r"\?list=[^&]*&", "?", url)
    url = re.sub(r"\?list=[^&]*$", "", url)
    return url

def log_skipped_video(video_info, reason, skipped_log_path):
    """Log skipped videos to a CSV file"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'video_url': video_info.get('video_url', ''),
        'video_title': video_info.get('video_title', ''),
        'video_id': video_info.get('video_id', ''),
        'reason': reason,
        'channel_name': video_info.get('channel_name', ''),
    }
    
    log_df = pd.DataFrame([log_entry])
    
    try:
        if Path(skipped_log_path).exists():
            existing_log = pd.read_csv(skipped_log_path)
            log_df = pd.concat([existing_log, log_df], ignore_index=True)
        log_df.to_csv(skipped_log_path, index=False)
    except Exception as e:
        print(f"  Warning: Could not write to log file: {e}")

def save_clean_csv(df, output_path):
    """Save with specific column order"""
    # Ensure all required columns exist
    for col in DOWNLOAD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Select and reorder columns
    clean_df = df[DOWNLOAD_COLUMNS].copy()
    clean_df.to_csv(output_path, index=False)
    return clean_df

# ============================================================
# MAIN
# ============================================================
def download_videos(input_csv, output_csv, video_dir, cookie_file=None, 
                   apptainer_image=None, resume=True, retry_failed=False, start_index=1):
    """
    Download videos from YouTube URLs listed in a CSV file.
    
    Args:
        input_csv: Path to input CSV file with video URLs
        output_csv: Path to save download results
        video_dir: Directory to save downloaded videos
        cookie_file: Optional path to cookies.txt for age-restricted videos
        apptainer_image: Optional path to Apptainer/Singularity image with yt-dlp
        resume: Skip already downloaded videos (default: True)
        retry_failed: Retry previously failed downloads (default: False)
        start_index: 1-based index to start from (default: 1)
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine skipped log path from output_csv
    output_path = Path(output_csv)
    skipped_log = output_path.parent / f"{output_path.stem}_skipped.csv"
    
    # Check for Apptainer
    use_apptainer = apptainer_image and Path(apptainer_image).exists()
    
    # Check for cookies
    use_cookies = cookie_file and Path(cookie_file).exists()

    print("=" * 70)
    print("VIDEO DOWNLOAD FROM YOUTUBE")
    print(f"Resume mode: {resume} | Retry failed: {retry_failed} | Start index: {start_index}")
    
    if use_apptainer:
        print(f"Using Apptainer: {apptainer_image}")
        print("  JavaScript runtime included (best compatibility)")
    else:
        print("Using system yt-dlp")
        if apptainer_image:
            print(f"  Note: Apptainer image not found: {apptainer_image}")
    
    if use_cookies:
        print(f"Cookies found: {cookie_file}")
        print("  Age-restricted videos will be downloaded")
    else:
        print("No cookies file")
        if cookie_file:
            print(f"  Note: Cookie file not found: {cookie_file}")
        print("  Age-restricted videos will be skipped")
    
    print(f"Skipped videos log: {skipped_log}")
    print("=" * 70)

    if not Path(input_csv).exists():
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)
    
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} videos\n")

    # Ensure required columns exist
    for col in DOWNLOAD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reconcile existing files
    print("Checking for already downloaded files...")
    fixed = 0
    for idx, row in df.iterrows():
        expected = row.get("video_file")
        url_val = row.get("video_url")
        
        if pd.isna(url_val) or not url_val:
            continue
            
        if (not expected or pd.isna(expected)):
            vid = None
            try:
                vid = clean_url(url_val).split("v=")[-1][:11]
            except Exception:
                vid = None
            if vid:
                matches = list(video_dir.glob(f"{vid}.*"))
                if matches:
                    filepath = matches[0]
                    df.at[idx, "video_file"] = filepath.name
                    df.at[idx, "video_id"] = vid
                    df.at[idx, "download_status"] = "success"
                    
                    try:
                        duration_result = subprocess.run(
                            ['ffprobe', '-v', 'error', '-show_entries', 
                            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                            str(filepath)],
                            capture_output=True,
                            text=True, 
                            timeout=30
                        )
                        if duration_result.returncode == 0:
                            duration = float(duration_result.stdout.strip())
                            df.at[idx, "duration_seconds"] = duration
                    except:
                        pass
                    
                    fixed += 1
        
        # Verify file still exists and get duration if missing
        if df.at[idx, "download_status"] == "success":
            vf = df.at[idx, "video_file"]
            if not vf or not (video_dir / vf).exists():
                df.at[idx, "download_status"] = None
            else:
                if pd.isna(df.at[idx, "duration_seconds"]):
                    try:
                        filepath = video_dir / vf
                        duration_result = subprocess.run(
                            ['ffprobe', '-v', 'error', '-show_entries', 
                            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                            str(filepath)],
                            capture_output=True,
                            text=True, 
                            timeout=30
                        )
                        if duration_result.returncode == 0:
                            duration = float(duration_result.stdout.strip())
                            df.at[idx, "duration_seconds"] = duration
                    except:
                        pass

    if fixed:
        print(f"Reconciled {fixed} files from disk\n")

    save_clean_csv(df, output_csv)
    print(f"Updated CSV with duration info\n")


    success = 0
    fail = 0
    skipped = 0
    no_url = 0
    total = len(df)
    
    # Iterate through videos
    for idx in range(start_index - 1, total):
        row = df.iloc[idx]
        title = str(row.get("video_title", ""))[:65]
        
        # Check if URL exists
        url_val = row.get("video_url")
        if pd.isna(url_val) or not url_val or str(url_val).strip() == "":
            print(f"[{idx+1}/{total}] ⚪ Skipping: No URL for '{title}'")
            no_url += 1
            continue
        
        url = clean_url(url_val)
        
        if not url or url.strip() == "":
            print(f"[{idx+1}/{total}] ⚪ Skipping: Invalid URL for '{title}'")
            no_url += 1
            continue
        
        status_now = df.at[idx, "download_status"]

        # Skip already successful downloads
        if status_now == "success":
            vf = df.at[idx, "video_file"]
            if vf and (video_dir / vf).exists():
                if resume:
                    print(f"[{idx+1}/{total}] Skip (already downloaded): {title}")
                    success += 1
                    continue
            else:
                df.at[idx, "download_status"] = None
                status_now = None

        # Handle previously failed/unavailable
        if status_now in ("failed", "unavailable"):
            if retry_failed:
                print(f"[{idx+1}/{total}] Retrying previously failed: {title}")
                df.at[idx, "download_status"] = None
                status_now = None
            else:
                print(f"[{idx+1}/{total}] Skipping previously failed: {title}")
                fail += 1
                continue

        # Handle age-restricted
        if status_now == "age_restricted_skipped":
            if retry_failed:
                print(f"[{idx+1}/{total}] Retrying age-restricted: {title}")
                df.at[idx, "download_status"] = None
                status_now = None
            else:
                print(f"[{idx+1}/{total}] Skipping age-restricted: {title}")
                skipped += 1
                continue

        # Download the video
        print(f"\n[{idx+1}/{total}] Downloading: {title}...")

        try:
            # Build yt-dlp command
            if use_apptainer:
                cmd = [
                    'apptainer', 'exec',
                    '--bind', f'{video_dir.parent.parent}:{video_dir.parent.parent}:rw',
                    apptainer_image, 'yt-dlp',
                    '--no-playlist',
                    '--merge-output-format', 'mp4',
                    '-o', str(video_dir / '%(id)s.%(ext)s'),
                ]
            else:
                cmd = [
                    'yt-dlp',
                    '--remote-components', 'ejs:github',
                    '--no-playlist',
                    '--merge-output-format', 'mp4',
                    '-o', str(video_dir / '%(id)s.%(ext)s'),
                ]
            
            if use_cookies:
                cmd.extend(['--cookies', cookie_file])
            
            cmd.append(url)
            
            # Execute download - NO capture_output!
            result = subprocess.run(
                cmd, 
                text=True, 
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed with return code {result.returncode}")
            
            # Extract video_id from URL
            video_id = clean_url(url).split("v=")[-1][:11]
            
            # Find the downloaded file
            matches = list(video_dir.glob(f"{video_id}.*"))
            if not matches:
                raise RuntimeError("Downloaded file not found")
            
            filename = matches[0].name
            filepath = matches[0]
            
            # Update dataframe
            df.at[idx, "download_status"] = "success"
            df.at[idx, "video_file"] = filename
            df.at[idx, "video_id"] = video_id
            
            # Get duration using ffprobe
            try:
                duration_result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 
                     'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                     str(filepath)],
                    capture_output=True,
                    text=True, 
                    timeout=30
                )
                if duration_result.returncode == 0:
                    duration = float(duration_result.stdout.strip())
                    df.at[idx, "duration_seconds"] = duration
                else:
                    df.at[idx, "duration_seconds"] = None
            except:
                df.at[idx, "duration_seconds"] = None

            size_mb = filepath.stat().st_size / (1024 ** 2)
            print(f"  Success: {filename} ({size_mb:.1f} MB)")
            success += 1

        except subprocess.TimeoutExpired:
            print(f"  Download timeout (>5 minutes)")
            df.at[idx, "download_status"] = "failed"
            fail += 1
            
        except Exception as e:
            err = str(e)
            print(f"  Download failed: {err[:200]}")
            
            # Check for age restriction
            if any(keyword in err.lower() for keyword in ["sign in", "age", "login_required", "confirm your age"]):
                if use_cookies:
                    print(f"  Age-restricted but cookies failed")
                    df.at[idx, "download_status"] = "failed"
                    fail += 1
                else:
                    print(f"  Age-restricted video - SKIPPING (no cookies)")
                    df.at[idx, "download_status"] = "age_restricted_skipped"
                    
                    video_info = {
                        'video_url': url,
                        'video_title': row.get('video_title', ''),
                        'video_id': row.get('video_id', ''),
                        'channel_name': row.get('channel_name', ''),
                    }
                    log_skipped_video(video_info, 'age_restricted', skipped_log)
                    skipped += 1
            else:
                df.at[idx, "download_status"] = "failed"
                fail += 1

        # Periodic save with clean columns
        if (idx + 1) % 5 == 0:
            save_clean_csv(df, output_csv)
            print(f"Saved (Success: {success}, Failed: {fail}, Skipped: {skipped}, No URL: {no_url})")
        
        # Rate limiting
        time.sleep(random.uniform(1.0, 2.5))

    # Final save with clean columns
    clean_df = save_clean_csv(df, output_csv)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    status = clean_df["download_status"].value_counts().to_dict()
    print(f"\nTotal: {len(clean_df)}")
    print(f"Success: {status.get('success', 0)}")
    print(f"Failed: {status.get('failed', 0)}")
    print(f"Age-restricted (skipped): {status.get('age_restricted_skipped', 0)}")
    print(f"No URL: {no_url}")
    none_count = clean_df["download_status"].isna().sum()
    if none_count > 0:
        print(f"Not attempted: {none_count}")
    
    downloaded_files = list(video_dir.glob("*.*"))
    if downloaded_files:
        total_gb = sum(f.stat().st_size for f in downloaded_files) / (1024 ** 3)
        print(f"\nDownloaded: {total_gb:.2f} GB ({len(downloaded_files)} files)")
    print(f"Location: {video_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"  Columns: {list(clean_df.columns)}")
    print(f"Skipped log: {skipped_log}")

# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube using yt-dlp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python download_videos.py --input videos.csv --output-dir ./videos
  python download_videos.py --input videos.csv --output-dir ./videos --cookies cookies.txt
  python download_videos.py --input videos.csv --output-dir ./videos --no-resume --retry-failed
        """)
    
    parser.add_argument("--input", required=True,
                        help="Input CSV file with video URLs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save downloaded videos (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV file with download results (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--cookies", default=None,
                        help="Path to cookies.txt file for age-restricted videos")
    parser.add_argument("--apptainer", default=None,
                        help="Path to Apptainer/Singularity image with yt-dlp")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not skip already downloaded videos")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry previously failed downloads")
    parser.add_argument("--start", dest="start_index", type=int, default=1,
                        help="1-based index to start from (default: 1)")
    
    args = parser.parse_args()
    
    try:
        download_videos(
            input_csv=args.input,
            output_csv=args.output_csv,
            video_dir=args.output_dir,
            cookie_file=args.cookies,
            apptainer_image=args.apptainer,
            resume=args.resume,
            retry_failed=args.retry_failed,
            start_index=args.start_index
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving state...")
        try:
            df = pd.read_csv(args.input)
            save_clean_csv(df, args.output_csv)
            print("Progress saved")
        except Exception as e:
            print(f"Could not save: {e}")
        sys.exit(1)