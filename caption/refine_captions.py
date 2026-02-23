#!/usr/bin/env python3
"""
Caption Refinement with LLM

Refines raw captions using LLM to improve medical terminology, grammar, and clarity.
Removes filler words, fixes incomplete sentences, and ensures professional quality.

Usage:
    python refine_captions.py --input captions.csv --output refined.csv
    python refine_captions.py --input captions.csv --output refined.csv --model Qwen/Qwen2-7B-Instruct
"""

import os
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/captions.csv"
DEFAULT_OUTPUT_CSV = "./data/refined_captions.csv"
DEFAULT_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_NUM_EXAMPLES = 5

# ============================================================
# REFINEMENT PROMPT
# ============================================================
REFINEMENT_SYSTEM = """You are an expert medical editor specializing in TCCC (Tactical Combat Casualty Care) training materials. Your task is to refine raw video captions into clear, professional medical descriptions."""

REFINEMENT_PROMPT_TEMPLATE = """Raw Caption (from video transcription):
{raw_caption}

Video Metadata:
- Title: {video_title}

Task: Refine this raw caption into a clear, professional description suitable for medical training.

Requirements:
1. Remove filler words (um, uh, like, you know)
2. Remove background noise descriptions ([Music], [Applause], [coughs])
3. Fix incomplete sentences and grammar
4. Use proper medical terminology
5. Keep the content factually accurate to what was said
6. Provide a clear, detailed description (4-7 sentences)
7. Focus on the medical procedure, equipment, and key steps

Refined Caption:"""

# ============================================================
# MODEL SETUP
# ============================================================
def setup_model(model_name, cache_dir):
    """Load LLM for caption refinement"""
    # Set up cache directory
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

# ============================================================
# REFINEMENT
# ============================================================
def refine_caption(model, tokenizer, device, raw_caption, metadata):
    """Refine a raw caption using LLM"""
    
    prompt = REFINEMENT_PROMPT_TEMPLATE.format(
        raw_caption=raw_caption,
        video_title=metadata.get('video_title', 'N/A')
    )
    
    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=False
        )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()

# ============================================================
# COMPARISON DISPLAY
# ============================================================
def display_comparisons(df, num_examples=5):
    """Display before/after comparisons"""
    
    print(f"\nRefinement Examples (showing {num_examples}):")
    print("=" * 70)
    
    examples = df[df['refined_caption'].notna()].head(num_examples)
    
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"\nExample {i}: {row['chunk_id']}")
        print(f"Video: {row['video_title'][:60]}...")
        print(f"Time: {row['start_time']:.1f}s - {row['end_time']:.1f}s")
        
        print(f"\n  Raw ({len(row['caption_text'])} chars):")
        print(f"    {row['caption_text'][:150]}...")
        
        print(f"\n  Refined ({len(row['refined_caption'])} chars):")
        print(f"    {row['refined_caption'][:150]}...")
        
        if i < num_examples:
            print("\n" + "-" * 70)

# ============================================================
# MAIN
# ============================================================
def refine_captions(input_csv, output_csv, model_name=DEFAULT_MODEL_NAME,
                   cache_dir=DEFAULT_CACHE_DIR, num_examples=DEFAULT_NUM_EXAMPLES):
    """
    Refine raw captions using LLM.
    
    Args:
        input_csv: Path to CSV with raw captions
        output_csv: Path to save refined captions
        model_name: Hugging Face model name
        cache_dir: Directory for model cache
        num_examples: Number of comparison examples to display
    """
    print("=" * 70)
    print("CAPTION REFINEMENT")
    print("Refines raw captions using LLM for improved quality")
    print("=" * 70)
    
    # Load captions
    if not Path(input_csv).exists():
        print(f"ERROR: Input file not found: {input_csv}")
        print("Please run caption extraction first")
        return
    
    df = pd.read_csv(input_csv)
    
    # Filter chunks with captions
    df_with_captions = df[df['caption_text'].notna()].copy()
    
    print(f"\nLoaded {len(df)} chunks")
    print(f"Found {len(df_with_captions)} chunks with captions to refine")
    
    # Setup model
    model, tokenizer, device = setup_model(model_name, cache_dir)
    
    # Add refined caption column
    df['refined_caption'] = None
    
    # Statistics
    stats = {
        'refined': 0,
        'failed': 0
    }
    
    print("\nRefining captions...")
    
    # Process each chunk with caption
    for idx, row in tqdm(df_with_captions.iterrows(), total=len(df_with_captions), desc="Processing"):
        try:
            metadata = {
                'video_title': row.get('video_title', '')
            }
            
            raw_caption = row['caption_text']
            refined_caption = refine_caption(model, tokenizer, device, raw_caption, metadata)
            
            df.at[idx, 'refined_caption'] = refined_caption
            stats['refined'] += 1
            
        except Exception as e:
            print(f"\nWarning: Error refining {row['chunk_id']}: {e}")
            stats['failed'] += 1
            continue
    
    # Save
    df.to_csv(output_csv, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("CAPTION REFINEMENT COMPLETE")
    print("=" * 70)
    
    print(f"\nTotal chunks: {len(df)}")
    print(f"  Successfully refined: {stats['refined']}")
    print(f"  Failed: {stats['failed']}")
    
    if stats['refined'] > 0:
        avg_len_raw = df_with_captions['caption_text'].str.len().mean()
        avg_len_refined = df[df['refined_caption'].notna()]['refined_caption'].str.len().mean()
        print(f"\nAverage caption length:")
        print(f"  Raw: {avg_len_raw:.0f} chars")
        print(f"  Refined: {avg_len_refined:.0f} chars")
    
    # Display comparisons
    display_comparisons(df, num_examples=num_examples)
    
    print(f"\nOutput: {output_csv}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine raw captions using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python refine_captions.py --input captions.csv --output refined.csv
  python refine_captions.py --input captions.csv --output refined.csv --model Qwen/Qwen2-7B-Instruct
        """)
    
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV,
                        help=f"Input CSV file with raw captions (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV file with refined captions (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help=f"Number of comparison examples to display (default: {DEFAULT_NUM_EXAMPLES})")
    
    args = parser.parse_args()
    
    refine_captions(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model,
        cache_dir=args.cache_dir,
        num_examples=args.num_examples
    )