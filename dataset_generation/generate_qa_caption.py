#!/usr/bin/env python3
"""
Q/A Generation from Captions

Generates educational question-answer pairs from refined captions using LLM.
Creates diverse questions covering procedures, equipment, timing, protocols, and scenarios.

Usage:
    python generate_qa_caption.py --input classified.csv --output qa_caption.csv
"""

import os
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
import argparse

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/classified_captions.csv"
DEFAULT_OUTPUT_CSV = "./data/qa_from_captions.csv"
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-Instruct"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_BATCH_SIZE = 8  # Number of chunks to process at once

# TCCC-specific question types
QUESTION_TYPES = [
    "procedural",      # How to perform procedures
    "equipment",       # Equipment requirements
    "scenario",        # Scenario-based decisions
    "march_protocol",  # MARCH protocol application
    "sequence",        # Step sequencing
    "timing",          # When to perform actions
    "recognition"      # Condition/injury recognition
]

# ============================================================
# Q/A GENERATION PROMPT
# ============================================================
QA_SYSTEM_PROMPT = """You are an expert in Tactical Combat Casualty Care (TCCC) education. Your task is to generate high-quality educational question-answer pairs based on combat medical training content."""

def create_qa_prompt(row, num_qa):
    """Create prompt for Q/A generation (caption-based)"""
    return f"""You are a TCCC expert instructor creating high-quality training questions from a video transcript.

TCCC (Tactical Combat Casualty Care) follows the MARCH protocol:
- M: Massive hemorrhage - Life-threatening bleeding control (tourniquets, pressure dressings)
- A: Airway - Airway management in combat conditions  
- R: Respiration - Breathing/chest injuries (needle decompression, chest seals)
- C: Circulation - IV access, fluid resuscitation
- H: Hypothermia prevention - Body temperature management

Key principles: Care under fire, tactical field care, resource constraints, multiple casualties.

---

**Video Segment Information:**
- MARCH Category: {row.get('march_category', 'N/A')}
- Care Phase: {row.get('care_phase', 'N/A')}
- Skill Level: {row.get('skill_level', 'N/A')}
- Content Type: {row.get('content_type', 'N/A')}

**Refined Caption (Transcript):**
{row['refined_caption']}

---

Generate {num_qa} question-answer pairs that cover different aspects: procedures, equipment, timing, protocols, and applied scenarios.

AVOID GENERIC ANSWERS LIKE THESE:
[BAD] "It is important to apply the tourniquet correctly to prevent complications."
[BAD] "Proper technique ensures the procedure is performed safely and effectively."
[BAD] "This step is critical for patient survival and should not be skipped."
[BAD] "Always follow TCCC protocols when treating a casualty in the field."

INSTEAD, BE SPECIFIC TO THE CAPTION CONTENT:
[GOOD] Cite exact steps, sequences, or conditions mentioned in the transcript
[GOOD] Include specific timing windows (e.g., "within 2 hours of injury"), dosages, or measurements when present
[GOOD] Reference the exact equipment names, anatomical sites, or failure conditions described
[GOOD] Connect procedural steps to their clinical rationale (WHY, not just WHAT)

Additional guidelines:
[GOOD] Ground every answer directly in the caption - do not hallucinate information not present
[GOOD] Range from basic recall to applied scenario questions
[GOOD] Use precise military/medical terminology consistent with the transcript
[GOOD] Vary question types: don't generate all procedural or all scenario questions
[AVOID] Don't ask questions whose answers are not inferable from the provided caption
[AVOID] Don't duplicate the same concept across multiple questions

Write naturally - don't force every answer into the same structure.

**Output Format:**
Return ONLY a valid JSON array:
[
  {{
    "question": "Your specific question here?",
    "answer": "Detailed answer grounded in the caption, explaining clinical rationale where relevant.",
    "question_type": "procedural|equipment|scenario|march_protocol|sequence|timing|recognition"
  }}
]

Generate {num_qa} Q/A pairs now:"""

# ============================================================
# MODEL SETUP
# ============================================================
def setup_model():
    """Load llama-405B model"""
    print(f"Loading model: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    # Padding setup (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Decoder-only models use left padding
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer

# ============================================================
# Q/A GENERATION (batch processing)
# ============================================================
def generate_qa_batch(model, tokenizer, rows, num_qa):
    """Generate Q/A pairs for a batch of chunks"""

    # Generate all prompts in the batch
    texts = []
    for row in rows:
        prompt = create_qa_prompt(row, num_qa)
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # Batch tokenization (left padding to match lengths)
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only generated part (exclude input tokens)
    responses = []
    for i in range(len(texts)):
        input_len = model_inputs.input_ids[i].shape[0]
        gen_ids = generated_ids[i][input_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(response)

    return responses


def parse_qa_response(response):
    """Parse LLM response to extract Q/A pairs"""
    
    try:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None
        
        json_str = response[start_idx:end_idx]
        qa_pairs = json.loads(json_str)
        
        if not isinstance(qa_pairs, list):
            return None
        
        for qa in qa_pairs:
            if not all(k in qa for k in ['question', 'answer', 'question_type']):
                return None
        
        return qa_pairs
    
    except json.JSONDecodeError:
        return None

# ============================================================
# MAIN
# ============================================================
def generate_all_qa(output_csv, num_qa):
    """Generate Q/A pairs with specified number per chunk"""
    
    print("=" * 70)
    print("PHASE 5.1: Q/A GENERATION FROM CAPTIONS")
    print(f"Generating {num_qa} Q/A pairs per chunk")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)
    
    if not Path(INPUT_CSV).exists():
        print(f"ERROR: Phase 4 output not found: {INPUT_CSV}")
        print("Please run Phase 4 first")
        return
    
    df = pd.read_csv(INPUT_CSV)
    df_with_captions = df[df['refined_caption'].notna()].copy()
    
    print(f"\nLoaded {len(df)} chunks")
    print(f"Found {len(df_with_captions)} chunks with refined captions")
    
    model, tokenizer = setup_model()
    
    results = []
    stats = {'processed': 0, 'failed': 0}
    
    print("\nGenerating Q/A pairs...")
    
    # Process in batches
    rows_list = [row for _, row in df_with_captions.iterrows()]
    
    for batch_start in tqdm(range(0, len(rows_list), BATCH_SIZE), desc="Processing batches"):
        batch_rows = rows_list[batch_start: batch_start + BATCH_SIZE]
        
        try:
            responses = generate_qa_batch(model, tokenizer, batch_rows, num_qa)
            
            for row, response in zip(batch_rows, responses):
                qa_pairs = parse_qa_response(response)
                
                if qa_pairs is None:
                    stats['failed'] += 1
                    continue
                
                for qa in qa_pairs:
                    results.append({
                        'chunk_id': row.get('chunk_id', 'unknown'),
                        'video_id': row.get('video_id', 'unknown'),
                        'video_title': row.get('video_title', ''),
                        'march_category': row.get('march_category', 'N/A'),
                        'care_phase': row.get('care_phase', 'N/A'),
                        'skill_level': row.get('skill_level', 'N/A'),
                        'content_type': row.get('content_type', 'N/A'),
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'question_type': qa['question_type'],
                        'source_caption': row['refined_caption'],
                        'source': 'caption'
                    })
                
                stats['processed'] += 1
        
        except Exception as e:
            print(f"\nError processing batch starting at {batch_start}: {e}")
            stats['failed'] += len(batch_rows)
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 70)
    print("PHASE 5.1 COMPLETE")
    print("=" * 70)
    
    print(f"\nTotal chunks with captions: {len(df_with_captions)}")
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\nTotal Q/A pairs generated: {len(results_df)}")
    
    if len(results_df) > 0:
        print(f"Average Q/A per chunk: {len(results_df)/max(stats['processed'],1):.2f}")
        
        print(f"\nQuestion Type Distribution:")
        print(results_df['question_type'].value_counts())
        
        print(f"\nMARCH Category Distribution:")
        print(results_df['march_category'].value_counts())
        
        print(f"\nQ/A Examples:")
        examples = results_df.head(3)
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            print(f"\nExample {i} ({row['question_type']}):")
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer'][:100]}...")
    
    print(f"\nOutput: {output_csv}")
    print(f"  Columns: {list(results_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.1: Generate Q/A pairs from refined captions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--num-qa",
        type=int,
        default=3,
        help="Number of Q/A pairs to generate per chunk (3, 4, 5, 6, 7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of chunks to process in one batch (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Update globals
    MODEL_NAME = args.model
    MAX_NEW_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    
    OUTPUT_CSV = f"/hpc/home/jkim1/workspace/TCCC/data/phase5.1_qa_pairs_{args.num_qa}qa_llama.csv"
    
    print(f"\nOutput will be saved to: {OUTPUT_CSV}\n")
    
    generate_all_qa(OUTPUT_CSV, args.num_qa)