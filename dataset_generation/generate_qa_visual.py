#!/usr/bin/env python3
"""
Visual Q/A Generation

Generates question-answer pairs from video frames using vision-language models.
Processes video chunks to create visual understanding questions.

Usage:
    python generate_qa_visual.py --input classified.csv --output qa_visual.csv --video-dir ./videos
"""

import os
import sys

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_INPUT_CSV = "./data/classified_captions.csv"
DEFAULT_OUTPUT_CSV = "./data/qa_from_visual.csv"
DEFAULT_VIDEO_DIR = "./data/videos/raw"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_CACHE_DIR = "./cache/huggingface"
DEFAULT_MAX_FRAMES = 64
DEFAULT_TARGET_SIZE = 448
DEFAULT_MAX_NEW_TOKENS = 800
DEFAULT_NUM_QA = 5

# Cache setup
def setup_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================
# PROMPT
# ============================================================
def create_visual_qa_prompt():
    """Create prompt for visual Q/A generation (num_qa fixed to NUM_QA)"""

    return f"""You are a TCCC expert instructor analyzing this training video.

TCCC (Tactical Combat Casualty Care) follows the MARCH protocol:
- M: Massive hemorrhage - Life-threatening bleeding control (tourniquets, pressure dressings)
- A: Airway - Airway management in combat conditions
- R: Respiration - Breathing/chest injuries (needle decompression, chest seals)
- C: Circulation - IV access, fluid resuscitation
- H: Hypothermia prevention - Body temperature management

Key principles: Care under fire, tactical field care, resource constraints, multiple casualties.

---

Watch the video carefully and identify visual elements that:
1. Show critical hand positions, body mechanics, or techniques that affect patient safety
2. Demonstrate visual verification steps that confirm correct procedure execution
3. Reveal common errors that are visible and must be avoided
4. Display anatomical landmarks or equipment positioning that impacts medical outcomes
5. Show sequential steps where visual confirmation is essential for success

Generate {NUM_QA} questions that:
[GOOD] Test understanding of WHY specific visual details matter clinically
[GOOD] Help distinguish correct vs incorrect technique through visual observation
[GOOD] Focus on visual cues that directly impact patient survival or treatment outcomes
[GOOD] Can ONLY be answered by actually watching the video demonstration
[GOOD] Connect visual observations to clinical reasoning and TCCC protocols

[AVOID] Trivial questions about colors, object counts, or simple descriptions
[AVOID] DON'T ask "what do you see" - ask "WHY is this technique/positioning critical"
[AVOID] DON'T duplicate information easily learned from audio narration alone

AVOID GENERIC ANSWERS LIKE THESE:

[BAD] "Visual confirmation of the tourniquet's position is essential to ensure 
   correct application and prevent complications."
   
[BAD] "Proper hand positioning is crucial for effective technique execution and 
   patient safety."
   
[BAD] "This technique ensures the procedure is performed correctly, which could 
   lead to better outcomes."

[BAD] "Maintaining proper body mechanics helps prevent injury to both the rescuer 
   and the patient."

INSTEAD, BE SPECIFIC:

[GOOD] "The tourniquet sits 2-3 inches proximal to the wound on bare skin, allowing 
   arterial compression against the humerus bone for maximum hemostatic effect."

[GOOD] "The medic's hands form a C-grip at the cricothyroid membrane, palpating the 
   thyroid and cricoid cartilages to identify the safe incision site."

[GOOD] "The needle enters perpendicular to the chest wall at the second intercostal 
   space, midclavicular line, to access the pleural space while avoiding the 
   subclavian vessels."

Include measurements, landmarks, and specific techniques when visible. 
Write naturally - don't force every answer into the same structure.

Output as JSON:
{{
  "questions": [
    {{
      "question": "Your detailed clinical question here?",
      "answer": "Your detailed answer explaining clinical significance and visual observations",
      "type": "clinical_visual"
    }}
  ]
}}

Generate {NUM_QA} clinically-focused questions now:"""


# ============================================================
# VIDEO LOADING (PyAV)
# ============================================================
def load_video(video_path, start_time, end_time, max_frames, target_size):
    """Load video using PyAV - works for AV1, VP9, H264"""
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
        logger.error(f"PyAV error {Path(str(video_path)).name}: {e}")
        return None


# ============================================================
# THREADING PREFETCH
# ============================================================
def video_loader_thread(tasks, queue, max_frames, target_size):
    for task in tasks:
        frames = load_video(
            task['video_path'],
            task['start_time'],
            task['end_time'],
            max_frames,
            target_size
        )
        queue.put({'frames': frames, 'row': task['row']})
    queue.put(None)


# ============================================================
# MODEL
# ============================================================
def setup_model():
    print("Loading model...")

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    print("Model loaded\n")
    return model, processor


# ============================================================
# Q/A GENERATION (single call)
# ============================================================
def generate_qa(frames, model, processor):
    """Generate NUM_QA Q/A pairs in a single forward pass"""
    try:
        prompt = create_visual_qa_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        video_inputs = [
            c.get("video")
            for msg in messages
            for c in msg["content"]
            if isinstance(c, dict) and c.get("type") == "video"
        ]

        inputs = processor(
            text=[text],
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )

        output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse JSON
        json_objects = []
        i = 0
        while i < len(output_text):
            if output_text[i] == '{':
                json_start = i
                brace_count = 0
                j = i
                while j < len(output_text):
                    if output_text[j] == '{':
                        brace_count += 1
                    elif output_text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_objects.append((json_start, j + 1))
                            i = j + 1
                            break
                    j += 1
                if brace_count != 0:
                    i += 1
            else:
                i += 1

        if not json_objects:
            return []

        json_start, json_end = json_objects[-1]
        qa_data = json.loads(output_text[json_start:json_end])
        return qa_data.get('questions', [])

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return []


# ============================================================
# MAIN
# ============================================================
def main(limit):

    print("=" * 70)
    print("PHASE 5.2: VISUAL QA - OPTIMIZED (num_qa=5 fixed)")
    print(f"num_qa: {NUM_QA} | max_new_tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    df_with_video = df[df['video_file'].notna()].copy()

    if limit:
        df_with_video = df_with_video.head(limit)
        print(f"\nProcessing only {limit} chunks")

    print(f"Total chunks: {len(df_with_video)}\n")

    model, processor = setup_model()

    # Prepare tasks
    tasks = []
    for idx, row in df_with_video.iterrows():
        video_path = VIDEO_DIR / row['video_file']
        if not video_path.exists():
            continue
        tasks.append({
            'video_path': video_path,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'row': row.to_dict()
        })

    print(f"Prepared {len(tasks)} tasks\n")

    all_results = []
    stats = {'processed': 0, 'failed': 0, 'total': len(tasks)}
    base_dir = Path("/hpc/home/jkim1/workspace/TCCC/data")

    # Threading prefetch (increased queue size)
    queue = Queue(maxsize=8)  # 3 -> 8
    loader = Thread(
        target=video_loader_thread,
        args=(tasks, queue, MAX_FRAMES, TARGET_SIZE),
        daemon=True
    )
    loader.start()

    while True:
        item = queue.get()
        if item is None:
            break

        row = item['row']
        frames = item['frames']

        if frames is None:
            logger.warning(f"Failed to load: {row.get('video_file', '')}")
            stats['failed'] += 1
            continue

        # Single generate call
        qa_pairs = generate_qa(frames, model, processor)

        if not qa_pairs:
            logger.warning(f"No Q/A generated: {row.get('chunk_id', '')}")
            stats['failed'] += 1
            continue

        for qa in qa_pairs:
            all_results.append({
                'chunk_id': row.get('chunk_id', ''),
                'video_id': row.get('video_id', ''),
                'video_title': row.get('video_title', ''),
                'video_file': row.get('video_file', ''),
                'video_url': row.get('video_url', ''),
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'chunk_duration': row['end_time'] - row['start_time'],
                'march_category': row.get('march_category', 'N/A'),
                'care_phase': row.get('care_phase', 'N/A'),
                'skill_level': row.get('skill_level', 'N/A'),
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': qa.get('type', 'clinical_visual'),
                'source': 'visual_grounded'
            })

        stats['processed'] += 1
        done = stats['processed'] + stats['failed']
        print(f"[{done}/{stats['total']}] Processed: {stats['processed']} | Failed: {stats['failed']}")

        # empty_cache + checkpoint: every 50 samples
        if stats['processed'] % 50 == 0:
            torch.cuda.empty_cache()
            ckpt_path = base_dir / f"phase5.2_visual_qa_{NUM_QA}_72B_checkpoint.csv"
            pd.DataFrame(all_results).to_csv(ckpt_path, index=False)
            logger.info(f"Checkpoint saved: {stats['processed']} processed")

    # Final save
    output_path = base_dir / f"phase5.2_visual_qa_{NUM_QA}_72B.csv"
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Output:    {output_path}")
    print(f"Rows:      {len(df_out)}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed:    {stats['failed']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.2: Visual QA Generation (num_qa=5 fixed)"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of chunks to process (for testing)")
    args = parser.parse_args()

    main(args.limit)