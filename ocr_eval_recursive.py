import os
import sys
import concurrent.futures
import subprocess
from pathlib import Path

# === CONFIG ===
GROUND_TRUTH_DIR = Path("ORIGINAL_TEXT")  # path to your GT folder
OCR_DIR = Path("submission")              # path to your OCR folder
RESULTS_FILE = Path("evaluation_results.txt")  # where to save the results
MAX_WORKERS = 32  # ⬅️ run up to 10 processes at the same time

# === DISCOVER TASKS ===
tasks = []  # (rel_path:str, gt_path:Path, ocr_path:Path)

for gt_path in GROUND_TRUTH_DIR.rglob("*.txt.text"):
    rel_path = gt_path.relative_to(GROUND_TRUTH_DIR)
    ocr_path = OCR_DIR / rel_path
    if not ocr_path.exists():
        print(f"⚠️ Missing OCR output for: {rel_path}")
        continue
    tasks.append((str(rel_path), gt_path, ocr_path))

total = len(tasks)
if total == 0:
    print("❌ No evaluation pairs found. Check your directories or file extensions.")
    sys.exit(1)

print(f"🚀 Starting evaluations for {total} file pair(s) with up to {MAX_WORKERS} parallel workers...")

# === WORKER ===
def run_eval(task):
    rel_path, gt_path, ocr_path = task
    try:
        # Use the current interpreter to avoid PATH issues
        proc = subprocess.run(
            [sys.executable, "ocr_eval_20250903.py", str(gt_path), str(ocr_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",   # keep your encoding preference
            errors="replace"
        )
        header = f"=== Evaluation: {rel_path} ==="
        block = header + "\n" + proc.stdout + ("\n" if not proc.stdout.endswith("\n") else "") + ("=" * 80) + "\n"
        return (rel_path, proc.returncode, block, proc.stderr)
    except Exception as e:
        header = f"=== Evaluation: {rel_path} ==="
        block = header + f"\n[EXCEPTION] {e}\n" + ("=" * 80) + "\n"
        return (rel_path, 1, block, str(e))

# === RUN PARALLEL ===
results_blocks = []
fail_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    future_to_task = {ex.submit(run_eval, t): t for t in tasks}
    done = 0
    for future in concurrent.futures.as_completed(future_to_task):
        rel_path, code, block, stderr = future.result()
        results_blocks.append(block)
        done += 1
        if code != 0:
            fail_count += 1
            # Print a short line; full details are already in the results file block
            print(f"❗ {rel_path} failed (code {code}).")
            if stderr:
                # Show a one-line hint from stderr if available
                first_line = stderr.strip().splitlines()[0] if stderr.strip() else ""
                if first_line:
                    print(f"   ↳ {first_line}")
        else:
            print(f"✅ {rel_path} ({done}/{total})")

# === SAVE ALL RESULTS ===
RESULTS_FILE.write_text("".join(results_blocks), encoding="utf-8")
print("🎉 All evaluations complete!")
print(f"📊 Results saved to: {RESULTS_FILE.resolve()}")
print(f"🔎 Summary: {total - fail_count} succeeded, {fail_count} failed.")
