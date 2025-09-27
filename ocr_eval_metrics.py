import re
import sys
from typing import Optional, Tuple

RESULTS_FILE = "evaluation_results.txt"


def _extract_first_group(pattern: str, text: str) -> Optional[Tuple[int, ...]]:
    m = re.search(pattern, text)
    if not m:
        return None
    groups = tuple(int(x) for x in m.groups())
    return groups


def parse_blocks(content: str):
    # Split by each file block; the header marker precedes the file-relative path
    raw_blocks = content.strip().split("=== Evaluation:")
    for block in raw_blocks:
        if "TOTAL ERROR SCORE" not in block:
            continue

        # Try to get a short file id from the header line if present
        first_line = block.strip().splitlines()[0].strip(" =")
        file_id = first_line if first_line else "(unknown)"

        # Totals and per-file sizes
        total_error = _extract_first_group(r"TOTAL ERROR SCORE:\s*(\d+)", block)
        chars_in_file = _extract_first_group(r"NO\. OF CHARS IN FILE:\s*(\d+)\s*,\s*(\d+)", block)
        lines_in_file = _extract_first_group(r"NO\. OF LINES IN FILE:\s*(\d+)\s*,\s*(\d+)", block)

        # Error breakdown at character level (S, D, I)
        err_breakdown = _extract_first_group(r"ERROR SCORE BREAK DOWN\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", block)

        # Counts triple lines/chars (first value often equals GT - deletions). We'll still use D/I from here if available.
        chars_trip = _extract_first_group(r"NO\. OF CHARS\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", block)
        lines_trip = _extract_first_group(r"NO\. OF LINES\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", block)

        if not (total_error and chars_in_file and lines_in_file and err_breakdown and chars_trip and lines_trip):
            # Skip malformed block
            continue

        yield {
            "file_id": file_id,
            "total_error": total_error[0],
            "gt_chars": chars_in_file[0],
            "ocr_chars": chars_in_file[1],
            "gt_lines": lines_in_file[0],
            "ocr_lines": lines_in_file[1],
            # char-level S, D, I
            "char_S": err_breakdown[0],
            "char_D": chars_trip[1],
            "char_I": chars_trip[2],
            # line-level aligned/matched proxy and D/I
            "line_aligned": lines_trip[0],  # proxy for matched/aligned lines
            "line_D": lines_trip[1],
            "line_I": lines_trip[2],
        }


def safe_div(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_FILE

    with open(results_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_gt_chars = 0
    total_ocr_chars = 0
    total_char_errors = 0

    total_gt_lines = 0
    total_ocr_lines = 0
    total_line_aligned = 0

    # For precision/recall estimates at char-level using S/D/I; substitutions impact correctness but not lengths
    sum_char_S = 0
    sum_char_D = 0
    sum_char_I = 0

    file_count = 0

    for rec in parse_blocks(content):
        file_count += 1
        total_gt_chars += rec["gt_chars"]
        total_ocr_chars += rec["ocr_chars"]
        total_char_errors += rec["total_error"]

        total_gt_lines += rec["gt_lines"]
        total_ocr_lines += rec["ocr_lines"]
        total_line_aligned += rec["line_aligned"]

        sum_char_S += rec["char_S"]
        sum_char_D += rec["char_D"]
        sum_char_I += rec["char_I"]

    # Micro-averaged CER
    cer = safe_div(total_char_errors, total_gt_chars)
    accuracy = 1.0 - cer

    # Char-level precision/recall/F1 (estimated)
    # Using GT = C + S + D and OCR = C + S + I => C_est from GT side
    char_correct_est = max(0, total_gt_chars - sum_char_S - sum_char_D)
    char_recall = safe_div(char_correct_est, total_gt_chars)
    char_precision = safe_div(char_correct_est, total_ocr_chars)
    char_f1 = safe_div(2 * char_precision * char_recall, (char_precision + char_recall)) if (char_precision + char_recall) else 0.0

    # Line-level precision/recall/F1 treating 'aligned' as matched lines
    line_recall = safe_div(total_line_aligned, total_gt_lines)
    line_precision = safe_div(total_line_aligned, total_ocr_lines)
    line_f1 = safe_div(2 * line_precision * line_recall, (line_precision + line_recall)) if (line_precision + line_recall) else 0.0

    # ---- Report ----
    print("📊 Overall OCR Evaluation Summary")
    print("=====================================")
    print(f"Blocks parsed:                {file_count}")
    print("")
    print("Character-level (micro)")
    print(f"  GT chars:                   {total_gt_chars}")
    print(f"  OCR chars:                  {total_ocr_chars}")
    print(f"  Total edit errors:          {total_char_errors}")
    print(f"  CER:                        {cer*100:.2f}%")
    print(f"  Accuracy:                   {accuracy*100:.2f}%")
    print(f"  Precision (est):            {char_precision*100:.2f}%")
    print(f"  Recall (est):               {char_recall*100:.2f}%")
    print(f"  F1 (est):                   {char_f1*100:.2f}%")
    print("")
    print("Line-level (micro)")
    print(f"  GT lines:                   {total_gt_lines}")
    print(f"  OCR lines:                  {total_ocr_lines}")
    print(f"  Aligned/matched lines:      {total_line_aligned}")
    print(f"  Precision:                  {line_precision*100:.2f}%")
    print(f"  Recall:                     {line_recall*100:.2f}%")
    print(f"  F1:                         {line_f1*100:.2f}%")


if __name__ == "__main__":
    main()
