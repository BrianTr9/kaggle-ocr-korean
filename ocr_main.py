import os
import cv2
import easyocr
import numpy as np

# ===============================
# 📁 CONFIGURATION
# ===============================
INPUT_DIR = "TEST_FOR_PHASE1"
OUTPUT_DIR = "submission"
USE_GPU = True

# ===============================
# 1. PREPROCESSING (Optimized Version)
# ===============================
def _estimate_smallest_text_height_optimized(gray: np.ndarray) -> float | None:
    """Optimized version using NumPy vectorization to avoid explicit loops."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=4)
    if num_labels <= 1: return None
    stats = stats[1:]  # Skip background

    h = stats[:, cv2.CC_STAT_HEIGHT]
    w = stats[:, cv2.CC_STAT_WIDTH]
    area = stats[:, cv2.CC_STAT_AREA]

    mask = (h >= 4) & (h <= 150) & (w >= 3) & (area >= np.maximum(10, h * 2))
    valid_heights = h[mask]

    if valid_heights.size == 0: return None
    
    return float(np.percentile(valid_heights, 10))

def image_stats(gray):
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    bright_ratio = float(np.mean(gray >= 240))  # near-white region ratio
    dark_ratio = float(np.mean(gray <= 15))     # near-black region ratio
    return mean, std, bright_ratio, dark_ratio

def select_alpha_by_stats(gray):
    mean, std, bright_ratio, dark_ratio = image_stats(gray)
    # Very bright or glare: reduce contrast
    if mean > 200 or bright_ratio > 0.02:
        return 0.7
    # Very dark: slightly increase contrast
    if mean < 80 and dark_ratio > 0.02:
        return 1.25
    # Low contrast, mid-tone: slightly increase contrast
    if std < 35 and 80 <= mean <= 170:
        return 1.25
    # Default: slightly reduce for textured backgrounds
    return 0.9

def preprocess_image(image, min_text_px=13, max_scale=2.5):
    """Preprocessing function calling the optimized estimator."""
    h, w = image.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    smallest = _estimate_smallest_text_height_optimized(image)

    if smallest is not None and smallest > 0 and smallest < min_text_px:
        scale = min(max_scale, max(1.0, (min_text_px / smallest)))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    alpha = select_alpha_by_stats(image)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return image

# ===============================
# 2. OCR & CHUNKING
# ===============================
def run_ocr(img_path, reader, chunk_height=1000, overlap=1): # overlap = 1 13 17
    """Processes an image by chunking and preprocessing each chunk."""
    full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if full_image is None:
        print(f"⚠️ Warning: Could not read image at {img_path}. Skipping.")
        return []
    h, _ = full_image.shape
    all_texts = []
    y_start = 0
    
    while y_start < h:
        y_end = min(y_start + chunk_height, h)
        print(f"   - Processing chunk from y={y_start} to y={y_end}")
        
        chunk = preprocess_image(full_image[y_start:y_end, :])

        results = reader.readtext(chunk, paragraph=False, detail=1)
        chunk_texts = [res[1] for res in results]
        all_texts.extend(chunk_texts)
        
        if y_end == h: break
        y_start += (chunk_height - overlap)

    # Simple deduplication
    # all_texts = list(dict.fromkeys(all_texts))
    return all_texts

# ===============================
# 3. FOLDER PROCESSING
# ===============================
def process_folder(input_dir, output_dir, reader):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                
                # Convert images_<name> to texts_<name>
                if rel_path.startswith("images_"):
                    rel_path = rel_path.replace("images_", "texts_", 1)
                
                output_folder = os.path.join(output_dir, rel_path)
                os.makedirs(output_folder, exist_ok=True)
                base_name, _ = os.path.splitext(file) 
                output_path = os.path.join(output_folder, f"{base_name}.txt")

                print(f"🔍 OCR: {input_path} -> {output_path}")
                try:
                    texts = run_ocr(input_path, reader)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(texts))
                except Exception as e:
                    print(f"❌ Failed to process {input_path}: {e}")

# ===============================
# 4. MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    print("🚀 Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['ko', 'en'], gpu=USE_GPU)
    # reader = easyocr.Reader(['ko', 'en'], gpu=USE_GPU, recog_network='korean_g2')
    print(f"📂 Input folder:  {INPUT_DIR}")
    print(f"📁 Output folder: {OUTPUT_DIR}")

    # Uncomment the line below to process the entire folder
    process_folder(INPUT_DIR, OUTPUT_DIR, reader)

    # # Run a single image for testing
    # text = run_ocr("ORIGINAL_IMAGE/images_hyecho/TCA20184_01.jpg", reader)
    # output_path = "ocr_result.txt"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("\n".join(text))
    # print(f"OCR result saved to: {output_path}")

    print("✅ OCR processing complete!")