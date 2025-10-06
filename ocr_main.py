import os
import cv2
import easyocr
import numpy as np
import shutil

# ===============================
# 📁 CONFIGURATION
# ===============================
INPUT_DIR = "TEST_FOR_PHASE1"
OUTPUT_DIR = "submission"
USE_GPU = True

# ===============================
# 🎛️ FINE-TUNING PARAMETERS
# ===============================
class OCRConfig:
    # Text Size & Scaling
    MIN_TEXT_PX = 18          # Minimum text height in pixels (increase for larger text datasets)
    MAX_SCALE = 3.5           # Maximum upscaling factor (increase for very small text)
    MAX_DIMENSION = 3000      # Maximum image dimension (balance quality vs speed)
    
    # Chunking Parameters
    CHUNK_HEIGHT = 900        # Height of each processing chunk (adjust for memory/overlap)
    OVERLAP = 1               # Pixel overlap between chunks (increase to prevent text loss)
    
    # Text Detection
    MIN_TEXT_HEIGHT = 8       # Minimum detected text height
    MAX_TEXT_HEIGHT = 300     # Maximum detected text height (increase for large headers)
    MIN_TEXT_WIDTH = 5        # Minimum character width
    MIN_AREA_RATIO = 0.25     # Minimum area ratio for valid text (0.2-0.4 range)
    
    # Thresholds for Contrast Adjustment
    BRIGHT_THRESHOLD = 235    # Glare detection (220-240 range)
    DARK_THRESHOLD = 20       # Shadow detection (15-25 range)
    
    # Contrast Enhancement Factors
    BRIGHT_BG_ALPHA = 0.70    # Alpha for very bright backgrounds (0.6-0.8)
    DARK_BG_ALPHA = 1.50      # Alpha for dark backgrounds (1.3-1.7)
    LOW_CONTRAST_ALPHA = 1.35 # Alpha for low contrast images (1.2-1.5)
    
    # Noise Reduction
    NOISE_STD_THRESHOLD = 60  # Apply noise reduction if std > this (50-80)
    SHARPENING_STD_THRESHOLD = 50  # Apply sharpening if std > this (40-60)

# ===============================
# 1. PREPROCESSING (Optimized Version)
# ===============================
def _validate_image_input(image):
    """Validate and normalize image input for preprocessing functions."""
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check for valid dimensions
    if image.size == 0 or min(image.shape) < 10:
        return None
        
    return image

def _estimate_smallest_text_height_optimized(gray: np.ndarray) -> float | None:
    """
    Optimized version for complex Korean layouts with text on photographic backgrounds.
    Uses adaptive thresholding and multiple detection methods.
    """
    gray = _validate_image_input(gray)
    if gray is None:
        return None
    
    # Multi-method approach for complex backgrounds
    text_heights = []
    
    # Method 1: OTSU thresholding (good for clear text)
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th_otsu) > 127:
        th_otsu = cv2.bitwise_not(th_otsu)
    
    # Method 2: Adaptive thresholding (better for varying backgrounds)
    th_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 2)
    if np.mean(th_adaptive) > 127:
        th_adaptive = cv2.bitwise_not(th_adaptive)
    
    # Analyze both methods
    for th_method in [th_otsu, th_adaptive]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th_method, connectivity=8)
        if num_labels <= 1:
            continue
            
        stats = stats[1:]  # Skip background
        h = stats[:, cv2.CC_STAT_HEIGHT]
        w = stats[:, cv2.CC_STAT_WIDTH]
        area = stats[:, cv2.CC_STAT_AREA]

        # Use configurable parameters for text detection
        min_height, max_height = OCRConfig.MIN_TEXT_HEIGHT, OCRConfig.MAX_TEXT_HEIGHT
        min_width = OCRConfig.MIN_TEXT_WIDTH
        min_area_ratio = OCRConfig.MIN_AREA_RATIO
        
        mask = (
            (h >= min_height) & (h <= max_height) &
            (w >= min_width) & (w <= h * 4) &  # Allow wider Korean characters
            (area >= np.maximum(15, h * w * min_area_ratio)) &
            (h >= w * 0.3)  # Ensure reasonable height-to-width ratio
        )
        
        valid_heights = h[mask]
        if valid_heights.size > 0:
            text_heights.extend(valid_heights)
    
    if len(text_heights) == 0:
        return None
    
    # Use 20th percentile for more stable estimation with varied text sizes
    return float(np.percentile(text_heights, 20))

def image_stats(gray):
    """
    Calculate comprehensive image statistics for preprocessing decisions.
    Returns statistics optimized for Korean text images.
    """
    gray = _validate_image_input(gray)
    if gray is None:
        return 128.0, 50.0, 0.0, 0.0  # Default values
    
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    
    # Use configurable thresholds
    bright_ratio = float(np.mean(gray >= OCRConfig.BRIGHT_THRESHOLD))
    dark_ratio = float(np.mean(gray <= OCRConfig.DARK_THRESHOLD))
    
    return mean, std, bright_ratio, dark_ratio

def select_alpha_by_stats(gray):
    """
    Select optimal alpha (contrast) value for Korean advertisement/tourism images.
    Handles colored text on photographic backgrounds and complex layouts.
    """
    mean, std, bright_ratio, dark_ratio = image_stats(gray)
    
    # Handle bright photographic backgrounds (like sky in the sample image)
    if mean > 190 and bright_ratio > 0.05:
        return OCRConfig.BRIGHT_BG_ALPHA  # Strong contrast reduction for very bright backgrounds
    elif mean > 170 and bright_ratio > 0.02:
        return 0.85  # Moderate reduction for bright backgrounds
    
    # Handle mid-tone photographic backgrounds (common in tourism images)
    if 100 <= mean <= 170:
        if std < 40:  # Low contrast overlaid text
            return 1.25  # Boost contrast for better text separation
        elif std > 70:  # High variation (complex background)
            return 1.10  # Gentle enhancement to maintain balance
    
    # Handle darker backgrounds or shadows
    if mean < 80:
        if dark_ratio > 0.05:
            return OCRConfig.DARK_BG_ALPHA  # Strong enhancement for very dark areas
        else:
            return 1.30  # Moderate enhancement
    elif mean < 120 and std < 35:
        return 1.20  # Boost contrast for dark, flat areas
    
    # Handle high contrast images (mixed bright/dark areas)
    if std > 85:
        return 0.90  # Slight reduction to prevent over-enhancement
    
    # Handle very low contrast (fade/overexposed areas)
    if std < 25:
        return OCRConfig.LOW_CONTRAST_ALPHA  # Strong boost for very flat images
    
    # Default for balanced images
    return 1.0

def preprocess_image(image, min_text_px=None, max_scale=None):
    """
    Comprehensive preprocessing for Korean tourism/advertisement images.
    Handles complex layouts with text on photographic backgrounds.
    
    Args:
        image: Input grayscale image
        min_text_px: Minimum desired text height in pixels (uses config if None)
        max_scale: Maximum upscaling factor (uses config if None)
        
    Returns:
        Preprocessed image optimized for Korean OCR
    """
    # Use config values if not specified
    if min_text_px is None:
        min_text_px = OCRConfig.MIN_TEXT_PX
    if max_scale is None:
        max_scale = OCRConfig.MAX_SCALE
        
    image = _validate_image_input(image)
    if image is None:
        return image
    
    original_shape = image.shape
    h, w = original_shape[:2]
    
    # Step 1: Initial size management for high-quality tourism images
    max_dimension = OCRConfig.MAX_DIMENSION
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"   📏 Resized from {w}x{h} to {new_w}x{new_h} (scale: {scale:.3f})")

    # Step 2: Apply noise reduction for photographic backgrounds
    mean, std, _, _ = image_stats(image)
    if std > OCRConfig.NOISE_STD_THRESHOLD:  # High variation suggests photographic content
        # Gentle bilateral filter to reduce background noise while preserving text edges
        image = cv2.bilateralFilter(image, 5, 80, 80)
        print(f"   🔧 Applied noise reduction for complex background")

    # Step 3: Estimate text size with improved method
    smallest = _estimate_smallest_text_height_optimized(image)
    
    # Step 4: More aggressive upscaling for small Korean text
    if smallest is not None and smallest > 0 and smallest < min_text_px:
        scale = min(max_scale, max(1.0, min_text_px / smallest))
        if scale > 1.05:  # Apply even small improvements
            h_new, w_new = image.shape[:2]
            new_w, new_h = int(w_new * scale), int(h_new * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"   🔍 Upscaled for Korean text: {scale:.2f}x (text height: {smallest:.1f}px → {smallest*scale:.1f}px)")

    # Step 5: Enhanced contrast adjustment for complex layouts
    alpha = select_alpha_by_stats(image)
    if abs(alpha - 1.0) > 0.03:  # Apply even subtle adjustments
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        print(f"   🎨 Contrast adjusted for layout: α={alpha:.2f}")
    
    # Step 6: Sharpening for better text definition on photographic backgrounds
    if std > OCRConfig.SHARPENING_STD_THRESHOLD:  # Only for complex backgrounds
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        # Blend original and sharpened (30% sharpening)
        image = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
        print(f"   ⚡ Applied text sharpening for photographic background")
    
    return image

# ===============================
# 2. OCR & CHUNKING
# ===============================
def run_ocr(img_path, reader, chunk_height=None, overlap=None): # configurable parameters
    """Processes an image by chunking and preprocessing each chunk."""
    # Use config values if not specified
    if chunk_height is None:
        chunk_height = OCRConfig.CHUNK_HEIGHT
    if overlap is None:
        overlap = OCRConfig.OVERLAP
        
    full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if full_image is None:
        print(f"⚠️ Warning: Could not read image at {img_path}. Skipping.")
        return []
    h, _ = full_image.shape
    all_texts = []
    all_confidences = []
    y_start = 0
    
    while y_start < h:
        y_end = min(y_start + chunk_height, h)
        print(f"   - Processing chunk from y={y_start} to y={y_end}")
        
        chunk = preprocess_image(full_image[y_start:y_end, :])

        results = reader.readtext(chunk, paragraph=False, detail=1)
        chunk_texts = [res[1] for res in results if res[2] >= 0.08]
        chunk_confidences = [res[2] for res in results]  # Extract confidence scores
        all_texts.extend(chunk_texts)
        all_confidences.extend(chunk_confidences)
        
        if y_end == h: break
        y_start += (chunk_height - overlap)

    # Calculate and display confidence statistics for this image
    if all_confidences:
        avg_confidence = sum(all_confidences) / len(all_confidences)
        min_confidence = min(all_confidences)
        max_confidence = max(all_confidences)
        print(f"   📊 Confidence stats: Avg={avg_confidence:.3f}, Min={min_confidence:.3f}, Max={max_confidence:.3f} ({len(all_confidences)} detections)")
    else:
        print(f"   📊 No text detected in image")

    # Simple deduplication
    # all_texts = list(dict.fromkeys(all_texts))
    return all_texts

# ===============================
# 3. FOLDER PROCESSING
# ===============================
def clear_submission_folder(output_dir):
    """Clear all files and folders in the submission directory before processing."""
    if os.path.exists(output_dir):
        print(f"🧹 Clearing submission folder: {output_dir}")
        try:
            # Remove all contents in the directory
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print("✅ Submission folder cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing submission folder: {e}")
    else:
        print(f"📁 Submission folder {output_dir} doesn't exist yet, will be created.")

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

    # Clear submission folder before processing
    clear_submission_folder(OUTPUT_DIR)

    # Uncomment the line below to process the entire folder
    process_folder(INPUT_DIR, OUTPUT_DIR, reader)

    # # Run a single image for testing
    # text = run_ocr("ORIGINAL_IMAGE/images_hyecho/TCA20184_01.jpg", reader)
    # output_path = "ocr_result.txt"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("\n".join(text))
    # print(f"OCR result saved to: {output_path}")

    print("✅ OCR processing complete!")