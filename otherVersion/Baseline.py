import os
import cv2
import easyocr
import numpy as np
import re

# ===============================
# 📁 CONFIGURATION (Edit here)
# ===============================
INPUT_DIR = "ORIGINAL_IMAGE"     # folder containing images
OUTPUT_DIR = "ocr_output"       # folder where OCR text files will be saved
USE_GPU = True                  # set True if GPU is available and you want to use it

# ===============================
# 1. Preprocess image (Improved with multiple steps)
# ===============================
def preprocess_image(img_path):
    """
    Tiền xử lý ảnh đơn giản nhất cho EasyOCR:
    - Chỉ đọc ảnh grayscale và resize khi cần
    - Đảm bảo kích thước text nằm trong vùng tối ưu của EasyOCR (10-50px height)
    """
    # Đọc ảnh grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Điều chỉnh kích thước nếu cần
    min_height = 600  # Để text nhỏ nhất ~12px
    max_height = 1200 # Giới hạn để tối ưu thời gian xử lý
    
    height = image.shape[0]
    if height < min_height:
        scale = min_height / height
        image = cv2.resize(image, None, fx=scale, fy=scale, 
                         interpolation=cv2.INTER_CUBIC)
    elif height > max_height:
        scale = max_height / height
        image = cv2.resize(image, None, fx=scale, fy=scale, 
                         interpolation=cv2.INTER_AREA)
    
    return image

# ===============================
# 2. Run OCR on a single image
# ===============================
def run_ocr(img_path, reader):
    """
    Chạy OCR trên một ảnh với EasyOCR.
    Chỉ trả về text, không có hậu xử lý.
    """
    processed_img = preprocess_image(img_path)
    results = reader.readtext(processed_img, detail=1)
    
    texts = []
    for result in results:
        if len(result) >= 2:  # Lấy text từ kết quả
            texts.append(result[1])
    return texts

# ===============================
# 3. Process all images recursively
# ===============================
def process_folder(input_dir, output_dir, reader):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(root, file)

                # Preserve folder structure
                rel_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, rel_path)
                os.makedirs(output_folder, exist_ok=True)

                # Output file name: {original_name}.txt.text
                base_name, _ = os.path.splitext(file)
                output_path = os.path.join(output_folder, f"{base_name}.txt.text")

                print(f"🔍 OCR: {input_path} → {output_path}")

                try:
                    texts = run_ocr(input_path, reader)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(texts))
                except Exception as e:
                    print(f"❌ Failed to process {input_path}: {e}")

# ===============================
# 4. Main execution
# ===============================
if __name__ == "__main__":
    print("🚀 Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['ko', 'en'], gpu=USE_GPU)

    print(f"📂 Input folder:  {INPUT_DIR}")
    print(f"📁 Output folder: {OUTPUT_DIR}")

    # process_folder(INPUT_DIR, OUTPUT_DIR, reader)
    text = run_ocr("ORIGINAL_IMAGE/images_hyecho/TAF20161_00.png", reader)
    output_path = "ocr_result.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"Kết quả OCR đã lưu tại: {output_path}")

    print("✅ OCR processing complete!")
