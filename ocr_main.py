"""
Korean OCR Pipeline - Optimized for Stylized Advertisement Images
Handles complex layouts with text on photographic backgrounds
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import easyocr
import numpy as np


# ===============================
# CONFIGURATION
# ===============================
@dataclass
class OCRConfig:
    """Centralized configuration for OCR pipeline"""
    
    # Directories
    input_dir: str = "TEST_FOR_PHASE1"
    output_dir: str = "submission"
    use_gpu: bool = True
    
    # Text Size & Scaling
    min_text_px: int = 18
    max_scale: float = 3.5
    max_dimension: int = 3000
    
    # Chunking
    chunk_height: int = 900
    overlap: int = 1
    
    # Text Detection Thresholds
    min_text_height: int = 8
    max_text_height: int = 300
    min_text_width: int = 5
    min_area_ratio: float = 0.25
    
    # Contrast Thresholds
    bright_threshold: int = 235
    dark_threshold: int = 20
    
    # Contrast Enhancement
    bright_bg_alpha: float = 0.80
    dark_bg_alpha: float = 1.35
    low_contrast_alpha: float = 1.15
    
    # Preprocessing
    noise_std_threshold: int = 60
    sharpening_std_threshold: int = 50
    
    # OCR
    confidence_threshold: float = 0.085
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['ko', 'en']


# ===============================
# IMAGE STATISTICS
# ===============================
class ImageAnalyzer:
    """Analyzes image properties for preprocessing decisions"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Validate and normalize image input"""
        if image is None:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check validity
        if image.size == 0 or min(image.shape) < 10:
            return None
            
        return image
    
    def calculate_stats(self, gray: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate mean, std, bright_ratio, dark_ratio"""
        gray = self.validate_image(gray)
        if gray is None:
            return 128.0, 50.0, 0.0, 0.0
        
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        bright_ratio = float(np.mean(gray >= self.config.bright_threshold))
        dark_ratio = float(np.mean(gray <= self.config.dark_threshold))
        
        return mean, std, bright_ratio, dark_ratio
    
    def estimate_text_height(self, gray: np.ndarray) -> Optional[float]:
        """Estimate smallest text height using multi-method approach"""
        gray = self.validate_image(gray)
        if gray is None:
            return None
        
        text_heights = []
        
        # Method 1: OTSU thresholding
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th_otsu) > 127:
            th_otsu = cv2.bitwise_not(th_otsu)
        
        # Method 2: Adaptive thresholding
        th_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        if np.mean(th_adaptive) > 127:
            th_adaptive = cv2.bitwise_not(th_adaptive)
        
        # Analyze both methods
        for th_method in [th_otsu, th_adaptive]:
            heights = self._extract_text_heights(th_method)
            text_heights.extend(heights)
        
        if not text_heights:
            return None
        
        return float(np.percentile(text_heights, 20))
    
    def _extract_text_heights(self, binary_image: np.ndarray) -> List[float]:
        """Extract valid text heights from binary image"""
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        if num_labels <= 1:
            return []
        
        stats = stats[1:]  # Skip background
        h = stats[:, cv2.CC_STAT_HEIGHT]
        w = stats[:, cv2.CC_STAT_WIDTH]
        area = stats[:, cv2.CC_STAT_AREA]
        
        # Filter valid text components
        mask = (
            (h >= self.config.min_text_height) & 
            (h <= self.config.max_text_height) &
            (w >= self.config.min_text_width) & 
            (w <= h * 4) &
            (area >= np.maximum(15, h * w * self.config.min_area_ratio)) &
            (h >= w * 0.3)
        )
        
        return h[mask].tolist()
    
    def select_contrast_alpha(self, gray: np.ndarray) -> float:
        """Select optimal contrast adjustment factor"""
        mean, std, bright_ratio, dark_ratio = self.calculate_stats(gray)
        
        # Very bright backgrounds
        if mean > 190 and bright_ratio > 0.05:
            return self.config.bright_bg_alpha
        elif mean > 170 and bright_ratio > 0.02:
            return 0.85
        
        # Mid-tone backgrounds
        if 100 <= mean <= 170:
            if std < 40:
                return 1.25
            elif std > 70:
                return 1.10
        
        # Dark backgrounds
        if mean < 80:
            return self.config.dark_bg_alpha if dark_ratio > 0.05 else 1.30
        elif mean < 120 and std < 35:
            return 1.20
        
        # High contrast
        if std > 85:
            return 0.90
        
        # Very low contrast
        if std < 25:
            return self.config.low_contrast_alpha
        
        return 1.0


# ===============================
# IMAGE PREPROCESSOR
# ===============================
class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.analyzer = ImageAnalyzer(config)
    
    def preprocess(self, image: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Apply full preprocessing pipeline"""
        image = self.analyzer.validate_image(image)
        if image is None:
            return image
        
        h, w = image.shape[:2]
        
        # Step 1: Resize if too large
        image = self._resize_if_needed(image, h, w, verbose)
        
        # Step 2: Calculate stats ONCE (used by noise reduction and sharpening)
        mean, std, _, _ = self.analyzer.calculate_stats(image)
        
        # Step 3: Noise reduction
        image = self._apply_noise_reduction(image, std, verbose)
        
        # Step 4: Upscaling for small text
        image = self._upscale_if_needed(image, verbose)
        
        # Step 5: Contrast adjustment
        image = self._adjust_contrast(image, verbose)
        
        # Step 6: Sharpening (reuse std from step 2)
        image = self._apply_sharpening(image, std, verbose)
        
        return image
    
    def _resize_if_needed(self, image: np.ndarray, h: int, w: int, verbose: bool) -> np.ndarray:
        """Resize image if it exceeds maximum dimension"""
        if max(h, w) <= self.config.max_dimension:
            return image
        
        scale = self.config.max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if verbose:
            print(f"   📏 Resized from {w}x{h} to {new_w}x{new_h} (scale: {scale:.3f})")
        
        return image
    
    def _apply_noise_reduction(self, image: np.ndarray, std: float, verbose: bool) -> np.ndarray:
        """Apply bilateral filter for noise reduction"""
        if std > self.config.noise_std_threshold:
            image = cv2.bilateralFilter(image, 5, 80, 80)
            if verbose:
                print(f"   🔧 Applied noise reduction for complex background")
        
        return image
    
    def _upscale_if_needed(self, image: np.ndarray, verbose: bool) -> np.ndarray:
        """Upscale image if text is too small"""
        smallest = self.analyzer.estimate_text_height(image)
        
        if smallest is not None and 0 < smallest < self.config.min_text_px:
            scale = min(self.config.max_scale, max(1.0, self.config.min_text_px / smallest))
            
            if scale > 1.05:
                h, w = image.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                if verbose:
                    print(f"   🔍 Upscaled for Korean text: {scale:.2f}x "
                          f"(text height: {smallest:.1f}px → {smallest*scale:.1f}px)")
        
        return image
    
    def _adjust_contrast(self, image: np.ndarray, verbose: bool) -> np.ndarray:
        """Adjust image contrast based on statistics"""
        alpha = self.analyzer.select_contrast_alpha(image)
        
        if abs(alpha - 1.0) > 0.03:
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            if verbose:
                print(f"   🎨 Contrast adjusted for layout: α={alpha:.2f}")
        
        return image
    
    def _apply_sharpening(self, image: np.ndarray, std: float, verbose: bool) -> np.ndarray:
        """Apply sharpening for better text definition"""
        if std > self.config.sharpening_std_threshold:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            image = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            if verbose:
                print(f"   ⚡ Applied text sharpening for photographic background")
        
        return image


# ===============================
# OCR PROCESSOR
# ===============================
class OCRProcessor:
    """Handles OCR operations with chunking support"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.reader = None
    
    def initialize_reader(self):
        """Initialize EasyOCR reader"""
        print("🚀 Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(
            self.config.languages, 
            gpu=self.config.use_gpu
        )
    
    def process_image(self, img_path: str) -> List[str]:
        """Process a single image with chunking"""
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if full_image is None:
            print(f"⚠️ Warning: Could not read image at {img_path}")
            return []
        
        h, _ = full_image.shape
        all_texts = []
        all_confidences = []
        
        # Process in chunks
        for y_start, y_end in self._generate_chunks(h):
            print(f"   - Processing chunk from y={y_start} to y={y_end}")
            
            chunk = full_image[y_start:y_end, :]
            chunk = self.preprocessor.preprocess(chunk)
            
            # Run OCR
            results = self.reader.readtext(chunk, paragraph=False, detail=1)
            
            # Filter by confidence
            for bbox, text, conf in results:
                if conf >= self.config.confidence_threshold:
                    all_texts.append(text)
                all_confidences.append(conf)
        
        # Display statistics
        self._display_stats(all_confidences)
        
        return all_texts
    
    def _generate_chunks(self, image_height: int):
        """Generate chunk boundaries for processing"""
        y_start = 0
        chunk_height = self.config.chunk_height
        overlap = self.config.overlap
        
        while y_start < image_height:
            y_end = min(y_start + chunk_height, image_height)
            yield y_start, y_end
            
            if y_end == image_height:
                break
            
            y_start += (chunk_height - overlap)
    
    def _display_stats(self, confidences: List[float]):
        """Display confidence statistics"""
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            print(f"   📊 Confidence stats: Avg={avg_conf:.3f}, "
                  f"Min={min_conf:.3f}, Max={max_conf:.3f} "
                  f"({len(confidences)} detections)")
        else:
            print(f"   📊 No text detected in image")


# ===============================
# BATCH PROCESSOR
# ===============================
class BatchProcessor:
    """Handles batch processing of image folders"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.ocr_processor = OCRProcessor(config)
    
    def process_folder(self):
        """Process all images in input folder"""
        input_path = Path(self.config.input_dir)
        output_path = Path(self.config.output_dir)
        
        print(f"📂 Input folder:  {input_path}")
        print(f"📁 Output folder: {output_path}")
        
        # Clear output folder
        self._clear_output_folder(output_path)
        
        # Initialize OCR reader
        self.ocr_processor.initialize_reader()
        
        # Process all images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for img_file in input_path.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                self._process_single_file(img_file, input_path, output_path)
        
        print("✅ OCR processing complete!")
    
    def _clear_output_folder(self, output_path: Path):
        """Clear output folder before processing"""
        if output_path.exists():
            print(f"🧹 Clearing submission folder: {output_path}")
            try:
                shutil.rmtree(output_path)
                print("✅ Submission folder cleared successfully!")
            except Exception as e:
                print(f"❌ Error clearing submission folder: {e}")
        
        output_path.mkdir(parents=True, exist_ok=True)
    
    def _process_single_file(self, img_file: Path, input_path: Path, output_path: Path):
        """Process a single image file"""
        # Calculate relative path
        rel_path = img_file.parent.relative_to(input_path)
        
        # Convert images_* to texts_*
        parts = list(rel_path.parts)
        if parts and parts[0].startswith('images_'):
            parts[0] = parts[0].replace('images_', 'texts_', 1)
            rel_path = Path(*parts) if parts else Path('.')
        
        # Create output paths
        output_folder = output_path / rel_path
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"{img_file.stem}.txt"
        
        print(f"🔍 OCR: {img_file} -> {output_file}")
        
        try:
            # Process image
            texts = self.ocr_processor.process_image(str(img_file))
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
                
        except Exception as e:
            print(f"❌ Failed to process {img_file}: {e}")


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Main execution function"""
    # Create configuration
    config = OCRConfig()
    
    # Process all images
    processor = BatchProcessor(config)
    processor.process_folder()


if __name__ == "__main__":
    main()
