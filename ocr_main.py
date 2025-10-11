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
from difflib import SequenceMatcher


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

    # Debug & Logging Controls
    enable_debug_images: bool = True   # Generate debug chunk images 
    enable_verbose_logging: bool = True  # Show preprocessing details
    debug_images_dir: str = "debug_chunks"  # Directory for debug images
    
    # Chunking
    chunk_height: int = 900
    overlap: int = 15 # Set 1 if no deduplicate needed
    
    # Deduplication
    enable_deduplication: bool = True  # Default off to preserve baseline evaluation
    lcs_threshold: float = 0.40

    # Text Size & Scaling
    min_text_px: int = 18
    max_scale: float = 3.5
    max_dimension: int = 3000
    
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
    sharpening_std_threshold: int = 60
    
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
    
    def preprocess(self, image: np.ndarray, verbose: bool = None) -> np.ndarray:
        """Apply full preprocessing pipeline"""
        # Use config setting if verbose not explicitly provided
        if verbose is None:
            verbose = self.config.enable_verbose_logging
            
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
        try:
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.use_gpu
            )
        except Exception as e:
            # Fallback to CPU mode if GPU initialization fails
            print(f"⚠️ EasyOCR initialization failed with gpu={self.config.use_gpu}: {e}")
            if self.config.use_gpu:
                print("ℹ️ Falling back to CPU mode for EasyOCR reader")
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=False
            )
    
    def process_image(self, img_path: str) -> List[str]:
        """Process image with Y-coordinate based LCS deduplication"""
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if full_image is None:
            print(f"⚠️ Warning: Could not read image at {img_path}")
            return []
        
        h, _ = full_image.shape
        # Store detected text grouped by logical lines to allow safe line-level removal
        all_lines: List[List[str]] = []  # each entry is a list of text blocks belonging to one visual line
        all_confidences: List[float] = []  # flat list for global stats

    # (no baseline collection here; when deduplication is disabled we
    # keep the original chunk-by-chunk detection order)

        # Simple approach: track last line of previous chunk
        prev_last_line = []  # List of texts from last line of previous chunk
        prev_last_conf = 0.0
        
        for chunk_idx, (y_start, y_end) in enumerate(self._generate_chunks(h)):
            print(f"   - Processing chunk {chunk_idx} from y={y_start} to y={y_end}")
            
            # Extract original chunk from full image (coords in original image space)
            chunk = full_image[y_start:y_end, :]
            orig_h, orig_w = chunk.shape[:2]

            # Preprocess may resize the chunk; we need the processed image for OCR
            proc_chunk = self.preprocessor.preprocess(chunk)
            proc_h, proc_w = proc_chunk.shape[:2]

            # Compute scale factors to map processed bbox coords back to original chunk coords
            # Avoid division by zero
            scale_x = proc_w / orig_w if orig_w and proc_w else 1.0
            scale_y = proc_h / orig_h if orig_h and proc_h else 1.0

            # Use proc_chunk for OCR
            chunk = proc_chunk
            
            # Run OCR
            results = self.reader.readtext(chunk, paragraph=False, detail=1)
            
            # Debug: Save all chunks with bounding boxes (if enabled)
            if self.config.enable_debug_images:
                self._save_debug_image(chunk, results, img_path, chunk_idx, self.config.confidence_threshold)
            
            # If deduplication is disabled: use baseline flow -> append texts in detection order
            if not self.config.enable_deduplication:
                # Preserve previous behaviour (each detection becomes its own line)
                for bbox, text, conf in results:
                    # Normalize confidence when possible
                    try:
                        conf_val = float(conf)
                    except Exception:
                        conf_val = conf

                    all_confidences.append(conf_val)
                    if conf_val >= self.config.confidence_threshold:
                        all_lines.append([text])
                continue

            # Extract texts with absolute Y coordinates and confidence filtering (advanced flow)
            valid_detections = []
            for bbox, text, conf in results:
                # Normalize/conf tracking
                try:
                    conf_val = float(conf)
                except Exception:
                    conf_val = conf

                all_confidences.append(conf_val)
                if conf_val >= self.config.confidence_threshold:
                    # Calculate absolute top-left Y and X coordinates
                    ys_proc = [p[1] for p in bbox]
                    xs_proc = [p[0] for p in bbox]
                    # map processed coords back to original chunk coords
                    y_top_orig = int(min(ys_proc) / scale_y)
                    x_left_orig = int(min(xs_proc) / scale_x)
                    y_absolute = y_start + y_top_orig
                    valid_detections.append((text, y_absolute, x_left_orig, conf_val))

            if not valid_detections:
                continue

            # Sort by top-left coordinate: first by Y (top to bottom), then by X (left to right)
            valid_detections.sort(key=lambda x: (x[1], x[2]))

            # Group detections into lines (each line: list of (text, y_abs, x_abs, conf))
            lines = self._group_detections_into_lines(valid_detections)

            if chunk_idx == 0:
                # First chunk: add all lines (ensure left->right within each line)
                for line in lines:
                    # line items are (text, y, x, conf)
                    line_sorted = sorted(line, key=lambda it: it[2])
                    all_lines.append([t for t, _, _, _ in line_sorted])

                # Store last line texts for next chunk
                prev_last_line = [t for t, _, _, _ in lines[-1]] if lines else []
                prev_last_conf = (sum(c for _, _, _, c in lines[-1]) / len(lines[-1])) if lines and lines[-1] else 0.0
            else:
                # Current first line
                curr_first = lines[0] if lines else []
                curr_first_texts = [t for t, _, _, _ in curr_first]
                curr_first_conf = (sum(c for _, _, _, c in curr_first) / len(curr_first)) if curr_first else 0.0

                # Only attempt deduplication if the first line of the current chunk
                # is within the top overlap region of this chunk. This avoids
                # removing legitimate lines that are not in the overlapped area.
                first_line_y = None
                if curr_first:
                    # detections in curr_first are tuples (text, y_abs, conf) or (text, y_abs, x_abs, conf)
                    # y is at index 1 for both shapes
                    first_line_y = curr_first[0][1]

                if first_line_y is not None and (first_line_y - y_start) <= self.config.overlap:
                    # Deduplicate using SequenceMatcher-based similarity + confidence
                    action = self._deduplicate_with_lcs(prev_last_line, curr_first_texts, prev_last_conf, curr_first_conf)

                    if action == 1:
                        # Remove last logical line from all_lines
                        if all_lines:
                            removed = all_lines.pop()
                            print(f"      🔄 LCS: Removed prev line ({len(removed)} texts): {removed}")

                    elif action == 2:
                        # Remove first logical line from current chunk (skip adding it)
                        print(f"      🔄 LCS: Removed curr first line ({len(curr_first_texts)} texts): {curr_first_texts}")
                        lines = lines[1:]
                else:
                    # Skip deduplication for this chunk (first line not in overlap)
                    if first_line_y is not None:
                        print(f"      ℹ️ Skipping dedupe: first line y={first_line_y:.1f} not within top overlap {self.config.overlap}px of chunk starting at {y_start}")

                # Add remaining logical lines from current chunk (preserve left-to-right order)
                for line in lines:
                    line_sorted = sorted(line, key=lambda it: it[2])
                    all_lines.append([t for t, _, _, _ in line_sorted])

                # Update last line for next iteration
                if lines:
                    prev_last_line = [t for t, _, _, _ in lines[-1]]
                    prev_last_conf = (sum(c for _, _, _, c in lines[-1]) / len(lines[-1])) if lines[-1] else 0.0
                else:
                    prev_last_line = []
                    prev_last_conf = 0.0
            
        # Display statistics
        self._display_stats(all_confidences)

        # Flatten grouped lines to the original output format (one detection per line)
        flattened = [t for line in all_lines for t in line]
        return flattened

    def _group_detections_into_lines(self, detections: List[tuple], tol: float = 5.0):
        """Group detections into logical lines based on Y coordinate proximity.

        Supports detection tuples of shape (text, y, x, conf) or ((x,y), text, conf) or (text, y, conf).

        Returns list of lines; each line is a list of detection tuples in the original shape.
        """
        if not detections:
            return []

        # Helper to extract Y from different tuple shapes
        def extract_y(det):
            if len(det) == 4:
                return det[1]
            if len(det) == 3 and isinstance(det[0], tuple):
                # ( (x,y), text, conf )
                return det[0][1]
            if len(det) == 3:
                # (text, y, conf)
                return det[1]
            raise ValueError("Unsupported detection tuple shape")

        lines = []
        current_line = [detections[0]]
        current_y = extract_y(detections[0])

        for det in detections[1:]:
            y_abs = extract_y(det)
            if abs(y_abs - current_y) <= tol:
                current_line.append(det)
            else:
                lines.append(current_line)
                current_line = [det]
                current_y = y_abs

        if current_line:
            lines.append(current_line)

        return lines
    
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
    
    def _deduplicate_with_lcs(self, prev_last_line, curr_first_line, prev_conf: float = 0.0, curr_conf: float = 0.0):
        """
        Deduplicate using LCS algorithm
        
        Args:
            prev_last_line: List of texts from last line of previous chunk
            curr_first_line: List of texts from first line of current chunk
            
        Returns:
            0: No duplicate - keep both lines
            1: Remove last line from all_texts (prev chunk line is duplicate)
            2: Remove first line from current chunk (curr chunk line is duplicate)
        """
        if not prev_last_line or not curr_first_line:
            return 0
        
        # Concatenate texts from each line with space and normalize
        prev_line_text = " ".join(prev_last_line).strip()
        curr_line_text = " ".join(curr_first_line).strip()

        print(f"      ⚖️ Comparing lines:")
        print(f"         Prev: '{prev_line_text}'")
        print(f"         Curr: '{curr_line_text}'")

        if not prev_line_text or not curr_line_text:
            print(f"      ✅ No duplicate found (empty)")
            return 0

        # Compute similarity ratio using SequenceMatcher (fast, low memory)
        ratio = SequenceMatcher(None, prev_line_text, curr_line_text).ratio()

        # Combine ratio with confidence information to reduce false removals
        # Normalize confidence difference into [-1,1]
        conf_diff = 0.0
        try:
            conf_diff = (curr_conf - prev_conf) / (max(prev_conf, curr_conf, 1e-6))
        except Exception:
            conf_diff = 0.0

        # Weighted score: prefer similarity but give some weight to confidence difference
        weighted_score = 0.9 * ratio + 0.1 * (1.0 - max(0.0, -conf_diff))

        print(f"         Sim ratio: {ratio:.3f}, prev_conf={prev_conf:.3f}, curr_conf={curr_conf:.3f}, weighted={weighted_score:.3f} (threshold: {self.config.lcs_threshold})")

        if weighted_score >= self.config.lcs_threshold:
            # Remove the line with lower average confidence; fall back to shorter length
            if curr_conf < prev_conf:
                return 2
            elif prev_conf < curr_conf:
                return 1
            else:
                # same confidence -> remove shorter (fewer chars)
                if len(prev_line_text) < len(curr_line_text):
                    return 1
                else:
                    return 2

        print(f"      ✅ No duplicate found")
        return 0
    
    
    
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
            print(f"   🌸 No text detected in image")
    
    def _save_debug_image(self, chunk: np.ndarray, results: list, img_path: str, chunk_idx: int, confidence_threshold: float):
        """Save debug image with bounding boxes for all chunks"""
        import cv2
        import os
        
        # Create debug output directory
        debug_dir = Path("otherVersion") / self.config.debug_images_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy chunk for drawing
        debug_img = chunk.copy()
        if len(debug_img.shape) == 2:  # Convert grayscale to color
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes and text
        for i, (bbox, text, conf) in enumerate(results):
            # Convert bbox to integer coordinates
            pts = np.array(bbox, dtype=np.int32)
            
            # Choose color based on confidence
            if conf >= confidence_threshold:
                color = (0, 255, 0)  # Green for accepted
                status = "ACCEPT"
            else:
                color = (0, 0, 255)  # Red for rejected
                status = "REJECT"
            
            # Draw bounding box
            cv2.polylines(debug_img, [pts], True, color, 2)
            
            # Draw text info
            x, y = pts[0]
            label = f"{i+1}: {text[:10]}... ({conf:.3f}) {status}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(debug_img, (x, y-text_h-5), (x+text_w, y), color, -1)
            cv2.putText(debug_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save debug image with chunk index
        img_name = Path(img_path).stem
        debug_path = debug_dir / f"{img_name}_chunk{chunk_idx:02d}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"      💾 Debug image saved: {debug_path}")


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
        
        # Clear debug chunks folder (if debug enabled)
        if self.config.enable_debug_images:
            self._clear_debug_folder()
        
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
    
    def _clear_debug_folder(self):
        """Clear debug chunks folder before processing"""
        debug_path = Path("otherVersion") / self.config.debug_images_dir
        if debug_path.exists():
            print(f"🧹 Clearing debug chunks folder: {debug_path}")
            try:
                shutil.rmtree(debug_path)
                print("✅ Debug chunks folder cleared successfully!")
            except Exception as e:
                print(f"❌ Error clearing debug chunks folder: {e}")
    
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
        
        print(f" 🎯 OCR: {img_file} -> {output_file}")
        
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
