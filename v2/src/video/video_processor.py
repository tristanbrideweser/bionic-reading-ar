"""
video_processor.py
------------------
Optimized video processing with region tracking, caching, and frame sampling.
Handles live video capture and pre-recorded video with bionic text overlay.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time

# Local imports
from src.ocr.ocr_processor import extract_text_from_image
from src.bionify.bionic_formatter import BionicFormatter
from src.tracking.text_region_tracker import TextRegionTracker, TextRegion


class VideoProcessor:
    """Main video processing pipeline with optimizations."""
    
    def __init__(self,
                 ocr_engine: str = "tesseract",
                 process_every_n_frames: int = 3,
                 emphasis_ratio: float = 0.4):
        """
        Initialize the video processor.
        
        Args:
            ocr_engine: "tesseract" or "easyocr"
            process_every_n_frames: Only detect/OCR every N frames
            emphasis_ratio: Bionic text emphasis ratio
        """
        self.ocr_engine = ocr_engine
        self.process_every_n_frames = process_every_n_frames
        self.frame_count = 0
        
        # Initialize components
        self.bionic_formatter = BionicFormatter(emphasis_ratio)
        self.tracker = TextRegionTracker(
            iou_threshold=0.5,
            max_frames_missing=10,
            stable_frames_before_skip=5
        )
        
        # Performance metrics
        self.fps_history = []
        self.last_time = time.time()
    
    def detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions likely to contain text using edge detection and contours.
        
        Args:
            frame: Video frame (BGR format from OpenCV)
            
        Returns:
            List of bounding boxes as (x, y, w, h) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        bboxes = []
        h_img, w_img = frame.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (not too small, not too large)
            if w < 20 or h < 10:
                continue
            if w > w_img * 0.95 or h > h_img * 0.95:
                continue
            
            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 15:
                continue
            
            # Add some padding
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w_img - x, w + 2*pad)
            h = min(h_img - y, h + 2*pad)
            
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect text, track regions, OCR, and overlay.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Annotated frame with bionic text overlay
        """
        self.frame_count += 1
        
        # Only detect/OCR every N frames
        if self.frame_count % self.process_every_n_frames == 0:
            # Detect text regions
            detected_bboxes = self.detect_text_regions(frame)
            
            # Update tracker and get regions that need OCR
            regions_needing_ocr = self.tracker.update(detected_bboxes)
            
            # Run OCR on regions that need it
            for region in regions_needing_ocr:
                x, y, w, h = region.bbox
                roi = frame[y:y+h, x:x+w]
                
                try:
                    # Extract text (simplified preprocessing for speed)
                    text = self._fast_ocr(roi)
                    
                    if text.strip():
                        # Format as bionic text
                        bionic_text = self.bionic_formatter.format_text(text)
                        
                        # Update tracker with results
                        self.tracker.set_text_for_region(region.region_id, text, bionic_text)
                except Exception as e:
                    print(f"OCR error for region {region.region_id}: {e}")
        
        # Overlay all tracked regions (even if not OCR'd this frame)
        annotated_frame = self.annotate_frame(frame)
        
        # Calculate FPS
        self._update_fps()
        
        return annotated_frame
    
    def _fast_ocr(self, roi: np.ndarray) -> str:
        """
        Simplified OCR with minimal preprocessing for speed.
        
        Args:
            roi: Region of interest to OCR
            
        Returns:
            Extracted text
        """
        # Simple preprocessing only
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Light denoising
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Extract text
        from src.ocr.ocr_processor import extract_text_from_image
        text = extract_text_from_image(gray, engine=self.ocr_engine)
        
        return text
    
    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bionic text overlays on the frame.
        
        Args:
            frame: Original frame
            
        Returns:
            Frame with text overlays
        """
        annotated = frame.copy()
        
        for region in self.tracker.get_all_regions():
            if region.bionic_text:
                x, y, w, h = region.bbox
                
                # Draw semi-transparent background
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                
                # Draw border
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Render bionic text (simplified - just show as regular text)
                # For full bionic rendering, you'd need to parse <b> tags
                clean_text = region.bionic_text.replace('<b>', '').replace('</b>', '')
                
                # Calculate font size based on region height
                font_scale = h / 40.0
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = max(1, int(font_scale * 1.5))
                
                # Draw text with bold simulation (multiple draws with offset)
                text_y = y + h // 2 + 5
                
                # Draw bold parts (from bionic formatting)
                # For now, just draw all text in white
                cv2.putText(annotated, clean_text, (x + 5, text_y), 
                           font, font_scale, (255, 255, 255), thickness)
        
        # Draw FPS counter
        fps = self._get_current_fps()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw stats
        num_regions = len(self.tracker.get_all_regions())
        cache_size = self.bionic_formatter.get_cache_size()
        cv2.putText(annotated, f"Regions: {num_regions} | Cache: {cache_size}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return annotated
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        self.last_time = current_time
    
    def _get_current_fps(self) -> float:
        """Get averaged FPS."""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0


def process_video_file(video_path: str, output_path: Optional[str] = None):
    """
    Process a pre-recorded video file.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output (optional)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize processor
    processor = VideoProcessor(process_every_n_frames=5)
    
    # Initialize video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated = processor.process_frame(frame)
        
        # Write to output
        if writer:
            writer.write(annotated)
        
        # Display
        cv2.imshow('Bionic Video Processing', annotated)
        
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Processed {frame_num} frames...")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Done! Processed {frame_num} frames.")


def process_live_camera(camera_index: int = 0):
    """
    Process live camera feed.
    
    Args:
        camera_index: Camera device index (0 for default)
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_index}")
    
    processor = VideoProcessor(process_every_n_frames=3)
    
    print("Starting live camera processing. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        annotated = processor.process_frame(frame)
        cv2.imshow('Live Bionic Processing', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process video file
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        process_video_file(video_path, output_path)
    else:
        # Live camera
        process_live_camera()