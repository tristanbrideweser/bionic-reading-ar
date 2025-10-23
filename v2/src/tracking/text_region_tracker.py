"""
text_region_tracker.py
----------------------
Tracks text regions across video frames to avoid redundant OCR.
Only triggers OCR when regions are new or have changed significantly.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class TextRegion:
    """Represents a tracked text region."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], region_id: int):
        self.bbox = bbox  # (x, y, w, h)
        self.region_id = region_id
        self.text: Optional[str] = None
        self.bionic_text: Optional[str] = None
        self.frames_stable = 0  # How many frames this region hasn't moved
        self.needs_ocr = True
        self.confidence = 0.0
    
    def update_bbox(self, new_bbox: Tuple[int, int, int, int], threshold: float = 0.15):
        """
        Update bbox and determine if OCR is needed.
        
        Args:
            new_bbox: New bounding box
            threshold: Position change threshold to trigger re-OCR
        """
        old_center = (self.bbox[0] + self.bbox[2]/2, self.bbox[1] + self.bbox[3]/2)
        new_center = (new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2)
        
        # Calculate distance as percentage of region size
        distance = np.sqrt((old_center[0] - new_center[0])**2 + 
                          (old_center[1] - new_center[1])**2)
        relative_distance = distance / max(self.bbox[2], self.bbox[3])
        
        if relative_distance > threshold:
            self.needs_ocr = True
            self.frames_stable = 0
        else:
            self.frames_stable += 1
        
        self.bbox = new_bbox


class TextRegionTracker:
    """Tracks text regions across frames and manages OCR requests."""
    
    def __init__(self, 
                 iou_threshold: float = 0.5,
                 max_frames_missing: int = 5,
                 stable_frames_before_skip: int = 3):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU to match regions across frames
            max_frames_missing: Remove region after this many frames without detection
            stable_frames_before_skip: Skip OCR after region stable for N frames
        """
        self.regions: Dict[int, TextRegion] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.stable_frames_before_skip = stable_frames_before_skip
        self.frames_missing: Dict[int, int] = {}
    
    def update(self, detected_bboxes: List[Tuple[int, int, int, int]]) -> List[TextRegion]:
        """
        Update tracked regions with new detections.
        
        Args:
            detected_bboxes: List of (x, y, w, h) tuples from current frame
            
        Returns:
            List of TextRegion objects that need OCR processing
        """
        # Match detected boxes to existing regions
        matched_regions = set()
        matched_bboxes = set()
        
        for region_id, region in list(self.regions.items()):
            best_match_idx = None
            best_iou = 0
            
            for idx, bbox in enumerate(detected_bboxes):
                if idx in matched_bboxes:
                    continue
                    
                iou = self._calculate_iou(region.bbox, bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Update existing region
                region.update_bbox(detected_bboxes[best_match_idx])
                matched_regions.add(region_id)
                matched_bboxes.add(best_match_idx)
                self.frames_missing[region_id] = 0
                
                # Skip OCR if region has been stable
                if region.frames_stable > self.stable_frames_before_skip:
                    region.needs_ocr = False
            else:
                # Region not found in this frame
                self.frames_missing[region_id] = self.frames_missing.get(region_id, 0) + 1
        
        # Remove regions that haven't been seen in a while
        for region_id in list(self.regions.keys()):
            if self.frames_missing.get(region_id, 0) > self.max_frames_missing:
                del self.regions[region_id]
                del self.frames_missing[region_id]
        
        # Add new regions for unmatched bboxes
        for idx, bbox in enumerate(detected_bboxes):
            if idx not in matched_bboxes:
                new_region = TextRegion(bbox, self.next_id)
                self.regions[self.next_id] = new_region
                self.frames_missing[self.next_id] = 0
                self.next_id += 1
        
        # Return regions that need OCR
        return [r for r in self.regions.values() if r.needs_ocr]
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1, bbox2: (x, y, w, h) tuples
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_all_regions(self) -> List[TextRegion]:
        """Get all currently tracked regions."""
        return list(self.regions.values())
    
    def set_text_for_region(self, region_id: int, text: str, bionic_text: str):
        """
        Update text and bionic text for a region after OCR.
        
        Args:
            region_id: The region ID
            text: Plain OCR text
            bionic_text: Formatted bionic text
        """
        if region_id in self.regions:
            self.regions[region_id].text = text
            self.regions[region_id].bionic_text = bionic_text
            self.regions[region_id].needs_ocr = False
    
    def clear(self):
        """Clear all tracked regions."""
        self.regions.clear()
        self.frames_missing.clear()
        self.next_id = 0