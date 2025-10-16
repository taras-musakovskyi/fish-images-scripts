#!/usr/bin/env python3
"""
Video Frame Extraction Script with Quality Filtering
Extracts high-quality frames from video files with customizable filtering
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import time
from datetime import datetime

class VideoFrameExtractor:
    def __init__(self, quality_threshold=0.7):
        self.quality_threshold = quality_threshold
        
    def _is_good_quality_frame(self, frame: np.ndarray) -> tuple[bool, dict]:
        """
        Assess frame quality based on multiple criteria
        Returns (is_good, quality_metrics)
        """
        if frame.size == 0:
            return False, {}
            
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Quality metrics
        metrics = {}
        
        # 1. Brightness check
        brightness = np.mean(gray)
        metrics['brightness'] = brightness
        brightness_ok = 70 <= brightness <= 180  # Avoid too dark or overexposed
        
        # 2. Contrast check
        contrast = np.std(gray)
        metrics['contrast'] = contrast
        contrast_ok = contrast >= 30  # Good detail definition
        
        # 3. Saturation check
        saturation = np.mean(hsv[:, :, 1])
        metrics['saturation'] = saturation
        saturation_ok = 40 <= saturation <= 150  # Avoid oversaturated or dull
        
        # 4. Sharpness check (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = laplacian_var
        sharpness_ok = laplacian_var >= 100  # Good focus
        
        # 5. Glare detection
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        glare_ratio = bright_pixels / total_pixels
        metrics['glare_ratio'] = glare_ratio
        glare_ok = glare_ratio <= 0.05  # Less than 5% very bright pixels
        
        # 6. Motion blur detection (edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics['edge_density'] = edge_density
        motion_blur_ok = edge_density >= 0.05  # Good edge definition
        
        # Overall quality score
        quality_checks = [brightness_ok, contrast_ok, saturation_ok, 
                         sharpness_ok, glare_ok, motion_blur_ok]
        quality_score = sum(quality_checks) / len(quality_checks)
        metrics['quality_score'] = quality_score
        
        is_good = quality_score >= self.quality_threshold
        
        return is_good, metrics
        
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30, max_frames: int = None,
                      save_rejected: bool = False, verbose: bool = True,
                      crop_bottom_percent: float = 0, filename_prefix: str = ""):
        """
        Extract quality frames from video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (default: 30)
            max_frames: Maximum number of frames to extract (None = unlimited)
            save_rejected: Save rejected frames to separate folder
            verbose: Print progress information
            crop_bottom_percent: Remove bottom N% of frame (0-30)
            filename_prefix: Optional prefix for frame filenames
        """
        
        # Setup paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_rejected:
            rejected_path = output_path / "rejected"
            rejected_path.mkdir(exist_ok=True)
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        if verbose:
            print(f"Video: {video_path}")
            print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
            print(f"Extracting every {frame_interval} frames with quality threshold {self.quality_threshold}")
            print(f"Output directory: {output_path}")
            
        # Frame extraction
        frame_count = 0
        extracted_count = 0
        rejected_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every Nth frame
                if frame_count % frame_interval == 0:
                    # Crop bottom if specified
                    if crop_bottom_percent > 0:
                        crop_height = int(frame.shape[0] * (1 - crop_bottom_percent / 100))
                        frame = frame[:crop_height, :]
                    
                    is_good, metrics = self._is_good_quality_frame(frame)
                    
                    if is_good:
                        # Save good frame
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        timestamp_str = f"{int(timestamp_ms//1000):04d}_{int(timestamp_ms%1000):03d}"
                        
                        prefix_part = f"{filename_prefix}_" if filename_prefix else ""
                        filename = f"{prefix_part}frame_{extracted_count+1:05d}_t{timestamp_str}_q{metrics['quality_score']:.3f}.jpg"
                        filepath = output_path / filename
                        
                        # Save with high quality
                        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        extracted_count += 1
                        
                        if verbose and extracted_count % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"Extracted {extracted_count} frames in {elapsed:.1f}s")
                            
                    else:
                        rejected_count += 1
                        if save_rejected:
                            prefix_part = f"{filename_prefix}_" if filename_prefix else ""
                            filename = f"{prefix_part}rejected_{rejected_count:05d}_q{metrics['quality_score']:.3f}.jpg"
                            filepath = rejected_path / filename
                            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                        
                frame_count += 1
                
        finally:
            cap.release()
            
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nExtraction complete!")
            print(f"Processed {frame_count} total frames in {elapsed_time:.2f}s")
            print(f"Extracted {extracted_count} good quality frames")
            print(f"Rejected {rejected_count} poor quality frames")
            print(f"Quality rate: {extracted_count/(extracted_count+rejected_count)*100:.1f}%")
            
        return {
            'extracted_count': extracted_count,
            'rejected_count': rejected_count,
            'total_processed': frame_count,
            'processing_time': elapsed_time
        }


def main():
    parser = argparse.ArgumentParser(description='Extract quality frames from video')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('output_dir', help='Output directory for frames')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Extract every N frames (default: 30)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to extract (default: unlimited)')
    parser.add_argument('--prefix', type=str, default="",
                       help='Prefix for frame filenames (e.g. "cam1", "test1")')
    parser.add_argument('--crop-bottom', type=float, default=0,
                       help='Remove bottom N%% of frame (default: 0, max: 30)')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                       help='Quality threshold 0-1 (default: 0.7)')
    parser.add_argument('--save-rejected', action='store_true',
                       help='Save rejected frames to separate folder')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output messages')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_video):
        print(f"Error: Video file not found: {args.input_video}")
        return 1
        
    # Create extractor
    extractor = VideoFrameExtractor(quality_threshold=args.quality_threshold)
    
    try:
        results = extractor.extract_frames(
            video_path=args.input_video,
            output_dir=args.output_dir,
            frame_interval=args.interval,
            max_frames=args.max_frames,
            save_rejected=args.save_rejected,
            verbose=not args.quiet,
            crop_bottom_percent=args.crop_bottom,
            filename_prefix=args.prefix
        )
        
        print(f"\nSummary: {results['extracted_count']} frames extracted successfully")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())