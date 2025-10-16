import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from datetime import datetime

def crop_frame_margins(frame, left_pct=0.24, right_pct=0.24, top_pct=0.10, bottom_pct=0.10):
    """
    Crop margins from frame to focus on center background area
    
    Args:
        frame: Input frame
        left_pct: Percentage to crop from left (0-1)
        right_pct: Percentage to crop from right (0-1)
        top_pct: Percentage to crop from top (0-1)
        bottom_pct: Percentage to crop from bottom (0-1)
    
    Returns:
        Cropped frame
    """
    h, w = frame.shape[:2]
    
    left = int(w * left_pct)
    right = w - int(w * right_pct)
    top = int(h * top_pct)
    bottom = h - int(h * bottom_pct)
    
    return frame[top:bottom, left:right]

def divide_frame_into_regions(frame, grid_size=4):
    """
    Divide frame into grid regions
    
    Args:
        frame: Input frame
        grid_size: NxN grid (default 4x4 = 16 regions)
    
    Returns:
        List of (region, x, y, w, h) tuples
    """
    h, w = frame.shape[:2]
    region_h = h // grid_size
    region_w = w // grid_size
    
    regions = []
    for i in range(grid_size):
        for j in range(grid_size):
            y = i * region_h
            x = j * region_w
            
            # Handle last row/column to include remaining pixels
            if i == grid_size - 1:
                region_h_actual = h - y
            else:
                region_h_actual = region_h
                
            if j == grid_size - 1:
                region_w_actual = w - x
            else:
                region_w_actual = region_w
            
            region = frame[y:y+region_h_actual, x:x+region_w_actual]
            regions.append((region, x, y, region_w_actual, region_h_actual))
    
    return regions

def has_motion_in_region(region, prev_region, threshold=200):
    """
    Check if region has significant motion
    
    Args:
        region: Current region (BGR)
        prev_region: Previous region (BGR)
        threshold: Motion threshold (lower = more sensitive)
    
    Returns:
        True if motion detected, False otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    
    prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)
    
    # Compute difference
    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Count changed pixels
    changed_pixels = cv2.countNonZero(thresh)
    
    return changed_pixels > threshold

def extract_background_crops(video_path, output_dir, 
                             sample_rate=10, 
                             grid_size=4,
                             motion_threshold=200,
                             min_crop_size=256,
                             max_crops=3000,
                             crop_left=0.24,
                             crop_right=0.24,
                             crop_top=0.10,
                             crop_bottom=0.10):
    """
    Extract background-only crops from video by detecting motion-free regions
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save background crops
        sample_rate: Check every Nth frame
        grid_size: Divide frame into NxN grid (default 4x4)
        motion_threshold: Motion threshold per region (lower = more sensitive)
        min_crop_size: Minimum crop dimension in pixels
        max_crops: Maximum crops to save
        crop_left: Percentage to crop from left margin (0-1)
        crop_right: Percentage to crop from right margin (0-1)
        crop_top: Percentage to crop from top margin (0-1)
        crop_bottom: Percentage to crop from bottom margin (0-1)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Checking every {sample_rate} frames")
    print(f"Frame crop margins: L{crop_left*100:.0f}% R{crop_right*100:.0f}% T{crop_top*100:.0f}% B{crop_bottom*100:.0f}%")
    print(f"Grid size: {grid_size}x{grid_size} = {grid_size*grid_size} regions per frame")
    print(f"Motion threshold: {motion_threshold} (lower = more sensitive)")
    
    saved_count = 0
    frame_idx = 0
    prev_frame = None
    
    stats = {
        'frames_processed': 0,
        'regions_checked': 0,
        'regions_with_motion': 0,
        'crops_saved': 0,
        'crops_too_small': 0
    }
    
    pbar = tqdm(total=min(total_frames // sample_rate, max_crops), desc="Extracting backgrounds")
    
    while saved_count < max_crops:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        stats['frames_processed'] += 1
        
        # Crop margins from frame to focus on center area
        frame = crop_frame_margins(frame, crop_left, crop_right, crop_top, crop_bottom)
        
        # Skip first frame (need previous for comparison)
        if prev_frame is None:
            prev_frame = frame.copy()
            frame_idx += 1
            continue
        
        # Divide into regions
        current_regions = divide_frame_into_regions(frame, grid_size)
        prev_regions = divide_frame_into_regions(prev_frame, grid_size)
        
        # Check each region for motion
        for (curr_reg, x, y, w, h), (prev_reg, _, _, _, _) in zip(current_regions, prev_regions):
            stats['regions_checked'] += 1
            
            # Skip regions that are too small
            if w < min_crop_size or h < min_crop_size:
                stats['crops_too_small'] += 1
                continue
            
            # Check for motion
            has_motion = has_motion_in_region(curr_reg, prev_reg, motion_threshold)
            
            if has_motion:
                stats['regions_with_motion'] += 1
                continue
            
            
	    # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	    
            # No motion detected - save as background crop
            crop_filename = f"bg_crop_f{timestamp}_{frame_idx:06d}_x{x}_y{y}.jpg"
            crop_path = output_dir / crop_filename
            cv2.imwrite(str(crop_path), curr_reg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            saved_count += 1
            stats['crops_saved'] += 1
            pbar.update(1)
            
            if saved_count >= max_crops:
                break
        
        prev_frame = frame.copy()
        frame_idx += 1
        
        if saved_count >= max_crops:
            break
    
    pbar.close()
    cap.release()
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Regions checked: {stats['regions_checked']}")
    print(f"Regions with motion (skipped): {stats['regions_with_motion']}")
    print(f"Regions too small (skipped): {stats['crops_too_small']}")
    print(f"Background crops saved: {stats['crops_saved']}")
    print(f"Output directory: {output_dir}")
    print(f"\nAverage crops per frame: {stats['crops_saved'] / max(stats['frames_processed'], 1):.1f}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract background-only crops from video using motion detection'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save background crops'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=10,
        help='Check every Nth frame (default: 10)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=4,
        help='Divide frame into NxN grid (default: 4, creates 16 regions)'
    )
    parser.add_argument(
        '--motion-threshold',
        type=int,
        default=200,
        help='Motion threshold per region in pixels (default: 200, lower=more sensitive)'
    )
    parser.add_argument(
        '--min-crop-size',
        type=int,
        default=256,
        help='Minimum crop dimension in pixels (default: 256)'
    )
    parser.add_argument(
        '--max-crops',
        type=int,
        default=3000,
        help='Maximum background crops to save (default: 3000)'
    )
    parser.add_argument(
        '--crop-left',
        type=float,
        default=24.0,
        help='Percentage to crop from left margin (default: 24.0)'
    )
    parser.add_argument(
        '--crop-right',
        type=float,
        default=24.0,
        help='Percentage to crop from right margin (default: 24.0)'
    )
    parser.add_argument(
        '--crop-top',
        type=float,
        default=10.0,
        help='Percentage to crop from top margin (default: 10.0)'
    )
    parser.add_argument(
        '--crop-bottom',
        type=float,
        default=10.0,
        help='Percentage to crop from bottom margin (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    print("Starting background crop extraction...")
    extract_background_crops(
        args.video_path,
        args.output_dir,
        sample_rate=args.sample_rate,
        grid_size=args.grid_size,
        motion_threshold=args.motion_threshold,
        min_crop_size=args.min_crop_size,
        max_crops=args.max_crops,
        crop_left=args.crop_left / 100.0,
        crop_right=args.crop_right / 100.0,
        crop_top=args.crop_top / 100.0,
        crop_bottom=args.crop_bottom / 100.0
    )

if __name__ == "__main__":
    main()
