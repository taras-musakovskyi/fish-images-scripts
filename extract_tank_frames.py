import cv2
from pathlib import Path

# Configuration
VIDEO_DIR = Path("/Users/tarasmusakovskyi/Downloads/augmented")
OUTPUT_DIR = Path("/Users/tarasmusakovskyi/Downloads/my-tank-underwater-scenes/frames")
EXTRACT_INTERVAL_SEC = 2.5  # Extract frame every 2.5 seconds

def crop_frame(frame):
    """
    Crop 25% from left and right, 12% from top and bottom.
    """
    height, width = frame.shape[:2]
    
    # Calculate crop boundaries
    left_crop = int(width * 0.25)
    right_crop = int(width * 0.75)
    top_crop = int(height * 0.12)
    bottom_crop = int(height * 0.88)
    
    # Crop
    cropped = frame[top_crop:bottom_crop, left_crop:right_crop]
    
    return cropped

def extract_frames_from_video(video_path, video_index):
    """
    Extract frames at regular time intervals from a single video.
    """
    print(f"\n{'='*70}")
    print(f"Processing video {video_index}: {video_path.name}")
    print('='*70)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    expected_frames = int(duration / EXTRACT_INTERVAL_SEC)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Expected extractions: ~{expected_frames} frames\n")
    
    video_name = video_path.stem
    frame_count = 0
    saved_count = 0
    last_saved_time = -999  # Force first frame to be saved
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Get timestamp
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamp_str = f"{int(timestamp_sec // 60):02d}m{int(timestamp_sec % 60):02d}s"
        
        # Check if enough time has passed since last saved frame
        if timestamp_sec - last_saved_time < EXTRACT_INTERVAL_SEC:
            continue
        
        # Crop frame
        cropped = crop_frame(frame)
        
        # Save frame with timestamp in filename
        output_path = OUTPUT_DIR / f"{video_name}_{timestamp_str}_frame{saved_count:04d}.jpg"
        cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        saved_count += 1
        last_saved_time = timestamp_sec
        
        # Progress update every 50 saved frames
        if saved_count % 50 == 0:
            print(f"  Saved: {saved_count} frames (at {timestamp_str})")
    
    cap.release()
    
    print(f"\nVideo complete - Saved: {saved_count} frames")
    
    return saved_count

def extract_frames():
    """
    Extract frames at regular intervals from all videos in directory.
    """
    print("=" * 70)
    print("TANK VIDEO FRAME EXTRACTOR - TIME-BASED SAMPLING")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = {'.mov', '.mp4', '.avi', '.mkv', '.MOV', '.MP4'}
    video_files = [
        f for f in VIDEO_DIR.iterdir()
        if f.is_file() and f.suffix in video_extensions
    ]
    
    if not video_files:
        print(f"\nNo video files found in {VIDEO_DIR}")
        return
    
    print(f"\nFound {len(video_files)} video(s) to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Extraction interval: Every {EXTRACT_INTERVAL_SEC} seconds")
    print(f"Crop: 25% left, 25% right, 12% top, 12% bottom\n")
    
    # Process each video
    total_saved = 0
    
    for idx, video_path in enumerate(sorted(video_files), 1):
        saved = extract_frames_from_video(video_path, idx)
        total_saved += saved
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL VIDEOS PROCESSED")
    print("=" * 70)
    print(f"Videos processed: {len(video_files)}")
    print(f"Total frames extracted: {total_saved}")
    print(f"\nOutput directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_frames()
