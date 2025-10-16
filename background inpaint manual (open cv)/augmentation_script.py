import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_aspect_ratio(img):
    """Calculate width/height aspect ratio"""
    return img.width / img.height

def is_acceptably_square(aspect_ratio, threshold=1.15):
    """Check if image is close enough to square"""
    return aspect_ratio <= threshold

def extract_and_stretch_background(img, position='top', target_height=100, source_height=80):
    """
    Extract background strip and stretch it to target height
    
    Args:
        img: PIL Image
        position: 'top' or 'bottom'
        target_height: Height needed to fill
        source_height: Height of strip to extract (will be stretched)
    """
    if position == 'top':
        strip = img.crop((0, 0, img.width, source_height))
    else:  # bottom
        strip = img.crop((0, img.height - source_height, img.width, img.height))
    
    # Stretch to target height
    stretched = strip.resize((img.width, target_height), Image.LANCZOS)
    
    # Add subtle noise to reduce uniformity
    strip_array = np.array(stretched)
    noise = np.random.normal(0, 3, strip_array.shape).astype(np.int16)
    strip_array = np.clip(strip_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(strip_array)

def apply_boundary_blending(img, top_boundary, original_height, blend_height=20):
    """
    Apply smooth Gaussian blending at boundaries between stretched and original
    
    Args:
        img: PIL Image with stretched strips and original in middle
        top_boundary: Y position where top strip meets original
        original_height: Height of original image portion
        blend_height: Height of blend region in pixels
    """
    img_array = np.array(img).astype(np.float32)
    height, width = img_array.shape[:2]
    
    # Create blend mask for top boundary
    if top_boundary > 0 and blend_height > 0:
        blend_start = max(0, top_boundary - blend_height)
        blend_end = min(height, top_boundary + blend_height)
        
        for y in range(blend_start, blend_end):
            # Calculate alpha (0 to 1) for smooth transition
            if y < top_boundary:
                alpha = (y - blend_start) / (top_boundary - blend_start) if top_boundary > blend_start else 1.0
            else:
                alpha = 1.0 - (y - top_boundary) / blend_height if blend_height > 0 else 1.0
            
            alpha = np.clip(alpha, 0, 1)
            
            # Apply very subtle blur at transition
            if 0.3 < alpha < 0.7:
                # Slight blur only in transition zone
                if y > 0 and y < height - 1:
                    img_array[y] = (img_array[y-1] * 0.2 + img_array[y] * 0.6 + img_array[y+1] * 0.2)
    
    # Create blend mask for bottom boundary
    bottom_boundary = top_boundary + original_height
    if bottom_boundary < height and blend_height > 0:
        blend_start = max(0, bottom_boundary - blend_height)
        blend_end = min(height, bottom_boundary + blend_height)
        
        for y in range(blend_start, blend_end):
            # Calculate alpha for smooth transition
            if y < bottom_boundary:
                alpha = (y - blend_start) / (bottom_boundary - blend_start) if bottom_boundary > blend_start else 1.0
            else:
                alpha = 1.0 - (y - bottom_boundary) / blend_height if blend_height > 0 else 1.0
            
            alpha = np.clip(alpha, 0, 1)
            
            # Apply very subtle blur at transition
            if 0.3 < alpha < 0.7:
                if y > 0 and y < height - 1:
                    img_array[y] = (img_array[y-1] * 0.2 + img_array[y] * 0.6 + img_array[y+1] * 0.2)
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def get_similar_width_images(target_width, width_grouped_images, tolerance=0.05):
    """
    Get images with similar width to target
    
    Args:
        target_width: Width to match
        width_grouped_images: Dict mapping width to list of images
        tolerance: Width tolerance (0.05 = 5%)
    """
    min_width = target_width * (1 - tolerance)
    max_width = target_width * (1 + tolerance)
    
    similar_images = []
    for width, images in width_grouped_images.items():
        if min_width <= width <= max_width:
            similar_images.extend(images)
    
    return similar_images

def augment_to_square(img, width_grouped_images, img_width, strip_percentage=0.14):
    """
    Augment image by adding background to top/bottom to make it square
    
    Args:
        img: PIL Image to augment
        width_grouped_images: Dict of images grouped by width
        img_width: Width of current image for finding similar images
        strip_percentage: Percentage of image height to use as source strip (default 0.14 = 14%)
    """
    width, height = img.size
    aspect = width / height
    
    if aspect <= 1.0:  # Already taller or square
        return img
    
    # Calculate needed height to make square
    target_height = width  # Make height equal to width for 1:1 aspect
    needed_height = target_height - height
    
    if needed_height <= 0:
        return img
    
    # Split needed height between top and bottom
    top_add = needed_height // 2
    bottom_add = needed_height - top_add
    
    # Extract source strip height based on percentage parameter
    source_strip_height = int(height * strip_percentage)
    source_strip_height = max(20, min(source_strip_height, 150))  # Clamp between 20-150px
    
    # 85% use own background, 15% use similar-width image from same folder
    if random.random() < 0.85:
        bg_source = img
    else:
        # Get images with similar width (±5%)
        similar_images = get_similar_width_images(img_width, width_grouped_images, tolerance=0.05)
        
        if similar_images:
            bg_source = random.choice(similar_images)
            # Verify width matches exactly, if not use own background
            if bg_source.width != width:
                bg_source = img
        else:
            bg_source = img
    
    # Extract and stretch background strips to exact needed height
    top_strip = extract_and_stretch_background(bg_source, 'top', top_add, source_strip_height)
    bottom_strip = extract_and_stretch_background(bg_source, 'bottom', bottom_add, source_strip_height)
    
    # Create new canvas
    new_img = Image.new('RGB', (width, target_height))
    
    # Paste stretched top strip
    new_img.paste(top_strip, (0, 0))
    
    # Paste original image in middle
    new_img.paste(img, (0, top_add))
    
    # Paste stretched bottom strip
    new_img.paste(bottom_strip, (0, top_add + height))
    
    # Apply Gaussian blur at boundaries for smooth blending
    new_img = apply_boundary_blending(new_img, top_add, height, blend_height=20)
    
    return new_img

def load_folder_images_by_width(folder_path, max_samples_per_width=10):
    """
    Load sample images from folder grouped by width for background reference
    
    Returns dict: {width: [list of PIL Images with that width]}
    """
    image_paths = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
    
    # Group by width
    width_groups = {}
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            width = img.width
            
            if width not in width_groups:
                width_groups[width] = []
            
            # Limit samples per width to avoid memory issues
            if len(width_groups[width]) < max_samples_per_width:
                width_groups[width].append(img)
            else:
                img.close()
                
        except:
            continue
    
    return width_groups

def process_dataset(dataset_root, num_to_process='all', aspect_threshold=1.15, strip_percentage=0.14):
    """
    Process dataset and create augmented versions
    
    Args:
        dataset_root: Path to dataset root (contains small/medium/wide folders)
        num_to_process: 'all' or integer number of images to process per folder
        aspect_threshold: Aspect ratio threshold for considering image "square enough"
        strip_percentage: Percentage of image height to use as source strip (default 0.14 = 14%)
    """
    dataset_root = Path(dataset_root)
    folders = ['small', 'medium', 'wide']
    
    stats = {
        'total': 0,
        'skipped_square': 0,
        'augmented': 0,
        'errors': 0
    }
    
    for folder_name in folders:
        folder_path = dataset_root / folder_name
        if not folder_path.exists():
            print(f"Skipping {folder_name} - folder not found")
            continue
        
        # Create output folder
        output_folder = dataset_root / f"{folder_name}_augmented"
        output_folder.mkdir(exist_ok=True)
        
        # Get all images
        image_paths = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        
        # Determine how many to process
        if num_to_process == 'all':
            paths_to_process = image_paths
        else:
            paths_to_process = random.sample(image_paths, min(int(num_to_process), len(image_paths)))
        
        print(f"\nProcessing folder: {folder_name}")
        print(f"Total images: {len(image_paths)}, Processing: {len(paths_to_process)}")
        
        # Load reference images grouped by width for background sampling
        print("Loading reference images grouped by width...")
        width_grouped_images = load_folder_images_by_width(folder_path)
        print(f"Loaded {sum(len(imgs) for imgs in width_grouped_images.values())} reference images across {len(width_grouped_images)} width groups")
        
        # Process images
        for img_path in tqdm(paths_to_process, desc=f"Augmenting {folder_name}"):
            try:
                img = Image.open(img_path).convert('RGB')
                aspect = get_aspect_ratio(img)
                img_width = img.width
                
                stats['total'] += 1
                
                # Check if augmentation needed
                if is_acceptably_square(aspect, aspect_threshold):
                    stats['skipped_square'] += 1
                    continue
                
                # Augment image
                augmented_img = augment_to_square(img, width_grouped_images, img_width, strip_percentage)
                
                # Save augmented image
                output_path = output_folder / img_path.name
                augmented_img.save(output_path, quality=95)
                
                stats['augmented'] += 1
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                stats['errors'] += 1
                continue
    
    # Print summary
    print("\n" + "="*50)
    print("AUGMENTATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {stats['total']}")
    print(f"Already square (skipped): {stats['skipped_square']}")
    print(f"Successfully augmented: {stats['augmented']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nAspect ratio threshold: {aspect_threshold:.2f}")
    print(f"Strip percentage used: {strip_percentage*100:.1f}%")
    print(f"Images with aspect ≤ {aspect_threshold:.2f} were kept as-is")

def main():
    parser = argparse.ArgumentParser(
        description='Augment fish crop images by adding background to make them square'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset root folder (containing small/medium/wide folders)'
    )
    parser.add_argument(
        '--num',
        type=str,
        default='all',
        help='Number of images to process per folder (default: all)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.15,
        help='Aspect ratio threshold - images with aspect ≤ this are kept as-is (default: 1.15)'
    )
    parser.add_argument(
        '--strip-percent',
        type=float,
        default=14.0,
        help='Percentage of image height to use as source strip for stretching (default: 14.0)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print(f"Dataset path: {dataset_path}")
    print(f"Processing: {args.num} images per folder")
    print(f"Aspect ratio threshold: {args.threshold}")
    print(f"Strip percentage: {args.strip_percent}%")
    print("\nStarting augmentation process...")
    
    process_dataset(dataset_path, args.num, args.threshold, args.strip_percent / 100.0)

if __name__ == "__main__":
    main()
