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

def extract_background_strip(img, position='top', strip_height=50):
    """Extract background strip from top or bottom of image"""
    if position == 'top':
        return img.crop((0, 0, img.width, strip_height))
    else:  # bottom
        return img.crop((0, img.height - strip_height, img.width, img.height))

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

def augment_to_square(img, width_grouped_images, img_width):
    """
    Augment image by adding background to top/bottom to make it square
    
    Args:
        img: PIL Image to augment
        width_grouped_images: Dict of images grouped by width
        img_width: Width of current image for finding similar images
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
    
    # Extract background strips
    strip_height = min(height // 4, 50)  # Use up to 25% of image or 50px
    
    # 78% use own background, 22% use similar-width image from same folder
    if random.random() < 0.78:
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
    
    top_strip = extract_background_strip(bg_source, 'top', strip_height)
    bottom_strip = extract_background_strip(bg_source, 'bottom', strip_height)
    
    # Create new canvas
    new_img = Image.new('RGB', (width, target_height))
    
    # Fill top section by tiling top strip
    current_y = 0
    while current_y < top_add:
        tile_height = min(strip_height, top_add - current_y)
        tile = top_strip.crop((0, 0, width, tile_height))
        new_img.paste(tile, (0, current_y))
        current_y += tile_height
    
    # Paste original image in middle
    new_img.paste(img, (0, top_add))
    
    # Fill bottom section by tiling bottom strip
    current_y = top_add + height
    while current_y < target_height:
        tile_height = min(strip_height, target_height - current_y)
        tile = bottom_strip.crop((0, strip_height - tile_height, width, strip_height))
        new_img.paste(tile, (0, current_y))
        current_y += tile_height
    
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

def process_dataset(dataset_root, num_to_process='all', aspect_threshold=1.15):
    """
    Process dataset and create augmented versions
    
    Args:
        dataset_root: Path to dataset root (contains small/medium/wide folders)
        num_to_process: 'all' or integer number of images to process per folder
        aspect_threshold: Aspect ratio threshold for considering image "square enough"
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
                augmented_img = augment_to_square(img, width_grouped_images, img_width)
                
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
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print(f"Dataset path: {dataset_path}")
    print(f"Processing: {args.num} images per folder")
    print(f"Aspect ratio threshold: {args.threshold}")
    print("\nStarting augmentation process...")
    
    process_dataset(dataset_path, args.num, args.threshold)

if __name__ == "__main__":
    main()
