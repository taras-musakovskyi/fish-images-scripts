import os
from pathlib import Path
from PIL import Image
import math

# Configuration
SOURCE_DIR = Path("/Users/tarasmusakovskyi/Downloads/shutterstock")
OUTPUT_DIR = SOURCE_DIR / "processed"

# Target resolutions
FULL_SCENE_SIZE = 1024
CONTEXTUAL_SIZE = 768
DETAIL_SIZE = 768

# Base repeat weights (will be adjusted per image)
BASE_WEIGHTS = {
    'full': 8,
    'contextual': 4,
    'detail': 2
}

def calculate_adaptive_weights(width, height):
    """
    Calculate repeat counts based on image size.
    Larger images = more crops justified.
    """
    total_pixels = width * height
    megapixels = total_pixels / 1_000_000
    
    # Reference: 4K image (~8MP) as baseline
    size_factor = megapixels / 8.0
    
    # Adaptive weights
    weights = {
        'full': max(1, round(BASE_WEIGHTS['full'] * min(size_factor, 1.5))),
        'contextual': max(1, round(BASE_WEIGHTS['contextual'] * size_factor)),
        'detail': max(1, round(BASE_WEIGHTS['detail'] * size_factor))
    }
    
    return weights, megapixels

def calculate_crop_counts(width, height, megapixels):
    """
    Determine how many contextual and detail crops to make.
    Simple version with conservative counts.
    """
    # Contextual crops: larger images get more
    if megapixels > 12:  # >12MP (e.g., 4500x3000)
        contextual_count = 4
    elif megapixels > 8:  # 8-12MP
        contextual_count = 3
    else:  # <8MP
        contextual_count = 2
    
    # Detail crops: based on available resolution
    min_dim = min(width, height)
    if min_dim >= 3000:
        detail_count = 4
    elif min_dim >= 2000:
        detail_count = 3
    else:
        detail_count = 2
    
    return contextual_count, detail_count

def create_full_scene(img, size=FULL_SCENE_SIZE):
    """
    Resize entire image to target size, preserving aspect ratio with center crop.
    """
    # Calculate dimensions for center crop to square
    width, height = img.size
    crop_size = min(width, height)
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
    
    return img_resized

def create_contextual_crops(img, count, output_base, size=CONTEXTUAL_SIZE):
    """
    Create large contextual crops that maintain scene coherence.
    Strategy: divide image into overlapping regions.
    """
    width, height = img.size
    crops = []
    
    # Target crop size before resize (larger to maintain context)
    crop_dim = int(min(width, height) * 0.7)  # 70% of smaller dimension
    
    if count == 2:
        # Left and right halves (or top/bottom if portrait)
        if width >= height:
            positions = [
                (0, height // 2 - crop_dim // 2),
                (width - crop_dim, height // 2 - crop_dim // 2)
            ]
        else:
            positions = [
                (width // 2 - crop_dim // 2, 0),
                (width // 2 - crop_dim // 2, height - crop_dim)
            ]
    
    elif count == 3:
        # Center + two corners
        positions = [
            (width // 2 - crop_dim // 2, height // 2 - crop_dim // 2),  # Center
            (0, 0),  # Top-left
            (width - crop_dim, height - crop_dim)  # Bottom-right
        ]
    
    else:  # count == 4
        # Four quadrants with overlap
        positions = [
            (0, 0),  # Top-left
            (width - crop_dim, 0),  # Top-right
            (0, height - crop_dim),  # Bottom-left
            (width - crop_dim, height - crop_dim)  # Bottom-right
        ]
    
    for i, (x, y) in enumerate(positions[:count]):
        # Ensure within bounds
        x = max(0, min(x, width - crop_dim))
        y = max(0, min(y, height - crop_dim))
        
        crop = img.crop((x, y, x + crop_dim, y + crop_dim))
        crop_resized = crop.resize((size, size), Image.Resampling.LANCZOS)
        crops.append((f"contextual_{i+1:02d}_old", crop_resized))
    
    return crops

def create_detail_crops(img, count, output_base, size=DETAIL_SIZE):
    """
    Create native resolution detail crops focusing on interesting areas.
    Strategy: sample from different regions at full resolution.
    """
    width, height = img.size
    crops = []
    
    # If image is smaller than detail size, skip or resize
    if min(width, height) < size:
        return []
    
    # Strategic positions for detail crops
    positions = []
    margin = size // 2
    
    # Always include center
    positions.append((width // 2 - size // 2, height // 2 - size // 2))
    
    if count >= 2:
        # Add top-left
        positions.append((margin, margin))
    
    if count >= 3:
        # Add bottom-right
        positions.append((width - size - margin, height - size - margin))
    
    if count >= 4:
        # Add top-right
        positions.append((width - size - margin, margin))
    
    for i, (x, y) in enumerate(positions[:count]):
        # Ensure within bounds
        x = max(0, min(x, width - size))
        y = max(0, min(y, height - size))
        
        crop = img.crop((x, y, x + size, y + size))
        crops.append((f"detail_{i+1:02d}_old", crop))
    
    return crops

def process_image(img_path):
    """
    Process a single image: create full scene, contextual, and detail versions.
    """
    print(f"\nProcessing: {img_path.name}")
    
    # Open image
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    print(f"  Original size: {width}x{height}")
    
    # Calculate adaptive weights and crop counts
    weights, megapixels = calculate_adaptive_weights(width, height)
    contextual_count, detail_count = calculate_crop_counts(width, height, megapixels)
    
    print(f"  Megapixels: {megapixels:.1f}MP")
    print(f"  Repeat weights - Full: {weights['full']}x, Contextual: {weights['contextual']}x, Detail: {weights['detail']}x")
    print(f"  Creating - Contextual: {contextual_count}, Detail: {detail_count}")
    
    # Create output directories
    img_name = img_path.stem
    img_output_dir = OUTPUT_DIR / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track what we created
    created_files = []
    
    # 1. Full scene version
    full_scene = create_full_scene(img)
    full_scene_path = img_output_dir / f"{img_name}_full_scene_old.jpg"
    full_scene.save(full_scene_path, quality=95)
    created_files.append(('full', full_scene_path, weights['full']))
    print(f"  ✓ Created full scene: {full_scene_path.name}")
    
    # 2. Contextual crops
    contextual_crops = create_contextual_crops(img, contextual_count, img_name)
    for name, crop in contextual_crops:
        crop_path = img_output_dir / f"{img_name}_{name}.jpg"
        crop.save(crop_path, quality=95)
        created_files.append(('contextual', crop_path, weights['contextual']))
    print(f"  ✓ Created {len(contextual_crops)} contextual crops")
    
    # 3. Detail crops
    detail_crops = create_detail_crops(img, detail_count, img_name)
    for name, crop in detail_crops:
        crop_path = img_output_dir / f"{img_name}_{name}.jpg"
        crop.save(crop_path, quality=95)
        created_files.append(('detail', crop_path, weights['detail']))
    print(f"  ✓ Created {len(detail_crops)} detail crops")
    
    return created_files, weights

def create_summary(all_files, image_weights):
    """
    Create a summary text file with statistics and repeat information.
    """
    summary_path = OUTPUT_DIR / "processing_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SHUTTERSTOCK IMAGE PROCESSING SUMMARY (SIMPLE VERSION)\n")
        f.write("=" * 70 + "\n\n")
        
        total_full = sum(1 for t, _, _ in all_files if t == 'full')
        total_contextual = sum(1 for t, _, _ in all_files if t == 'contextual')
        total_detail = sum(1 for t, _, _ in all_files if t == 'detail')
        
        f.write(f"Total images processed: {len(image_weights)}\n")
        f.write(f"Total files created: {len(all_files)}\n")
        f.write(f"  - Full scenes: {total_full}\n")
        f.write(f"  - Contextual crops: {total_contextual}\n")
        f.write(f"  - Detail crops: {total_detail}\n\n")
        
        # Calculate effective training exposures
        total_exposures = 0
        for img_idx, (img_name, weights) in enumerate(image_weights.items(), 1):
            img_files = [f for f in all_files if img_name in str(f[1])]
            
            full_count = sum(1 for t, _, _ in img_files if t == 'full')
            contextual_count = sum(1 for t, _, _ in img_files if t == 'contextual')
            detail_count = sum(1 for t, _, _ in img_files if t == 'detail')
            
            img_exposures = (
                full_count * weights['full'] +
                contextual_count * weights['contextual'] +
                detail_count * weights['detail']
            )
            total_exposures += img_exposures
            
            f.write(f"\n{img_idx}. {img_name}:\n")
            f.write(f"   Weights: Full={weights['full']}x, Contextual={weights['contextual']}x, Detail={weights['detail']}x\n")
            f.write(f"   Files: {full_count} full, {contextual_count} contextual, {detail_count} detail\n")
            f.write(f"   Effective exposures: {img_exposures}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"TOTAL EFFECTIVE TRAINING EXPOSURES: {total_exposures}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("1. Review all processed images in subfolders\n")
        f.write("2. Delete any unsatisfactory crops\n")
        f.write("3. Run augmentation script to apply transforms\n")
        f.write("4. Use repeat weights during LoRA training\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    return total_exposures

def main():
    """
    Main processing function.
    """
    print("=" * 70)
    print("SHUTTERSTOCK IMAGE PROCESSOR (SIMPLE VERSION)")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"\nNo images found in {SOURCE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    all_files = []
    image_weights = {}
    
    # Process each image
    for img_path in sorted(image_files):
        try:
            created, weights = process_image(img_path)
            all_files.extend(created)
            image_weights[img_path.stem] = weights
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    # Create summary
    total_exposures = create_summary(all_files, image_weights)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files created: {len(all_files)}")
    print(f"Effective training exposures: {total_exposures}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nReview the images and delete any you don't want to use.")
    print("Then proceed with augmentation.")

if __name__ == "__main__":
    main()
