from pathlib import Path
from PIL import Image, ImageEnhance
import random
import numpy as np

# Configuration
SOURCE_DIR = Path("/Users/tarasmusakovskyi/Downloads/my-tank-underwater-scenes/frames")
OUTPUT_DIR = Path("/Users/tarasmusakovskyi/Downloads/my-tank-underwater-scenes/my-tank-underwater-scenes-augmented")

# Moderate augmentation for tank images
TANK_VERSIONS = 4  # 3-4 versions per frame

def augment_tank_frame(img, version):
    """
    Moderate augmentation for tank frames.
    Balance between preserving real underwater quality and adding variety.
    """
    aug_img = img.copy()
    
    # Horizontal flip
    if version % 2 == 0:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Brightness (±12%)
    brightness_factor = random.uniform(0.88, 1.12)
    enhancer = ImageEnhance.Brightness(aug_img)
    aug_img = enhancer.enhance(brightness_factor)
    
    # Rotation (±5 degrees)
    angle = random.uniform(-5, 5)
    aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Contrast adjustment (±10%)
    contrast_factor = random.uniform(0.90, 1.10)
    enhancer = ImageEnhance.Contrast(aug_img)
    aug_img = enhancer.enhance(contrast_factor)
    
    # Saturation adjustment (±10%)
    sat_factor = random.uniform(0.90, 1.10)
    enhancer = ImageEnhance.Color(aug_img)
    aug_img = enhancer.enhance(sat_factor)
    
    # Color temperature shift (moderate)
    if version % 3 == 0:
        # Warm shift - increase red
        img_array = np.array(aug_img, dtype=np.float32)
        shift = random.uniform(1.03, 1.08)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * shift, 0, 255)
        aug_img = Image.fromarray(img_array.astype(np.uint8))
    elif version % 3 == 1:
        # Cool shift - increase blue
        img_array = np.array(aug_img, dtype=np.float32)
        shift = random.uniform(1.03, 1.08)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * shift, 0, 255)
        aug_img = Image.fromarray(img_array.astype(np.uint8))
    
    return aug_img

def process_frame(img_path):
    """
    Process a single tank frame.
    """
    print(f"  Processing: {img_path.name}")
    
    # Open image
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save original copy
    base_name = img_path.stem
    ext = img_path.suffix
    original_output = OUTPUT_DIR / f"{base_name}_v00{ext}"
    img.save(original_output, quality=95)
    
    # Generate augmented versions
    for i in range(1, TANK_VERSIONS):
        aug_img = augment_tank_frame(img, i)
        output_path = OUTPUT_DIR / f"{base_name}_v{i:02d}{ext}"
        aug_img.save(output_path, quality=95)
    
    return TANK_VERSIONS

def main():
    """
    Main augmentation function for tank frames.
    """
    print("=" * 70)
    print("TANK FRAMES AUGMENTATION")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"\nNo images found in {SOURCE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} frames to process")
    print(f"Creating {TANK_VERSIONS} versions per frame\n")
    
    # Process all frames
    total_created = 0
    for img_path in sorted(image_files):
        try:
            count = process_frame(img_path)
            total_created += count
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Source frames: {len(image_files)}")
    print(f"Total images created: {total_created}")
    print(f"Expected: {len(image_files) * TANK_VERSIONS}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nAugmentation strategy: Moderate")
    print("  - Brightness: ±12%")
    print("  - Rotation: ±5°")
    print("  - Contrast: ±10%")
    print("  - Saturation: ±10%")
    print("  - Color temperature shifts (moderate)")

if __name__ == "__main__":
    main()
