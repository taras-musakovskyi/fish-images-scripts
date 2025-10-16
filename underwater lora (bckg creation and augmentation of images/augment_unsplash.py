from pathlib import Path
from PIL import Image, ImageEnhance
import random
import numpy as np

# Configuration
SOURCE_DIR = Path("/Users/tarasmusakovskyi/Downloads/flooded_apartments_unsplash")

OUTPUT_DIR = SOURCE_DIR / "augmented-unsplash"

# More aggressive augmentation for free sources
UNSPLASH_VERSIONS = 5  # More variety needed from each image

def augment_unsplash(img, version):
    """
    More aggressive augmentation for Unsplash images.
    Generate variety from free sources.
    """
    aug_img = img.copy()
    
    # Horizontal flip
    if version % 2 == 0:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Brightness (±15% - more aggressive)
    brightness_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Brightness(aug_img)
    aug_img = enhancer.enhance(brightness_factor)
    
    # Rotation (±8 degrees - more than Shutterstock)
    angle = random.uniform(-8, 8)
    aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Contrast adjustment (±15%)
    contrast_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Contrast(aug_img)
    aug_img = enhancer.enhance(contrast_factor)
    
    # Saturation adjustment (±20%)
    sat_factor = random.uniform(0.80, 1.20)
    enhancer = ImageEnhance.Color(aug_img)
    aug_img = enhancer.enhance(sat_factor)
    
    # Color temperature shift (vectorized with numpy)
    if version % 3 == 0:
        # Warm shift - increase red
        img_array = np.array(aug_img, dtype=np.float32)
        shift = random.uniform(1.05, 1.15)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * shift, 0, 255)
        aug_img = Image.fromarray(img_array.astype(np.uint8))
    elif version % 3 == 1:
        # Cool shift - increase blue
        img_array = np.array(aug_img, dtype=np.float32)
        shift = random.uniform(1.05, 1.15)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * shift, 0, 255)
        aug_img = Image.fromarray(img_array.astype(np.uint8))
    
    return aug_img

def process_image(img_path):
    """
    Process a single Unsplash image.
    """
    print(f"  Processing: {img_path.name}")
    
    # Open image
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 768px (shorter side) for faster processing
    width, height = img.size
    if min(width, height) > 768:
        if width < height:
            new_width = 768
            new_height = int(height * (768 / width))
        else:
            new_height = 768
            new_width = int(width * (768 / height))
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Save original copy
    base_name = img_path.stem
    ext = img_path.suffix
    original_output = OUTPUT_DIR / f"{base_name}_v00{ext}"
    img.save(original_output, quality=95)
    
    # Generate augmented versions
    for i in range(1, UNSPLASH_VERSIONS):
        aug_img = augment_unsplash(img, i)
        output_path = OUTPUT_DIR / f"{base_name}_v{i:02d}{ext}"
        aug_img.save(output_path, quality=95)
    
    return UNSPLASH_VERSIONS

def main():
    """
    Main augmentation function for Unsplash images.
    """
    print("=" * 70)
    print("UNSPLASH IMAGE AUGMENTATION")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"\nNo images found in {SOURCE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Creating {UNSPLASH_VERSIONS} versions per image\n")
    
    # Process all images
    total_created = 0
    for img_path in sorted(image_files):
        try:
            count = process_image(img_path)
            total_created += count
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Source images: {len(image_files)}")
    print(f"Total images created: {total_created}")
    print(f"Expected: {len(image_files) * UNSPLASH_VERSIONS}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nAugmentation strategy: Aggressive")
    print("  - Brightness: ±15%")
    print("  - Rotation: ±8°")
    print("  - Contrast: ±15%")
    print("  - Saturation: ±20%")
    print("  - Color temperature shifts")

if __name__ == "__main__":
    main()
