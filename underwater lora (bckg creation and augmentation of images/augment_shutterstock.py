from pathlib import Path
from PIL import Image, ImageEnhance
import random
import re

# Configuration
SOURCE_DIR = Path("/Users/tarasmusakovskyi/Downloads/shutterstock/total")
OUTPUT_DIR = SOURCE_DIR / "augmented"

# Augmentation counts
ORIGINAL_VERSIONS = 10  # 8-10 for originals
CONTEXTUAL_VERSIONS = 3  # 2-3 for contextual
DETAIL_VERSIONS = 2      # 1-2 for detail

def identify_file_type(filename):
    """
    Identify if file is original, contextual, detail, or skip.
    Returns: 'original', 'contextual', 'detail', or 'skip'
    """
    # Skip full_scene files
    if 'full_scene' in filename:
        return 'skip'
    
    # Check if it's a detail or contextual crop
    if '_detail_' in filename or '_contextual_' in filename:
        if '_detail_' in filename:
            return 'detail'
        else:
            return 'contextual'
    
    # Original files: shutterstock_NUMBER.jpg (no additional underscores)
    pattern = r'^shutterstock_\d+\.jpg$'
    if re.match(pattern, filename):
        return 'original'
    
    return 'skip'

def augment_original(img, version):
    """
    Gentle augmentation for original full-res images.
    Preserves composition and quality.
    """
    aug_img = img.copy()
    
    # Horizontal flip (50% chance)
    if version % 2 == 0:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Brightness adjustment (±8%)
    brightness_factor = random.uniform(0.92, 1.08)
    enhancer = ImageEnhance.Brightness(aug_img)
    aug_img = enhancer.enhance(brightness_factor)
    
    # Color temperature shift (±150K equivalent)
    # Warm = more red, Cool = more blue
    if version % 3 == 0:
        color_shift = random.uniform(0.95, 1.05)
        # Adjust red channel slightly
        pixels = aug_img.load()
        width, height = aug_img.size
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                r = int(min(255, r * color_shift))
                pixels[x, y] = (r, g, b)
    
    # Slight contrast adjustment
    contrast_factor = random.uniform(0.95, 1.05)
    enhancer = ImageEnhance.Contrast(aug_img)
    aug_img = enhancer.enhance(contrast_factor)
    
    return aug_img

def augment_contextual(img, version):
    """
    Moderate augmentation for contextual crops.
    """
    aug_img = img.copy()
    
    # Horizontal flip
    if version % 2 == 0:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Brightness (±10%)
    brightness_factor = random.uniform(0.90, 1.10)
    enhancer = ImageEnhance.Brightness(aug_img)
    aug_img = enhancer.enhance(brightness_factor)
    
    # Rotation (±3 degrees)
    angle = random.uniform(-3, 3)
    aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Color adjustments
    if version % 2 == 1:
        # Saturation
        sat_factor = random.uniform(0.95, 1.05)
        enhancer = ImageEnhance.Color(aug_img)
        aug_img = enhancer.enhance(sat_factor)
    
    return aug_img

def augment_detail(img, version):
    """
    Minimal augmentation for detail crops.
    """
    aug_img = img.copy()
    
    # Just flip
    if version % 2 == 0:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Very subtle brightness
    brightness_factor = random.uniform(0.95, 1.05)
    enhancer = ImageEnhance.Brightness(aug_img)
    aug_img = enhancer.enhance(brightness_factor)
    
    return aug_img

def process_file(img_path):
    """
    Process a single file with appropriate augmentation strategy.
    """
    file_type = identify_file_type(img_path.name)
    
    if file_type == 'skip':
        return 0
    
    # Determine augmentation count
    if file_type == 'original':
        num_versions = ORIGINAL_VERSIONS
        aug_function = augment_original
        print(f"  Original: {img_path.name} -> {num_versions} versions")
    elif file_type == 'contextual':
        num_versions = CONTEXTUAL_VERSIONS
        aug_function = augment_contextual
        print(f"  Contextual: {img_path.name} -> {num_versions} versions")
    else:  # detail
        num_versions = DETAIL_VERSIONS
        aug_function = augment_detail
        print(f"  Detail: {img_path.name} -> {num_versions} versions")
    
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
    created_count = 1  # Count original
    for i in range(1, num_versions):
        aug_img = aug_function(img, i)
        output_path = OUTPUT_DIR / f"{base_name}_v{i:02d}{ext}"
        aug_img.save(output_path, quality=95)
        created_count += 1
    
    return created_count

def main():
    """
    Main augmentation function.
    """
    print("=" * 70)
    print("SHUTTERSTOCK IMAGE AUGMENTATION")
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
    
    print(f"\nFound {len(image_files)} files in source directory\n")
    
    # Categorize files
    originals = []
    contextuals = []
    details = []
    skipped = []
    
    for f in image_files:
        ftype = identify_file_type(f.name)
        if ftype == 'original':
            originals.append(f)
        elif ftype == 'contextual':
            contextuals.append(f)
        elif ftype == 'detail':
            details.append(f)
        else:
            skipped.append(f)
    
    print(f"File breakdown:")
    print(f"  Originals: {len(originals)}")
    print(f"  Contextual crops: {len(contextuals)}")
    print(f"  Detail crops: {len(details)}")
    print(f"  Skipped (full_scene): {len(skipped)}\n")
    
    # Process files
    total_created = 0
    
    print("Processing originals...")
    for img_path in sorted(originals):
        try:
            count = process_file(img_path)
            total_created += count
        except Exception as e:
            print(f"  ✗ Error: {img_path.name}: {e}")
    
    print("\nProcessing contextual crops...")
    for img_path in sorted(contextuals):
        try:
            count = process_file(img_path)
            total_created += count
        except Exception as e:
            print(f"  ✗ Error: {img_path.name}: {e}")
    
    print("\nProcessing detail crops...")
    for img_path in sorted(details):
        try:
            count = process_file(img_path)
            total_created += count
        except Exception as e:
            print(f"  ✗ Error: {img_path.name}: {e}")
    
    # Summary
    expected_from_originals = len(originals) * ORIGINAL_VERSIONS
    expected_from_contextual = len(contextuals) * CONTEXTUAL_VERSIONS
    expected_from_detail = len(details) * DETAIL_VERSIONS
    expected_total = expected_from_originals + expected_from_contextual + expected_from_detail
    
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Total images created: {total_created}")
    print(f"  From originals: ~{expected_from_originals}")
    print(f"  From contextual: ~{expected_from_contextual}")
    print(f"  From detail: ~{expected_from_detail}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nReady for LoRA training with repeat weights:")
    print("  Originals: 8x weight")
    print("  Contextual: 4x weight")
    print("  Detail: 2x weight")

if __name__ == "__main__":
    main()
