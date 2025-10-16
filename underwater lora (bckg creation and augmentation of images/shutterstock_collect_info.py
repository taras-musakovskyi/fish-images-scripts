from pathlib import Path
from PIL import Image
import os

# Configuration
SOURCE_DIR = Path("/Users/tarasmusakovskyi/Downloads/shutterstock")
OUTPUT_FILE = SOURCE_DIR / "image_info.txt"

def get_image_info(img_path):
    """
    Extract detailed information about an image.
    """
    img = Image.open(img_path)
    
    # Basic info
    width, height = img.size
    mode = img.mode
    format_type = img.format
    
    # File size
    file_size_bytes = os.path.getsize(img_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Megapixels
    megapixels = (width * height) / 1_000_000
    
    # DPI info
    dpi = img.info.get('dpi', (None, None))
    
    # Aspect ratio
    aspect_ratio = width / height
    
    return {
        'filename': img_path.name,
        'width': width,
        'height': height,
        'megapixels': megapixels,
        'aspect_ratio': aspect_ratio,
        'mode': mode,
        'format': format_type,
        'dpi': dpi,
        'file_size_mb': file_size_mb
    }

def main():
    print("Extracting image information...")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {SOURCE_DIR}")
        return
    
    # Collect info
    all_info = []
    for img_path in sorted(image_files):
        try:
            info = get_image_info(img_path)
            all_info.append(info)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Write to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SHUTTERSTOCK IMAGES - DETAILED INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total images: {len(all_info)}\n\n")
        
        # Individual image details
        for i, info in enumerate(all_info, 1):
            f.write(f"{i}. {info['filename']}\n")
            f.write(f"   Dimensions: {info['width']} × {info['height']} pixels\n")
            f.write(f"   Megapixels: {info['megapixels']:.2f} MP\n")
            f.write(f"   Aspect ratio: {info['aspect_ratio']:.2f}\n")
            f.write(f"   File size: {info['file_size_mb']:.2f} MB\n")
            f.write(f"   Color mode: {info['mode']}\n")
            f.write(f"   Format: {info['format']}\n")
            
            if info['dpi'][0]:
                f.write(f"   DPI: {info['dpi'][0]} × {info['dpi'][1]}\n")
            else:
                f.write(f"   DPI: Not specified\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        total_mp = sum(info['megapixels'] for info in all_info)
        avg_mp = total_mp / len(all_info)
        max_mp = max(info['megapixels'] for info in all_info)
        min_mp = min(info['megapixels'] for info in all_info)
        
        max_width = max(info['width'] for info in all_info)
        max_height = max(info['height'] for info in all_info)
        min_width = min(info['width'] for info in all_info)
        min_height = min(info['height'] for info in all_info)
        
        f.write(f"Average megapixels: {avg_mp:.2f} MP\n")
        f.write(f"Range: {min_mp:.2f} MP - {max_mp:.2f} MP\n\n")
        
        f.write(f"Width range: {min_width} - {max_width} pixels\n")
        f.write(f"Height range: {min_height} - {max_height} pixels\n\n")
        
        total_size = sum(info['file_size_mb'] for info in all_info)
        f.write(f"Total file size: {total_size:.2f} MB\n")
    
    print(f"\n✓ Image information saved to: {OUTPUT_FILE}")
    print(f"  Total images analyzed: {len(all_info)}")
    print(f"  Average resolution: {avg_mp:.2f} MP")

if __name__ == "__main__":
    main()
