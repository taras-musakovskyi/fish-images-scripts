from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

# Configuration
DATASET_DIR = Path("/Users/tarasmusakovskyi/Downloads")
SIZE_FOLDERS = ['small', 'medium', 'wide']

def extract_species_from_filename(filename):
    """
    Extract species name from YOLO crop filename.
    Format: {species}_{timestamp}_{number}_conf{confidence}.jpg
    Example: guppy_female_20250926_090114_936_conf0.688.jpg
    """
    # Remove extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    # Find where timestamp starts (8-digit number)
    species_parts = []
    for part in parts:
        if part.isdigit() and len(part) == 8:
            # Found timestamp, stop here
            break
        species_parts.append(part)
    
    if not species_parts:
        return 'Unknown'
    
    # Join species parts and format
    species_name = ' '.join(species_parts)
    
    # Title case each word
    species_name = ' '.join(word.capitalize() for word in species_name.split())
    
    return species_name

def analyze_size_folder(folder_path):
    """
    Analyze images in a single size folder.
    Returns: dict with dimensions, species distribution, etc.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        return None
    
    # Collect statistics
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    species_counts = defaultdict(int)
    
    print(f"  Analyzing {len(image_files)} images...")
    
    for img_path in image_files:
        try:
            # Get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                areas.append(width * height)
                aspect_ratios.append(width / height if height > 0 else 0)
            
            # Get species
            species = extract_species_from_filename(img_path.name)
            species_counts[species] += 1
            
        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")
    
    # Calculate statistics
    return {
        'count': len(image_files),
        'dimensions': {
            'width': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'median': np.median(widths),
            },
            'height': {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'median': np.median(heights),
            },
            'area': {
                'min': min(areas),
                'max': max(areas),
                'mean': np.mean(areas),
                'median': np.median(areas),
            },
            'aspect_ratio': {
                'min': min(aspect_ratios),
                'max': max(aspect_ratios),
                'mean': np.mean(aspect_ratios),
                'median': np.median(aspect_ratios),
            }
        },
        'species': dict(species_counts)
    }

def print_statistics(stats, folder_name):
    """Pretty print statistics for a size folder."""
    print(f"\n{'='*70}")
    print(f"{folder_name.upper()} FOLDER")
    print('='*70)
    print(f"Total images: {stats['count']}")
    
    print(f"\nDimensions:")
    print(f"  Width:  {stats['dimensions']['width']['min']}-{stats['dimensions']['width']['max']} px "
          f"(avg: {stats['dimensions']['width']['mean']:.1f}, median: {stats['dimensions']['width']['median']:.1f})")
    print(f"  Height: {stats['dimensions']['height']['min']}-{stats['dimensions']['height']['max']} px "
          f"(avg: {stats['dimensions']['height']['mean']:.1f}, median: {stats['dimensions']['height']['median']:.1f})")
    print(f"  Area:   {stats['dimensions']['area']['min']}-{stats['dimensions']['area']['max']} px² "
          f"(avg: {stats['dimensions']['area']['mean']:.0f})")
    print(f"  Aspect: {stats['dimensions']['aspect_ratio']['min']:.2f}-{stats['dimensions']['aspect_ratio']['max']:.2f} "
          f"(avg: {stats['dimensions']['aspect_ratio']['mean']:.2f})")
    
    print(f"\nSpecies distribution:")
    sorted_species = sorted(stats['species'].items(), key=lambda x: x[1], reverse=True)
    for species, count in sorted_species:
        percentage = (count / stats['count']) * 100
        print(f"  {species:20s}: {count:4d} ({percentage:5.1f}%)")

def recommend_architecture(all_stats):
    """
    Recommend multi-scale VAE architecture based on dataset analysis.
    """
    print(f"\n{'='*70}")
    print("ARCHITECTURE RECOMMENDATIONS")
    print('='*70)
    
    total_images = sum(stats['count'] for stats in all_stats.values())
    print(f"\nTotal dataset size: {total_images:,} images")
    
    # Check if sizes are well-separated
    small_max_area = all_stats['small']['dimensions']['area']['max']
    medium_min_area = all_stats['medium']['dimensions']['area']['min']
    medium_max_area = all_stats['medium']['dimensions']['area']['max']
    wide_min_area = all_stats['wide']['dimensions']['area']['min']
    
    print(f"\nSize separation:")
    print(f"  Small:  up to {small_max_area:,.0f} px²")
    print(f"  Medium: {medium_min_area:,.0f} - {medium_max_area:,.0f} px²")
    print(f"  Wide:   from {wide_min_area:,.0f} px²")
    
    overlap_small_medium = small_max_area > medium_min_area
    overlap_medium_wide = medium_max_area > wide_min_area
    
    if overlap_small_medium or overlap_medium_wide:
        print(f"  ⚠️  Warning: Size categories overlap!")
    else:
        print(f"  ✓ Clean separation between size categories")
    
    print(f"\n{'='*70}")
    print("RECOMMENDED APPROACH FOR PART 5 (Fish → Apartment Inpainting):")
    print('='*70)
    
    print(f"\n📋 OPTION A: Three Separate VAEs (RECOMMENDED)")
    print(f"   Pros:")
    print(f"   ✓ Simplest to implement")
    print(f"   ✓ Each VAE specialized for its size range")
    print(f"   ✓ Easy to train (standard VAE architecture)")
    print(f"   ✓ Can mix latents from different VAEs")
    print(f"   Cons:")
    print(f"   ✗ Need to train 3 separate models")
    print(f"   ✗ Larger total model size")
    
    print(f"\n   Training:")
    print(f"   1. Train VAE-small on small folder (resize to 64×64 latent)")
    print(f"   2. Train VAE-medium on medium folder (resize to 128×128 latent)")
    print(f"   3. Train VAE-wide on wide folder (resize to 192×192 latent)")
    print(f"   4. For Part 5: Select VAE based on desired fish size in scene")
    
    print(f"\n📋 OPTION B: Single Multi-Input VAE")
    print(f"   Pros:")
    print(f"   ✓ One model handles all sizes")
    print(f"   ✓ Learns shared fish features across scales")
    print(f"   Cons:")
    print(f"   ✗ More complex architecture")
    print(f"   ✗ Harder to train (need careful resizing strategy)")
    
    print(f"\n   Architecture:")
    print(f"   - Resize all inputs to common size (e.g., 256×256)")
    print(f"   - Encoder with size-conditioning (embed size category)")
    print(f"   - Single shared latent space")
    print(f"   - For Part 5: Condition on desired fish size")
    
    print(f"\n📋 OPTION C: Hierarchical Multi-Scale VAE")
    print(f"   Pros:")
    print(f"   ✓ Most sophisticated - multiple latent scales")
    print(f"   ✓ Best quality preservation across sizes")
    print(f"   Cons:")
    print(f"   ✗ Very complex implementation")
    print(f"   ✗ Longest training time")
    print(f"   ✗ Overkill for your use case")
    
    print(f"\n{'='*70}")
    print(f"💡 MY RECOMMENDATION: Option A (Three Separate VAEs)")
    print('='*70)
    print(f"Why:")
    print(f"1. You have {total_images:,} images - enough for 3 separate VAEs")
    print(f"2. Clean size separation in your dataset")
    print(f"3. Simplest to implement and debug")
    print(f"4. For Part 5 inpainting: Pick VAE based on scene composition")
    print(f"   - Small fish far away → VAE-small")
    print(f"   - Medium fish mid-distance → VAE-medium")
    print(f"   - Large fish close-up → VAE-wide")
    print(f"5. Portfolio value: Shows you can train specialized models")
    
    print(f"\nEstimated training time per VAE: 2-4 hours on Colab")
    print(f"Total for 3 VAEs: 6-12 hours")

def main():
    """Main analysis function."""
    print("="*70)
    print("FISH DATASET ANALYSIS FOR MULTI-SCALE VAE")
    print("="*70)
    
    if not DATASET_DIR.exists():
        print(f"\n❌ Error: Dataset directory not found: {DATASET_DIR}")
        print("   Please update DATASET_DIR in the script")
        return
    
    # Analyze each size folder
    all_stats = {}
    for folder_name in SIZE_FOLDERS:
        folder_path = DATASET_DIR / folder_name
        
        if not folder_path.exists():
            print(f"\n⚠️  Warning: {folder_name} folder not found")
            continue
        
        print(f"\nAnalyzing {folder_name} folder...")
        stats = analyze_size_folder(folder_path)
        
        if stats:
            all_stats[folder_name] = stats
            print_statistics(stats, folder_name)
    
    if not all_stats:
        print("\n❌ No data found. Check your dataset path.")
        return
    
    # Provide recommendations
    recommend_architecture(all_stats)
    
    # Save results to JSON
    output_file = DATASET_DIR / "dataset_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
