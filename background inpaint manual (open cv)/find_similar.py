#!/usr/bin/env python3
"""
Fast image-similarity grouping using perceptual hashing (pHash)
with progress logs.
"""

import os
import argparse
from PIL import Image
import imagehash
from itertools import combinations
from tqdm import tqdm

def hamming_similarity(hash1, hash2):
    """Return similarity in percent based on Hamming distance."""
    dist = hash1 - hash2
    max_bits = len(hash1.hash) ** 2
    return 100 * (1 - dist / max_bits)

def group_similar_hashes(img_hashes, threshold):
    """Group images whose hash similarity ≥ threshold."""
    groups = []
    used = set()
    items = list(img_hashes.keys())
    total_pairs = len(items) * (len(items) - 1) // 2

    print(f"Comparing {len(items)} images (~{total_pairs:,} pairs)...")

    for img1, img2 in tqdm(combinations(items, 2), total=total_pairs, desc="Grouping", unit="pair", dynamic_ncols=True):
        if img1 in used or img2 in used:
            continue
        sim = hamming_similarity(img_hashes[img1], img_hashes[img2])
        if sim >= threshold:
            for g in groups:
                if img1 in g or img2 in g:
                    g.update({img1, img2})
                    break
            else:
                groups.append({img1, img2})
            used.add(img2)

    # add ungrouped images
    grouped = {i for g in groups for i in g}
    for img in items:
        if img not in grouped:
            groups.append({img})
    return groups

def main():
    parser = argparse.ArgumentParser(
        description="Find visually similar images in a directory using perceptual hashing.")
    parser.add_argument("directory", help="Path to image directory")
    parser.add_argument("--threshold", type=float, default=96.0,
                        help="Similarity threshold in percent (default: 96)")
    args = parser.parse_args()

    # Collect image files
    image_files = [
        os.path.join(args.directory, f)
        for f in os.listdir(args.directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    total = len(image_files)
    if total == 0:
        print("No images found.")
        return

    print(f"Found {total} images in '{args.directory}'.")
    print("Computing perceptual hashes...")

    img_hashes = {}
    for path in tqdm(image_files, desc="Hashing", unit="img", dynamic_ncols=True):
        try:
            with Image.open(path) as img:
                img_hashes[path] = imagehash.phash(img)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    print("Hashing complete. Grouping similar images...")

    groups = group_similar_hashes(img_hashes, args.threshold)

    print("\n==== SUMMARY ====")
    print(f"Total images: {total}")
    print(f"Unique after grouping (≥{args.threshold}%): {len(groups)}")
    print(f"Duplicates detected: {total - len(groups)}")

if __name__ == "__main__":
    main()

