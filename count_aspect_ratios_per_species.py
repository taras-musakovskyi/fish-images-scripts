import os
import sys
from PIL import Image
from collections import defaultdict

# Groups of interest
GROUPS = [
    "ancistrus",
    "gold_fish",
    "gold_molly",
    "guppy_female",
    "guppy_male",
    "dalmatian_molly",
    "black_molly"
]

def get_aspect_ratio(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w / h if h else None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main(base_dir):
    directories = [os.path.join(base_dir, name) for name in ("small", "medium", "wide")]
    ratios = defaultdict(list)

    for d in directories:
        if not os.path.isdir(d):
            print(f"Skipping missing directory: {d}")
            continue

        for fname in os.listdir(d):
            if not fname.lower().endswith(".jpg"):
                continue

            for group in GROUPS:
                if fname.startswith(group):
                    ratio = get_aspect_ratio(os.path.join(d, fname))
                    if ratio:
                        ratios[group].append(ratio)
                    break

    print("\nAverage aspect ratios:")
    for group in GROUPS:
        if ratios[group]:
            avg = sum(ratios[group]) / len(ratios[group])
            print(f"{group:<20}: {avg:.3f}  (count={len(ratios[group])})")
        else:
            print(f"{group:<20}: no images found")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aspect_ratios.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    if not os.path.isdir(base_directory):
        print(f"Error: {base_directory} is not a valid directory.")
        sys.exit(1)

    main(base_directory)

