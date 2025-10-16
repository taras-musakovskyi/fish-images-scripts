import os
import zipfile
from tqdm import tqdm

CACHE_ZIP = "r_caustic_dataset.zip"
TARGET_DIR = "r_caustic_indoor"
INCLUDE_FOLDERS = ["dataset_1_L/"]  # likely wall/building-like caustics
EXCLUDE_SUFFIXES = [".cont.jpg", ".diff.jpg", ".bin.jpg"]

def main():
    if not os.path.exists(CACHE_ZIP):
        raise FileNotFoundError(f"{CACHE_ZIP} not found. Run the main downloader first.")

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Scanning archive: {CACHE_ZIP}")

    with zipfile.ZipFile(CACHE_ZIP) as z:
        all_files = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        # Filter likely indoor ones
        selected = []
        for f in all_files:
            fl = f.lower()
            if any(sub in fl for sub in INCLUDE_FOLDERS) and not any(fl.endswith(s) for s in EXCLUDE_SUFFIXES):
                selected.append(f)

        print(f"Found {len(selected)} likely indoor/building-related images.")

        for name in tqdm(selected, desc="Extracting indoor-like images", unit="img"):
            target_path = os.path.join(TARGET_DIR, os.path.basename(name))
            with z.open(name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    print(f"\n✅ Extracted {len(selected)} images to {TARGET_DIR}")


if __name__ == "__main__":
    main()

