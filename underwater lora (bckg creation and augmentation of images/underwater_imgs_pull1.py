import os
import io
import zipfile
import random
import requests
from tqdm import tqdm

RECORD_ID = "6467282"  # R-CAUSTIC dataset
TARGET_DIR = "r_caustic_samples"
CACHE_ZIP = "r_caustic_dataset.zip"
MAX_IMAGES = 50


def download_with_progress(url, dest_path):
    """Download file with a progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading ZIP",
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{RECORD_ID}"
    print(f"Fetching metadata from: {api_url}")
    r = requests.get(api_url)
    r.raise_for_status()
    data = r.json()

    files = data.get("files", [])
    zip_files = [f for f in files if f["key"].lower().endswith(".zip")]
    if not zip_files:
        raise RuntimeError("No ZIP files found in record metadata.")

    zip_url = zip_files[0]["links"]["self"]

    # Cache check
    if os.path.exists(CACHE_ZIP):
        print(f"✅ Using cached archive: {CACHE_ZIP}")
    else:
        print(f"Downloading ZIP from {zip_url}")
        download_with_progress(zip_url, CACHE_ZIP)
        print("✅ Download complete, cached locally.")

    # Extract subset of images
    print("Extracting images...")
    with zipfile.ZipFile(CACHE_ZIP) as z:
        image_files = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Archive contains {len(image_files)} images total.")
        random.shuffle(image_files)
        subset = image_files[:MAX_IMAGES]

        for name in tqdm(subset, desc="Extracting images", unit="img"):
            target_path = os.path.join(TARGET_DIR, os.path.basename(name))
            with z.open(name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    print(f"\n✅ Extracted {len(subset)} random images to {TARGET_DIR}")


if __name__ == "__main__":
    main()

