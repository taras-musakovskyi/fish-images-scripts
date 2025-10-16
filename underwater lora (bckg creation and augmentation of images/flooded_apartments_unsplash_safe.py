import os
import time
import requests
from tqdm import tqdm

# 🔑 Fill in your own Unsplash Access Key here
UNSPLASH_ACCESS_KEY = "McqsZe6SuzUtd7Nb_e4cw4knnhDjzEUFkXCLPqexxfA"

# Where to store results
OUTPUT_DIR = "flooded_apartments_unsplash"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keywords focused on indoor + water effects
QUERY_TERMS = [
    "underwater furniture",
    "sofa underwater",
    "caustics on wall",
    "aquarium reflection",
    "water reflections interior",
]

MAX_IMAGES = 150        # hard safety limit per run
DELAY_BETWEEN_CALLS = 1.2  # seconds between API requests (polite pacing)
HEADERS = {"Accept-Version": "v1", "User-Agent": "FloodedRoomsDataset/1.0"}

def search_unsplash(query, per_page=20, pages=3):
    """Yield image URLs gently, respecting Unsplash rate limits."""
    total_fetched = 0
    for page in range(1, pages + 1):
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "orientation": "landscape",
            "client_id": UNSPLASH_ACCESS_KEY,
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ API error ({r.status_code}): {r.text[:200]}")
            break

        results = r.json().get("results", [])
        if not results:
            break

        for res in results:
            yield res["urls"]["full"]
            total_fetched += 1
            if total_fetched >= per_page * pages:
                return
        time.sleep(DELAY_BETWEEN_CALLS)

def download_images():
    seen = set()
    total_downloaded = 0

    for term in QUERY_TERMS:
        print(f"\n🔹 Searching for '{term}'...")
        for img_url in tqdm(search_unsplash(term), desc=term):
            if img_url in seen:
                continue
            seen.add(img_url)
            fname = os.path.join(OUTPUT_DIR, os.path.basename(img_url.split("?")[0]))
            if os.path.exists(fname):
                continue

            try:
                resp = requests.get(img_url, stream=True, headers=HEADERS, timeout=30)
                if resp.status_code == 200:
                    with open(fname, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                    total_downloaded += 1
                    if total_downloaded >= MAX_IMAGES:
                        print(f"\n✅ Limit reached ({MAX_IMAGES} images).")
                        return
            except Exception as e:
                print(f"⚠️ Error downloading {img_url}: {e}")

            # Wait politely between downloads
            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n✅ Done! Total downloaded: {total_downloaded}")

if __name__ == "__main__":
    download_images()

