# utils/download_data.py — Fixed image downloader using actual image_url column
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from tqdm import tqdm
from config import IMAGE_DIR, CSV_TRAIN

def download_images(tsv_path, image_dir, max_images=2000):
    os.makedirs(image_dir, exist_ok=True)
    df = pd.read_csv(tsv_path, sep='\t')

    # Filter rows that actually have images
    df = df[df['hasImage'] == True].reset_index(drop=True)
    print(f"Total rows with images: {len(df)}")

    downloaded = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=min(len(df), max_images), desc="Downloading images"):
        if downloaded >= max_images:
            break

        post_id  = str(row.get('id', ''))
        img_url  = str(row.get('image_url', ''))
        img_path = os.path.join(image_dir, f"{post_id}.jpg")

        # Skip if already downloaded
        if os.path.exists(img_path):
            downloaded += 1
            continue

        if not img_url or img_url == 'nan':
            failed += 1
            continue

        try:
            resp = requests.get(img_url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(img_path, 'wb') as f:
                    f.write(resp.content)
                downloaded += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"\n✅ Downloaded: {downloaded} | ❌ Failed: {failed}")
    print(f"📁 Images saved to: {image_dir}")

if __name__ == "__main__":
    print("Downloading images using image_url column...")
    download_images(CSV_TRAIN, IMAGE_DIR, max_images=2000)