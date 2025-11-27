#!/usr/bin/env python3
"""
Download training images from the web for rock-paper-scissors gestures
Uses DuckDuckGo image search (no API key needed)
"""

import os
import requests
from PIL import Image
import io
import time
from urllib.parse import urlparse

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Installing duckduckgo-search...")
    import subprocess
    subprocess.check_call(["pip", "install", "duckduckgo-search"])
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True

def validate_image(img):
    """Basic validation to ensure image is suitable for training"""
    if img is None:
        return False
    
    # Check minimum size (too small images are not useful)
    if img.width < 100 or img.height < 100:
        return False
    
    # Check aspect ratio (very extreme ratios might not be hand gestures)
    aspect_ratio = max(img.width, img.height) / min(img.width, img.height)
    if aspect_ratio > 5:  # Too wide or too tall
        return False
    
    return True

def download_image(url, timeout=5):
    """Download an image from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if it's an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return None
        
        # Load and verify it's a valid image
        img_data = response.content
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (keep aspect ratio)
        max_size = 800
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Validate image quality
        if not validate_image(img):
            return None
        
        return img
    except Exception as e:
        return None

def download_images_for_class(class_name, num_images=200):
    """Download images for a specific class"""
    print(f"\n{'='*50}")
    print(f"Downloading {num_images} images for: {class_name.upper()}")
    print(f"{'='*50}")
    
    # Create folder if it doesn't exist
    os.makedirs(f"data/{class_name}", exist_ok=True)
    
    # Search queries - specific hand gesture variations for rock-paper-scissors
    if class_name == "rock":
        search_queries = [
            "rock hand gesture fingers closed fist",
            "rock paper scissors rock hand sign",
            "rock hand gesture closed fist",
            "rock hand pose fist closed",
            "rock hand gesture isolated",
            "rock hand sign white background",
            "rock gesture hand close up",
            "rock hand gesture tutorial"
        ]
    elif class_name == "paper":
        search_queries = [
            "paper hand gesture palm open flat",
            "rock paper scissors paper hand sign",
            "paper hand gesture open palm",
            "paper hand pose flat palm",
            "paper hand gesture isolated",
            "paper hand sign white background",
            "paper gesture hand close up",
            "paper hand gesture tutorial"
        ]
    elif class_name == "scissors":
        search_queries = [
            "scissors hand gesture two fingers",
            "rock paper scissors scissors hand sign",
            "scissors hand gesture peace sign",
            "scissors hand pose two fingers",
            "scissors hand gesture isolated",
            "scissors hand sign white background",
            "scissors gesture hand close up",
            "scissors hand gesture tutorial"
        ]
    else:
        search_queries = [
            f"{class_name} hand gesture",
            f"{class_name} hand sign",
            f"rock paper scissors {class_name}",
            f"{class_name} hand pose"
        ]
    
    downloaded = 0
    skipped = 0
    seen_urls = set()
    
    # Get existing count
    existing_count = len([f for f in os.listdir(f"data/{class_name}") if f.endswith('.jpg')])
    start_count = existing_count
    
    print(f"Starting from {start_count} existing images...")
    
    with DDGS() as ddgs:
        for query in search_queries:
            if downloaded >= num_images:
                break
                
            print(f"\nSearching: '{query}'...")
            
            try:
                # Add delay to avoid rate limits
                time.sleep(2)
                
                # Search for images
                results = list(ddgs.images(
                    keywords=query,
                    max_results=min(100, num_images - downloaded + 20)  # Get more results per query
                ))
                
                for result in results:
                    if downloaded >= num_images:
                        break
                    
                    url = result.get('image', '')
                    if not url or url in seen_urls:
                        continue
                    
                    seen_urls.add(url)
                    
                    # Download image
                    img = download_image(url)
                    if img:
                        # Save image
                        filename = f"data/{class_name}/{start_count + downloaded}.jpg"
                        img.save(filename, 'JPEG', quality=85)
                        downloaded += 1
                        
                        if downloaded % 10 == 0:
                            print(f"  Downloaded {downloaded}/{num_images} images...")
                        
                        # Small delay to be respectful
                        time.sleep(0.5)  # Increased delay to avoid rate limits
                    else:
                        skipped += 1
                        
            except Exception as e:
                error_msg = str(e)
                if 'Ratelimit' in error_msg or '403' in error_msg or '202' in error_msg:
                    print(f"  Rate limited. Waiting 10 seconds before next search...")
                    time.sleep(10)
                else:
                    print(f"  Error searching '{query}': {e}")
                continue
    
    print(f"\n✓ Finished downloading for {class_name}")
    print(f"  Downloaded: {downloaded} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Total in folder: {start_count + downloaded} images")
    
    return downloaded

def main():
    print("="*60)
    print("Rock-Paper-Scissors Training Image Downloader")
    print("="*60)
    print("\nThis script downloads hand gesture images from the web for training.")
    print("It searches for specific rock, paper, and scissors hand gestures.\n")
    
    classes = ["rock", "paper", "scissors"]
    images_per_class = 200
    
    print(f"Target: {images_per_class} images per class")
    print("\nNote: This may take several minutes due to rate limiting.")
    print("The script will automatically handle delays and retries.\n")
    
    # Show current counts
    print("Current image counts:")
    for cls in classes:
        if os.path.exists(f"data/{cls}"):
            count = len([f for f in os.listdir(f"data/{cls}") if f.endswith('.jpg')])
            print(f"  {cls}: {count} images")
        else:
            print(f"  {cls}: 0 images (folder will be created)")
    
    print("\nStarting download...")
    print("="*60)
    
    total_downloaded = 0
    for cls in classes:
        downloaded = download_images_for_class(cls, images_per_class)
        total_downloaded += downloaded
        time.sleep(2)  # Brief pause between classes to avoid rate limits
    
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"Total new images downloaded: {total_downloaded}")
    print(f"{'='*60}")
    
    # Show summary
    print("\nFinal counts:")
    for cls in classes:
        if os.path.exists(f"data/{cls}"):
            count = len([f for f in os.listdir(f"data/{cls}") if f.endswith('.jpg')])
            print(f"  {cls}: {count} images")
        else:
            print(f"  {cls}: 0 images")
    
    print("\n" + "="*60)
    print("Next step: Run 'python3 train_model.py' to train the model!")
    print("="*60)

if __name__ == "__main__":
    main()

