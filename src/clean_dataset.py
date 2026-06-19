"""
clean_dataset.py  —  Dataset Quality Filtering Helper
======================================================
Scans a traffic sign image dataset (structured as class folders) to identify:
  1. Corrupt/unreadable images
  2. Extremely low-resolution images (< 16px)
  3. Aspect-ratio distorted images (width/height ratio > 3.0)
  4. Flat/solid-color boxes (grayscale pixel standard deviation < 8.0)

These poor-quality images are typically noise or corrupt crops that confuse
classifiers and lower accuracy.
"""

import os
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict

def scan_image(path, min_size=16, min_std=8.0, max_aspect=3.0):
    """
    Analyzes a single image for corruption, low size, flat colors, or distortion.
    Returns:
        is_bad (bool): True if the image is poor quality.
        reason (str): Reason for flagging the image, empty string if fine.
    """
    try:
        # Check 1: Corruption check (read header)
        with Image.open(path) as img:
            img.verify()
            
        # Re-open to read pixels for quality check (verify() closes the file pointer)
        with Image.open(path) as img:
            width, height = img.size
            
            # Check 2: Resolution check (extremely small images are useless noise)
            if width < min_size or height < min_size:
                return True, f"Too small ({width}x{height})"
                
            # Check 3: Aspect ratio distortion (signs are circular/square, aspect should be ~1.0)
            aspect = max(width, height) / max(1, min(width, height))
            if aspect > max_aspect:
                return True, f"Extreme aspect ratio ({aspect:.1f}:1)"
                
            # Check 4: Flat/blank image (detects solid grey, white or black squares)
            gray = img.convert("L")
            arr  = np.array(gray)
            std  = np.std(arr)
            if std < min_std:
                return True, f"Flat/blank color (std={std:.1f})"
                
        return False, ""
    except Exception as e:
        return True, f"Corrupt/unreadable ({type(e).__name__})"


def clean_dataset(data_dir, backup_dir=None, dry_run=True, min_size=16, min_std=8.0, max_aspect=3.0):
    """
    Scans data_dir, finds flagged images, and optionally moves them to backup_dir.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Source directory does not exist: {data_dir}")
        return

    if not dry_run and backup_dir is None:
        # Default backup directory: sibling folder of data_dir
        parent_dir = os.path.dirname(os.path.abspath(data_dir))
        backup_dir = os.path.join(parent_dir, os.path.basename(data_dir) + "_cleaned_backup")

    print("=" * 72)
    print(f"  Dataset Cleaner (Dry Run: {dry_run})")
    print(f"  Source   : {data_dir}")
    if not dry_run:
        print(f"  Backup   : {backup_dir}")
    print("-" * 72)
    print("  Scanning dataset files…")

    bad_files = []
    reason_counts = defaultdict(int)
    class_flagged_counts = defaultdict(int)
    total_scanned = 0

    # Walk directory
    for root, _, files in os.walk(data_dir):
        # Sort files to keep output stable
        for filename in sorted(files):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                total_scanned += 1
                filepath = os.path.join(root, filename)
                is_bad, reason = scan_image(filepath, min_size, min_std, max_aspect)
                
                if is_bad:
                    class_folder = os.path.basename(root)
                    bad_files.append((filepath, class_folder, filename, reason))
                    reason_counts[reason.split(" (")[0]] += 1
                    class_flagged_counts[class_folder] += 1

    print(f"  Total scanned       : {total_scanned:,} images")
    print(f"  Total flagged (bad) : {len(bad_files):,} images ({len(bad_files)/max(1, total_scanned):.1%})")
    print("-" * 72)

    if not bad_files:
        print("  ✓ No bad or low-quality images found!")
        print("=" * 72)
        return

    # Print summary of reasons
    print("  Flag reasons summary:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {reason:<25} : {count:>4} images")
    print()

    # Print top flagged classes
    print("  Top classes with flagged images:")
    for folder, count in sorted(class_flagged_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    Class {folder:<28} : {count:>4} images")
    print("-" * 72)

    # Perform move if not a dry run
    if dry_run:
        print(f"  [DRY RUN] Would have moved {len(bad_files):,} images.")
        print("  To actually clean the dataset, run with dry_run=False.")
    else:
        print(f"  Moving {len(bad_files):,} images to backup...")
        moved_count = 0
        for filepath, class_folder, filename, reason in bad_files:
            # Recreate class folder structure in backup
            dest_folder = os.path.join(backup_dir, class_folder)
            os.makedirs(dest_folder, exist_ok=True)
            
            dest_path = os.path.join(dest_folder, filename)
            try:
                shutil.move(filepath, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"    [WARNING] Failed to move {filename}: {e}")
                
        print(f"  ✓ Successfully moved {moved_count:,} images to {backup_dir}")
        print("  You can now inspect the backup folder to confirm they are indeed bad crops.")
    
    print("=" * 72)


if __name__ == "__main__":
    import sys
    src = "data/Indian_Dataset" if len(sys.argv) < 2 else sys.argv[1]
    clean_dataset(src, dry_run=True)
