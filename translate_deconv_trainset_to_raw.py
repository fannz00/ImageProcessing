import os
import shutil
from pathlib import Path

from tqdm import tqdm

def find_image_in_crops_folders(image_name, root_dir, debug=False):
    """
    Search for an image in Crops folder of the profile directory with closest timestamp.
    Only considers profiles where the image timestamp is after the profile timestamp.
    
    Args:
        image_name: Name of the image file to find (e.g., '20220423-22243682_002.168bar_26.66C_112.png')
        root_dir: Root directory containing Profile folders (e.g., M181)
    
    Returns:
        str: Path to the found image or None if not found
    """
    img_date = int(image_name[:8])
    img_time = int(image_name[9:15])
    img_seconds = (img_time // 10000) * 3600 + ((img_time // 100) % 100) * 60 + (img_time % 100)
    img_timestamp = img_date * 86400 + img_seconds
    
    closest_profile = None
    smallest_diff = float('inf')
    closest_profile_diff = None
    
    for profile_dir in os.listdir(root_dir):
        try:
            datetime_str = profile_dir.split('_')[-1]
            profile_date = int(datetime_str[:8])
            profile_time = int(datetime_str[9:13])
            profile_seconds = (profile_time // 100) * 3600 + (profile_time % 100) * 60
            profile_timestamp = profile_date * 86400 + profile_seconds
            
            # Calculate time difference considering midnight boundary
            if profile_date == img_date:
                # Same day - simple difference
                time_diff = abs(profile_seconds - img_seconds)
            elif profile_date == img_date - 1:
                # Profile from previous day
                if profile_seconds > 22*3600:  # Profile after 22:00
                    # Compare with wraparound at midnight
                    time_diff = abs((86400 - profile_seconds) + img_seconds)
                else:
                    # Too far apart, skip
                    continue
            elif profile_date == img_date + 1:
                # Profile from next day
                if img_seconds > 22*3600:  # Image after 22:00
                    # Compare with wraparound at midnight
                    time_diff = abs((86400 - img_seconds) + profile_seconds)
                else:
                    # Too far apart, skip
                    continue
            else:
                # More than one day apart, skip
                continue
            
            if time_diff < smallest_diff:
                smallest_diff = time_diff
                closest_profile = profile_dir
                closest_profile_diff = time_diff/3600  # Store diff in hours
                
                # if debug:
                #     tqdm.write(f"\nNew closest match for {image_name}:")
                #     tqdm.write(f"Image time: {img_time//10000:02d}:{(img_time//100)%100:02d}:{img_time%100:02d}")
                #     tqdm.write(f"Profile: {profile_dir}")
                #     tqdm.write(f"Profile time: {profile_time//100:02d}:{profile_time%100:02d}")
                #     tqdm.write(f"Time diff: {closest_profile_diff:.1f} hours")
        except (ValueError, IndexError):
            continue
    
    if closest_profile:
        profile_path = os.path.join(root_dir, closest_profile)
        # Look for exact filename match in Crops folder
        for crops_dir in Path(profile_path).rglob('Crops'):
            image_path = os.path.join(crops_dir, image_name)
            if os.path.isfile(image_path):
                return str(image_path)
        
        # Only print debug info if image was not found
        if debug:
            tqdm.write(f"\nCould not find: {image_name}")
            tqdm.write(f"Closest profile was: {closest_profile} (diff: {closest_profile_diff:.1f} hours)")
    
    return None

def recreate_image_structure(source_root, target_root, crops_root):
    """
    Recreate folder structure and copy images from Crops folders.
    
    Args:
        source_root: Path to the original image category structure
        target_root: Path where to create the new structure
        crops_root: Root directory containing Profile folders with Crops subfolders
    """
    # Create target root if it doesn't exist
    os.makedirs(target_root, exist_ok=True)
    
    # First, collect all image files
    print("Collecting image files...")
    all_images = []
    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                rel_path = os.path.relpath(dirpath, source_root)
                all_images.append((rel_path, filename))
    
    missing_files = []
    pbar = tqdm(all_images, desc="Processing images", unit="image")
    
    for rel_path, filename in pbar:
        target_dir = os.path.join(target_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        new_image_path = find_image_in_crops_folders(filename, crops_root, debug=True)
        
        if new_image_path:
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(new_image_path, target_path)
        else:
            missing_files.append(os.path.join(rel_path, filename))
            pbar.write(f"Warning: Could not find {filename}")
    
    # Print summary
    total_files = len(all_images)
    missing_count = len(missing_files)
    success_count = total_files - missing_count
    
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Successfully copied: {success_count}")
    print(f"Files not found: {missing_count} ({missing_count/total_files*100:.1f}%)")
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"  - {file}")


if __name__ == "__main__":
    # Example usage
    source_root = "/home/veit/Downloads/export__TSV_17108_20250616_1412/"
    target_root = "/home/veit/Documents/M181_raw_trainset/"
    crops_root = "/home/veit/PIScO_dev/Segmentation_results/M181"
    
    recreate_image_structure(source_root, target_root, crops_root)