import os
import shutil
import random
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

class Logger:
    """Simple logger that writes to both terminal and file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def log(self, message: str):
        """Print to terminal and write to log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()  # Ensure it's written immediately
    
    def close(self):
        """Close the log file."""
        self.log_file.close()

def extract_pressure_from_filename(filename: str) -> float:
    """Extract pressure (in dbar) from image filename."""
    if 'dbar' in filename:
        try:
            # Split at 'dbar', get the part before it
            parts = filename.split('dbar')
            # Split by both '-' and '_' to get numeric pressure value
            prefix = parts[0].replace('-', '_').split('_')[-1]
            return float(prefix)
        except (ValueError, IndexError):
            pass
    
    if 'bar' in filename:
        try:
            parts = filename.split('bar')
            prefix = parts[0].split('_')[-1]
            return float(prefix)
        except (ValueError, IndexError):
            pass
    
    return float('inf')

def collect_images_from_profile(profile_path: str) -> List[Tuple[str, float]]:
    """Collect all images from a single profile with their depths."""
    images_with_depth = []
    
    # Look for image directories with various naming patterns
    image_dirs = []
    for item in os.listdir(profile_path):
        item_path = os.path.join(profile_path, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()
            # Match: "PNG", "images-png", "Images", etc.
            if 'image' in item_lower or item_lower == 'png':
                image_dirs.append(item_path)
    
    # Always also check the profile directory itself (for HE570 direct placement)
    image_dirs.append(profile_path)
    
    for img_dir in image_dirs:
        # Only non-recursive - check directly in each directory
        png_files = glob(os.path.join(img_dir, "*.png"))
        for img_file in png_files:
            pressure = extract_pressure_from_filename(os.path.basename(img_file))
            if pressure != float('inf'):
                images_with_depth.append((img_file, pressure))
    
    # Sort by pressure (depth)
    images_with_depth.sort(key=lambda x: x[1])
    return images_with_depth

def build_benchmark_dataset(source_root: str, output_root: str, 
                           profiles_per_cruise: int = 5):
    """
    Build benchmark dataset with full profiles.
    
    Strategy:
    - Takes 5 complete profiles from each cruise (except M181)
    - All images from selected profiles are copied
    - Preserves background correction by keeping complete profiles
    """
    os.makedirs(output_root, exist_ok=True)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_root, f"benchmark_creation_{timestamp}.log")
    logger = Logger(log_path)
    
    logger.log(f"Benchmark Dataset Creation Log")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Source: {source_root}")
    logger.log(f"Output: {output_root}")
    logger.log(f"Profiles per cruise: {profiles_per_cruise}")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    cruises = [d for d in os.listdir(source_root) 
               if os.path.isdir(os.path.join(source_root, d)) 
               and d != 'PISCO_KOSMOS_2020_Peru'
               and d != 'M181'  # Exclude M181
               and not d.startswith('.')]
    
    global_stats = {}
    
    for cruise in sorted(cruises):
        cruise_path = os.path.join(source_root, cruise)
        cruise_output = os.path.join(output_root, cruise)
        os.makedirs(cruise_output, exist_ok=True)
        
        logger.log(f"\n{'='*70}")
        logger.log(f"CRUISE: {cruise}")
        logger.log(f"{'='*70}")
        
        # Find PISCO-Profiles directory
        profiles_base = os.path.join(cruise_path, f"{cruise}-PISCO-Profiles")
        
        if not os.path.exists(profiles_base):
            logger.log(f"  ❌ No PISCO-Profiles directory found")
            continue
        
        # Get all profiles
        profile_dirs = sorted([d for d in os.listdir(profiles_base) 
                              if os.path.isdir(os.path.join(profiles_base, d))
                              and not d.startswith('.')])
        
        if not profile_dirs:
            logger.log(f"  ❌ No profiles found in {profiles_base}")
            continue
        
        logger.log(f"  📊 Total profiles available: {len(profile_dirs)}")
        
        # Collect images from each profile
        profile_images = {}
        
        for profile_dir in profile_dirs:
            profile_path = os.path.join(profiles_base, profile_dir)
            images = collect_images_from_profile(profile_path)
            if images:
                profile_images[profile_dir] = images
        
        logger.log(f"  📷 Profiles with images: {len(profile_images)}")
        
        if not profile_images:
            logger.log(f"  ❌ No images found in any profile")
            continue
        
        # Select profiles_per_cruise random profiles
        num_profiles_to_select = min(profiles_per_cruise, len(profile_images))
        selected_profile_names = random.sample(list(profile_images.keys()), 
                                              num_profiles_to_select)
        
        logger.log(f"  🎯 Randomly selecting {len(selected_profile_names)} profiles")
        
        # Copy all images from selected profiles
        total_images_copied = 0
        profile_stats = {}
        
        for profile_name in sorted(selected_profile_names):
            images_list = profile_images[profile_name]
            num_images = len(images_list)
            
            copied_count = 0
            for img_path, _ in images_list:
                try:
                    img_name = os.path.basename(img_path)
                    dest_path = os.path.join(cruise_output, img_name)
                    shutil.copy2(img_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    logger.log(f"    ⚠️  Error copying {img_path}: {e}")
            
            profile_stats[profile_name] = {
                "total": copied_count,
                "expected": num_images
            }
            
            logger.log(f"    ✓ {profile_name}: {copied_count} images copied")
            total_images_copied += copied_count
        
        global_stats[cruise] = {
            "total": total_images_copied,
            "profiles_used": len(selected_profile_names),
            "profiles": profile_stats
        }
        
        logger.log(f"  ✅ Total images from {len(selected_profile_names)} profiles: {total_images_copied}")
    
    # Print final summary
    logger.log(f"\n{'='*70}")
    logger.log("BENCHMARK DATASET CREATION SUMMARY")
    logger.log(f"{'='*70}")
    
    total_all = 0
    for cruise in sorted(global_stats.keys()):
        stats = global_stats[cruise]
        logger.log(f"{cruise:30s}: {stats['total']:5d} imgs ({stats['profiles_used']} profiles)")
        total_all += stats['total']
    
    logger.log(f"{'-'*70}")
    logger.log(f"{'TOTAL':30s}: {total_all:5d} images")
    logger.log(f"{'='*70}")
    logger.log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log saved to: {log_path}")
    
    # Close the logger
    logger.close()

if __name__ == "__main__":
    source_root = "/mnt/filer"
    output_root = "/home/veit/Documents/PIScO_benchmark_dataset_v3"
    
    build_benchmark_dataset(source_root, output_root, profiles_per_cruise=5)