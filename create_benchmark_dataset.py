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

def select_batches_from_profile(images_with_depth: List[Tuple[str, float]], 
                               num_batches: int = 1,
                               batch_size: int = 200,
                               shallow_threshold: float = 200.0) -> Tuple[List[str], Dict]:
    """
    Select consecutive batches from a single profile.
    Prefer shallow batches if available.
    
    Returns: (selected_image_paths, stats_dict)
    """
    if not images_with_depth or num_batches == 0:
        return [], {"total": 0, "shallow": 0, "deep": 0}
    
    selected = []
    stats = {"total": 0, "shallow": 0, "deep": 0, "batches": []}
    
    # Separate by depth
    shallow = [(p, d) for p, d in images_with_depth if d <= shallow_threshold]
    deep = [(p, d) for p, d in images_with_depth if d > shallow_threshold]
    
    # Try to get batches from shallow first (60% of batches)
    shallow_batches_wanted = max(1, int(num_batches * 0.6))
    deep_batches_wanted = num_batches - shallow_batches_wanted
    
    # Get shallow batches
    if shallow:
        max_shallow_batches = len(shallow) // batch_size
        for _ in range(min(shallow_batches_wanted, max_shallow_batches)):
            start_idx = random.randint(0, len(shallow) - batch_size)
            batch = [p for p, _ in shallow[start_idx:start_idx + batch_size]]
            selected.extend(batch)
            stats["shallow"] += len(batch)
            stats["batches"].append(("shallow", start_idx, len(batch)))
    
    # Get deep batches
    if deep:
        max_deep_batches = len(deep) // batch_size
        for _ in range(min(deep_batches_wanted, max_deep_batches)):
            start_idx = random.randint(0, len(deep) - batch_size)
            batch = [p for p, _ in deep[start_idx:start_idx + batch_size]]
            selected.extend(batch)
            stats["deep"] += len(batch)
            stats["batches"].append(("deep", start_idx, len(batch)))
    
    stats["total"] = len(selected)
    return selected, stats

def build_benchmark_dataset(source_root: str, output_root: str, 
                           target_per_cruise: int = 1000,
                           batch_size: int = 200):
    """
    Build benchmark dataset with proper profile distribution.
    
    Strategy:
    - Collects from multiple profiles within each cruise
    - Takes consecutive batches from within each profile (preserves background correction)
    - Randomly selects which profiles to use
    - Prefers shallow depths (top 200m) across all profiles
    - Guarantees exactly target_per_cruise images per cruise
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
    logger.log(f"Target per cruise: {target_per_cruise} images")
    logger.log(f"Batch size: {batch_size} images")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    cruises = [d for d in os.listdir(source_root) 
               if os.path.isdir(os.path.join(source_root, d)) 
               and d != 'PISCO_KOSMOS_2020_Peru'
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
        total_images_available = 0
        
        for profile_dir in profile_dirs:
            profile_path = os.path.join(profiles_base, profile_dir)
            images = collect_images_from_profile(profile_path)
            if images:
                profile_images[profile_dir] = images
                total_images_available += len(images)
        
        logger.log(f"  📷 Profiles with images: {len(profile_images)}")
        logger.log(f"  📊 Total images available: {total_images_available}")
        
        if not profile_images:
            logger.log(f"  ❌ No images found in any profile")
            continue
        
        # Collect images until we have exactly target_per_cruise
        selected_images = []
        profile_stats = {}
        remaining_needed = target_per_cruise
        
        # Start with a reasonable number of profiles
        num_batches_total = (target_per_cruise + batch_size - 1) // batch_size
        num_profiles_to_use = min(len(profile_images), max(3, num_batches_total // 2))
        
        # Keep trying with more profiles if we don't get enough images
        attempt = 0
        max_attempts = 5
        
        while remaining_needed > 0 and attempt < max_attempts:
            attempt += 1
            
            # Randomly select profiles for diversity
            selected_profile_names = random.sample(list(profile_images.keys()), 
                                                   min(num_profiles_to_use, len(profile_images)))
            
            logger.log(f"  🎯 Attempt {attempt}: Using {len(selected_profile_names)} profiles")
            logger.log(f"  🎯 Need {remaining_needed} more images")
            
            # Distribute batches across profiles
            num_batches_total = (remaining_needed + batch_size - 1) // batch_size
            batches_per_profile = num_batches_total // len(selected_profile_names)
            remainder = num_batches_total % len(selected_profile_names)
            
            attempt_selected = []
            attempt_stats = {}
            
            for i, profile_name in enumerate(selected_profile_names):
                # Give extra batch to first few profiles if there's a remainder
                batches_for_this = batches_per_profile + (1 if i < remainder else 0)
                
                if batches_for_this == 0:
                    continue
                
                profile_images_list = profile_images[profile_name]
                batch_images, stats = select_batches_from_profile(
                    profile_images_list,
                    num_batches=batches_for_this,
                    batch_size=batch_size
                )
                
                attempt_selected.extend(batch_images)
                attempt_stats[profile_name] = stats
                
                logger.log(f"    ✓ {profile_name}: {stats['total']} imgs " +
                          f"({stats['shallow']} shallow, {stats['deep']} deep)")
            
            # Add these to our selection
            selected_images.extend(attempt_selected)
            profile_stats.update(attempt_stats)
            remaining_needed = target_per_cruise - len(selected_images)
            
            # If we got enough, break
            if remaining_needed <= 0:
                break
            
            # Otherwise, try with more profiles
            num_profiles_to_use = min(len(profile_images), num_profiles_to_use + 2)
        
        # Copy exactly target_per_cruise images
        copied = 0
        for img_path in selected_images[:target_per_cruise]:
            try:
                img_name = os.path.basename(img_path)
                dest_path = os.path.join(cruise_output, img_name)
                shutil.copy2(img_path, dest_path)
                copied += 1
            except Exception as e:
                logger.log(f"    ⚠️  Error copying {img_path}: {e}")
        
        global_stats[cruise] = {
            "total": copied,
            "profiles_used": len(profile_stats),
            "profiles": profile_stats
        }
        
        if copied == target_per_cruise:
            logger.log(f"  ✅ Selected exactly {copied} images for benchmark")
        else:
            logger.log(f"  ⚠️  Selected {copied} images (target was {target_per_cruise})")
    
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
    output_root = "/home/veit/Documents/PIScO_benchmark_dataset_v2"
    
    build_benchmark_dataset(source_root, output_root, target_per_cruise=1000, batch_size=200)