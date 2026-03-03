import os
import shutil
from glob import glob
from datetime import datetime
from segmenter import run_segmenter

class Logger:
    """Simple logger that writes to both terminal and file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def log(self, message: str):
        """Print to terminal and write to log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()

def move_and_segment_benchmark(benchmark_dir, deconvolution=True, image_ext=".png"):
    """
    Organize and segment benchmark dataset.
    
    Strategy:
    - Move raw images to 'raw/' subdirectory
    - Run segmentation on raw images
    - Save results in a common results folder next to benchmark_dir
    - Log all operations to file
    """
    os.makedirs(benchmark_dir, exist_ok=True)

    # Shared results root next to benchmark_dir
    results_root = os.path.join(os.path.dirname(benchmark_dir), "benchmarkv2_segmented")
    os.makedirs(results_root, exist_ok=True)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(benchmark_dir, f"segmentation_{timestamp}.log")
    logger = Logger(log_path)
    
    logger.log(f"Benchmark Dataset Segmentation Log")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Benchmark directory: {benchmark_dir}")
    logger.log(f"Deconvolution: {deconvolution}")
    logger.log(f"Image extension: {image_ext}")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    cruises = sorted([d for d in os.listdir(benchmark_dir) 
                      if os.path.isdir(os.path.join(benchmark_dir, d))
                      and not d.startswith('.')])
    
    if not cruises:
        logger.log("❌ No cruise directories found")
        logger.close()
        return
    
    logger.log(f"📊 Found {len(cruises)} cruise(s)")
    logger.log(f"{'='*70}")
    
    global_stats = {}
    
    for cruise in cruises:
        logger.log(f"\nCRUISE: {cruise}")
        logger.log(f"{'-'*70}")
        
        cruise_path = os.path.join(benchmark_dir, cruise)
        raw_path = os.path.join(cruise_path, "raw")
        
        # Create raw directory
        os.makedirs(raw_path, exist_ok=True)
        
        # Move images to raw/
        images = glob(os.path.join(cruise_path, f"*{image_ext}"))
        images = [img for img in images if os.path.isfile(img) 
                  and os.path.dirname(img) == cruise_path]
        
        moved = 0  # track moved images
        
        if not images:
            raw_images = glob(os.path.join(raw_path, f"*{image_ext}"))
            if raw_images:
                logger.log(f"  ✅ Images already in raw/ ({len(raw_images)} found); skipping move")
            else:
                logger.log(f"  ⚠️  No images found in {cruise_path} or raw/")
                global_stats[cruise] = {"moved": 0, "segmented": 0, "status": "no_images"}
                continue
        else:
            logger.log(f"  📷 Found {len(images)} images to process")
            for img_path in images:
                try:
                    img_name = os.path.basename(img_path)
                    dest_path = os.path.join(raw_path, img_name)
                    shutil.move(img_path, dest_path)
                    moved += 1
                except Exception as e:
                    logger.log(f"    ⚠️  Error moving {img_path}: {e}")
            logger.log(f"  ✓ Moved {moved} images to raw/")
        
        # Create a unique results folder under the shared root, per cruise
        results_folder = os.path.join(results_root, cruise)
        os.makedirs(results_folder, exist_ok=True)
        logger.log(f"  ✓ Using results folder: {results_folder}")
        
        # Run segmentation
        logger.log(f"  🔄 Running segmenter...")
        try:
            run_segmenter(raw_path, results_folder, deconvolution)
            logger.log(f"  ✅ Segmentation completed")
            
            # Count segmented images
            segmented_images = glob(os.path.join(results_folder, f"*{image_ext}"))
            segmented_count = len(segmented_images)
            
            global_stats[cruise] = {
                "moved": moved,
                "segmented": segmented_count,
                "status": "completed"
            }
            
            logger.log(f"  📊 Results: {segmented_count} segmented images")
            
        except Exception as e:
            logger.log(f"  ❌ Segmentation failed: {e}")
            global_stats[cruise] = {
                "moved": moved,
                "segmented": 0,
                "status": "failed",
                "error": str(e)
            }
    
    # Print final summary
    logger.log(f"\n{'='*70}")
    logger.log("SEGMENTATION SUMMARY")
    logger.log(f"{'='*70}")
    
    total_moved = 0
    total_segmented = 0
    
    for cruise in sorted(global_stats.keys()):
        stats = global_stats[cruise]
        status = stats.get('status', 'unknown')
        moved = stats.get('moved', 0)
        segmented = stats.get('segmented', 0)
        
        logger.log(f"{cruise:30s}: {moved:5d} moved, {segmented:5d} segmented [{status}]")
        total_moved += moved
        total_segmented += segmented
    
    logger.log(f"{'-'*70}")
    logger.log(f"{'TOTAL':30s}: {total_moved:5d} moved, {total_segmented:5d} segmented")
    logger.log(f"{'='*70}")
    logger.log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log saved to: {log_path}")
    
    # Close the logger
    logger.close()

if __name__ == "__main__":
    # Segment the benchmark dataset
    move_and_segment_benchmark("/home/veit/Documents/PIScO_benchmark_dataset_v2", 
                               deconvolution=True)