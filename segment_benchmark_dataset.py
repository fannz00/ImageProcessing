import os
import shutil
from glob import glob
from datetime import datetime
from segmenter import run_segmenter

def move_and_segment_benchmark(benchmark_dir, deconvolution=True, image_ext=".png"):
    cruises = [d for d in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, d))]
    for cruise in cruises:
        cruise_path = os.path.join(benchmark_dir, cruise)
        raw_path = os.path.join(cruise_path, "raw")
        os.makedirs(raw_path, exist_ok=True)
        # Move images to raw/
        images = glob(os.path.join(cruise_path, f"*{image_ext}"))
        for img_path in images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(raw_path, img_name)
            shutil.move(img_path, dest_path)
        # Create a unique results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = os.path.join(cruise_path, f"results_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)
        # Run segmentation, saving results in the new folder
        print(f"Running segmenter for {cruise}...")
        run_segmenter(raw_path, results_folder, deconvolution)
        print(f"Done with {cruise}, results in {results_folder}")

# Example usage:
if __name__ == "__main__":
    # Specify the path to the benchmark dataset
    # Ensure that the dataset is structured as expected with subdirectories for each cruise
    move_and_segment_benchmark("/home/veit/Documents/PIScO_benchmark_dataset", deconvolution=True)