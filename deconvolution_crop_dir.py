import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import cv2 as cv
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import glob

from lucyd import LUCYD

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

## Load Deconv Model globally for faster inference on CPU
MODEL_NAME='lucyd-edof-plankton_231204.pth'

# Model setup
model = LUCYD(num_res=1)
model_path = os.path.join(current_dir, 'models', MODEL_NAME)

# Check if CUDA is available and load the model accordingly
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available, using GPU.")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
else:
    device = torch.device('cpu')
    print("Warning! CUDA is not available, using CPU.")

model.load_state_dict(torch.load(model_path, map_location=device))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)
model.eval()


def add_granular_noise(img: np.ndarray, noise_strength: float = 0.1) -> np.ndarray:
    """
    Add granular noise to an image, similar to background noise.
    
    Args:
        img: Input image array
        noise_strength: Strength of the noise (0-1 scale)
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(loc=0.9, scale=noise_strength, size=img.shape)
    noisy_img = np.clip(img.astype(np.float64) * noise, 0, 255).astype(np.uint8)
    noisy_img = cv.blur(noisy_img + 255, (40, 40))
    return noisy_img


def pad_to_even(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to have even dimensions if needed.
    
    Args:
        img: Input image array
        
    Returns:
        Tuple of (padded_image, (rows_removed, cols_removed))
    """
    rows_removed = 0
    cols_removed = 0
    
    if img.shape[0] % 2 != 0:
        img = img[1:]
        rows_removed = 1
    if img.shape[1] % 2 != 0:
        img = img[:, 1:]
        cols_removed = 1
        
    return img, (rows_removed, cols_removed)


def process_batch(batch_images: List[np.ndarray], 
                  batch_filenames: List[str],
                  batch_offsets: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, str]]:
    """
    Process a batch of images with potentially different sizes using centered granular noise padding.
    
    Args:
        batch_images: List of preprocessed numpy arrays
        batch_filenames: List of corresponding filenames
        batch_offsets: List of (top_offset, left_offset) for each image
        
    Returns:
        List of tuples (deconvolved_image, filename)
    """
    if not batch_images:
        return []
    
    # Find max dimensions in batch
    max_h = max(img.shape[0] for img in batch_images)
    max_w = max(img.shape[1] for img in batch_images)
    
    # Pad all images to same size and convert to tensors
    padded_tensors = []
    original_shapes = []
    offsets = []
    
    for img in batch_images:
        h, w = img.shape
        original_shapes.append((h, w))
        
        # Calculate padding for centering
        pad_h = max_h - h
        pad_w = max_w - w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        offsets.append((pad_top, pad_left))
        
        # Create padding with granular noise
        if pad_h > 0 or pad_w > 0:
            # Create noise-filled padding
            padded_img = np.ones((max_h, max_w), dtype=np.uint8) * 245
            padded_img = add_granular_noise(padded_img, noise_strength=0.1)
            
            # Place original image in center
            padded_img[pad_top:pad_top+h, pad_left:pad_left+w] = img
            img = padded_img
        
        # Normalize and convert to tensor
        x = img / 255.0
        x_t = torch.from_numpy(x).to(device)
        x_t = x_t.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        padded_tensors.append(x_t)
    
    # Concatenate into batch
    batch_tensor = torch.cat(padded_tensors, dim=0)
    
    # Run deconvolution
    with torch.no_grad():
        y_hat_batch, _, _ = model(batch_tensor.float())
    
    # Extract results and remove padding
    results = []
    for i in range(y_hat_batch.shape[0]):
        h, w = original_shapes[i]
        top_offset, left_offset = offsets[i]
        
        # Extract original size from padded result
        deconv = y_hat_batch[i, 0, top_offset:top_offset+h, left_offset:left_offset+w].detach().cpu().numpy()
        deconv = (deconv * 255).astype(np.uint8)
        
        results.append((deconv, batch_filenames[i]))
    
    return results


def deconvolve_directory(input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         batch_size: int = 4,
                         file_pattern: str = "*.png",
                         save_inverted: bool = False) -> int:
    """
    Deconvolve all images in a directory and save results.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save deconvolved images
        batch_size: Number of images to process in each batch
        file_pattern: Glob pattern for image files (e.g., "*.png", "*.jpg")
        save_inverted: Whether to save bitwise inverted version as well
        
    Returns:
        Number of images processed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob.glob(str(input_dir / file_pattern)))
    
    if not image_files:
        print(f"No images found in {input_dir} matching pattern {file_pattern}")
        return 0
    
    print(f"Found {len(image_files)} images to process")
    
    batch_images = []
    batch_filenames = []
    batch_offsets = []
    processed_count = 0
    
    for img_path in image_files:
        # Read image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping")
            continue
        
        # Check if image is not blank (stdev > 2)
        if np.std(img) <= 2:
            print(f"Skipping blank image: {img_path}")
            continue
        
        # Ensure even dimensions
        img, padding_info = pad_to_even(img)
        
        # Add to batch
        batch_images.append(img)
        batch_filenames.append(os.path.basename(img_path))
        batch_offsets.append((0, 0))  # Will be calculated in process_batch
        
        # Process batch when full
        if len(batch_images) == batch_size:
            results = process_batch(batch_images, batch_filenames, batch_offsets)
            
            # Save results
            for deconv_img, filename in results:
                output_path = output_dir / filename
                cv.imwrite(str(output_path), deconv_img)
                
                if save_inverted:
                    inverted = cv.bitwise_not(deconv_img)
                    inverted_path = output_dir / f"inverted_{filename}"
                    cv.imwrite(str(inverted_path), inverted)
                
                processed_count += 1
            
            print(f"Processed {processed_count}/{len(image_files)} images")
            
            # Clear batch
            batch_images.clear()
            batch_filenames.clear()
            batch_offsets.clear()
    
    # Process remaining images in last batch
    if batch_images:
        results = process_batch(batch_images, batch_filenames, batch_offsets)
        
        for deconv_img, filename in results:
            output_path = output_dir / filename
            cv.imwrite(str(output_path), deconv_img)
            
            if save_inverted:
                inverted = cv.bitwise_not(deconv_img)
                inverted_path = output_dir / f"inverted_{filename}"
                cv.imwrite(str(inverted_path), inverted)
            
            processed_count += 1
    
    print(f'Deconvolution finished. Processed {processed_count} images.')
    return processed_count


# Example usage
if __name__ == "__main__":
    input_directory = "/home/veit/PIScO_dev/Segmentation_results/SO298/SO298-PISCO-Profiles/TempResults/SO298_298-10-1_PISCO2_20230422-2334/SO298_298-10-1_PISCO2_20230422-2334_Results/Crops"
    output_directory = "/home/veit/PIScO_dev/Segmentation_results/deconvolution_test"
    
    deconvolve_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        batch_size=16,
        file_pattern="*.png",
        save_inverted=False
    )