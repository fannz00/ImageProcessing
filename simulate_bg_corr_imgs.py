import os
import random
import numpy as np
import cv2 as cv

def edof_sim(
    img: np.ndarray,
    depth: int = 50,
    object_z_pos: int = 25,
    dov: int = 1,
    blur_multiplier: int = 4,
    noise_strength: int = 10,
):
    res = np.zeros(img.shape, dtype=np.float64)
    counter = 0
    for t in np.arange(0, 2 * np.pi, 0.1):
        counter += 1
        f_z = depth / 2 * (1 - np.cos(t))
        z_dist_object = abs(f_z - object_z_pos)
        noise = np.random.sample(img.shape) * noise_strength
        if z_dist_object > dov:
            blurred = cv.blur(
                img,
                (
                    max(1, blur_multiplier * round(z_dist_object)),
                    max(1, blur_multiplier * round(z_dist_object)),
                ),
            )
            res += blurred
        else:
            res += img.astype(np.float64)
        res += noise
        res[np.where(res > 255 * counter)] = 255 * counter
    res /= counter
    res = res.astype(np.uint8)
    return res

def segment_and_soft_mask(img, edge_blur=21):
    _, mask = cv.threshold(img, 240, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    mask_obj = np.zeros_like(img)
    cv.drawContours(mask_obj, [max(contours, key=cv.contourArea)], -1, 255, -1)
    mask_blur = cv.GaussianBlur(mask_obj, (edge_blur, edge_blur), 0)
    mask_blur = mask_blur.astype(np.float32) / 255.0
    img_float = img.astype(np.float32)
    white_bg = np.ones_like(img_float) * 255
    soft_img = img_float * mask_blur + white_bg * (1 - mask_blur)
    return soft_img.astype(np.uint8), mask_blur

def michelson_contrast(img):
    Imax = np.max(img)
    Imin = np.min(img)
    return (Imax - Imin) / (Imax + Imin + 1e-8)

def adjust_michelson_contrast(img, target_contrast=0.5):
    img = img.astype(np.float32)
    mean = np.mean(img)
    Imax = np.max(img)
    Imin = np.min(img)
    current_contrast = (Imax - Imin) / (Imax + Imin + 1e-8)
    if current_contrast == 0:
        return img.astype(np.uint8)
    scale = target_contrast / current_contrast
    img = (img - mean) * scale + mean
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_crop_noise(img, noise_strength=0.05):
    noise = np.random.normal(loc=1.0, scale=noise_strength, size=img.shape)
    noisy_img = np.clip(img.astype(np.float32) * noise, 0, 255)
    return noisy_img.astype(np.uint8)

def process_and_save_random_crops(
    source_path, 
    output_path, 
    num_crops=100, 
    edge_blur=21, 
    edof_params=None,
    target_contrast=0.5, 
    contrast_jitter=0.1, 
    noise_strength=0.05
):
    if edof_params is None:
        edof_params = dict(depth=50, object_z_pos=25, dov=1, blur_multiplier=4, noise_strength=10)
    os.makedirs(output_path, exist_ok=True)
    all_files = []
    for root, dirs, files in os.walk(source_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, f))
    for i in range(num_crops):
        img_path = random.choice(all_files)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Segment and soft mask
        seg_img, mask_blur = segment_and_soft_mask(img, edge_blur=edge_blur)
        # EDOF blur (all blurring here)
        edof_img = edof_sim(
            seg_img,
            depth=edof_params.get("depth", 50),
            object_z_pos=random.uniform(0, edof_params.get("depth", 50)),
            dov=edof_params.get("dov", 1),
            blur_multiplier=edof_params.get("blur_multiplier", 4),
            noise_strength=edof_params.get("noise_strength", 10),
        )
        # Adjust Michelson contrast
        jitter = random.uniform(-contrast_jitter, contrast_jitter)
        adj_contrast = max(0.01, min(1.0, target_contrast + jitter))
        contrast_img = adjust_michelson_contrast(edof_img, target_contrast=adj_contrast)
        # Add slight noise to the crop
        noisy_crop = add_crop_noise(contrast_img, noise_strength=noise_strength)
        # Save
        out_path = os.path.join(output_path, f"crop_{i:04d}.png")
        cv.imwrite(out_path, noisy_crop)

if __name__ == "__main__":
    source_path = "/home/veit/Documents/PlanktonSet/FINAL_Plankton_Segments_12082014/copepod_calanoid"
    output_path = "/home/veit/Documents/edof_sim_250715_crops"
    process_and_save_random_crops(
        source_path,
        output_path,
        num_crops=100,
        edge_blur=1,
        edof_params=dict(depth=50, dov=1, blur_multiplier=4, noise_strength=10),
        target_contrast=0.5,
        contrast_jitter=0.1,
        noise_strength=0.03
    )