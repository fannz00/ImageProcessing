import cv2 as cv
import numpy as np
import os
import time
import csv

from multiprocessing import Queue
from process_pool import ProcessPool
from dataclasses import dataclass
#from tqdm import tqdm


@dataclass
class DetectionSettings:
    """
    A data class for storing settings relevant to the image detection process.

    Attributes:
        data_path (str): The path where detection data (e.g., CSV files) will be saved.
        raw_crop_path (str): The path where cropped image segments will be saved.
        deconv_crop_path (str): The path where cropped image segments will be saved.
        mask_path (str): The path where mask images will be saved.
        img_path (str): The path where processed images will be saved.
        min_area_to_save (float): Minimum area for a detected object to be saved.
        min_area_to_segment (float): Minimum area required for segmentation.
        n_sigma (float): Number of standard deviations for thresholding.
        save_bb_image (bool): Flag indicating whether to save images with bounding boxes.
        save_crops (bool): Flag indicating whether to save cropped images.
        equalize_hist (bool): Flag indicating whether to apply histogram equalization.
        resize (bool): Flag indicating whether images should be resized.
        clear_save_path (bool): Flag indicating whether to clear the save path before processing.
        mask_img (bool): Flag indicating whether to apply a mask to images.
        mask_radius (int): Radius of the circular mask to be applied on images.
    """
    data_path: str
    raw_crop_path: str
    deconv_crop_path: str
    mask_path: str
    img_path: str
    min_area_to_save: float
    min_area_to_segment: float
    n_sigma: float
    save_bb_image: bool
    save_crops: bool
    equalize_hist: bool
    resize: bool
    clear_save_path: bool
    mask_img: bool
    mask_radius: int


def save_crop_data(path, data):
    """
    Save crop data to a CSV file.

    This function writes a list of data rows to a CSV file specified by `path`.

    Args:
        path (str): The file path where the CSV data will be saved.
        data (list): The data to be written to the CSV file, typically a list of lists.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)


def detect_on_img(input, settings: DetectionSettings, mask: np.ndarray, index=0):
    """
    Perform object detection on a single image and save relevant data.

    This function applies a detection algorithm on a given image, identifies objects,
    and saves the processed data. It uses settings from `DetectionSettings` to determine
    behavior such as masking, thresholding, and where to save outputs.

    Args:
        input (tuple): A tuple containing the corrected image, cleaned image, mean values, and filename.
        settings (DetectionSettings): An instance of DetectionSettings containing configuration options.
        mask (np.ndarray): A mask to be applied to the image if specified in settings.
        index (int, optional): Index of the image, used for logging or indexing purposes. Default is 0.
    """
    corrected, cleaned, mean_raw, fn = input
    raw_bg_corr = corrected
    corrected = cv.bitwise_not(cleaned)
    if mean_raw[1]>2:
        #print(fn)
        # hier bild maskieren
        color = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
        #cleaned = cv.resize(cleaned, (2560, 2560))

        if settings.mask_img:
            mean = np.mean(corrected[np.where(mask == 255)])
            std = np.std(corrected[np.where(mask == 255)])
            corrected[np.where(mask < 255)] = 0
        else:
            mean = np.mean(corrected)
            std = np.std(corrected)   

        c = 2 #add small constant value for empty images.

        thresh = cv.threshold(
            corrected,
            #mean + settings.n_sigma * std +c,
            10,#fixed value now since images are very similar in color
            255,
            cv.THRESH_BINARY,
        )[1]
        thresh = thresh.astype(np.uint8)

        cnts, hierachy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#[0]
        #cnts, hierachy = cv.findContours(dilated_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#[0]
        mask = np.zeros_like(thresh, dtype=np.uint8)

        areas = np.array([cv.contourArea(cnt) for cnt in cnts])
        if settings.resize:
            areas *= 4

        bounding_boxes = [cv.boundingRect(cnt) for cnt in cnts]
        # save center position and match bounding box to full width image
        bounding_boxes = np.array(
            [np.array((x + w / 2, y + h / 2, w, h)) for (x, y, w, h) in bounding_boxes]
        )
        if settings.resize:
            bounding_boxes *= 2

        crop_data = []
        c = 0
        for i, cnt in enumerate(cnts):
            if areas[i] < settings.min_area_to_segment:
                continue

            raw_crop_fn = os.path.join(settings.raw_crop_path, f"{fn[:-4]}_{c}.png")
            deconv_crop_fn = os.path.join(settings.deconv_crop_path, f"{fn[:-4]}_{c}.png")
            mask_fn = os.path.join(settings.mask_path, f"{fn[:-4]}_{c}.png")

            # if settings.save_bb_image:
            #     cv.drawContours(color, [cnt], -1, (0, 255, 0), 2)

            x, y, w, h = bounding_boxes[i]

            if areas[i] > settings.min_area_to_save:
                #mask = np.zeros_like(cleaned, dtype=np.uint8)
                crop_data.append([c, os.path.basename(raw_crop_fn),mean_raw[0],mean_raw[1],mean,std, areas[i], x, y, w, h, 1])

                x = int(x - w / 2)
                y = int(y - h / 2)
                h = int(h)
                w = int(w)

                if settings.save_crops:
                    crop = cleaned[y : y + h, x : x + w]
                    raw_crop = raw_bg_corr[y : y + h, x : x + w]
                    cv.drawContours(mask,[cnt],-1,(255), thickness=cv.FILLED)                    
                    c_mask = mask[y : y + h, x : x + w]
                    crop_mask = np.where(c_mask == 255, crop, 255)
                    #raw_crop_mask = np.where(c_mask == 255, raw_crop, 255)
                    
                    
                    cv.imwrite(deconv_crop_fn, crop_mask)#now only the object is saved and not objects close to it
                    cv.imwrite(raw_crop_fn, raw_crop)
                    cv.imwrite(mask_fn, c_mask)
            else:
                crop_data.append([c, os.path.basename(mask_fn),mean_raw[0],mean_raw[1],mean,std, areas[i], x, y, w, h, 0])

            c += 1

        if settings.save_bb_image:
            color = cv.resize(color, (1280, 1280))
            cv.imwrite(os.path.join(settings.img_path, fn), color)
            
            #cleaned = cv.resize(cleaned, (512, 512))
            # Create a mask from the thresholded image
            #thresh = np.where(thresh == 255, cleaned, 255).astype(np.uint8)

            # Create a white canvas with the same size as the original image
            #white_canvas = np.ones_like(cleaned)*255

            # Apply the mask to the original image
            #thresh = cv.bitwise_and(cleaned, white_canvas, mask=mask)
            #thresh = cv.bitwise_and(cleaned, mask=cv.bitwise_not(thresh))
            #cleaned = cv.resize((1000,1000), cleaned)
            #cv.imwrite(os.path.join(settings.img_path, 'cleaned', fn), cleaned)
            #cv.imwrite(os.path.join(settings.img_path, 'thresh_'+fn), thresh)

        save_crop_data(os.path.join(settings.data_path, fn[:-4] + ".csv"), crop_data)
    else:
        crop_data = []
        crop_data.append(['corrupt', os.path.basename(fn), 0, 0, 0, 0, 0, 0])
        save_crop_data(os.path.join(settings.data_path, fn[:-4] + ".csv"), ['corrupt image!'])


def run_detection(input: Queue, settings: DetectionSettings, n_cores, n_imgs, running):
    """
    Execute the detection process across multiple images using a process pool.

    This function manages the detection workflow, including image preparation,
    masking, and parallel processing across multiple cores. It uses the
    DetectionSettings to configure the detection behavior and outputs.

    Args:
        input (Queue): A queue containing images and metadata to be processed.
        settings (DetectionSettings): Configuration settings for detection.
        n_cores (int): Number of processor cores to use for parallel processing.
        n_imgs (int): Total number of images to process.
        running: A flag or condition indicating whether the process pool is active.
    """
    shape = (2560, 2560)
    if settings.resize:
        shape = (2560, 2560)
    mask = np.zeros(shape, dtype=np.uint8)
    if settings.mask_img:
        cv.circle(
            mask,
            (int(mask.shape[0] / 2), int(mask.shape[1] / 2)),
            settings.mask_radius,
            255,
            -1,
        )
    pool = ProcessPool(
        lambda input, index: detect_on_img(input, settings, mask, index),
        running,
        10,
    )
    pool.start(n_cores,'detection')
    for i in range(n_imgs):
        task = input.get()
        pool.add_task(task)
        print(i, task[-1]) #print filenames

    pool.stop(slow=True)

    while pool.is_running():
        print('.')
        time.sleep(1)

    print("detection finished")
