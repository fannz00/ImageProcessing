import time
import cv2 as cv
import numpy as np
import os

from process_pool import ProcessPool
from multiprocessing import Queue
from reader import ReaderOutput
from scipy.ndimage import uniform_filter


def is_ready(img_index: int, input: ReaderOutput, n_bg_imgs):
    """
    Check if a sufficient number of background images are available for processing.

    This function determines if a given image at `img_index` has a sufficient
    number of preceding images available to serve as background images for
    processing. If any of the required background images are missing (i.e., `None`),
    the function returns False.

    Args:
        img_index (int): The index of the image to check readiness for.
        input (ReaderOutput): An object containing a list of images and possibly
                              other metadata.
        n_bg_imgs (int): The number of background images required for processing.

    Returns:
        bool: True if the required background images are ready, False otherwise.
    """
    # Ensure we don't go out of bounds
    start = max(0, img_index - n_bg_imgs + 1)
    end = min(len(input.images), img_index + 1)
    
    # If we don't have enough images for background correction, skip
    if end - start < 2:  # At least need current image + 1 background
        return True
    
    # Check that all images in range are loaded and not None
    for i in range(start, end):
        if i < 0 or i >= len(input.images):
            return False
        if input.images[i] is None:
            return False
    return True


def correct_img(
    img_index: int, input: ReaderOutput, output: Queue, n_bg_imgs: int, index=0
):
    """
    Perform background correction on an image and put the result in an output queue.

    This function waits until the image at `img_index` is ready for processing,
    then performs background correction by calculating the mean and standard
    deviation of the image. If the standard deviation is above a threshold, it
    computes a background image using a set of preceding images and applies
    correction. The corrected image, its cleaned version, and related metadata
    are then placed into the `output` queue.

    Args:
        img_index (int): The index of the image to process.
        input (ReaderOutput): An object containing a list of images and possibly
                              other metadata.
        output (Queue): A queue to store the results of the background correction.
        n_bg_imgs (int): The number of images to use for creating the background.
        index (int, optional): An additional index used in processing, default is 0.

    Returns:
        None
    """
    n_bg_imgs *= 2
    timeout = 60  # 60 seconds timeout to prevent infinite hanging
    elapsed = 0
    while not is_ready(img_index, input, n_bg_imgs):
        time.sleep(0.1)
        elapsed += 0.1
        if elapsed > timeout:
            print(f'Timeout waiting for image {img_index} to be ready. Skipping.')
            output.put(([], [], [0, 0], 'TIMEOUT'))
            return
    
    # Check if image at img_index is None (corrupted/failed to load)
    if img_index >= len(input.images) or input.images[img_index] is None:
        print(f'Skipping corrupted/missing image at index {img_index}')
        output.put(([], [], [0, 0], 'CORRUPTED'))
        return
    
    try:
        img, fn = input.images[img_index]
    except (IndexError, TypeError) as e:
        print(f'Error accessing image {img_index}: {e}')
        output.put(([], [], [0, 0], 'ERROR'))
        return
    
    mean = np.mean(img)
    stdev = np.std(img)
    if stdev > 2:
        # Find valid background images
        start = max(0, img_index - n_bg_imgs + 1)
        end = min(len(input.images), img_index + 1)
        
        bg_imgs: list[np.ndarray] = []
        # Collect valid background images, skipping any None entries
        for i in range(start, end - 1):  # -1 to not include current image
            if i >= 0 and i < len(input.images) and input.images[i] is not None:
                try:
                    bg_imgs.append(input.images[i][0])
                except (TypeError, IndexError):
                    continue
        
        # Only attempt background correction if we have at least one valid background image
        if len(bg_imgs) > 0:
            bg = np.max(bg_imgs, axis=0)
            correct_img = cv.absdiff(img, bg)
            cleaned_img = cv.bitwise_not(correct_img)
            bg_corr_img = cleaned_img
            output.put((bg_corr_img, cleaned_img, [mean, stdev], fn))
        else:
            print(f'No valid background images available for image {img_index}: {fn}')
            output.put(([], [], [mean, stdev], fn))
    else:
        print(f'Found corrupt image (low stdev): {fn}')
        output.put(([], [], [mean, stdev], fn))


def run_bg_correction(input: ReaderOutput, output: Queue, n_bg_imgs: int, running):
    """
    Start a process pool to perform background correction on a set of images.

    This function initializes a process pool and distributes the task of
    background correction across multiple processes. Each image in the `input`
    is checked and, if suitable, processed to correct the background. The results
    are stored in the `output` queue. The function handles exceptions that occur
    when adding images to the pool.

    Args:
        input (ReaderOutput): An object containing a list of images and possibly
                              other metadata.
        output (Queue): A queue to store the results of the background correction.
        n_bg_imgs (int): The number of images to use for creating the background.
        running: A flag or condition indicating whether the process pool is active.

    Returns:
        None
    """
    pool = ProcessPool(
        lambda img_index, index: correct_img(
            img_index, input, output, n_bg_imgs, index
        ),
        running,
        -1,
    )
    pool.start(3,'bg_corr') #n_processes was 3
    #print(len(input.images))
    for i in range(len(input.images)):
        try:
            pool.add_task(i)
        except Exception as e:
            print(f"Exception when adding image {i} to bg correction pool: {e}")
    pool.stop(slow=True)

    # while pool.is_running():
    #     time.sleep(1)

    # print("Bg correction finished")
