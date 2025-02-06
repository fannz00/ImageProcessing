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
    start = (
        img_index
        if img_index + n_bg_imgs < len(input.images)
        else len(input.images) - n_bg_imgs
    )
    for i in range(start, start + n_bg_imgs):
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
    while not is_ready(img_index, input, n_bg_imgs):
        time.sleep(0.1)
    img, fn = input.images[img_index]
    mean = np.mean(img)
    stdev = np.std(img)
    if stdev >2:
        start = (img_index if img_index + n_bg_imgs < len(input.images) else len(input.images) - n_bg_imgs)
        bg_imgs: list[np.ndarray] = []
        for i in range(1, n_bg_imgs):
            bg_imgs.append(
                np.min([input.images[start + i - 1][0], input.images[start + i][0]], axis=0)
            )
        bg = np.max(bg_imgs, axis=0)
        
        correct_img = cv.absdiff(img, bg)
        cleaned_img = cv.bitwise_not(correct_img)
        bg_corr_img = cleaned_img
        #fixing problem where very bright objects (plankton acts as lens?!) print through on following images: see mattermost board "Segmenter Problem: Durchdrucken großer/dunkler Objekte im MaxSegmenter"
        #cleaned_img[np.where(cleaned_img >= 250)] = np.mean(bg) 
        
        output.put((bg_corr_img, cleaned_img, [mean,stdev], fn))
    else:
        print('found corrupt image: ', fn)

        output.put(([], [], [mean,stdev], fn))


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
