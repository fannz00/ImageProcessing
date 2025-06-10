import cv2 as cv
import numpy as np
import time
import os
import threading
from process_pool import ProcessPool
from multiprocessing import Manager
from thread_pool import ThreadPool

import cProfile
from PIL import Image
#import torchvision.transforms.functional as TF



class ReaderOutput:
    """
    A class to manage and store the outputs of image reading operations.

    This class uses a shared list to store images and their filenames, allowing
    for thread-safe operations using a lock.

    Attributes:
        lock (threading.Lock): A lock to ensure thread-safe access to the images list.
        images (Manager.list): A list shared among processes, initialized to hold
                               `n_images` number of placeholders.
    """
    def __init__(self, n_images, manager: Manager) -> None:
        """
        Initialize the ReaderOutput with a specified number of image slots.

        Args:
            n_images (int): The number of images that will be read and stored.
            manager (Manager): A multiprocessing manager for creating a shared list.
        """
        self.lock = threading.Lock()
        self.images = manager.list([None for _ in range(n_images)])

    def add_output(self, img, fn, index):
        """
        Add an image and its filename to the specified index in the images list.

        This method ensures that updates to the images list are thread-safe.

        Args:
            img (numpy.ndarray): The image data to store.
            fn (str): The filename of the image.
            index (int): The index at which to store the image and filename.
        """
        with self.lock:
            self.images[index] = (img, fn)


def read_img(output: ReaderOutput, input, thread_index=0):
    """
    Read an image from a file and store it in a ReaderOutput object.

    This function reads an image file, resizes it, and stores the image data
    along with its filename into a `ReaderOutput` object at the specified index.
    It handles cases where the file might be empty or not a valid image.

    Args:
        output (ReaderOutput): The object to store the read image data.
        input: Either a file path string or a tuple of (file_path, img_index)
        thread_index (int, optional): The index of the thread executing this function,
                                      used for logging purposes. Default is 0.
    """
    if isinstance(input, tuple):
        file, img_index = input
    else:
        file = input
        img_index = thread_index  # fallback to using thread index as image index

    fn = os.path.basename(file)
    try:
        # Check if the file size is 0 bytes
        if os.path.getsize(file) == 0:
            print(f'Thread {thread_index}: File {file} is an empty image file')
            return

        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Thread {thread_index}: Failed to read image {file}")
            return
            
        img = cv.resize(img, (2560, 2560))
        output.add_output(img, fn, img_index)
    except Exception as e:
        print(f'Thread {thread_index}: Exception occurred reading {file}: {str(e)}')
        return

def run_reader(files, output: ReaderOutput, n_threads: int, resize:bool):
    """
    Read multiple image files concurrently using a thread pool.

    Args:
        files (iterable): An iterable of file paths to be read.
        output (ReaderOutput): The object to store the read image data.
        n_threads (int): The number of threads to use in the thread pool.
        resize (bool): A flag indicating whether images should be resized.
    """
    pool = ThreadPool(lambda input, index: read_img(output, input, index), 100)
    pool.start(n_threads)
    
    try:
        for i, file in enumerate(files):
            try:
                # Handle both string and tuple inputs
                if isinstance(file, tuple):
                    file_path = file[0]  # Extract the file path from tuple
                else:
                    file_path = file
                pool.add_task((file_path, i))
            except Exception as e:
                print(f"Exception when adding file {file} to reader pool: {e}")
        
        print('all files in batch added to reader pool')
    finally:
        # Ensure pool is stopped even if exceptions occur
        pool.stop(slow=True)
        print("Reader pool stopped")

def profiled_run_reader(batch, reader_output, num_threads, resize):
    # Set up the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the original function
    run_reader(batch, reader_output, num_threads, resize)

    # Disable the profiler and save the results to a file
    profiler.disable()
    profile_filename = f"profile_run_reader_{threading.current_thread().name}.prof"
    profiler.dump_stats(profile_filename)
    print(f"Profiled run_reader saved to {profile_filename}")

    # Für batchwise das untere auskommentieren, bessere Performance

    # while pool.is_running():
    #      time.sleep(1)

    # print("Reader finished")
