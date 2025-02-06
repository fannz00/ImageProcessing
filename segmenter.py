import cv2 as cv
import os
import time
import numpy as np
import csv
from threading import Thread
from multiprocessing import Queue, Manager, Process, Value
import imghdr
#from tqdm import tqdm

from reader import run_reader, ReaderOutput, profiled_run_reader
from bg_correction import run_bg_correction
from deconvolution import profiled_run_deconvolution
from detection import run_detection, DetectionSettings

import tkinter as tk
from tkinter import filedialog
import pickle

def compute_radius(files):
    """
    Computes the radius of the background mask from the first 10 images in the given list of files.

    This function reads the first 10 image files from the provided list, resizes them to a fixed 
    dimension of 2560x2560 pixels, and calculates a background image by finding the maximum pixel 
    intensity at each position across these images. It then creates a binary mask to isolate 
    background elements and iteratively adjusts the radius of a circular region until a specified 
    amount of background pixels is covered.

    Parameters:
        files (list): List of file paths to images. Assumes that the list contains at least 10 
                      image files.

    Returns:
        int: The computed radius that covers the desired amount of background pixels.
    """
    imgs = [cv.resize(cv.imread(file, cv.IMREAD_GRAYSCALE),(2560,2560)) for file in files[:10]]
    #print(imgs)
    bg = np.max(imgs, axis=0)

    bg = cv.threshold(cv.bitwise_not(bg), 180, 255, cv.THRESH_BINARY)[1]
    bg = cv.bitwise_not(bg)
    bg[np.where(bg == 0)] = 10

    mask = np.zeros(bg.shape, dtype=np.uint8)
    count_bg_px = 0
    radius = 1000
    while count_bg_px < 10000:
        radius += 50
        cv.circle(mask, (1280, 1280), radius, (255), -1)
        masked = cv.bitwise_and(bg, mask)
        count_bg_px = len(np.where(masked == 10)[0])

    radius -= 50
    return radius

def save_detection_settings_to_csv(settings, file_path, src_path):
    """
    Save the fields and values of a DetectionSettings dataclass to a CSV file.

    This function writes the field names and their corresponding values from
    a `DetectionSettings` instance into a CSV file. The first row of the CSV
    file contains the column headers: "Field Name" and "Value".

    Args:
        settings: An instance of the DetectionSettings dataclass containing
                  the configuration to be saved.
        file_path (str): The file path where the CSV will be saved.
        src_path: The path to the data source folder
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Field Name", "Value"])
        writer.writerow(["data source", src_path])
        for field_name, field_value in settings.__annotations__.items():
            writer.writerow([field_name, getattr(settings, field_name)])

### write file checking function!!

def timesort(file):
    """
    Extract a timestamp from a filename for sorting purposes.

    This function attempts to parse a timestamp from a filename by extracting
    specific substrings that represent date and time. It is designed to handle
    filenames with a specific format where the timestamp is embedded.

    Args:
        file (str): The filename from which to extract the timestamp.

    Returns:
        int: An integer representation of the timestamp for sorting. If parsing
             fails due to an IndexError, it attempts an alternative format.
    """
    try:
        return (int(file.split("_")[4][:8]+file.split("_")[4][9:17] ))
    except IndexError:
        return (int(file.split("_")[0][:8]+file.split("_")[0][9:17] ))
    
def run_segmenter(src_path: str, save_path: str, deconvolution: bool):
    """
    Run the segmentation process on a set of images, optionally including deconvolution.

    This function organizes the segmentation workflow, including reading images,
    applying background correction, optional deconvolution, and detection. It
    also saves the settings used for detection into a CSV file. The process is
    executed in batches to manage large sets of images efficiently.

    Args:
        src_path (str): The source directory path containing the images to segment.
        save_path (str): The directory path where outputs will be saved, including
                         crops, masks, and detection data.
        deconvolution (bool): A flag indicating whether to perform deconvolution
                              on the images before detection.

    Workflow:
        - Creates necessary directories for saving outputs.
        - Filters and sorts image files from the source directory.
        - Computes necessary parameters like mask radius.
        - Initializes detection settings and saves them to a CSV file.
        - Processes images in batches, applying background correction and detection.
        - Optionally applies deconvolution if specified.

    Note:
        - This function assumes the presence of helper functions and classes like
          `compute_radius`, `run_reader`, `run_bg_correction`, `run_deconvolution`,
          `run_detection`, and `ReaderOutput`.
        - The batch processing and use of multiprocessing or threading require
          proper management of shared resources and synchronization mechanisms.
    """
    #save_path = os.path.join(save_path, os.path.basename(src_path))
    os.makedirs(save_path, exist_ok=True)

    raw_crop_path = os.path.join(save_path, "Crops")
    deconv_crop_path = os.path.join(save_path, "Deconv_crops")
    mask_path = os.path.join(save_path, "Masks")
    data_path = os.path.join(save_path, "Data")
    img_path = os.path.join(save_path, "Images")
    os.makedirs(raw_crop_path, exist_ok=True)
    os.makedirs(deconv_crop_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    files_unsorted = os.listdir(src_path)
    files_unsorted = [os.path.join(src_path, file) for i, file in enumerate(files_unsorted)]
    files = []
    segmented_files = os.listdir(data_path)
    segmented_files = [os.path.splitext(name)[0] for name in segmented_files]
    print('checking files...')
    for file in files_unsorted:#tqdm(files_unsorted):
        if os.path.splitext(os.path.basename(file))[0] in segmented_files:
            continue
        elif (file.endswith('.png') or file.endswith('.jpg')) and imghdr.what(file) is not None:
            files.append(file)
    
    #post message about the status of already segmented images
    print(f'{len(segmented_files)} of {len(files)} images already segmented')
    
    try:
        files.sort(
            key=lambda x: float(os.path.basename(x).split("_")[0].split("-")[1]) #sort the "old" way
            #key = timesort #the new way, thank you @Timm Schöning: applied 240801-19:14 (UTC+2)
        )  # sort after time
    except:
        print("Key-based sort failed")
        files.sort()
    #files = [os.path.join(src_path, file) for i, file in enumerate(files)]
    print('start segmentation...')
    #print(files[:10])

    radius = compute_radius(files)
    print("Radius: ",radius)

    manager = Manager()
    settings = DetectionSettings(
        data_path,  #data_path: str
        raw_crop_path,  #crop_path: str
        deconv_crop_path,
        mask_path,
        img_path,   #img_path: str
        400,        #min_area_to_save: float
        10,        #min_area_to_segment: float
        1,          #n_sigma: float
        False,       #save_bb_image: bool
        True,       #save_crops: bool
        True,       #equalize_hist: bool
        False,       #resize: bool
        True,       #clear_save_path: bool
        True,       #mask_img: bool
        radius,       #mask_radius: int
    )

    # Specify the path where you want to save the CSV file
    csv_file_path = os.path.join(save_path,"settings.csv")

    # Call the function to save the data class to the CSV file
    save_detection_settings_to_csv(settings, csv_file_path, src_path)

    batch_size = 200
    batches = []
    for i in range(len(files) // batch_size - 1):
        batches.append(files[i * batch_size : (i + 1) * batch_size])
    batches.append(files[batch_size * (len(files) // batch_size - 1) :])
    batch_n = 0
    for batch in batches:
        batch_n += 1
        print(f'Batch {batch_n} of {len(batches)}')
        start = time.perf_counter()
        batch = [(img, i) for i, img in enumerate(batch)]
        reader_output = ReaderOutput(len(batch), manager)
        bg_output_queue = Queue(10)
        deconv_output_queue = Queue(10)
        corr_running = Value("i",1)
        detect_running = Value("i",1)


        # Thread(
        #     target=run_reader, 
        #     args=(batch, reader_output, 8, settings.resize)#input, output, n_threads
        #     ).start()
        # Use the profiled version in the thread
        thread = Thread(
            target=profiled_run_reader,  # Use the profiled version
            args=(batch, reader_output, 8, settings.resize)
        )
        thread.start()

        bg_size = 6

        run_bg_correction(reader_output, bg_output_queue, bg_size, corr_running) 

        if deconvolution:
            Thread(
                target=profiled_run_deconvolution,
                args=(bg_output_queue, deconv_output_queue, len(batch), 12), #last argument is deconv batch_size has to match with image batch size 
            ).start()
        else:
            deconv_output_queue = bg_output_queue
        
        #print(len(batch))
        n_cores = 16 #8 on Seavision

        run_detection(deconv_output_queue, settings, n_cores, len(batch), detect_running)

        end = time.perf_counter()
        duration = end - start
        print(f'Time per image: {duration / len(batch)}')
        #print(bg_output_queue.empty(),deconv_output_queue.empty())
        #print(len(reader_output.images))
        bg_output_queue.close()
        deconv_output_queue.close()
    print('finished segmentation...') 

def process_image_folders(source_paths, dest_base, deconvolution=True):
    """
    Processes image folders by running the segmenter on each folder.
    
    Args:
        source_paths (str or list): Path(s) to the image folders.
        dest_base (str): Path to the destination folder.
        deconvolution (bool, optional): Whether to perform deconvolution. Defaults to True.
    """
    # If source_paths is a string, convert it to a list
    if isinstance(source_paths, str):
        source_paths = [source_paths]

    # Loop through each folder
    for folder in source_paths:
        # Get the source and destination folder paths
        source_folder = os.path.join(folder, 'PNG')
        destination_folder = os.path.join(dest_base, os.path.basename(folder))

        # If the folder is already in the destination folder
        if os.path.basename(folder) in os.listdir(dest_base):
            # If the number of images in the destination folder is the same as the source folder
            if len(os.listdir(os.path.join(destination_folder,'Data'))) == (len(os.listdir(source_folder))):
                print(folder, 'already fully segmented in destination folder! Skipping...')
                continue
            # If there are more images in the destination folder than the source folder
            elif len(os.listdir(os.path.join(destination_folder,'Data'))) > (len(os.listdir(source_folder))):
                print(folder, 'more images in destination folder! Please check manually! Skipping...')
                continue
            # If the folder is a directory
            elif os.path.isdir(folder):
                print(folder)
                # Run the segmenter
                run_segmenter(source_folder, destination_folder, deconvolution)
        # If the folder is a directory
        elif os.path.isdir(folder):
            print(folder)
            # Run the segmenter
            run_segmenter(source_folder, destination_folder, deconvolution)

def select_directories():
    """
    Opens a file dialog to select directories and returns a list of selected directories.
    
    Returns:
        list: A list of selected directories.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    dirs = []
    while True:
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")], title="Select a pickle file")
        if not file_path:  # User clicked 'cancel'
            break
        if file_path.endswith('.pkl'):
            # Load the list from the file
            with open(file_path, 'rb') as f:
                dirs.extend(pickle.load(f))
    while True:
        dir = filedialog.askdirectory(title="Select directories")
        if not dir:  # User clicked 'cancel'
            break
        else:
            dirs.append(dir)
    
    # Save the list to a file
    with open('selected_list.pkl', 'wb') as f:
        pickle.dump(dirs, f)
    
    #print(dirs)

    return dirs


# if __name__ == "__main__":
#     dest_folder = '/media/plankton/30781fe1-cea5-4503-ae00-1986beb935d2/Segmentation_results/M181/results_240328'
#     source_folders = []
#     for folder in os.listdir('/media/plankton/30781fe1-cea5-4503-ae00-1986beb935d2/M181_raw'):
#         dir = os.path.join('/media/plankton/30781fe1-cea5-4503-ae00-1986beb935d2/M181_raw', folder)
#         source_folders.append(dir)
#     selected_folders = select_directories()
#     source_folders.extend(selected_folders)
#     print(source_folders)
    
#     process_image_folders(source_folders, dest_folder, True)

if __name__ == "__main__":
    run_segmenter(
        "/media/veit/30781fe1-cea5-4503-ae00-1986beb935d2/M181_raw/M181-107-1_CTD-036_00°00S-007°00W_20220504-0756/PNG", #source folder
        "/home/veit/Data_test/M181_test_results/", #destination folder
        True, #deconvolution        
    )

# if __name__ == "__main__":
#     source_base = "/media/plankton/Elements"
#     dest_base = "/media/plankton/30781fe1-cea5-4503-ae00-1986beb935d2/Segmentation_results/M181"

#     for folder in os.listdir(source_base):
#         dir = os.path.join(source_base,folder)
#         source_folder = os.path.join(dir, 'PNG')
#         destination_folder = source_folder.replace(source_base, dest_base)
#         if folder == '$RECYCLE.BIN':
#             continue
#         elif folder == 'System Volume Information':
#             continue
#         elif folder == '.Trash-1000':
#             continue
#         #elif folder in ['M181-204-1_CTD-056_00°00S-024°00W_20220511-0913']:
#         #    continue
#         if folder in os.listdir(dest_base):
#             if len(os.listdir(os.path.join(destination_folder,'PNG','Data'))) >= (len(os.listdir(source_folder))+1):
#                 print(folder, 'already fully segmented in destination folder! Skipping...')
#                 continue
#             elif len(os.listdir(os.path.join(destination_folder,'PNG','Data'))) < (len(os.listdir(source_folder))+1):
#                 print(folder, 'already more images in destination folder! Please check manually! Skipping...')
#                 continue
#             elif os.path.isdir(dir):
#                 print(folder)

#                 # Assuming the rest of the parameters are constant for all runs
#                 deconvolution = True
#                 depth_range = None  # Set your depth range here
#                 log_file = None     # Set your log file location here

#                 run_segmenter(
#                     source_folder, 
#                     destination_folder, 
#                     deconvolution, 
#                     #depth_range, 
#                     #log_file
#                     )
                
#         elif os.path.isdir(dir):
#             print(folder)
#             source_folder = os.path.join(dir, 'PNG')
#             destination_folder = source_folder.replace(source_base, dest_base)

#             # Assuming the rest of the parameters are constant for all runs
#             deconvolution = True
#             depth_range = None  # Set your depth range here
#             log_file = None     # Set your log file location here

#             run_segmenter(
#                 source_folder, 
#                 destination_folder, 
#                 deconvolution, 
#                 #depth_range, 
#                 #log_file
#                 )
            