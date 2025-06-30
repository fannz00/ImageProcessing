# This script is intended to use after segmentation to create a df and add it to the sql database. Some columns are still missing, they are added in load_profiles_from_database.ipynb
#TODO: Add the missing operations here.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import datetime
from PIL import Image, ImageDraw, ImageFont
import zipfile
import shutil

from pandas.errors import EmptyDataError

from sqlalchemy import create_engine
from sqlalchemy import text

import inspect
from skimage import measure
from skimage.io import imread
import cv2
import umap
import pickle
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from datasets import Features, Image as ImageFeature, Value
from datasets import Dataset, Features, Image as ImageFeature, Value

# For type hints (optional but recommended)
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("analyze_profiles.log"),
    logging.StreamHandler()
])


### fixed variables
VOLUME_PER_IMAGE = 50 * (3.5)**2 * np.pi  # in cubic centimeters
VOLUME_PER_IMAGE_LITERS = VOLUME_PER_IMAGE / 1000  # convert to liters
IMAGE_SIZE = 2560

#Setup Database
#Setup connection parameters
username = 'plankton'
password = 'piscodisco'
host = 'deepseavision.geomar.de'  # or the IP address of your database server
port = '5432'       # default port for PostgreSQL
database = 'pisco_crops_db'

# Create an engine that connects to the PostgreSQL server
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
logging.info("Connected to the database")


def extract_lat_lon_from_profile(profile_name):
    """
    Extract latitude and longitude from the profile name.
    Example: 'M181-252-1_CTD-066_00deg00S-032deg00W_20220514-1919'
    Returns: (lat, lon) as floats
    """
    match = re.search(r'(\d+)°(\d+)([NS])-(\d+)°(\d+)([EW])', profile_name)
    if match:
        lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = match.groups()
        lat = int(lat_deg) + int(lat_min) / 60.0
        lon = int(lon_deg) + int(lon_min) / 60.0
        if lat_dir == 'S':
            lat *= -1
        if lon_dir == 'W':
            lon *= -1
        return lat, lon
    return None, None

def gen_crop_df(path:str, small:bool, size_filter:int = 0, pressure_unit:str = 'dbar', absolute_pressure:bool = True, cruise:str = None):
    """
    A function to generate a DataFrame from a directory of CSV files, with options to filter out small objects.
    
    Parameters:
    path (str): The path to the directory containing the CSV files.
    small (bool): A flag indicating whether to filter out small objects.
    size_filter (int): The size filter for the objects. Default is 0.
    pressure_unit (str): The unit of pressure. Default is 'dbar'.
    
    Returns:
    pandas.DataFrame: The concatenated and processed DataFrame with additional columns for analysis.
    """
    
    def area_to_esd(area: float) -> float:
        pixel_size = 13.5*2 #in µm/pixel @ 2560x2560 
        return 2 * np.sqrt(area * pixel_size**2 / np.pi)

    # Function to concatenate directory and filename
    def join_strings(dir, filename):
        return os.path.join(dir, filename)

    directory = os.path.dirname(path)
    directory = os.path.join(directory,'Crops')

    files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    dataframes = []
    empty_file_counter = 0
    id = 1
    for file in tqdm(files):
        try:
            # Define which columns should be strings (filename is column 2)
            string_columns = [1]  # Add other string column indices if needed
            
            # Read CSV with dtype specifications
            df = pd.read_csv(file, delimiter=",", header=None)
            
            if len(df.columns) == 44:
                # Convert non-string columns to numeric
                for i in range(len(df.columns)):
                    if i not in string_columns:
                        df[i] = pd.to_numeric(df[i], errors='coerce')
                    else:
                        df[i] = df[i].astype(str)
                        
                df.insert(0, 'source_file_id', id)
                dataframes.append(df)
                id += 1
            else:
                continue
        except EmptyDataError:
            empty_file_counter += 1
            #print(f"File {file} is empty")
    
    df = pd.concat(dataframes, ignore_index=True)
    #headers = ["img_id","index", "filename", "area", "x", "y", "w", "h", "saved"]
    headers = ["img_id","index", "filename", "mean_raw", "std_raw", "mean", "std", "area", "x", "y", "w", "h", 
               "saved", "object_bound_box_w", "object_bound_box_h", "bound_box_x", "bound_box_y", "object_circularity", "object_area_exc", 
               "object_area_rprops", "object_%area", "object_major_axis_len", "object_minor_axis_len", "object_centroid_y", "object_centroid_x", 
               "object_convex_area", "object_min_intensity", "object_max_intensity", "object_mean_intensity", "object_int_density", "object_perimeter", 
               "object_elongation", "object_range", "object_perim_area_excl", "object_perim_major", "object_circularity_area_excl", "object_angle", 
               "object_boundbox_area", "object_eccentricity", "object_equivalent_diameter", "object_euler_nr", "object_extent", 
               "object_local_centroid_col", "object_local_centroid_row", "object_solidity"
    ]
    df.columns = headers
    df.reset_index(drop=True, inplace=True)
    df.drop("index", axis=1, inplace=True)

    if not small:
        df = df[df["saved"] == 1]
    df_unique = df.drop_duplicates(subset=['img_id'])
    print(len(df_unique))
    #df.drop("saved", axis=1, inplace=True)

    # Split the 'filename' column SO298_298-6-1_PISCO2_0009.82dbar-02.00S-089.00W-28.54C_20230418-18023076_13.png
    df['filename'] = df['filename'].astype(str)
    split_df = df['filename'].str.split('_', expand=True)
    if small:# bug fix for segmenter where small objects are saved with _mask.png extension instead of .png: needs to be fixed if segmenter is fixed
        headers = ["cruise", "dship_id", "instrument", "pressure", "mask_ext"]
        split_df.columns = headers
        split_df.drop("mask_ext", axis=1, inplace=True)
    
    elif cruise == "HE570":
        headers = ["cruise", "dship_id", "instrument", "pressure", "date", "time", "index"]
        split_df.columns = headers
        # Reformat the 'date-time' column to match the expected format
        split_cols = split_df['pressure'].str.split('-', expand=True)
        split_df[['pressure', 'lat', 'lon']] = split_cols
        split_df['pressure'] = split_df['pressure'].str.replace(pressure_unit, '', regex=False).astype(float)
        # Convert and combine date and time columns to datetime format
        split_df['date-time'] = pd.to_datetime(split_df['date'].astype(str) + split_df['time'].astype(str).str.replace('.', ''), format='%Y%m%d%H%M%S%f')
        # Format the datetime to match required format '20230418-18023076'
        split_df['date-time'] = split_df['date-time'].dt.strftime('%Y%m%d-%H%M%S%f')
        split_df['index'] = split_df['index'].str.replace('.png', '', regex=False).astype(int)
    
    elif cruise == "PISCO_KOSMOS_2020_Peru":
        headers = ["cruise", "dship_id", "instrument-date", "time", "pressure", "index"]
        split_df.columns = headers
        # Split instrument-date into instrument and date
        split_df['instrument'] = split_df['instrument-date'].str.extract(r'([A-Za-z]+)')
        split_df['date'] = split_df['instrument-date'].str.extract(r'(\d{8})')
        split_df.drop('instrument-date', axis=1, inplace=True)
        split_df['pressure'] = split_df['pressure'].str.replace(pressure_unit, '', regex=False).astype(float)       
        # Convert and combine date and time columns to datetime format
        split_df['date-time'] = pd.to_datetime(split_df['date'].astype(str) + split_df['time'].astype(str).str.replace('.', ''), format='%Y%m%d%H%M%S%f')
        # Format the datetime to match required format '20230418-18023076'
        split_df['date-time'] = split_df['date-time'].dt.strftime('%Y%m%d-%H%M%S%f')
        split_df['index'] = split_df['index'].str.replace('.png', '', regex=False).astype(int)

    elif cruise == "M181":
        headers = ["date-time", "pressure", "temperature", "index"]
        split_df.columns = headers
        split_df['pressure'] = split_df['pressure'].str.replace(pressure_unit, '', regex=False).astype(float)
        split_df['temperature'] = split_df['temperature'].str.replace('C', '', regex=False).astype(float)
        split_df['index'] = split_df['index'].str.replace('.png', '', regex=False).astype(int)


    else:
        try:
            headers = ["cruise", "dship_id", "instrument", "pressure","date-time","index"]
            split_df.columns = headers
            split_cols = split_df['pressure'].str.split('-', expand=True)
            split_df[['pressure', 'lat', 'lon', 'temperature']] = split_cols
            split_df['pressure'] = split_df['pressure'].str.replace(pressure_unit, '', regex=False).astype(float)
            split_df['temperature'] = split_df['temperature'].str.replace('C', '', regex=False).astype(float)
            split_df['index'] = split_df['index'].str.replace('.png', '', regex=False).astype(int)
           
        except ValueError:
            print(f"Error processing file {file} . Please check the format of the 'filename' column.")

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([split_df, df], axis=1)

    # Extend the original 'filename' column
    df['full_path'] = df.apply(lambda x: join_strings(directory, x['filename']), axis=1)
    #df = df.drop('filename', axis=1)

    df['esd'] = df['area'].apply(area_to_esd).round(2)
    
    if pressure_unit == 'bar':
        df['pressure'] = (df['pressure'])*10

    if not absolute_pressure:
        df['pressure'] = df['pressure'] - 10.1325

    df.rename(columns={'pressure': 'pressure [dbar]'}, inplace=True)

    # Sort the DataFrame by the 'date-time' column
    print(df.head())
    df = df.sort_values(by=['date-time','index'], ascending=True)
    df.reset_index(drop=True, inplace=True)

    #filter the df for objects where 1 dimension is larger than ca. 1mm
    df = df[(df['w'] > size_filter) | (df['h'] > size_filter)]
    df_unique = df.drop_duplicates(subset=['img_id'])
    print(f'{empty_file_counter} files were empty and were dropped; Number of uniue images: {len(df_unique)}')

    logging.info('crop_df created')
    return df

def get_ctd_profile_id(cruise_base: str, profile: str) -> str:
    """
    Extract CTD profile ID from metadata CSV file or prompt user for input.
    
    Args:
        cruise_base (str): Base directory for the cruise
        profile (str): Profile folder name
        
    Returns:
        str: CTD profile ID or None if not found/provided
    """
    # Construct path to metadata CSV
    csv_path = os.path.join(
        cruise_base,
        profile,
        profile + "_Metadata",
        profile + ".csv"
    )
    
    try:
        # Try to read the CSV file
        with open(csv_path, 'r') as f:
            for line in f:
                if 'CTDprofileid' in line:
                    # Extract profile ID after comma
                    ctd_id = line.split(',')[1].strip()[-3:]
                    return ctd_id
                    
    except FileNotFoundError:
        print(f"Warning: Metadata CSV not found: {csv_path}")
    except Exception as e:
        print(f"Error reading metadata for {profile}: {str(e)}")
    
    # If we get here, we need user input
    # Use input with timeout to avoid blocking
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError
    
    def get_user_input(profile):
        return input(f"\nPlease enter CTD profile ID for {profile} (or press Enter to skip): ")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_user_input, profile)
        try:
            ctd_id = future.result(timeout=600)  # 10 second timeout
            return ctd_id if ctd_id else None
        except TimeoutError:
            print(f"\nSkipping {profile} - no CTD ID provided within timeout")
            return None

def add_ctd_data(ctd_data_loc:str, crop_df):
    #crop_df.drop('depth [m]',axis=1)
    # Reading the specified header line (line 124) to extract column names
    def find_header_and_data_start(file_path):
        try:
            with open(file_path, 'r') as file:
                # Read the first 10 lines to find the header
                for line_num, line in enumerate(file):
                    line = line.strip()
                    # Find the header line that starts with "Columns = "
                    if line.startswith('Columns  ='):
                        header_line = line
                        column_names = line.split(' = ')[1].split(':')
                        # Read next line to check for data start
                        next_line = next(file).strip()
                        # Try to convert first element to float to verify it's a data row
                        try:
                            float(next_line.split()[0])
                            return line_num + 1, column_names
                        except ValueError:
                            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return None, None
    
    try:
        data_start_line, column_names = find_header_and_data_start(ctd_data_loc)
    except Exception as e:
        print(f"Error reading CTD data file: {e}")
        return crop_df  # Return original DataFrame if error occurs
    
    if data_start_line is not None:
        ctd_df = pd.read_csv(ctd_data_loc,
                        delim_whitespace=True, 
                        header=None, 
                        skiprows=data_start_line, 
                        names=column_names)
    else: 
        raise ValueError("Could not find valid header and data section in file")

    ctd_df['z_factor']=ctd_df['z']/ctd_df['p']
    
    # Function to interpolate a column based on closest pressure values
    def interpolate_column(pressure, column):
        # Sort 'p' values based on the distance from the current pressure
        closest_ps = ctd_df['p'].iloc[(ctd_df['p'] - pressure).abs().argsort()[:2]]
        
        # Get corresponding column values
        column_values = ctd_df.loc[closest_ps.index, column]
        
        # Linear interpolation
        return np.interp(pressure, closest_ps, column_values)
    
    # Columns to interpolate
    chl_col = list(ctd_df.filter(like='chl').columns)[0]
    columns = ['s', 'o', 't', chl_col, 'z_factor']

    # Identify unique pressures and calculate their interpolated 's' values
    unique_pressures = crop_df['pressure [dbar]'].unique()

    # Interpolate for each column and store the results in a dictionary
    interpolated_columns = {column: {pressure: interpolate_column(pressure, column) 
                                    for pressure in unique_pressures}
                            for column in columns}

    for column in columns:
        new_col_name = f'interpolated_{column}'
        crop_df[new_col_name] = crop_df['pressure [dbar]'].map(interpolated_columns[column])
    # Determine the position of pressure column
    position = crop_df.columns.get_loc('pressure [dbar]') + 1

    # Insert a new column. For example, let's insert a column named 'new_column' with a constant value
    crop_df.insert(position, 'depth [m]', (crop_df['pressure [dbar]']*crop_df['interpolated_z_factor']).round(3))

    logging.info('CTD data added to crop_df')
    return crop_df

def add_prediction(crop_df, prediction_df):
    """
    Merge prediction DataFrame with crop DataFrame based on filename.
    
    Args:
        crop_df (pd.DataFrame): DataFrame containing crop data
        prediction_df (pd.DataFrame): DataFrame containing predictions
        
    Returns:
        pd.DataFrame: Merged DataFrame with predictions added to crop data
        
    Raises:
        ValueError: If DataFrames have different lengths or missing filename column
    """
    # Verify both DataFrames have filename column
    if 'filename' not in crop_df.columns or 'filename' not in prediction_df.columns:
        raise ValueError("Both DataFrames must have 'filename' column")
        
    # Verify DataFrames have same length
    if len(crop_df) != len(prediction_df):
        raise ValueError(f"DataFrames have different lengths: {len(crop_df)} vs {len(prediction_df)}")
    
    # Merge DataFrames on filename
    merged_df = pd.merge(
        crop_df,
        prediction_df,
        on='filename',
        how='left',
        validate='1:1'  # Ensures one-to-one merge
    )
    
    # Verify no rows were lost in merge
    if len(merged_df) != len(crop_df):
        raise ValueError("Rows were lost during merge. Check for duplicate filenames.")
        
    logging.info('Predictions added to crop_df')
    return merged_df
   

def resize_to_larger_edge(image, target_size):
    # Get the original dimensions of the image
    original_width, original_height = image.size
    
    # Determine which dimension is larger
    larger_edge = max(original_width, original_height)
    
    # Compute the scale factor to resize the larger edge to the target size
    scale_factor = target_size / larger_edge
    
    # Compute new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    try:
    # Resize the image
        resized_image = F.resize(image, (new_height, new_width))
    except(ValueError):
        #print(image.size,new_height,new_width)
        logging.info(f"Skipping: {image}: image size: {image.size}, new height: {new_height}, new width: {new_width}")
        return None        
    return resized_image

def custom_image_processor(image, target_size=(224, 224), padding_color=255, size_bar=False):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if size_bar:
        # Step 0: Remove scale bar by cropping the bottom 50 pixels
        width, height = image.size
        image = image.crop((0, 0, width, height - 50))  # Crop out the scale bar area
    
    # Step 1: Resize the image
    resized_image = resize_to_larger_edge(image,224)

    if resized_image is None:  # Skip processing if resizing failed
        #print(f"Skipping image due to resize failure: {image.size}")
        return None  # This allows to filter out bad images later

    #Step 2: Calculate padding
    new_width, new_height = resized_image.size
    padding_left = (target_size[0] - new_width) // 2
    padding_right = target_size[0] - new_width - padding_left
    padding_top = (target_size[1] - new_height) // 2
    padding_bottom = target_size[1] - new_height - padding_top

    # Step 3: Apply padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    pad_transform = transforms.Pad(padding, fill=padding_color)
    padded_image = pad_transform(resized_image)

    # Step 4: Apply other transformations
    transform_chain = transforms.Compose([
        transforms.RandomRotation(degrees=180,fill=255),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    # Apply the transformations
    return transform_chain(padded_image)

# Example function to process a batch
def process_batch(example_batch):
    # Process each image in the batch
    processed_images = [
        custom_image_processor(img) for img in example_batch['image']
        if custom_image_processor(img) is not None
        ]
    
    # Convert to a batch tensor
    inputs = torch.stack(processed_images)
    
    # Include labels (assuming they are present and you want to keep them)
    return {'pixel_values': inputs, 'label': example_batch['label']}

def load_unclassified_images(data_dir):
    """
    Load unclassified images from a directory, filtering for valid image files.
    
    Args:
        data_dir (str): Path to directory containing images
        
    Returns:
        Dataset: HuggingFace dataset containing images
    """
    # Define valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    # List all files in the directory and filter for valid image files
    image_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if os.path.isfile(os.path.join(data_dir, f))
        and f.lower().endswith(valid_extensions)
    ]

    # Prepare data for the dataset
    data = {
        'image': image_files,
        'label': image_files 
    }
    
    features = Features({
        'image': ImageFeature(),
        'label': Value('string')
    })
    
    # Create the dataset
    dataset = Dataset.from_dict(data, features=features)
    return dataset

def get_predictions_with_entropy_ood(dataset, save_dir, entropy_threshold=1.0, temperature=1.5, batch_size=64):
    print("Initializing model for predictions")
    vit = ViTForImageClassification.from_pretrained(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)
    vit.eval()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    predictions = []
    filenames = []
    probabilities = []
    entropy_scores = []
    ood_flags = []
    
    with torch.amp.autocast('cuda'):
        for batch in tqdm(dataloader, desc="Processing dataset"):
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            
            with torch.no_grad():
                outputs = vit(pixel_values=inputs)
            
            # Apply temperature scaling
            scaled_logits = outputs.logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            top_probs, top_indices = torch.topk(probs, 5, dim=-1)
            
            batch_labels = [[vit.config.id2label[idx.item()] for idx in indices] 
                          for indices in top_indices]
            
            # Flag samples as OOD if entropy is above threshold
            batch_ood = (entropy > entropy_threshold).cpu().numpy()
            
            predictions.extend(batch_labels)
            filenames.extend(batch['label'])
            probabilities.extend(top_probs.cpu().numpy())
            entropy_scores.extend(entropy.cpu().numpy())
            ood_flags.extend(batch_ood)
    
    return filenames, predictions, probabilities, entropy_scores, ood_flags

def plot_histogram(df, plot_path:str):
    """
    Plot a histogram of the 'esd' column from the given DataFrame df and save the plot to the specified plot_path.

    Parameters:
    - df: DataFrame containing the data to plot
    - plot_path: str, the path to save the plot image

    Returns:
    None
    """

    log_bins = np.logspace(np.log10(df["esd"].min()), np.log10(df["esd"].max()), num=500)
    plt.hist(df["esd"], bins=log_bins, log=True)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("ESD")
    plt.ylabel("Frequency")
    plt.title("Histrogram of esds")
    plt.savefig(os.path.join(plot_path,'esd_hist.png'))
    plt.close()

def populate_esd_bins(df, esd_bins=np.array([125,250,500,1000,100000]), depth_bin_size=5):
    """
    Generate histograms and pivot tables based on the provided dataframe and specified bin sizes.
    
    Parameters:
    - df: DataFrame containing the data to be processed
    - esd_bins: Array of bin edges for Equivalent Spherical Diameter (ESD) values
    - depth_bin_size: Size of the depth bins
    
    Returns:
    - histogram: DataFrame with counts of occurrences for each 'esd_bin' and 'depth_bin' combination
    - pivoted_df: Pivoted DataFrame with normalized counts for different 'esd_bin' values as columns
    """

    # Define depth bins with an interval of 5 m
    max_depth = df['depth [m]'].max()
    min_depth = df['depth [m]'].min()
    depth_bins = np.arange(min_depth, max_depth + depth_bin_size, depth_bin_size)

    # Assign each 'esd' and 'depth [m]' value to its respective bin
    df['esd_bin'] = np.digitize(df['esd'], esd_bins)
    df['depth_bin'] = np.digitize(df['depth [m]'], depth_bins)

    # Group by the depth bin and count unique values in 'depth [m]'
    depth_bin_volumes = df.groupby('depth_bin')['img_id'].nunique()*VOLUME_PER_IMAGE_LITERS

    # Group by 'esd_bin' and 'depth_bin' and count occurrences
    histogram = df.groupby(['esd_bin', 'depth_bin']).size().reset_index(name='count')
    histogram['normalized_count'] = histogram.apply(lambda row: row['count'] / depth_bin_volumes.get(row['depth_bin'], 1), axis=1)

    
    # Pivot the dataframe to make 'esd_bin' values as column headers
    pivoted_df = histogram.pivot(index='depth_bin', columns='esd_bin', values='normalized_count').reset_index()

    # Rename columns for clarity
    pivoted_df.columns = ['depth [m]', 'ESD<125um', 'ESD 125-250um', 'ESD 250-500um', 'ESD 500-1000um', 'ESD >1000um']

    return histogram, pivoted_df

def populate_esd_bins_pressure(df,  depth_bin_size, esd_bins=np.array([0,125,250,500,1000,100000])):

    # Define depth bins with an interval of 5 m
    max_depth = df['pressure [dbar]'].max()
    min_depth = df['pressure [dbar]'].min()
    depth_bins = np.arange(min_depth, max_depth + depth_bin_size, depth_bin_size)

    # Assign each 'esd' and 'depth [m]' value to its respective bin
    df['esd_bin'] = np.digitize(df['esd'], esd_bins)
    df['depth_bin'] = np.digitize(df['pressure [dbar]'], depth_bins)

    # Group by the depth bin and count unique values in 'depth [m]'
    depth_bin_volumes = df.groupby('depth_bin')['img_id'].nunique()*VOLUME_PER_IMAGE_LITERS

    # Group by 'esd_bin' and 'depth_bin' and count occurrences
    histogram = df.groupby(['esd_bin', 'depth_bin']).size().reset_index(name='count')
    histogram['normalized_count'] = histogram.apply(lambda row: row['count'] / depth_bin_volumes.get(row['depth_bin'], 1), axis=1)
    
    # Pivot the dataframe to make 'esd_bin' values as column headers
    pivoted_df = histogram.pivot(index='depth_bin', columns='esd_bin', values='normalized_count').reset_index()

    return histogram, pivoted_df

def plot_particle_dist(grouped, stationID:str, plot_path:str, depth_bin_size=5, preliminary=True, depth_min=0, maximum_y_value=None):
    """
    Generate a particle distribution plot based on the provided data.

    Parameters:
    - grouped: DataFrame containing the grouped data
    - stationID: str, the ID of the station
    - plot_path: str, the path where the plot will be saved
    - depth_bin_size: int, optional, the size of the depth bins, default is 5
    - preliminary: bool, optional, flag indicating if the plot is preliminary, default is True
    - depth_min: int, optional, the minimum depth value, default is 0
    """

    fig, ax1 = plt.subplots(figsize=(10,15))
    fig.subplots_adjust(top=0.97) # Adjust the value as needed
    fig.subplots_adjust(bottom=0.2)

    ax1.invert_yaxis()

    ax2 =ax1.twiny()
    ax3 =ax1.twiny()
    ax4 =ax1.twiny()
    ax5 =ax1.twiny()

    axes = [ax1,ax2,ax3,ax4,ax5]
    esd_bins = [1,2,3,4,5]
    esd_bin_names = ['<125um','125-250um','250-500um','500-1000um','>1000um']
    colors = ['red', 'blue', 'lime', 'cyan', 'black']
    positions = [0,40,80,120,160]


    for ax,i,name,color,pos in zip(axes,esd_bins,esd_bin_names,colors,positions):
        if i in grouped.columns:
            ax.plot(grouped[i], grouped['depth_bin']*depth_bin_size+depth_min, color=color, label=name)
            ax.set_xlabel(f'normalized abundance LPM {name} [#/L]',color=color)
            ax.spines['bottom'].set_color(color)
            ax.tick_params(axis='x', colors=color)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', pos))
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()
            if maximum_y_value is None:
                maximum_y_value = grouped['depth_bin'].max() * depth_bin_size + depth_min
            ax.set_ylim(top=0, bottom=maximum_y_value)

    # Set the labels and legend
    #plt.xlabel('LPM Abundance')
    if preliminary:
        ax1.set_ylabel('binned depth [dbar]')
        ax1.set_title('Preliminary PIScO LPM Distribution ' + stationID)
    else:
        ax1.set_ylabel('binned depth [m]')
        ax1.set_title('LPM Distribution')
    #ax1.legend(loc='best')

    # Show the plot
    if maximum_y_value is None:
        fig.savefig(os.path.join(plot_path, 'particle_dist.png'))
    else:
        fig.savefig(os.path.join(plot_path, 'particle_dist_zoom.png'))
    plt.close(fig)

def plot_ctd_data(df, stationID:str, plot_path:str, maximum_y_value=None):
    """
    Generate a particle distribution plot based on the provided data.

    Parameters:
    - grouped: DataFrame containing the grouped data
    - stationID: str, the ID of the station
    - plot_path: str, the path where the plot will be saved
    - depth_bin_size: int, optional, the size of depth bins, default is 5
    - preliminary: bool, optional, flag indicating if the plot is preliminary, default is True
    - depth_min: int, optional, the minimum depth value, default is 0
    """

    fig, ax1 = plt.subplots(figsize=(10,15))
    fig.subplots_adjust(top=0.97) # Adjust the value as needed
    fig.subplots_adjust(bottom=0.2)

    ax1.invert_yaxis()

    ax2 =ax1.twiny()
    ax3 =ax1.twiny()
    ax4 =ax1.twiny()
    ax5 =ax1.twiny()

    axes = [ax1,ax2,ax3,ax4,ax5]
    names = ['interpolated_s','interpolated_t','interpolated_o','interpolated_chl','interpolated_z_factor']
    colors = ['red', 'blue', 'lime', 'cyan', 'black']
    positions = [0,40,80,120,160]


    for ax,name,color,pos in zip(axes,names,colors,positions):
        if name in df.columns:
            ax.plot(df[name], df['pressure [dbar]'], color=color, label=name)
            ax.set_xlabel(f' {name}',color=color)
            ax.spines['bottom'].set_color(color)
            ax.tick_params(axis='x', colors=color)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', pos))
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()
            if maximum_y_value is None:
                ylim = df['pressure [dbar]'].max()
            else:
                ylim = maximum_y_value
            ax.set_ylim(top=0, bottom=ylim)

    # Set the labels and legend
    #plt.xlabel('LPM Abundance')
    
    ax1.set_ylabel('depth [dbar]')
    ax1.set_title('Preliminary CTD data ' + stationID)

    # Show the plot
    if maximum_y_value is None:
        fig.savefig(os.path.join(plot_path, 'ctd.png'))
    else:
        fig.savefig(os.path.join(plot_path, 'ctd_zoom.png'))
    plt.close(fig)

def plot_position_hist(df,plot_path):
    plt.subplot(121)
    plt.hist(df["x"], bins=100)
    plt.xlim([0, IMAGE_SIZE])
    plt.xlabel("x-positions")
    plt.ylabel("Frequency")
    plt.title("Histrogram of \n x-positions")
    plt.subplot(122)
    plt.hist(df["y"], bins=100)
    plt.xlim([0, IMAGE_SIZE])
    plt.xlabel("y-positions")
    plt.title("Histrogram of \n y-positions")
    plt.savefig(os.path.join(plot_path,'position_hist.png'))
    plt.close()

def plot_2d_histogram(df,plot_path):
    hist2d = plt.hist2d(df["x"], df["y"], bins=np.linspace(0, IMAGE_SIZE-1, 500), norm=mcolors.PowerNorm(0.5))
    plt.colorbar()
    gca = plt.gca()
    gca.invert_xaxis()
    gca.invert_yaxis()
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.title("2D Histogram of object positions")
    plt.savefig(os.path.join(plot_path, '2d_hist.png'))
    plt.close()

def add_hist_value(df):
    def computeHist(x, y, nBins):
        step = IMAGE_SIZE / nBins
        hist = [[0 for j in range(nBins)] for i in range(nBins)]
        hist = np.zeros((nBins, nBins),dtype=np.int64)
        for row in range(len(x)):
            try:
                x_index = int(x[row] // step)
                y_index = int(y[row] // step)
                hist[y_index, x_index] += 1
            except:
                pass
        return hist
    
    hist = computeHist(df["x"], df["y"], 500)

    def get_hist_value(x, y):
        step = 2560 / 500
        x_index = int(x // step )
        y_index = int(y // step )
        return hist[x_index][y_index]
    
    df["position_hist_value"] = df.apply(lambda row: get_hist_value(row["x"], row["y"]), axis=1)

    return df

def calculate_regionprops(row):
    def regionprop2zooprocess(prop):
        """
        Calculate zooprocess features from skimage regionprops.
        Taken from morphocut

        Notes:
            - date/time specify the time of the sampling, not of the processing.
        """
        return {
            #"label": prop.label,
            # width of the smallest rectangle enclosing the object
            "width": prop.bbox[3] - prop.bbox[1],
            # height of the smallest rectangle enclosing the object
            "height": prop.bbox[2] - prop.bbox[0],
            # X coordinates of the top left point of the smallest rectangle enclosing the object
            "bx": prop.bbox[1],
            # Y coordinates of the top left point of the smallest rectangle enclosing the object
            "by": prop.bbox[0],
            # circularity : (4∗π ∗Area)/Perim^2 a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon
            "circ.": (4 * np.pi * prop.filled_area) / prop.perimeter ** 2,
            # Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
            "area_exc": prop.area,
            # Surface area of the object in square pixels
            "area_rprops": prop.filled_area,
            # Percentage of object’s surface area that is comprised of holes, defined as the background grey level
            "%area": 1 - (prop.area / prop.filled_area),
            # Primary axis of the best fitting ellipse for the object
            "major": prop.major_axis_length,
            # Secondary axis of the best fitting ellipse for the object
            "minor": prop.minor_axis_length,
            # Y position of the center of gravity of the object
            "centroid_y": prop.centroid[0],
            # X position of the center of gravity of the object
            "centroid_x": prop.centroid[1],
            # The area of the smallest polygon within which all points in the objet fit
            "convex_area": prop.convex_area,
            # Minimum grey value within the object (0 = black)
            "min_intensity": prop.intensity_min,
            # Maximum grey value within the object (255 = white)
            "max_intensity": prop.intensity_max,
            # Average grey value within the object ; sum of the grey values of all pixels in the object divided by the number of pixels
            "mean_intensity": prop.intensity_mean,
            # Integrated density. The sum of the grey values of the pixels in the object (i.e. = Area*Mean)
            "intden": prop.filled_area * prop.mean_intensity,
            # The length of the outside boundary of the object
            "perim.": prop.perimeter,
            # major/minor
            "elongation": np.divide(prop.major_axis_length, prop.minor_axis_length),
            # max-min
            "range": prop.max_intensity - prop.min_intensity,
            # perim/area_exc
            "perimareaexc": prop.perimeter / prop.area,
            # perim/major
            "perimmajor": prop.perimeter / prop.major_axis_length,
            # (4 ∗ π ∗ Area_exc)/perim 2
            "circex": np.divide(4 * np.pi * prop.area, prop.perimeter ** 2),
            # Angle between the primary axis and a line parallel to the x-axis of the image
            "angle": prop.orientation / np.pi * 180 + 90,
            # # X coordinate of the top left point of the image
            # 'xstart': data_object['raw_img']['meta']['xstart'],
            # # Y coordinate of the top left point of the image
            # 'ystart': data_object['raw_img']['meta']['ystart'],
            # Maximum feret diameter, i.e. the longest distance between any two points along the object boundary
            # 'feret': data_object['raw_img']['meta']['feret'],
            # feret/area_exc
            # 'feretareaexc': data_object['raw_img']['meta']['feret'] / property.area,
            # perim/feret
            # 'perimferet': property.perimeter / data_object['raw_img']['meta']['feret'],
            "bounding_box_area": prop.bbox_area,
            "eccentricity": prop.eccentricity,
            "equivalent_diameter": prop.equivalent_diameter,
            "euler_number": prop.euler_number,
            "extent": prop.extent,
            "local_centroid_col": prop.local_centroid[1],
            "local_centroid_row": prop.local_centroid[0],
            "solidity": prop.solidity,
        }
    #check if image is saved
    if row['saved']==1:
        # Load image
        img_path = row['full_path']
        img = imread(img_path)

        # Convert to grayscale if image is RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Reproduce Threshold from Segmenter
        thresh = cv2.threshold(
            cv2.bitwise_not(img),
            10,
            255,
            cv2.THRESH_BINARY,
        )[1]
        thresh = thresh.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key = cv2.contourArea)

        # Create a mask for the largest contour
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # Use regionprops on the mask
        props = measure.regionprops(mask, intensity_image=img)

        # Get all valid attributes from RegionProperties
        #valid_attributes = inspect.getmembers(props[0], lambda a:not(inspect.isroutine(a)))

        # Filter out the methods
        #valid_attributes = [a[0] for a in valid_attributes if not(a[0].startswith('_'))]

        # Only include valid attributes in the dictionary comprehension
        #print(props[0])
        region_data = regionprop2zooprocess(props[0])#{attr: getattr(props[0], attr) for attr in valid_attributes}
        #region_data['filename'] = row['filename']
        
        return pd.Series(region_data)
    
    else:
        return None

def modify_full_path(path):
    dirname, base_name = os.path.split(path)
    base_parts = base_name.split('_')
    new_base_name = '_'.join(base_parts[:-1]) + '.png'
    return os.path.join(dirname.replace('Crops', 'Images'), new_base_name)

# Function to reformat timestamp
def reformat_timestamp(timestamp):
    # Format: YYYYMMDD_HHh_MMm_SSs to YYYYMMDD-HHMMSS
    formatted_timestamp = re.sub(r'(\d{4})(\d{2})(\d{2})_(\d{2})h_(\d{2})m_(\d{2})s', r'\1\2\3-\4\5\6', timestamp)
    return formatted_timestamp

def parse_line(line, row):
    if line.startswith("b'TT"):
        temp_values = line[2:].rstrip("'").split('_')
        row['TT']= float(temp_values[1])
        row['T1']= float(temp_values[3])
        row['T2']= float(temp_values[5])
        row['TH']= float(temp_values[7])

def create_log_df(file_path):
    # Read the log file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    logging.info(f"Reading log file: {file_path}")

    data = []
    temp_data = {}
    indicator = 0
    counter = 1

    # Iterate through lines
    for line in lines:
        line = line.rstrip('\n')
        #print(line)
        # Check if line is a timestamp
        if "h" in line and "m" in line and "s" in line:
            if temp_data and 'timestamp' in temp_data:
                # If current timestamp already exists in data, update the existing dictionary
                if any(d['timestamp'] == temp_data['timestamp'] for d in data):
                    existing_data = [d for d in data if d['timestamp'] == temp_data['timestamp']][0]
                    existing_data.update(temp_data)
                else:
                    data.append(temp_data)
                temp_data = {}
            temp_data['timestamp'] = line
        else:
            # Parse line according to message type
            if line.startswith("b'TT"):
                temp_values = line[2:].rstrip("'").split('_')
                if len(temp_values) != 20: #adding this for debugging...
                    logging.warning(
                        f"Unexpected TT line format: {line}. "
                        f"Parsing TT data: {temp_values}, "
                        f"number of values: {len(temp_values)}"
                    )
                    logging.info(f"skipping line in file {file_path}: {line}")
                    continue
                else:
                    temp_data['TT'] = float(temp_values[1])
                    temp_data['T1'] = float(temp_values[3])
                    temp_data['T2'] = float(temp_values[5])
                    temp_data['C1'] = float(temp_values[7])
                    temp_data['C2'] = float(temp_values[9])
                    temp_data['TH'] = float(temp_values[11])
            elif line.startswith('Restart Tag'):
                temp_data['restart'] = True
                indicator = 0
            elif line == 'Relock':
                temp_data['relock'] = True
                indicator = counter
                counter += 1
            temp_data['TAG_event'] = indicator

    # If there is data waiting after the last line, add it
    if temp_data:
        if any(d['timestamp'] == temp_data['timestamp'] for d in data):
            existing_data = [d for d in data if d['timestamp'] == temp_data['timestamp']][0]
            existing_data.update(temp_data)
        else:
            data.append(temp_data)

    # Create a dataframe from the data list
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d_%Hh_%Mm_%Ss")
    df.set_index('timestamp', inplace=True)

    # Ensure 'restart' and 'relock' columns exist
    # (bc i got this error: KeyError: "None of [Index(['restart', 'relock'], dtype='object')] are in the [columns])
    if 'restart' not in df.columns:
        df['restart'] = False
    if 'relock' not in df.columns:
        df['relock'] = False

    # Replace NaN values in 'relock' and 'restart' columns with False
    df[['restart', 'relock']] = df[['restart', 'relock']].fillna(False).astype(bool)
    cols = ['TT', 'T1', 'T2', 'C1','C2','TH']  # list of your column names
    for col in cols:
        df[col] = df[col].interpolate()
    
    return df

def calculate_umap_embeddings(df, reducer, scaler):
    selected_features = [
       # 'pressure [dbar]', 
       # 'temperature', 
       # 'area', 
       # 'w', 
       # 'h', 
       'esd', 
       # 'interpolated_s', 
       # 'interpolated_o',
       # 'interpolated_t', 
       # 'interpolated_chl', 
       'object_area_exc', 
       'object_area_rprops', 
       'object_%area',
       'object_major_axis_len', 
       'object_minor_axis_len', 
       'object_centroid_y', 
       'object_centroid_x', 
       'object_convex_area',
       'object_min_intensity', 
       'object_max_intensity', 
       'object_mean_intensity', 
       'object_int_density', 
       'object_perimeter',
       'object_elongation', 
       'object_range', 
       'object_perim_area_excl', 
       'object_perim_major', 
       'object_circularity_area_excl', 
       'object_angle',
       'object_boundbox_area', 
       'object_eccentricity', 
       'object_equivalent_diameter',
       'object_euler_nr', 
       'object_extent', 
       'object_local_centroid_col', 
       'object_local_centroid_row',
       'object_solidity', 
       'TAG_event', 
       'part_based_filter'
    ]
    df_selected = df[selected_features]
    
    df_selected_scaled = scaler.transform(df_selected)

    # Then transform the UMAP model with the scaled data
    embedding = reducer.transform(df_selected_scaled)
    
    #add embedding to database
    df.drop(['umap_x', 'umap_y'], axis=1, inplace=True, errors='ignore')
    df['umap_x']=embedding[:, 0]
    df['umap_y']=embedding[:, 1]

    return df

def analyze_profiles(profiles_dir, dest_folder, engine, small=False, add_ctd=True, calc_props=True, calc_umap=True, plotting=True, log_directory=None):
    os.makedirs(dest_folder, exist_ok=True)
    if calc_umap:
        print('loading UMAP model...')
        reducer_path = '/home/fanny/UMAP_scaler/umap_reducer.pkl'
        scaler_path = '/home/fanny/UMAP_scaler/standard_scaler.pkl'
        with open(reducer_path, 'rb') as f:
            reducer = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    for folder in os.listdir(profiles_dir):
        if '°' in folder:
            folder_corr = folder.replace('°', 'deg')
        print('working on ', folder_corr)
        logging.info('working on ' + folder_corr)
        profile_data_dir = os.path.join(result_dir, folder, 'Data')
        df = gen_crop_df(profile_data_dir, small=small, size_filter=0)
        print(len(df.index), 'particles found.')
        df['overview_path'] = df['full_path'].apply(modify_full_path)
        #print(df['full_path'][0])
        ctd_file = os.path.join(os.path.dirname(result_dir), 'CTD_preliminary_calibrated', f'met_181_1_{folder.split("_")[1].split("-")[1]}.ctd')               
        if add_ctd:
            print('adding ctd data...')
            df = add_ctd_data(ctd_file, df)
        # Add predictions
        prediction_file = os.path.join(profile_data_dir, 'ViT_predictions.csv')  # Example path
        if os.path.exists(prediction_file):
            print('Adding predictions...')
            prediction_df = pd.read_csv(prediction_file)
            df = add_prediction(df, prediction_df)
        if calc_props:
            tqdm.pandas()
            print('calculating regionprops...')
            new_df = df.progress_apply(calculate_regionprops, axis=1)
            # Concatenate the original DataFrame with the new one
            df = pd.concat([df, new_df], axis=1)
        if log_directory is not None:
            print('adding log info...')
            timestamp = folder_corr[-13:]
            # Convert timestamp to datetime object
            date_time_obj = datetime.datetime.strptime(timestamp, '%Y%m%d-%H%M')
            min_diff = datetime.timedelta(days=365*1000)  # initialize with a big time difference
            closest_file = None

            # Iterate over all files in the directory
            for filename in os.listdir(log_directory):
                # Check if filename is a Templog
                if '__Templog.txt' in filename:
                    # Extract timestamp from filename and convert to datetime object
                    file_timestamp = filename[:16]
                    file_datetime = datetime.datetime.strptime(file_timestamp, '%Y%m%d_%Hh_%Mm')

                    # Calculate time difference
                    diff = abs(date_time_obj - file_datetime)

                    # If this file is closer, update min_diff and closest_file
                    if diff < min_diff:
                        min_diff = diff
                        closest_file = filename

            if closest_file is None:
                print("Logfile not found")
            else:
                file_path = os.path.join(log_directory, closest_file)
                file_size = os.path.getsize(file_path)  # Get file size in bytes
                print(f"Closest logfile: {closest_file}, Size: {file_size} bytes")
            
            # Read the log file and parse the relevant data

            df_log = create_log_df(file_path)

            # Match the data with the profile dataframe
            df.drop(['TT_x', 'T1_x', 'T2_x', 'TH_x', 'restart_x', 'relock_x', 'Time_log_x', 'TT_y', 'T1_y', 'T2_y', 'TH_y', 'restart_y', 'relock_y', 'Time_log_y', 'TT', 'T1', 'T2', 'TH', 'restart', 'relock', 'Time_log'], axis=1, inplace=True, errors='ignore')
            # Convert the timestamps in both dataframes to datetime format
            df['timestamp'] = df['date-time']

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d-%H%M%S%f')

            # Sort the dataframes by the timestamp
            df = df.sort_values('timestamp')
            df_log = df_log.sort_values('timestamp')

            # Use merge_asof to merge the two dataframes, finding the nearest match on the timestamp
            df_combined = pd.merge_asof(df, df_log, left_on='timestamp', right_on='timestamp', direction='backward')
            df_combined.drop('timestamp', axis=1, inplace=True)

        #Sort by filename and add obj_id
        sorted_df = df_combined.sort_values(by='filename')
        sorted_fn_list = sorted_df['filename'].tolist()
        obj_ids = []
        id_cnt = 0
        for img in sorted_fn_list:
            curr_id = id_cnt
            obj_ids.append('obj_'+str(curr_id))
            id_cnt = id_cnt+1
        sorted_df['obj_id'] = obj_ids

        #Add particle count based filter for filtering out images that are potentially obscured by schlieren or bubbles
        df_unique = sorted_df[['date-time', 'pressure [dbar]', 'depth [m]', 'img_id','temperature','overview_path','interpolated_s','interpolated_t','interpolated_o','interpolated_chl','interpolated_z_factor','restart','relock','TAG_event']].drop_duplicates()
        df_count = sorted_df.groupby('date-time').size().reset_index(name='count')
        df_unique = df_unique.merge(df_count, on='date-time', how='left')
        df_unique = df_unique.sort_values('pressure [dbar]')

        # Filter the data
        df_unique['part_based_filter'] = df_unique['count'].apply(lambda x: 0 if x < df_unique['count'].std() else 1)

        # Merge 'df_unique' back to 'sorted_df' to create 'part_based_filter' column in 'df'
        sorted_df = sorted_df.merge(df_unique[['date-time', 'part_based_filter']], on='date-time', how='left')

        #Calculate UMAP embeddings
        if calc_umap:
            print('calculating UMAP embeddings...')
            sorted_df = calculate_umap_embeddings(sorted_df, reducer, scaler)

        #Add to database
        sorted_df.to_sql(folder_corr, engine, if_exists='replace', index=False)
        print('... added to database.')
        logging.info('... added to database.')

        if plotting:
            print('plotting...')
            logging.info('generate plots...')
            plot_path = os.path.join(dest_folder, folder)
            os.makedirs(plot_path, exist_ok=True)
            plot_histogram(df, plot_path)
            plot_position_hist(df, plot_path)
            plot_2d_histogram(df, plot_path)
            press_min = df['pressure [dbar]'].min()-10
            depth_bin_size = 1
            _, pivoted_df = populate_esd_bins_pressure(df,  depth_bin_size=depth_bin_size, esd_bins=np.array([0,125,250,500,1000,100000]))
            plot_particle_dist(pivoted_df, folder, plot_path, depth_bin_size=depth_bin_size, preliminary=True, depth_min=press_min)
            plot_particle_dist(pivoted_df, folder, plot_path, depth_bin_size=depth_bin_size, preliminary=True, depth_min=press_min, maximum_y_value=500)
            plot_ctd_data(df, folder, plot_path)
            plot_ctd_data(df, folder, plot_path, maximum_y_value=500)
        
        print('Done.')
    
def convert_to_decimal(coord):
    # Check if the coordinate is valid
    if coord is None:
        return None
    # Extract degrees, minutes, and direction
    match = re.match(r'(\d+)°(\d+)([NSWE])', coord)
    if match:
        degrees, minutes, direction = match.groups()
        decimal = int(degrees) + int(minutes) / 60.0
        if direction in ['S', 'W']:  # South and West are negative
            decimal *= -1
        return decimal
    return None


def extract_coordinates(path):
    match = re.search(r'(\d+°\d+[NS])-(\d+°\d+[EW])', path)
    if match:
        lat, lon = match.groups()
        return lat, lon
    return None, None

def extract_coords_from_yaml(yaml_file):
    lat, lon = None, None
    with open(yaml_file, 'r') as f:
        for line in f:
            if 'image-latitude:' in line and lat is None:
                lat = float(line.split(':')[1].strip())
            elif 'image-longitude:' in line and lon is None:
                lon = float(line.split(':')[1].strip())
            if lat is not None and lon is not None:
                break
    return lat, lon


def process_crop_data(df, dship_id):
    """
    Process the crop data DataFrame by adding object IDs, splitting date-time, and extracting coordinates.
    Parameters:
    - df: DataFrame containing the crop data
    - dship_id: str, the ID for the action of the research vessel
    """
    # Add object ID column
    df['object_id'] = dship_id + df['img_id'].astype(str) + '_' + df['index'].astype(str) 

    #split date-time
    df[['date', 'time']] = df['date-time'].str.split('-', expand=True)


    # Apply the extraction function to the full_path column
    df[['lat', 'lon']] = df['full_path'].apply(
        lambda x: pd.Series(extract_coordinates(x))
    )

    # Convert latitude and longitude to decimal format
    df['lat'] = df['lat'].apply(convert_to_decimal)
    df['lon'] = df['lon'].apply(convert_to_decimal)
    return df

def filter_defect_crops(df):
    df_filtered = df[(df['TAG_event'] == 0) & (df['part_based_filter'] == 0)]
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def rename_for_ecotaxa(df, mapping_csv=None, sep="\t", sample_profile_id=None, predicted=False):    
    if mapping_csv is not None:
        # Load CSV files
        polytaxo_classes_df = pd.read_csv(mapping_csv, sep=sep)

        # Add annotation status
        df['object_annotation_status'] = 'predicted'

        # Create mapping dictionary
        mapping_dict = dict(zip(
            polytaxo_classes_df["Dataset Class NamePolyTaxo Description"],
            polytaxo_classes_df["PolyTaxo Description"]
        ))

        # Columns to update
        columns_to_replace = ["top1", "top2", "top3", "top4", "top5"]

        # Define regex pattern to split on space, semicolon, colon, or slash
        split_pattern = r"[ ;:/]"

        # Replace values using mapping_dict, extract first word, and replace underscores with spaces
        df[columns_to_replace] = df[columns_to_replace].replace(mapping_dict).apply(
            lambda col: col.astype(str).apply(
                lambda x: re.split(split_pattern, x)[0].replace("_", " ") if pd.notna(x) else x
            )
        )

        # Adjust header names
        rename_mapping = {
            'pressure [dbar]': 'object_pressure',
            'date': 'object_date',
            'time': 'object_time',
            'filename': 'img_file_name',
            'depth [m]': 'object_depth_min',
            'area': 'object_area',
            'esd': 'object_esd',
            'top1': 'object_annotation_category',
            'top2': 'object_annotation_category_2',
            'top3': 'object_annotation_category_3',
            'top4': 'object_annotation_category_4',
            'top5': 'object_annotation_category_5',
            'prob1': 'object_prob_1',
            'prob2': 'object_prob_2',
            'prob3': 'object_prob_3',
            'prob4': 'object_prob_4',
            'prob5': 'object_prob_5',
            'lat': 'object_lat',
            'lon': 'object_lon',
            'w': 'object_width',
            'h': 'object_height',
            'interpolated_s': 'object_interpolated_s',
            'interpolated_o': 'object_interpolated_o',
            'interpolated_chl2_raw': 'object_interpolated_chl',
            'interpolated_t': 'object_interpolated_t',
            'img_id': 'img_rank',
            'cruise': 'sample_cruise',
            'dship_id': 'sample_dship_id',
            'instrument': 'sample_instrument',
            'date-time': 'object_date-time',
            'index': 'object_index',
            'temperature': 'object_temperature',
            'mean_raw': 'object_mean_raw',
            'std_raw': 'object_std_raw',
            'mean': 'object_mean',
            'std': 'object_std',
            'full_path': 'object_full_path',
            'saved': 'object_saved',
            'x': 'object_x',
            'y': 'object_y',
            'bound_box_x': 'object_bound_box_x',
            'bound_box_y': 'object_bound_box_y',
            'fullframe_path': 'object_fullframe_path',
            'interpolated_z_factor': 'object_interpolated_z_factor',
            'TT': 'object_tt',
            'T1': 'object_t1',
            'T2': 'object_t2',
            'TH': 'object_th',
            'C1': 'object_c1',
            'C2': 'object_c2',
            'part_based_filter': 'object_part_based_filter',
            'restart': 'object_restart',
            'relock': 'object_relock',
            'TAG_event': 'object_TAG_event',
        }
        df.rename(columns=rename_mapping, inplace=True)
        df['object_depth_max'] = df['object_depth_min']
        df['sample_id'] = sample_profile_id
        
        # Ensure 'object_time' has 8 elements by padding with leading zeros
        df['object_time'] = df['object_time'].apply(lambda x: x.zfill(8) if isinstance(x, str) else x)

        # Define annotation columns
        annotation_columns = ['object_annotation_category', 'object_annotation_category_2', 'object_annotation_category_3', 'object_annotation_category_4', 'object_annotation_category_5']

        # Find rows with empty cells in annotation columns
        rows_to_drop = df[annotation_columns].isnull().any(axis=1)

        # Get the filenames of the images to drop
        images_to_drop = df.loc[rows_to_drop, 'img_file_name'].tolist()

        # Drop rows with empty annotation cells
        df = df[~rows_to_drop].reset_index(drop=True)
        dtype_row = [determine_dtype(df.dtypes[col]) for col in df.columns]
        #df.loc[-1] = dtype_row  # Add the dtype row
        # Insert the dtype_row after the header (as the second row)
        df = pd.concat([df.iloc[:0], pd.DataFrame([dtype_row], columns=df.columns), df.iloc[0:]]).reset_index(drop=True)

    else:
        if predicted:
            # Add annotation status
            df['object_annotation_status'] = 'predicted'

        # Adjust header names without mapping
        rename_mapping = {
            'pressure [dbar]': 'object_pressure',
            'date': 'object_date',
            'time': 'object_time',
            'filename': 'img_file_name',
            'depth [m]': 'object_depth_min',
            'area': 'object_area',
            'esd': 'object_esd',
            'lat': 'object_lat',
            'lon': 'object_lon',
            'w': 'object_width',
            'h': 'object_height',
            'interpolated_s': 'object_interpolated_s',
            'interpolated_o': 'object_interpolated_o',
            'interpolated_chl2_raw': 'object_interpolated_chl',
            'interpolated_t': 'object_interpolated_t',
            'img_id': 'img_rank',
            'cruise': 'sample_cruise',
            'dship_id': 'sample_dship_id',
            'instrument': 'sample_instrument',
            'date-time': 'object_date-time',
            'index': 'object_index',
            'temperature': 'object_temperature',
            'mean_raw': 'object_mean_raw',
            'std_raw': 'object_std_raw',
            'mean': 'object_mean',
            'std': 'object_std',
            'full_path': 'object_full_path',
            'saved': 'object_saved',
            'x': 'object_x',
            'y': 'object_y',
            'bound_box_x': 'object_bound_box_x',
            'bound_box_y': 'object_bound_box_y',
            'fullframe_path': 'object_fullframe_path',
            'interpolated_z_factor': 'object_interpolated_z_factor',
            'TT': 'object_tt',
            'T1': 'object_t1',
            'T2': 'object_t2',
            'TH': 'object_th',
            'C1': 'object_c1',
            'C2': 'object_c2',
            'part_based_filter': 'object_part_based_filter',
            'restart': 'object_restart',
            'relock': 'object_relock',
            'TAG_event': 'object_TAG_event',
            'top1': 'object_annotation_category',
            'top2': 'object_annotation_category_2',
            'top3': 'object_annotation_category_3',
            'top4': 'object_annotation_category_4',
            'top5': 'object_annotation_category_5',
            'prob1': 'object_prob_1',
            'prob2': 'object_prob_2',
            'prob3': 'object_prob_3',
            'prob4': 'object_prob_4',
            'prob5': 'object_prob_5',
            'is_ood': 'object_is_ood',
            'entropy': 'object_entropy',
            # Add any other columns that need renaming
        }
        df.rename(columns=rename_mapping, inplace=True)
        if "object_depth_min" in df.columns:
            df['object_depth_max'] = df['object_depth_min']
        df['sample_id'] = sample_profile_id

        # Keep only specified columns after renaming
        columns_to_keep = [
            'object_pressure',
            'object_date',
            'object_time',
            'img_file_name',
            'object_depth_min',
            'object_depth_max',
            'object_area',
            'object_esd',
            'object_lat',
            'object_lon',
            'object_width',
            'object_height',
            'object_bound_box_w', 
            'object_bound_box_h', 
            'object_circularity', 
            'object_area_exc', 
            'object_area_rprops', 
            'object_%area', 
            'object_major_axis_len', 'object_minor_axis_len', 'object_centroid_y', 
            'object_centroid_x', 'object_convex_area', 'object_min_intensity', 
            'object_max_intensity', 'object_mean_intensity', 'object_int_density', 
            'object_perimeter', 'object_elongation', 'object_range', 
            'object_perim_area_excl', 'object_perim_major', 
            'object_circularity_area_excl', 'object_angle', 'object_boundbox_area', 
            'object_eccentricity', 'object_equivalent_diameter', 'object_euler_nr', 
            'object_extent', 'object_local_centroid_col', 'object_local_centroid_row', 
            'object_solidity', 
            'object_interpolated_s',
            'object_interpolated_o',
            'object_interpolated_chl',
            'object_interpolated_t',
            'object_entropy',
            'object_annotation_category',
            'object_annotation_status',
            'object_prob_1',
            #'img_rank',
            'object_id',
            'sample_cruise',
            'sample_id',
            'object_full_path',
        ]
        
        # Drop all columns except those specified
        df = df.loc[:, df.columns.intersection(columns_to_keep)]
        
        # Ensure 'object_time' has 8 elements by padding with leading zeros
        def format_time(x):
            if pd.isna(x):
                return x
            try:
                # Convert to string if not already
                x = str(x).strip()
                # Remove any colons if present
                x = x.replace(':', '')
                # Pad with leading zeros to ensure 8 characters (HHMMSSMS)
                return x.zfill(8)
            except:
                return x
        # Apply the formatting function
        df.loc[:, 'object_time'] = df['object_time'].apply(format_time)

         # Add dtype row using MultiIndex
        dtypes = pd.Series(df.dtypes).map(lambda x: '[f]' if pd.api.types.is_numeric_dtype(x) else '[t]')
        df.columns = pd.MultiIndex.from_tuples(
            [(col, dtypes[col]) for col in df.columns],
            names=['header', 'dtype']
        )

        # dtype_row = [determine_dtype(df.dtypes[col]) for col in df.columns]
        # # Insert the dtype_row after the header (as the second row)
        # df = pd.concat([df.iloc[:0], pd.DataFrame([dtype_row], columns=df.columns), df.iloc[0:]]).reset_index(drop=True)

    return df

def get_ctd_profile_id(cruise_base: str, profile: str) -> str:
    """
    Extract CTD profile ID from metadata CSV file or prompt user for input.
    Log profiles with missing CTD IDs to a CSV file.
    
    Args:
        cruise_base (str): Base directory for the cruise
        profile (str): Profile folder name
        
    Returns:
        str: CTD profile ID or None if not found/provided
    """
    # Define path for logging missing CTD IDs
    #missing_ctd_log = os.path.join(os.path.dirname(cruise_base), "missing_ctd_profiles.csv")
    
    # Construct path to metadata CSV
    csv_path = os.path.join(
        cruise_base,
        profile,
        profile + "_Metadata",
        profile + ".csv"
    )
    
    try:
        # Try to read the CSV file
        with open(csv_path, 'r') as f:
            for line in f:
                if 'CTDprofileid' in line:
                    # Extract profile ID after comma
                    ctd_id = line.split(',')[1].strip()[-3:]
                    return ctd_id
                    
    except FileNotFoundError:
        print(f"Warning: Metadata CSV not found for {csv_path}")
    except Exception as e:
        print(f"Error reading metadata for {profile}: {str(e)}")
    
    return None

# def rename_for_ecotaxa(df, mapping_csv=None, sep="\t", sample_profile_id=None):
#     """
#     Rename and format DataFrame columns according to EcoTaxa requirements.
    
#     Args:
#         df (pd.DataFrame): Input DataFrame
#         mapping_csv (str, optional): Path to CSV with class name mappings
#         sep (str, optional): Separator used in mapping CSV. Defaults to "\t"
#         sample_profile_id (str, optional): Profile ID to add to the DataFrame
        
#     Returns:
#         pd.DataFrame: Reformatted DataFrame with EcoTaxa compatible column names
#     """
#     # Standard column renaming dictionary
#     rename_mapping = {
#         'pressure [dbar]': 'object_pressure',
#         'date': 'object_date',
#         'time': 'object_time',
#         'filename': 'img_file_name',
#         'depth [m]': 'object_depth_min',
#         'area': 'object_area',
#         'esd': 'object_esd',
#         'lat': 'object_lat',
#         'lon': 'object_lon',
#         'w': 'object_width',
#         'h': 'object_height',
#         'interpolated_s': 'object_interpolated_s',
#         'interpolated_o': 'object_interpolated_o',
#         'interpolated_chl2_raw': 'object_interpolated_chl',
#         'interpolated_t': 'object_interpolated_t',
#         'img_id': 'img_rank'
#     }

#     # Create copy to avoid modifying original
#     df = df.copy()
    
#     # Rename columns
#     df.rename(columns=rename_mapping, inplace=True)
    
#     if mapping_csv:
#         # Process class mappings
#         polytaxo_classes_df = pd.read_csv(mapping_csv, sep=sep)
#         mapping_dict = dict(zip(
#             polytaxo_classes_df["Dataset Class NamePolyTaxo Description"],
#             polytaxo_classes_df["PolyTaxo Description"]
#         ))
        
#         # Add annotation columns
#         df['object_annotation_status'] = 'predicted'
        
#         # Process prediction columns
#         pred_cols = ["top1", "top2", "top3", "top4", "top5"]
#         new_pred_cols = [f"object_annotation_category_{'' if i==1 else i}" for i in range(1,6)]
#         prob_cols = [f"object_prob_{i}" for i in range(1,6)]
        
#         # Rename prediction columns
#         for old, new in zip(pred_cols, new_pred_cols):
#             if old in df.columns:
#                 df[new] = df[old].map(mapping_dict).str.split(r"[ ;:/]").str[0].str.replace("_", " ")
#                 df.drop(old, axis=1, inplace=True)
        
#         # Rename probability columns if they exist
#         for i, prob in enumerate(["prob1", "prob2", "prob3", "prob4", "prob5"], 1):
#             if prob in df.columns:
#                 df[f"object_prob_{i}"] = df[prob]
#                 df.drop(prob, axis=1, inplace=True)
                
#     else:
#         # Keep only essential columns for non-prediction data
#         columns_to_keep = [
#             'object_pressure',
#             'object_date',
#             'object_time',
#             'img_file_name',
#             'object_depth_min',
#             'object_area',
#             'object_esd',
#             'object_lat',
#             'object_lon',
#             'object_width',
#             'object_height',
#             'object_interpolated_s',
#             'object_interpolated_o',
#             'object_interpolated_chl',
#             'object_interpolated_t',
#             'object_id',
#             'sample_cruise',
#             'sample_id',
#             'object_full_path'
#         ]
#         df = df[columns_to_keep]

#     # Add consistent fields
#     df['object_depth_max'] = df['object_depth_min']
#     if sample_profile_id:
#         df['sample_id'] = sample_profile_id

#     # Format time values
#     df['object_time'] = df['object_time'].apply(
#         lambda x: str(x).strip().replace(':', '').zfill(8) if pd.notna(x) else x
#     )

#     # Add dtype row using MultiIndex
#     dtypes = pd.Series(df.dtypes).map(lambda x: '[f]' if pd.api.types.is_numeric_dtype(x) else '[t]')
#     df.columns = pd.MultiIndex.from_tuples(
#         [(col, dtypes[col]) for col in df.columns],
#         names=['header', 'dtype']
#     )

#     return df

def determine_dtype(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return '[f]' 
    elif pd.api.types.is_string_dtype(dtype):
        return '[t]'
    else:
        return 'other'

def add_scale_bar(image_path, output_path, pixel_resolution=23, scale_length_mm=1):
    """
    Adds a scale bar below the image and saves it to the output path.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with the scale bar.
        pixel_resolution (int): Micrometers per pixel.
        scale_length_mm (int): Length of the scale bar in millimeters.
    """
    # Open the image
    img = Image.open(image_path)

    # Calculate the scale bar length in pixels
    scale_length_px = int((scale_length_mm * 1000) / pixel_resolution)

    # Define the height of the additional space for the scale bar and text
    extra_height = 50  # Space for the scale bar and text
    bar_height = 5     # Height of the scale bar in pixels
    margin = 20        # Margin from the bottom and sides

    # Calculate minimum required width
    min_width = scale_length_px + 2 * margin
    pad_left = pad_right = 0
    if img.width < min_width:
        pad_total = min_width - img.width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Pad the image
        padded_img = Image.new("RGB", (min_width, img.height), "white")
        padded_img.paste(img, (pad_left, 0))
        img = padded_img

    # Create a new image with extra space below
    new_img = Image.new("RGB", (img.width, img.height + extra_height), "white")
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    # Define the scale bar position and size
    bar_x_start = margin
    bar_x_end = bar_x_start + scale_length_px
    bar_y_start = img.height + (extra_height - bar_height) // 2
    bar_y_end = bar_y_start + bar_height

    # Draw the scale bar (black rectangle)
    draw.rectangle([bar_x_start, bar_y_start, bar_x_end, bar_y_end], fill="black")

    # Add text below the scale bar
    text = f"{scale_length_mm} mm"
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = bar_x_start + (scale_length_px - text_width) // 2
    text_y = bar_y_end + 5  # Slight margin below the bar
    draw.text((text_x, text_y), text, fill="black", font=font)

    # Save the modified image
    new_img.save(output_path)

def estimate_zip_size(file_paths, compression_ratio=1.0):
    total_size = sum(os.path.getsize(fp) for fp in file_paths)
    return total_size * compression_ratio

def split_files_by_zip_size(file_paths, max_zip_size_mb, compression_ratio=1.0):
    max_zip_size_bytes = max_zip_size_mb * 1024 * 1024
    groups = []
    current_group = []
    current_size = 0

    for fp in file_paths:
        file_size = os.path.getsize(fp) * compression_ratio
        if current_size + file_size > max_zip_size_bytes and current_group:
            groups.append(current_group)
            current_group = []
            current_size = 0
        current_group.append(fp)
        current_size += file_size
    if current_group:
        groups.append(current_group)
    return groups

def create_ecotaxa_zips(output_folder, df, profile_name, max_zip_size_mb=500, compression_ratio=1.0, copy_images=True, add_scale_bar_to_deconv=False, pixel_resolution=23, scale_length_mm=1):
    """Create EcoTaxa-compatible zip files with images and metadata."""
    for crop_type in ['crops', 'deconv_crops']:
        crop_folder = os.path.join(output_folder, "EcoTaxa", crop_type)
        excluded_folder = os.path.join(output_folder, f"{crop_type}_excluded")
        
        # Convert MultiIndex columns back to single level for processing
        df_single = df.copy()
        df_single.columns = df_single.columns.get_level_values('header')

        # Get source folder and images
        source_folder = None
        if crop_type == 'crops':
            source_images = set(os.path.basename(row['object_full_path']) for _, row in df_single.iterrows())
            source_folder = os.path.dirname(df_single.iloc[2]['object_full_path'])
        else:
            source_images = set(os.path.basename(row['object_full_path']) for _, row in df_single.iterrows())
            source_folder = os.path.dirname(df_single.iloc[2]['object_full_path']).replace('/Crops', '/Deconv_crops')

        if not source_folder or not os.path.exists(source_folder):
            print(f"Source folder not found for {crop_type}")
            continue

        # Prepare metadata DataFrame
        df_ET = df_single.drop(columns=['object_full_path'], errors='ignore')
        
        # Get image paths for included files
        # image_files = [f for f in os.listdir(source_folder) 
        #               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        # image_paths = [os.path.join(source_folder, f) for f in image_files]
        # Get image paths for included files
        image_paths = [os.path.join(source_folder, f) for f in source_images 
                    if os.path.exists(os.path.join(source_folder, f))]

        # Split into groups if needed
        groups = split_files_by_zip_size(image_paths, max_zip_size_mb, compression_ratio)
        print(f"Processing {len(groups)} groups for {source_folder}")

        if len(groups) == 1:
            # For single group, create TSV file directly in source folder
            metadata_file = f"ecotaxa_{profile_name}.tsv"
            metadata_path = os.path.join(source_folder, metadata_file)
            
            # Get dtype information
            dtypes = pd.Series(df_ET.dtypes).map(lambda x: '[f]' if pd.api.types.is_numeric_dtype(x) else '[t]')
            
            # Create the TSV file with proper header and dtype row
            with open(metadata_path, 'w') as f:
                f.write('\t'.join(df_ET.columns) + '\n')
                f.write('\t'.join(dtypes.values) + '\n')
                
            # Write data rows
            df_ET.to_csv(metadata_path, sep='\t', index=False, mode='a', header=False)

            # Create zip file from original folder
            zip_path = os.path.join(output_folder, f"{crop_type}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from source folder
                for file in os.listdir(source_folder):
                    file_path = os.path.join(source_folder, file)
                    if os.path.isfile(file_path):  # Only add files, not directories
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
            
            print(f"Created zip file from source folder: {zip_path}")

        else:
            # Create folders for multiple groups
            os.makedirs(crop_folder, exist_ok=True)
            os.makedirs(excluded_folder, exist_ok=True)

            # Move excluded files
            for filename in os.listdir(source_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    if filename not in source_images:
                        shutil.move(
                            os.path.join(source_folder, filename),
                            os.path.join(excluded_folder, filename)
                        )

            # Process each group
            for i, group in enumerate(groups):
                # Create part folder
                part_folder = os.path.join(output_folder,  
                                     f"{crop_type}_part{i+1}_upload" if len(groups) > 1 else crop_type)
                os.makedirs(part_folder, exist_ok=True)

                # Copy/process images
                for img_path in group:
                    dest_path = os.path.join(part_folder, os.path.basename(img_path))
                    if crop_type == 'deconv_crops' and add_scale_bar_to_deconv:
                        add_scale_bar(img_path, dest_path, pixel_resolution, scale_length_mm)
                    else:
                        shutil.copy2(img_path, dest_path)

                # Filter and save metadata
                part_images = [os.path.basename(fp) for fp in group]
                part_metadata = df_ET[df_ET['img_file_name'].isin(part_images)]
                
                # Save TSV with dtype row
                metadata_file = f"ecotaxa_{profile_name}.tsv"
                metadata_path = os.path.join(part_folder, metadata_file)
                
                # Get dtype information
                dtypes = pd.Series(part_metadata.dtypes).map(lambda x: '[f]' if pd.api.types.is_numeric_dtype(x) else '[t]')
                
                # Create the TSV file with proper header and dtype row
                with open(metadata_path, 'w') as f:
                    # Write header row
                    f.write('\t'.join(part_metadata.columns) + '\n')
                    # Write dtype row
                    f.write('\t'.join(dtypes.values) + '\n')
                    
                # Write data rows
                part_metadata.to_csv(metadata_path, sep='\t', index=False, mode='a', header=False)

                # Create zip file
                zip_path = f"{part_folder}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(part_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, part_folder)
                            zipf.write(file_path, arcname)
            
                print(f"Created zip file: {zip_path}")
                
                # Delete the unzipped folder after creating zip
                shutil.rmtree(part_folder)
                print(f"Deleted unzipped folder: {part_folder}")

        # Delete the EcoTaxa folder structure if empty
        ecotaxa_folder = os.path.join(output_folder, "EcoTaxa")
        if os.path.exists(ecotaxa_folder) and not os.listdir(ecotaxa_folder):
            shutil.rmtree(ecotaxa_folder)
            print(f"Deleted empty EcoTaxa folder structure: {ecotaxa_folder}")