# This script is intended to use after segmentation to create a df and add it to the sql database. Some columns are still missing, they are added in load_profiles_from_database.ipynb
#TODO: Add the missing operations here.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import datetime

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

def gen_crop_df(path:str, small:bool, size_filter:int = 0):
    """
    A function to generate a DataFrame from a directory of CSV files, with options to filter out small objects.
    
    Parameters:
    path (str): The path to the directory containing the CSV files.
    small (bool): A flag indicating whether to filter out small objects.
    
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
            df = pd.read_csv(file, delimiter=",", header=None)
            if len(df.columns) == 44:#was 8
                df.insert(0,'',id)            
                dataframes.append(df)
                id+=1
            else:
                continue
        except EmptyDataError:
            empty_file_counter += 1
            print(f"File {file} is empty")
    
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

    # Split the 'filename' column
    split_df = df['filename'].str.split('_', expand=True)
    if small:# bug fix for segmenter where small objects are saved with _mask.png extension instead of .png: needs to be fixed if segmenter is fixed
        headers = ["date-time", "pressure", "temperature", "index", "mask_ext"]
        split_df.columns = headers
        split_df.drop("mask_ext", axis=1, inplace=True)
    else:
        headers = ["date-time", "pressure", "temperature", "index"]
        split_df.columns = headers
    split_df['pressure'] = split_df['pressure'].str.replace('bar', '', regex=False).astype(float)
    split_df['temperature'] = split_df['temperature'].str.replace('C', '', regex=False).astype(float)
    split_df['index'] = split_df['index'].str.replace('.png', '', regex=False).astype(int)

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([split_df, df], axis=1)

    # Extend the original 'filename' column
    df['full_path'] = df.apply(lambda x: join_strings(directory, x['filename']), axis=1)
    #df = df.drop('filename', axis=1)

    df['esd'] = df['area'].apply(area_to_esd).round(2)
    df['pressure'] = (df['pressure']-1)*10
    df.rename(columns={'pressure': 'pressure [dbar]'}, inplace=True)

    # Sort the DataFrame by the 'date-time' column
    df = df.sort_values(by=['date-time','index'], ascending=True)
    df.reset_index(drop=True, inplace=True)

    #filter the df for objects where 1 dimension is larger than ca. 1mm
    df = df[(df['w'] > size_filter) | (df['h'] > size_filter)]
    df_unique = df.drop_duplicates(subset=['img_id'])
    print(f'{empty_file_counter} files were empty and were dropped; Number of uniue images: {len(df_unique)}')

    logging.info('crop_df created')
    return df


def add_ctd_data(ctd_data_loc:str, crop_df):
    #crop_df.drop('depth [m]',axis=1)
    # Reading the specified header line (line 124) to extract column names
    with open(ctd_data_loc, 'r') as file:
        for _ in range(123):
            next(file)  # Skip lines until the header
        header_line = next(file).strip()  # Read the header line

    # Processing the header line to get column names
    # The header line is expected to be in the format "Columns  = column1:column2:..."
    column_names = header_line.split(' = ')[1].split(':')
    ctd_df = pd.read_csv(ctd_data_loc, delim_whitespace=True, header=None, skiprows=124, names=column_names)
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
    columns = ['s', 'o', 't', 'chl', 'z_factor']

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
        
    # Sort both DataFrames by 'filename'
    crop_df_sorted = crop_df.sort_values(by='filename').reset_index(drop=True)
    prediction_df_sorted = prediction_df.sort_values(by='filename').reset_index(drop=True)

    # Concatenate data frames
    combined_df = pd.concat([crop_df_sorted, prediction_df_sorted], axis=1)
    
    logging.info('Prediction data added to crop_df')
    return combined_df

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
                if len(temp_values) != 8: #adding this for debugging...
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
                    temp_data['TH'] = float(temp_values[7])
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
    df[['restart', 'relock']] = df[['restart', 'relock']].fillna(False)
    cols = ['TT', 'T1', 'T2', 'TH']  # list of your column names
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
    

if __name__ == "__main__":
    dest_folder = '/home/fanny/Analysis_results'
    result_dir = '/home/fanny/M181_output'
    #result_dir = '/home/fanny/M181_output_test'
    log_dir = '/home/fanny/Templog'
    #log_dir = '/home/fanny/Templog_test'
    analyze_profiles(result_dir, dest_folder, engine, small=False, add_ctd=True, calc_props=False, plotting=True, log_directory=log_dir)
    print('All done.')
    logging.info('All done.')
    engine.dispose()
