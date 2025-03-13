import logging
from segmenter import run_segmenter
import pandas as pd
import os
import csv
from tqdm import tqdm
from pandas.errors import EmptyDataError
import numpy as np
import re
import shutil
import subprocess
from pyecotaxa.remote import Remote

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    directory = os.path.join(directory,'Data')

    files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    dataframes = []
    empty_file_counter = 0
    id = 1
    for file in tqdm(files):
        try:
            df = pd.read_csv(file, delimiter=",", header=None, index_col=None)
            if len(df.columns) == 44:
                df.insert(0,'',id)            
                dataframes.append(df)
                id+=1
            else:
                continue
        except EmptyDataError:
            empty_file_counter += 1
            print(f"File {file} is empty")

    df = pd.concat(dataframes, ignore_index=True)
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
    
    #df.drop("saved", axis=1, inplace=True)

    # Split the 'filename' column
    split_df = df['filename'].str.split('_', expand=True)
    num_elements = split_df.shape[1]
    
    if small:  # bug fix for segmenter where small objects are saved with _mask.png extension instead of .png: needs to be fixed if segmenter is fixed
        headers = ["date", "time", "pressure", "temperature", "index", "mask_ext"]
        split_df.columns = headers
        split_df.drop("mask_ext", axis=1, inplace=True)
    else:
        if num_elements == 5:
            headers = ["date-time", "pressure", "temperature", "index", "drop"]
            split_df.columns = headers
            split_df.drop("drop", axis=1, inplace=True)
        else:
            headers = ["date-time", "pressure", "temperature", "index"]
            split_df.columns = headers
    
    # split date-time
    split_df[['date', 'time']] = split_df['date-time'].str.split('-', expand=True)
    split_df.drop(columns=['date-time'], inplace=True)

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
    df = df.sort_values(by=['date', 'time','index'], ascending=True)
    df.reset_index(drop=True, inplace=True)

    #filter the df for objects where 1 dimension is larger than ca. 1mm
    df = df[(df['w'] > size_filter) | (df['h'] > size_filter)]
    df_unique = df.drop_duplicates(subset=['img_id'])
    print(f'{empty_file_counter} files were empty and were dropped; Number of unique images: {len(df_unique)}')

    return df

def prepare_prediction_data(prediction_csv, mapping_csv, sep="\t"):    
    # Load CSV files
    prediction_df = pd.read_csv(prediction_csv)
    polytaxo_classes_df = pd.read_csv(mapping_csv, sep=sep)

    # Add annotation status
    prediction_df['object_annotation_status'] = 'predicted'

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
    prediction_df[columns_to_replace] = prediction_df[columns_to_replace].replace(mapping_dict).apply(
        lambda col: col.astype(str).apply(
            lambda x: re.split(split_pattern, x)[0].replace("_", " ") if pd.notna(x) else x
        )
    )

    return prediction_df  # Return the processed DataFrame

def combine_segmentation_and_prediction(segmentation_df, prediction_df):
        
    # Sort both DataFrames by 'filename'
    segmentation_df_sorted = segmentation_df.sort_values(by='filename').reset_index(drop=True)
    prediction_df_sorted = prediction_df.sort_values(by='filename').reset_index(drop=True)

    # Concatenate data frames
    combined_df = pd.concat([segmentation_df_sorted, prediction_df_sorted], axis=1)
    
    # Remove duplicate columns (keeping the first occurrence)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]

    # Add object ID column
    combined_df['object_id'] = combined_df['img_id'].astype(str) + '_' + combined_df['index'].astype(str)

    # Define columns to delete
    columns_to_delete = [
        'temperature', 'mean_raw', 'std_raw', 'mean', 'std', 'x', 'y', 'w', 'h', 
        'saved', 'bound_box_x', 'bound_box_y', 'full_path', 'img_id', 'index'
    ]
    # Remove unwanted columns if they exist
    combined_df.drop(columns=[col for col in columns_to_delete if col in combined_df.columns], axis=1, inplace=True)

    # Adjust header names
    rename_mapping = {
        'pressure [dbar]': 'object_pressure',
        'date': 'object_date',
        'time': 'object_time',
        'filename': 'img_file_name',
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
        'prob5': 'object_prob_5'
    }
    combined_df.rename(columns=rename_mapping, inplace=True)

    return combined_df  # Return the processed DataFrame

def determine_dtype(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return '[f]' 
    elif pd.api.types.is_string_dtype(dtype):
        return '[t]'
    else:
        return 'other'

def zip_data(folder_path, zip_path, extra_file=None):
    """
    Zips a folder and optionally includes an extra file inside the zip.

    Parameters:
        folder_path (str): The folder to zip.
        zip_path (str): The destination zip file path.
        extra_file (str, optional): metadata file
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return
    
    # Move meta data file into the folder before zipping
    if extra_file and os.path.exists(extra_file):
        shutil.move(extra_file, os.path.join(folder_path, os.path.basename(extra_file)))

    # Create ZIP archive
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', folder_path)
    print(f"Zipped {folder_path} to {zip_path}")

# def ET_upload(remote, project_id, folder_path): #Uploads a zip file to an EcoTaxa project using terminal commands.
#     try:
#         remote.push([(folder_path, project_id)])
#         logging.info(f"Successfully uploaded {folder_path} to project {project_id}.")
#     except Exception as e:
#         logging.error(f"Error during upload: {e}")

if __name__ == "__main__":
    # Define file paths
    file_path = '/home/fanny/M181-3_output/Data'
    prediction_csv = '/home/fanny/M181-3_output/ViT_predictions.csv'
    mapping_csv = '/home/fanny/taxonomic_data/Polytaxo_classes.csv'
    eco_taxa_folder = "/home/fanny/M181-3_output/EcoTaxa"
    os.makedirs(eco_taxa_folder, exist_ok=True)

    # Generate crop DataFrame
    segmentation_df = gen_crop_df(file_path, False)

    # Prepare prediction data
    prediction_df = prepare_prediction_data(prediction_csv, mapping_csv)

    # Combine segmentation and prediction data
    segm_and_prediction_df = combine_segmentation_and_prediction(segmentation_df, prediction_df)

    # Determine data types and insert as the first row
    dtype_row = [determine_dtype(segm_and_prediction_df.dtypes[col]) for col in segm_and_prediction_df.columns]
    segm_and_prediction_df.loc[-1] = dtype_row  # Add the dtype row
    segm_and_prediction_df.index = segm_and_prediction_df.index + 1  # Shift index
    segm_and_prediction_df = segm_and_prediction_df.sort_index()  # Sort by index

    # Save combined data as TSV file
    metadata_path = os.path.join(eco_taxa_folder, 'ecotaxa_metadata.tsv')
    segm_and_prediction_df.to_csv(metadata_path, sep="\t", index=False)

    # Ensure the metadata file is generated
    if os.path.exists(metadata_path):
        logging.info(f"Metadata TSV file generated at {metadata_path}")
    else:
        logging.error("Failed to generate metadata TSV file")

    # Define paths for zipping
    deconv_crops_folder = '/home/fanny/M181-3_output/Deconv_crops'
    raw_crops_folder = '/home/fanny/M181-3_output/Crops'
    zip_path_deconv = os.path.join(eco_taxa_folder, "ecotaxa_upload_deconv.zip")
    zip_path_raw = os.path.join(eco_taxa_folder, "ecotaxa_upload_raw.zip")

    # Copy metadata to both folders before zipping
    shutil.copy(metadata_path, os.path.join(deconv_crops_folder, 'ecotaxa_metadata.tsv'))
    shutil.copy(metadata_path, os.path.join(raw_crops_folder, 'ecotaxa_metadata.tsv'))

    # Zip data
    zip_data(deconv_crops_folder, zip_path_deconv, metadata_path)
    zip_data(raw_crops_folder, zip_path_raw, metadata_path)

    logging.info("Folders have been zipped")

    # Commenting out the upload to EcoTaxa
    # remote = Remote()
    # remote.current_user()
    # ET_upload(remote, 15753, zip_path_raw)
    # ET_upload(remote, 15862, zip_path_deconv)