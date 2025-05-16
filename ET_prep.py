import logging
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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("ET_prep.log"),
    logging.StreamHandler()
])



def prepare_prediction_data(metadata_csv, mapping_csv, sep="\t"):    
    # Load CSV files
    df = pd.read_csv(metadata_csv)
    polytaxo_classes_df = pd.read_csv(mapping_csv, sep=sep)

    # Add annotation status
    df['object_annotation_status'] = 'predicted'

    # # Extract only the file name from the 'filename' column
    # prediction_df.rename(columns={'filename': 'full_path'}, inplace=True)
    # prediction_df['filename'] = prediction_df['full_path'].apply(lambda x: os.path.basename(x))

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
    
    return df


def process_predictions(root_folder, mapping_csv):
    combined_prediction_df = pd.DataFrame()
    for profile_folder in os.listdir(root_folder):
        profile_path = os.path.join(root_folder, profile_folder)
        if os.path.isdir(profile_path):
            prediction_csv = os.path.join(profile_path, 'ViT_predictions.csv')
            if os.path.exists(prediction_csv):
                logging.info(f"Processing: {prediction_csv}")
                try:
                    processed_df = prepare_prediction_data(prediction_csv, mapping_csv)
                    processed_df['profile'] = profile_folder
                    combined_prediction_df = pd.concat([combined_prediction_df, processed_df], ignore_index=True)
                except Exception as e:
                    logging.error(f"Error processing {prediction_csv}: {e}")
    return combined_prediction_df



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


def get_crop_data(crop_data):
    # Columns to include from the exported data file
    columns_to_include = [
        'date-time', 'pressure [dbar]', 'depth [m]', 'filename','area', 
        'object_bound_box_w', 'object_bound_box_h', 'object_circularity', 
        'object_area_exc', 'object_area_rprops', 'object_%area', 
        'object_major_axis_len', 'object_minor_axis_len', 'object_centroid_y', 
        'object_centroid_x', 'object_convex_area', 'object_min_intensity', 
        'object_max_intensity', 'object_mean_intensity', 'object_int_density', 
        'object_perimeter', 'object_elongation', 'object_range', 
        'object_perim_area_excl', 'object_perim_major', 
        'object_circularity_area_excl', 'object_angle', 'object_boundbox_area', 
        'object_eccentricity', 'object_equivalent_diameter', 'object_euler_nr', 
        'object_extent', 'object_local_centroid_col', 'object_local_centroid_row', 
        'object_solidity', 'esd', 'img_id', 'index', 'full_path'
    ]
    crop_df = pd.read_csv(crop_data, usecols=columns_to_include)

    return crop_df


def process_crop_data(df):
    # Add object ID column
    df['object_id'] = df['img_id'].astype(str) + '_' + df['index'].astype(str)
    print('object_id added')

    #split date-time
    df[['date', 'time']] = df['date-time'].str.split('-', expand=True)


    # Apply the extraction function to the full_path column
    df[['lat', 'lon']] = df['full_path'].apply(
        lambda x: pd.Series(extract_coordinates(x))
    )
    print('coordinates extracted')

    # Convert latitude and longitude to decimal format
    df['lat'] = df['lat'].apply(convert_to_decimal)
    df['lon'] = df['lon'].apply(convert_to_decimal)
    print('coordinates converted')

    df.drop(['date-time', 'index', 'img_id'], axis=1, inplace=True)
    print('columns removed')


def create_metadata_file(crop_df, prediction_df, output_path):
    logging.info(f"original crop_df length: {len(crop_df)}")
    logging.info(f"original prediction_df length: {len(prediction_df)}")
    #Remove duplicates based on the 'filename' column
    crop_df = crop_df.drop_duplicates(subset='filename')
    logging.info(f"Crop DataFrame length after removing duplicates: {len(crop_df)}")
    prediction_df = prediction_df.drop_duplicates(subset='filename')
    logging.info(f"Prediction DataFrame length after removing duplicates: {len(prediction_df)}")

    crop_df['filename'] = crop_df['filename'].str.strip()
    prediction_df['filename'] = prediction_df['filename'].str.strip()
    prediction_df = prediction_df.merge(crop_df[['filename']], on='filename', how='inner')
    logging.info('datasets merged')
    logging.info(f"Crop DataFrame length after merging: {len(crop_df)}")
    logging.info(f"Prediction DataFrame length after merging: {len(prediction_df)}")
    # Save the updated prediction_df
    prediction_df.to_csv("updated_prediction.csv", sep="\t", index=False)
    crop_df_sorted = crop_df.sort_values(by='filename').reset_index(drop=True)
    prediction_df_sorted = prediction_df.sort_values(by='filename').reset_index(drop=True)

    unmatched_in_crop = crop_df[~crop_df['filename'].isin(prediction_df['filename'])]
    unmatched_in_crop.to_csv("unmatched_in_crop.csv", index=False)
    logging.info(f"Number of unmatched rows in crop_df: {len(unmatched_in_crop)}")


    #debugging: Checking duplicates
    # duplicates = prediction_df[prediction_df['filename'].duplicated(keep=False)]
    # # Save the duplicates to a new file
    # duplicates.to_csv("duplicates_in_prediction.csv", sep="\t", index=False)
    # num_duplicates = prediction_df['filename'].duplicated().sum()
    # print(f"Duplicate filenames in prediction_df:{num_duplicates}")
    # duplicates = crop_df[crop_df['filename'].duplicated(keep=False)]
    # # Save the duplicates to a new file
    # duplicates.to_csv("duplicates_in_crop.csv", sep="\t", index=False)
    # num_duplicates = crop_df['filename'].duplicated().sum()
    # print(f"Duplicate filenames in crop_df:{num_duplicates}")
    # unmatched_in_prediction = prediction_df[~prediction_df['filename'].isin(crop_df['filename'])]
    # print(f"Number of unmatched rows in prediction_df: {len(unmatched_in_prediction)}")
    # unmatched_in_prediction.to_csv("unmatched_in_prediction.csv", index=False)

    # debugging: Checking skipped images due to resize failure


    # Ensure the DataFrames have the same number of rows
    if len(crop_df_sorted) != len(prediction_df_sorted):
        logging.error(f"Row counts: crop_df_sorted: {len(crop_df_sorted)}, prediction_df_sorted: {len(prediction_df_sorted)}")
        raise ValueError("Mismatch in row counts between crop and prediction data.")
    
    # Concatenate data frames
    combined_df = pd.concat([crop_df_sorted, prediction_df_sorted], axis=1)
    logging.INFO('dataframes concatenated')
    
    # Remove duplicate columns (keeping the first occurrence)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
    #combined_df.drop(columns=['full_path'], inplace=True)

    # Adjust header names
    rename_mapping = {
        'pressure [dbar]': 'object_pressure',
        'date': 'object_date',
        'time': 'object_time',
        'filename': 'img_file_name',
        'depth [m]': 'object_depth',
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
    dtype_row = [determine_dtype(combined_df.dtypes[col]) for col in combined_df.columns]
    combined_df.loc[-1] = dtype_row  # Add the dtype row

    combined_df.to_csv(output_path, sep="\t", index=False)
    logging.INFO(f"Metadata saved to: {output_path}")

    return combined_df


def determine_dtype(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return '[f]' 
    elif pd.api.types.is_string_dtype(dtype):
        return '[t]'
    else:
        return 'other'



if __name__ == "__main__":
    root_folder = "/home/fanny/M181_output"
    mapping_csv = "/home/fanny/taxonomic_data/Polytaxo_classes.csv"
    crop_data = "/home/fanny/exported_images/exported_images.csv"
    output_path = "/home/fanny/ET_metadata/combined_metadata.tsv"
    prediction_path = "/home/fanny/ET_metadata/combined_predictions.tsv"
    crop_data_path = "/home/fanny/ET_metadata/processed_crop_data.tsv"

    # pd.set_option('display.max_colwidth', None)
    # crop_df = pd.read_csv(crop_data_path, sep="\t")
    # print(crop_df['full_path'].head(10))
    # combined_prediction_df = pd.read_csv(prediction_path, sep="\t")
    # print(combined_prediction_df['full_path'].head(10))    

    # #process prediction data
    # logging.info("Starting to process prediction data...")
    # combined_prediction_df = process_predictions(root_folder, mapping_csv)
    # combined_prediction_df.to_csv(prediction_path,sep="\t", index=False)
    # logging.info("Complete prediction dataset generated.")

    # #process crop data
    # logging.info("Starting to process crop data...")
    # crop_df = get_crop_data(crop_data)  # Load crop data
    # process_crop_data(crop_df)  # Process crop data
    # crop_df.to_csv(crop_data_path, sep="\t", index=False)
    # logging.info("Crop data processed.")

    combined_prediction_df = pd.read_csv(prediction_path, sep="\t")
    print("Prediction data loaded")
    crop_df = pd.read_csv(crop_data_path, sep="\t")
    print("Crop data loaded")
    # Step 3: Combine and save metadata
    logging.info("Combining crop and prediction data into metadata...")
    combined_metadata_df = create_metadata_file(crop_df, combined_prediction_df, output_path)
    logging.info(f"Metadata saved to folder: ET_metadata.")


