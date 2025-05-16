from segmenter import run_segmenter
from EcoTaxa_preparation import gen_crop_df, add_ctd_data, prepare_prediction_data, combine_segmentation_and_prediction, zip_data, determine_dtype
from EcoTaxa_upload import upload_all_zips
from Classify_ViT import classify_images, classlist
from smb.SMBConnection import SMBConnection
import os
import pandas as pd
import logging
import subprocess
import shutil
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("pipeline.log"),
    logging.StreamHandler()
])


# Login to ET project
#login_cmd = ["pyecotaxa", "login"]
#subprocess.run(login_cmd, check=True)

def process_and_upload_folder(png_folder_path, destination_path, model_dir, classlist):
    run_segmenter(png_folder_path, destination_path, True)

    # Classify images using ViT
    crops_folder = os.path.join(destination_path, 'Deconv_crops')
    vit_predictions_csv = os.path.join(destination_path, 'ViT_predictions.csv')
    histogram_path = os.path.join(destination_path, 'top5_prediction_histogram.png')

    classify_images(crops_folder, vit_predictions_csv, model_dir, histogram_path, classlist)

    # # Generate crop DataFrame
    # file_path = os.path.join(destination_path, 'Data')
    # segmentation_df = gen_crop_df(file_path, False)

    # # Prepare prediction data
    # prediction_df = prepare_prediction_data(vit_predictions_csv, "/home/fanny/taxonomic_data/Polytaxo_classes.csv")

    # # Combine segmentation and prediction data
    # combined_df = combine_segmentation_and_prediction(segmentation_df, prediction_df)

    # # Determine data types and insert as the first row
    # dtype_row = [determine_dtype(combined_df.dtypes[col]) for col in combined_df.columns]
    # combined_df.loc[-1] = dtype_row  # Add the dtype row
    # combined_df.index = combined_df.index + 1  # Shift index
    # combined_df = combined_df.sort_index()  # Sort by index

    # # Save combined data as TSV file
    # eco_taxa_folder = os.path.join(destination_path, "EcoTaxa")
    # os.makedirs(eco_taxa_folder, exist_ok=True)
    # metadata_path = os.path.join(eco_taxa_folder, 'ecotaxa_metadata.tsv')
    # combined_df.to_csv(metadata_path, sep="\t", index=False)

    # # Ensure the metadata file is generated
    # if os.path.exists(metadata_path):
    #     logging.info(f"Metadata TSV file generated at {metadata_path}")
    # else:
    #     logging.error("Failed to generate metadata TSV file")

    # deconv_crops_folder = os.path.join(destination_path, 'Deconv_crops')
    # raw_crops_folder = os.path.join(destination_path, 'Crops')

    # # Copy metadata to both folders before zipping
    # shutil.copy(metadata_path, os.path.join(deconv_crops_folder, 'ecotaxa_metadata.tsv'))
    # shutil.copy(metadata_path, os.path.join(raw_crops_folder, 'ecotaxa_metadata.tsv'))

    # # zip paths
    # zip_path_deconv = os.path.join(eco_taxa_folder, "ecotaxa_upload_deconv.zip")
    # zip_path_raw = os.path.join(eco_taxa_folder, "ecotaxa_upload_raw.zip")

    # zip_data(deconv_crops_folder, zip_path_deconv, metadata_path)
    # zip_data(raw_crops_folder, zip_path_raw, metadata_path)

    # logging.info(f"Folder {os.path.basename(destination_path)} has been processed and zipped")

    # # Upload to EcoTaxa
    

def process_pipeline(main_folder, output_base_folder, model_dir, classlist):
    for root, dirs, files in os.walk(main_folder):
        if "PNG" in dirs:  # Check if a "PNG" folder exists in the current directory
            png_folder_path = os.path.join(root, "PNG")
            parent_folder_name = os.path.basename(root)
            destination_path = os.path.join(output_base_folder, parent_folder_name)
            logging.info(f"Processing folder: {parent_folder_name}")
            
            #Check if the output subfolder already exists
            if os.path.exists(destination_path):
                logging.info(f"Output folder for {parent_folder_name} already exists. Skipping...")
                continue

            os.makedirs(destination_path, exist_ok=True)
            process_and_upload_folder(png_folder_path, destination_path, model_dir, classlist)
'''

def process_pipeline(main_folder, output_base_folder, model_dir, classlist):
    # Establish SMB connection
    conn = SMBConnection('GEOMAR\\fbrodbek', 'local_machine', 'filer.geomar.de', use_ntlm_v2=True)
    try:
        conn.connect('filer.geomar.de', 445, timeout=30)
        logging.info("Connection established successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to SMB server: {e}", exc_info=True)
        return

    # List directories in the main folder
    try:
        logging.info(f"Attempting to list directories in: {main_folder}")
        directories = conn.listPath('projekte', main_folder)
        logging.info(f"Found directories: {[d.filename for d in directories if d.isDirectory]}")

        for directory in directories:
            if directory.isDirectory and directory.filename not in ['.', '..']:
                png_folder_path = f"{main_folder}/{directory.filename}/PNG"
                parent_folder_name = directory.filename
                destination_path = os.path.join(output_base_folder, parent_folder_name)
                logging.info(f"Processing folder: {parent_folder_name}")

                # Check if the output subfolder already exists
                if os.path.exists(destination_path):
                    logging.info(f"Output folder for {parent_folder_name} already exists. Skipping...")
                    continue

                # Check if the PNG folder exists in the SMB share
                logging.info(f"Checking for PNG folder at: {png_folder_path}")
                try:
                    conn.listPath('projekte', png_folder_path)
                except Exception as e:
                    logging.warning(f"PNG folder not found at {png_folder_path}: {e}")
                    continue

                os.makedirs(destination_path, exist_ok=True)
                process_and_upload_folder(png_folder_path, destination_path, model_dir, classlist)
    except Exception as e:
        logging.error(f"Error while listing directories: {e}")
    finally:
        conn.close()
'''
if __name__ == "__main__":
    f_path = '/mnt/Filer/M181/M181-PNG'
    #f_path = '/home/fanny/M181_test_set'
    output_base_folder = '/home/fanny/M181_output'
    model_dir = '/home/fanny/250129_ViT_custom_size_sensitive/best_model'
    process_pipeline(f_path, output_base_folder, model_dir, classlist)

