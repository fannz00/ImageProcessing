import pandas as pd
import numpy as np
import os
import zipfile
import shutil


def add_ctd_data(ctd_data_loc:str, crop_df):
    '''function that adds the ctd data from a given location to the crop dataframe '''

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

    return crop_df


def process_profile(profile_dir, ctd_data_dir, add_ctd=True):
    """
    Processes a single profile directory, unpacks the zip files, adds CTD data, and re-zips them.

    Parameters:
        profile_dir (str): Path to the profile directory.
        ctd_data_dir (str): Directory containing CTD files.
        add_ctd (bool): Whether to add CTD data to the metadata table.
    """
    eco_taxa_dir = os.path.join(profile_dir, "EcoTaxa")
    zip_files = ["ecotaxa_upload_deconv.zip", "ecotaxa_upload_raw.zip"]
    temp_dir = os.path.join(eco_taxa_dir, "temp_unzip")

    for zip_file_name in zip_files:
        zip_file_path = os.path.join(eco_taxa_dir, zip_file_name)

        if not os.path.exists(zip_file_path):
            print(f"Zip file not found: {zip_file_path}")
            continue

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        metadata_path = os.path.join(temp_dir, "EcoTaxa_metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found in {zip_file_path}")
            shutil.rmtree(temp_dir)
            continue

        # Load the metadata CSV
        metadata_df = pd.read_csv(metadata_path)

        # Find corresponding CTD file
        folder_name = os.path.basename(profile_dir)
        ctd_file = os.path.join(
            ctd_data_dir,
            'CTD_preliminary_calibrated',
            f'met_181_1_{folder_name.split("_")[1].split("-")[1]}.ctd'
        )

        # Add CTD data if enabled
        if add_ctd and os.path.exists(ctd_file):
            print(f"Adding CTD data for {folder_name} in {zip_file_name}...")
            metadata_df = add_ctd_data(ctd_file, metadata_df)
            metadata_df.to_csv(metadata_path, index=False)
        elif add_ctd:
            print(f"CTD file not found for {folder_name}")

        # Re-zip the folder
        with zipfile.ZipFile(zip_file_path, 'w') as zip_ref:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_ref.write(file_path, arcname)

        print(f"Processed and re-zipped: {zip_file_path}")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)


def run(profiles_dir, ctd_data_dir, add_ctd=True):
    """
    Processes all profiles in the given directory.

    Parameters:
        profiles_dir (str): Directory containing profile folders.
        ctd_data_dir (str): Directory containing CTD files.
        add_ctd (bool): Whether to add CTD data to the metadata table.
    """
    for folder in os.listdir(profiles_dir):
        profile_dir = os.path.join(profiles_dir, folder)
        if os.path.isdir(profile_dir):
            process_profile(profile_dir, ctd_data_dir, add_ctd)


if __name__ == '__main__':
    profiles_dir = '/home/fanny/test_set_Segm_Output'
    ctd_data_dir = '/home/fanny/CTD_preliminary_calibrated'
    run(profiles_dir, ctd_data_dir)
