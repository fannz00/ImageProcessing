#### External modules
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

#### Internal modules

from segmenter import run_segmenter
import analyze_profiles_seavision as ap

#Don't forget to mount the SMB share, in system terminal: sudo mount -t cifs //filer.geomar.de/projekte/ORTC-ST-PISCO /mnt/filer -o username=GEOM-svcPISCO_rw@geomar.de,password=2mZmhx-7GCGf 
# Select the base dir of the PIScO data for the cruise you want to process. Example: /mnt/filer/SO298/SO298-Logfiles_PISCO

cruise = "SO298"
ctd_dir = "/mnt/filer/SO298/SO298-CTD_UVP_ETC/SO298-CTD/calibrated/"
ctd_prefix = "son_298_1_"

log_directory = "/mnt/filer/SO298/SO298-Logfiles_PISCO/Templogs"

cruise_base = "/mnt/filer/SO298/SO298-PISCO-Profiles"
#cruise_base = "/home/veit/PIScO_dev/usb_mount/SO298-PNG"

intermediate_result_dir = "/home/veit/PIScO_dev/Segmentation_results/SO298/SO298-PISCO-Profiles/TempResults"

#cruise_base = "/app/usb_mount/SO298-PNG"
#intermediate_result_dir = "/app/Segmentation_results/SO298/SO298-PNG/TempResults"

####### MAIN LOOP ########
# Loop through all profiles in the cruise base directory
# and run the segmenter on each profile's image folder
# The segmenter will save the results in a new folder named "Results" inside the profile's directory

for profile in os.listdir(cruise_base):
        print(profile)
        profile_path = os.path.join(cruise_base, profile)
        profile_id = profile.split('_')[1]  # Extract the profile ID from the profile name
        if os.path.isdir(profile_path):
                img_folder = os.path.join(profile_path, profile+"_Images-PNG")
                #img_folder = profile_path
                results_folder = os.path.join(intermediate_result_dir, profile, profile+"_Results")
                if not os.path.exists(results_folder):
                        os.makedirs(results_folder)
                run_segmenter(img_folder, results_folder, deconvolution=False)

                # Load the results of the segmenter
                # and create a DataFrame with the particle data
                profile_data_dir = os.path.join(results_folder, 'Data')
                df = ap.gen_crop_df(profile_data_dir, small=small, size_filter=0)
                print(len(df.index), 'particles found.')
                df['fullframe_path'] = df['full_path'].apply(ap.modify_full_path)

                # Add ctd data to the DataFrame
                ctd_file = os.path.join(ctd_dir, f'{ctd_prefix}{int(profile.split("_")[1].split("-")[1]):03d}.ctd')               
                print('adding ctd data...')
                df = add_ctd_data(ctd_file, df)

                # Predict on deconvolved images
                print('adding predictions...')
                
                prediction_file = os.path.join(results_folder, 'ViT_predictions.csv')  # Example path
                if os.path.exists(prediction_file):
                    print('Adding predictions...')
                    prediction_df = pd.read_csv(prediction_file)
                    df = ap.add_prediction(df, prediction_df)

                # Add log data to the DataFrame
                if log_directory is not None:                
                    print('adding log info...')
                    timestamp = profile.split('_')[-1]
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

                    df_log = ap.create_log_df(file_path)

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
                df_unique['part_based_filter'] = df_unique['count'].apply(lambda x: 0 if x < df_unique['count'].std() else 1)
                sorted_df = sorted_df.merge(df_unique[['date-time', 'part_based_filter']], on='date-time', how='left')

                
                
                
                
                
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
                
