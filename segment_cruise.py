import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from segmenter import run_segmenter

#Don't forget to mount the SMB share, in system terminal: sudo mount -t cifs //filer.geomar.de/projekte/ORTC-ST-PISCO /mnt/filer -o username=GEOM-svcPISCO_rw@geomar.de,password=2mZmhx-7GCGf 
# Select the base dir of the PIScO data for the cruise you want to process. Example: /mnt/filer/SO298/SO298-Logfiles_PISCO

### KOSMOS Peru 2020
# cruise = "PISCO_KOSMOS_2020_Peru"
# intermediate_result_dir = "/home/veit/PIScO_dev/Segmentation_results/PISCO_KOSMOS_2020_Peru/TempResults"
# cruise_base = f"/mnt/filer/PISCO_KOSMOS_2020_Peru/PIScO_Peru_source"
# ctd_dir= None
# #ctd_prefix = None
# log_directory = None

### HE570
# cruise = "HE570"
# cruise_base = "/mnt/filer/HE570/HE570-PISCO-Profiles"
# intermediate_result_dir = f"/home/veit/PIScO_dev/Segmentation_results/HE570/HE570-PISCO-Profiles/TempResults"
# ctd_dir = None
# #ctd_prefix = "met_202_1_"
# log_directory = None

# ### M181
# cruise = "M181"
# cruise_base = "/home/veit/PIScO_dev/Segmentation_results/M181"
# intermediate_result_dir = f"/home/veit/PIScO_dev/Segmentation_results/M181"
# ctd_dir = "/home/veit/Downloads/CTD_preliminary_calibrated"
# ctd_prefix = "met_181_1_"
# log_directory = "/home/veit/Downloads/Templog"
# pressure_unit = "bar"  # or "dbar"

## SO_298
cruise = "SO298"
cruise_base = "/mnt/filer/SO298/SO298-PISCO-Profiles"
# #cruise_base = "/home/veit/PIScO_dev/usb_mount/SO298-PNG"
intermediate_result_dir = "/home/veit/PIScO_dev/Segmentation_results/SO298/SO298-PISCO-Profiles/TempResults"
ctd_dir = "/mnt/filer/SO298/SO298-CTD_UVP_ETC/SO298-CTD/calibrated/"
ctd_prefix = "son_298_1_"
log_directory = "/mnt/filer/SO298/SO298-Logfiles_PISCO/Templog"
pressure_unit = "dbar"

### MSM_126
# cruise = "MSM126"
# cruise_base = "/mnt/filer/MSM126/MSM126-PISCO-Profiles"
# intermediate_result_dir = f"/home/veit/PIScO_dev/Segmentation_results/MSM126/MSM126-PISCO-Profiles/TempResults/"
# ctd_dir = "/mnt/filer/MSM126/MSM126-Data-UVP-CTD-ADCP/CTD/msm_126_1_ctd"
# ctd_prefix = "msm_126_1_"
# log_directory = "/mnt/filer/MSM126/MSM126-PISCO_Logfilesetc/Logfiles"

## M202
# cruise = "M202"
# cruise_base = "/mnt/filer/M202/M202-PISCO-Profiles"
# intermediate_result_dir = f"/home/veit/PIScO_dev/Segmentation_results/M202/M202-PISCO-Profiles/TempResults"
# ctd_dir = "/mnt/filer/M202/M202-external-Data/M202-CTD/met_202_1_ctd/met_202_1_ctd"
# ctd_prefix = "met_202_1_"
# log_directory = "/mnt/filer/M202/M202-Pisco-Logfiles/Logfiles"

### SO_308
# cruise = "SO308"
# cruise_base = "/mnt/filer/SO308/SO308-PISCO-Profiles"
# intermediate_result_dir = "/home/veit/PIScO_dev/Segmentation_results/SO308/SO308-PISCO-Profiles/TempResults"
# ctd_dir = "/mnt/filer/SO308/ADCP-ETC/CTD/SO308_ctd_files"
# ctd_prefix = "son_308_1_"
# log_directory = "/mnt/filer/SO308/PISCO-Logfiles/PISCO-LOGFILES/Logfiles"
# pressure_unit = "dbar"  # or "bar"

for profile in os.listdir(cruise_base):
        print(profile)
        profile_path = os.path.join(cruise_base, profile)
        if os.path.isdir(profile_path):
                img_folder = os.path.join(profile_path, profile+"_Images-PNG")
                #img_folder = profile_path
                results_folder = os.path.join(intermediate_result_dir, profile, profile+"_Results")
                if not os.path.exists(img_folder):
                        print(f"Image folder {img_folder} does not exist, skipping profile {profile}.")
                        continue
                if not os.path.exists(results_folder):
                        os.makedirs(results_folder)
                run_segmenter(img_folder, results_folder, deconvolution=True)


