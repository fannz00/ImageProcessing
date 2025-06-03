import logging
import os
from pyecotaxa.remote import Remote

# Configure logging to both console and file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('/home/fanny/ImageProcessing/EcoTaxa_upload2.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def login_to_ecotaxa(username, password):
    try:
        remote = Remote()
        remote.login(username, password)
        logging.info("Successfully logged into EcoTaxa.")
        return remote
    except Exception as e:
        logging.error(f"Error during login: {e}")
        return None

def ET_upload(remote, project_id, folder_path):
    try:
        remote.push([(folder_path, project_id)])
        logging.info(f"Successfully uploaded {folder_path} to project {project_id}.")
    except Exception as e:
        logging.error(f"Error during upload: {e}")



def upload_all_zips(remote, base_folder):      
    for profile in os.listdir(base_folder):
        profile_path = os.path.join(base_folder, profile)
        if not os.path.isdir(profile_path):
            logging.warning(f"Skipping non-directory item: {profile_path}")
            continue

        logging.info(f"Processing EcoTaxa folder: {profile_path}")                

        # Find all zip files in the sub folder
        zip_files = [f for f in os.listdir(profile_path) if f.endswith('.zip')]
        # Check if there are zip files with "part" in their names
        part_zip_files = [f for f in zip_files if "part" in f]
    
        if part_zip_files:
            # If "part" zip files exist, upload only those
            for part_zip in part_zip_files:
                zip_path = os.path.join(profile_path, part_zip)
                project_id = 17108 if "deconv" in part_zip else 17109  # Adjust project IDs as needed
                ET_upload(remote, project_id, zip_path)
                logging.info(f"Uploaded {part_zip} to project {project_id}.")
        else:
            # Otherwise, upload the default crops.zip and deconv_crops.zip
            zip_path_deconv = os.path.join(profile_path, "deconv_crops.zip")
            ET_upload(remote, 17108, zip_path_deconv)
            logging.info(f"Uploaded deconv_crops.zip to project 17108.")
            zip_path_raw = os.path.join(profile_path, "crops.zip")
            ET_upload(remote, 17109, zip_path_raw)
            logging.info(f"Uploaded crops.zip to project 17109.")
            
                        


if __name__ == "__main__":
    # Define username and password
    USERNAME = 'your username'
    PASSWORD = 'your password'

    # remote = login_to_ecotaxa(USERNAME, PASSWORD)
    # if remote:
    #     zip_path_raw = '/home/fanny/EcoTaxa/M181-066-1_CTD-024_03deg30S-007deg15E_20220428-1514_updated/crops1.zip'
    #     zip_path_deconv = '/home/fanny/EcoTaxa/M181-005-1_CTD-002_16deg00S-011deg34E_20220422-0039_updated/deconv_crops.zip'
    #     ET_upload(remote, zip_path_deconv)
    #     ET_upload(remote, zip_path_raw)
    # else:
    #     logging.error("Failed to log into EcoTaxa. Exiting...")

    remote = login_to_ecotaxa(USERNAME, PASSWORD)
    if remote:
        output_base_folder = '/home/fanny/EcoTaxa'
        upload_all_zips(remote, output_base_folder)
    else:
        logging.error("Failed to log into EcoTaxa. Exiting...")
