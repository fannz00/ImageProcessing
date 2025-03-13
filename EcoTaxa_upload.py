import logging
import os
from pyecotaxa.remote import Remote

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def upload_to_ecotaxa(remote, zip_path_raw, zip_path_deconv):
    try:
        # upload to ET
        ET_upload(remote, 15753, zip_path_raw)
        ET_upload(remote, 15862, zip_path_deconv)
        logging.info("Upload to EcoTaxa completed successfully.")
    except Exception as e:
        logging.error(f"Error during upload: {e}")

def upload_all_zips(remote, output_base_folder):
    for root, dirs, files in os.walk(process_output_base_folder):
        if "EcoTaxa" in dirs:  # check if ecotaxa folder exists
            eco_taxa_folder = os.path.join(root, "EcoTaxa")
            parent_folder_name = os.path.basename(root)
            
            # Define paths for the zip files
            zip_path_deconv = os.path.join(eco_taxa_folder, "ecotaxa_upload_deconv.zip")
            zip_path_raw = os.path.join(eco_taxa_folder, "ecotaxa_upload_raw.zip")

            # Check if the zip files exist
            if os.path.exists(zip_path_deconv) and os.path.exists(zip_path_raw):
                upload_to_ecotaxa(remote, zip_path_raw, zip_path_deconv)
            else:
                logging.warning(f"Zip files not found for folder {parent_folder_name}")

if __name__ == "__main__":
    # Define username and password
    USERNAME = 'fbrodbek@geomar.de'
    PASSWORD = 'CopepodC0nspiracy!'

    remote = login_to_ecotaxa(USERNAME, PASSWORD)
    if remote:
        process_output_base_folder = '/home/fanny/M181-3_output'
        upload_all_zips(remote, process_output_base_folder)
    else:
        logging.error("Failed to log into EcoTaxa. Exiting...")