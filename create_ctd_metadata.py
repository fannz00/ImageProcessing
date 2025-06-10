import os
import datetime
import re
from pathlib import Path
import subprocess

def parse_profile_datetime(profile_name):
    """Extract datetime from profile name like SO298_298-6-1_PISCO2_20230418-1801"""
    timestamp_str = profile_name.split('_')[-1]
    date_str = timestamp_str[:8]
    time_str = timestamp_str[9:]
    
    # Convert to datetime object
    dt = datetime.datetime.strptime(f"{date_str}-{time_str}", "%Y%m%d-%H%M")
    return dt

def parse_ctd_file_datetime(ctd_file_path):
    """Extract datetime from CTD file"""
    with open(ctd_file_path, 'r') as f:
        date_line = ""
        time_line = ""
        for line in f:
            if line.startswith("Date"):
                date_line = line.strip()
            elif line.startswith("Time"):
                time_line = line.strip()
                break
                
    if date_line and time_line:
        date_str = date_line.split('=')[1].strip()
        time_str = time_line.split('=')[1].strip()
        
        # Convert to datetime object
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
        return dt
    return None

def find_matching_ctd_file(profile_datetime, ctd_dir):
    """Find closest matching CTD file based on timestamp"""
    min_time_diff = datetime.timedelta(days=1)  # Maximum 1 day difference
    matching_file = None
    
    for ctd_file in os.listdir(ctd_dir):
        if not ctd_file.endswith('.ctd'):
            continue
            
        ctd_path = os.path.join(ctd_dir, ctd_file)
        ctd_datetime = parse_ctd_file_datetime(ctd_path)
        
        if ctd_datetime:
            time_diff = abs(profile_datetime - ctd_datetime)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                matching_file = ctd_file
                
    return matching_file

def create_metadata_csv(profile_path, ctd_id):
    """Create metadata CSV file with CTD profile ID using sudo"""
    metadata_dir = os.path.join(profile_path, f"{os.path.basename(profile_path)}_Metadata")
    csv_path = os.path.join(metadata_dir, f"{os.path.basename(profile_path)}.csv")
    
    # Create content for CSV
    csv_content = f"CTDprofileid,{ctd_id}\n"
    
    # Create temporary file in home directory
    temp_path = os.path.expanduser("~/temp_metadata.csv")
    with open(temp_path, 'w') as f:
        f.write(csv_content)
    
    # Use sudo to create directory if it doesn't exist
    if not os.path.exists(metadata_dir):
        subprocess.run(['sudo', 'mkdir', '-p', metadata_dir])
    
    # Use sudo to move file to final location
    subprocess.run(['sudo', 'mv', temp_path, csv_path])
    subprocess.run(['sudo', 'chmod', '666', csv_path])

def main():
    cruise_base = "/mnt/filer/SO298/SO298-PISCO-Profiles"
    ctd_dir = "/mnt/filer/SO298/SO298-CTD_UVP_ETC/SO298-CTD/calibrated"
    
    for profile in os.listdir(cruise_base):
        profile_path = os.path.join(cruise_base, profile)
        if not os.path.isdir(profile_path):
            continue
            
        # Check if metadata CSV already exists
        metadata_csv = os.path.join(profile_path, f"{profile}_Metadata", f"{profile}.csv")
        if os.path.exists(metadata_csv):
            continue
            
        # Parse datetime from profile name
        profile_dt = parse_profile_datetime(profile)
        
        # Find matching CTD file
        matching_ctd = find_matching_ctd_file(profile_dt, ctd_dir)
        
        if matching_ctd:
            # Extract CTD ID from filename (e.g., "001" from "son_298_1_001.ctd")
            ctd_id = matching_ctd.split('_')[-1].replace('.ctd', '')
            print(f"Found match for {profile}: {matching_ctd} (ID: {ctd_id})")
            
            # Create metadata CSV
            create_metadata_csv(profile_path, ctd_id)
            print(f"Created metadata CSV for {profile}")
        else:
            print(f"No matching CTD file found for {profile}")

if __name__ == "__main__":
    main()