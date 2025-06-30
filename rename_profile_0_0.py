from pathlib import Path
import re
import sys
import argparse
import pandas as pd
import os
import glob

def rename_files(base_path, dry_run=True):
    base = Path(base_path)
    pattern = re.compile(r'_0_0_')
    renamed_count = 0
    modified_csv_count = 0
    
    print(f"Searching for files in: {base}")
    print(f"Mode: {'Dry run (no changes will be made)' if dry_run else 'Real run (files will be renamed)'}")
    
    # First collect all renames we need to do
    rename_mapping = {}
    for path in base.rglob('*'):
        if path.is_file():
            old_name = path.name
            if pattern.search(old_name):
                new_name = pattern.sub('_0-0_', old_name)
                rename_mapping[old_name] = new_name
                new_path = path.parent / new_name
                try:
                    if dry_run:
                        print(f"Would rename: {path} -> {new_path}")
                    else:
                        path.rename(new_path)
                        print(f"Renamed: {path} -> {new_path}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Error with {old_name}: {e}")
    
    # Then update CSV contents
    # Then update CSV contents
    data_folder = os.path.join(base_path, 'Data')
    if os.path.exists(data_folder):
        for csv_file in glob.glob(os.path.join(data_folder, '*.csv')):
            try:
                df = pd.read_csv(csv_file, header=None)
                df[1] = df[1].astype(str).apply(lambda x: x.replace('_0_0_', '_0-0_'))
                df.to_csv(csv_file, index=False, header=False)
                print(f"Updated CSV file: {csv_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

    print(f"\nTotal files {'to be ' if dry_run else ''}renamed: {renamed_count}")
    print(f"Total CSV files {'to be ' if dry_run else ''}modified: {modified_csv_count}")
    return renamed_count, modified_csv_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename PISCO files replacing _0_0_ with _0-0_ and update CSV contents')
    parser.add_argument('directory', help='Directory path to process')
    parser.add_argument('--execute', action='store_true', help='Execute the renaming (without this flag, runs in dry-run mode)')
    
    args = parser.parse_args()
    rename_files(args.directory, dry_run=not args.execute)