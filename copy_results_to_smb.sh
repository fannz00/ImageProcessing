#!/bin/bash

SRC_BASE="/home/veit/PIScO_dev/Segmentation_results/SO298/SO298-PISCO-Profiles/TempResults"
DST_BASE="/mnt/filer/SO298/SO298-PISCO-Profiles"

for profile in "$SRC_BASE"/*; do
    echo "Processing profile: $profile"
    # locate EcoTaxa (adjust -maxdepth if your structure is deeper)
    eco_dir=$(find "$profile" -maxdepth 3 -type d -name EcoTaxa -print -quit)

    if [ -n "$eco_dir" ]; then
        prof_name=$(basename "$profile")
        results_dirname=$(basename "$(dirname "$eco_dir")")  # e.g. <profile>_Results
        dst_folder="$DST_BASE/${prof_name}/${results_dirname}/EcoTaxa"

        echo "Cleaning $dst_folder"
        sudo mkdir -p "$dst_folder"
        sudo rm -f "$dst_folder"/*.zip "$dst_folder"/*.csv

        echo "Copying .zip and .csv files from $eco_dir -> $dst_folder"
        sudo rsync -a --progress -v \
            --include='*.zip' --include='*.csv' --exclude='*' \
            "$eco_dir/" "$dst_folder/"
    else
        echo "No EcoTaxa found under $profile"
    fi
done

echo "All .zip and .csv files from EcoTaxa folders copied."