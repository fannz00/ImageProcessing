#!/bin/bash
# filepath: /home/veit/PIScO_dev/ImageProcessing/remove_images.sh

# Script to remove all contents of /Images folders in subdirectories
# Usage: ./remove_images.sh [directory_path]

# Set the base directory (default to current directory if not provided)
BASE_DIR="${1:-.}"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' does not exist."
    exit 1
fi

echo "Searching for Images folders in subdirectories of: $BASE_DIR"

# Counter for tracking operations
found_count=0
removed_count=0

# Function to safely remove directory contents
safe_remove_contents() {
    local dir="$1"
    echo "  Removing contents of: $dir"
    
    # Method 1: Try using find with -delete (most efficient)
    if command -v find >/dev/null 2>&1; then
        find "$dir" -mindepth 1 -delete 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ Contents removed using find"
            return 0
        fi
    fi
    
    # Method 2: Use find with -exec rm (fallback)
    if command -v find >/dev/null 2>&1; then
        find "$dir" -mindepth 1 -type f -exec rm -f {} + 2>/dev/null
        find "$dir" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ Contents removed using find with exec"
            return 0
        fi
    fi
    
    # Method 3: Remove the entire directory and recreate it (last resort)
    local parent_dir=$(dirname "$dir")
    local dir_name=$(basename "$dir")
    rm -rf "$dir" 2>/dev/null
    mkdir -p "$dir" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ Directory recreated (contents removed)"
        return 0
    fi
    
    echo "  ✗ Failed to remove contents"
    return 1
}

# Find all subdirectories at two levels deep and check for Images folders
for subdir in "$BASE_DIR"/*/*/; do
    if [ -d "$subdir" ]; then
        images_dir="$subdir/Images"
        
        if [ -d "$images_dir" ]; then
            found_count=$((found_count + 1))
            echo "Found Images folder: $images_dir"
            
            # Check if the Images folder has contents
            if [ "$(ls -A "$images_dir" 2>/dev/null)" ]; then
                if safe_remove_contents "$images_dir"; then
                    removed_count=$((removed_count + 1))
                fi
            else
                echo "  ✓ Already empty"
            fi
        fi
    fi
done

echo ""
echo "Summary:"
echo "  Found $found_count Images folders"
echo "  Cleaned $removed_count folders"

if [ $found_count -eq 0 ]; then
    echo "No Images folders found at BASE_DIR/*/*/Images in $BASE_DIR"
fi