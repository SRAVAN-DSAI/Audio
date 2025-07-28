import os
import pandas as pd
import shutil
import sys
import tarfile # The UrbanSound8K dataset is in a .tar.gz archive
import requests # NEW: Library to make API calls

# --- 1. CONFIGURATION ---
# The main folder where the final structured dataset will be created.
ROOT_PATH = "UrbanSound8K_structured"
# The name of the downloaded archive file.
ARCHIVE_FILENAME = "UrbanSound8K.tar.gz"
# The name of the directory created after extraction.
EXTRACTED_DIR_NAME = "UrbanSound8K"
# The Zenodo record ID for the UrbanSound8K dataset
ZENODO_RECORD_ID = "1203745"


print("--- UrbanSound8K Automated Dataset Setup Script ---")

# --- 2. DOWNLOAD THE DATASET ---
# Check if the archive file already exists. If not, download it.
if not os.path.exists(ARCHIVE_FILENAME):
    print(f"\n--- Downloading UrbanSound8K dataset ({ARCHIVE_FILENAME}) ---")
    print("This is a large file (~6 GB) and will take some time.")
    
    # NEW: Use Zenodo API to get the correct download link
    try:
        api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
        print(f"Fetching download link from API: {api_url}")
        response = requests.get(api_url)
        response.raise_for_status() # Raises an error for bad responses
        record_data = response.json()
        
        # Find the download link for our specific file
        download_link = None
        for f in record_data['files']:
            if f['key'] == ARCHIVE_FILENAME:
                download_link = f['links']['self']
                break
        
        if not download_link:
            print(f"FATAL ERROR: Could not find a download link for '{ARCHIVE_FILENAME}' in the API response.")
            sys.exit(1)
            
        print(f"Download link found. Starting download...")
        # Use wget via os.system for a robust download with a progress bar.
        os.system(f"wget --progress=bar:force -O {ARCHIVE_FILENAME} '{download_link}'")

    except requests.exceptions.RequestException as e:
        print(f"\nFATAL ERROR: Could not connect to Zenodo API: {e}")
        print("Please check your network connection and try again.")
        sys.exit(1)

    # Verify that the download was successful
    if not os.path.exists(ARCHIVE_FILENAME) or os.path.getsize(ARCHIVE_FILENAME) < 1000000: # Check if file is tiny
        print(f"\nFATAL ERROR: Download failed. The file '{ARCHIVE_FILENAME}' is missing or empty.")
        print("Please check your network connection or try running the script again.")
        if os.path.exists(ARCHIVE_FILENAME):
            os.remove(ARCHIVE_FILENAME) # Clean up empty file
        sys.exit(1)
else:
    print(f"\n--- Dataset archive '{ARCHIVE_FILENAME}' already exists. Skipping download. ---")

# --- 3. EXTRACT THE DATASET ---
# Check if the dataset has already been extracted. If not, extract it.
if not os.path.exists(EXTRACTED_DIR_NAME):
    print(f"\n--- Extracting '{ARCHIVE_FILENAME}' (this may take a few minutes) ---")
    try:
        with tarfile.open(ARCHIVE_FILENAME, "r:gz") as tar:
            tar.extractall()
        print(f"Successfully extracted to '{EXTRACTED_DIR_NAME}/'")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to extract the archive. It may be corrupt.")
        print(f"Error details: {e}")
        print(f"Please try deleting '{ARCHIVE_FILENAME}' and running the script again to re-download.")
        sys.exit(1)
else:
    print(f"\n--- Dataset directory '{EXTRACTED_DIR_NAME}' already exists. Skipping extraction. ---")


# --- 4. STRUCTURE THE DATASET ---
print("\n--- Structuring the dataset into labeled folders ---")

# Create the main root folder for our new structured dataset
os.makedirs(ROOT_PATH, exist_ok=True)

# Load the metadata file which contains the mapping of filenames to classes
metadata_path = os.path.join(EXTRACTED_DIR_NAME, 'metadata', 'UrbanSound8K.csv')
try:
    metadata = pd.read_csv(metadata_path)
except FileNotFoundError:
    print(f"FATAL ERROR: Metadata file not found at '{metadata_path}'.")
    print("The extraction may have failed silently. Please check the contents of the 'UrbanSound8K' folder.")
    sys.exit(1)

total_files = len(metadata)
processed_files = 0
print(f"Found metadata for {total_files} files. Starting structuring process...")

# Iterate over each row in the metadata file
for index, row in metadata.iterrows():
    filename = row['slice_file_name']
    fold_number = 'fold' + str(row['fold'])
    class_name = row['class']
    
    # --- THIS IS THE CORRECTED LINE ---
    # The original path was missing the 'audio' subdirectory.
    source_path = os.path.join(EXTRACTED_DIR_NAME, 'audio', fold_number, filename)
    
    # Define the new directory for the class
    label_dir = os.path.join(ROOT_PATH, class_name)
    os.makedirs(label_dir, exist_ok=True)
    
    # Define the final destination path for the audio file
    destination_path = os.path.join(label_dir, filename)
    
    # Check if the source file actually exists before trying to move it
    if os.path.exists(source_path):
        # Use shutil.move to move the file. This is efficient.
        shutil.move(source_path, destination_path)
        processed_files += 1
    
    # Print a progress update every 500 files
    if (index + 1) % 500 == 0:
        print(f"Processed {index + 1}/{total_files} files...")

print(f"\nProcessed a total of {processed_files} files.")
print(f"--- Setup Complete! ---")
print(f"Dataset successfully structured in the '{ROOT_PATH}' folder.")
print(f"\nRecommendation: You can now delete the large '{ARCHIVE_FILENAME}' file and the original '{EXTRACTED_DIR_NAME}' folder to save space.")
