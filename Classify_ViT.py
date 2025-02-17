import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
import os
from datasets import Dataset, Features, Image, Value
import re
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

def extract_pressure_from_filename(filename):
    # Use a regular expression to extract the pressure value from the filename
    match = re.search(r'(\d{4}\.\d{2})dbar', filename)
    if match:
        pressure = float(match.group(1))
        return pressure - 10  # Correct the pressure by subtracting 10 dbar
    return None

def generate_histogram(df, save_path, classlist):
    # Extract pressures and correct them
    df['pressure'] = df['filename'].apply(extract_pressure_from_filename)

    # Define the pressure bins (0 to 1000 dbar in steps of 10)
    bins = np.arange(0, 1000, 10)

    # Initialize a dictionary to count occurrences per bin
    bin_counts = {label: np.zeros(len(bins) - 1) for label in classlist}

    # Loop through each row and count occurrences in the pressure bin
    for _, row in df.iterrows():
        pressures = row['pressure']
        
        # Get the bin index for the current pressure
        bin_index = np.digitize(pressures, bins) - 1
        
        # Loop over each top prediction
        for label_column in ['top1', 'top2', 'top3', 'top4', 'top5']:
            label = row[label_column]
            if 0 <= bin_index < len(bins) - 1:
                bin_counts[label][bin_index] += 1

    # Plot the histogram
    plt.figure(figsize=(15, 7))
    for label, counts in bin_counts.items():
        plt.bar(bins[:-1], counts, width=10, alpha=0.5, label=label)

    plt.xlabel('Pressure (dbar)')
    plt.ylabel('Count')
    plt.title('Counts of Top 5 Predictions per Pressure Bin')
    plt.legend()
    plt.grid()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")

# Define a custom function to load the dataset with filenames
def load_unclassified_images(data_dir):
    # List all files in the directory and create full paths
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    # Prepare data for the dataset
    data = {
        'image': image_files,
        'filename': image_files 
    }
    
    features = Features({
        'image': Image(),
        'filename': Value('string')
    })
    
    # Create the dataset
    dataset = Dataset.from_dict(data, features=features)
    return dataset

def resize_to_larger_edge(image, target_size):
    # Get the original dimensions of the image
    original_width, original_height = image.size
    
    # Determine which dimension is larger
    larger_edge = max(original_width, original_height)
    
    # Compute the scale factor to resize the larger edge to the target size
    scale_factor = target_size / larger_edge
    
    # Compute new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize the image
    resized_image = F.resize(image, (new_height, new_width))
    
    return resized_image

def custom_image_processor(image, target_size=(224, 224), padding_color=255):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Step 1: Resize the image
    #resize_transform = transforms.Resize(target_size)
    resized_image = resize_to_larger_edge(image,224)

    #Step 2: Calculate padding
    new_width, new_height = resized_image.size
    padding_left = (target_size[0] - new_width) // 2
    padding_right = target_size[0] - new_width - padding_left
    padding_top = (target_size[1] - new_height) // 2
    padding_bottom = target_size[1] - new_height - padding_top

    # Step 3: Apply padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    pad_transform = transforms.Pad(padding, fill=padding_color)
    padded_image = pad_transform(resized_image)

    # Step 4: Apply other transformations
    transform_chain = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=180,fill=255),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    # Apply the transformations
    return transform_chain(padded_image)

# Example function to process a batch
def process_batch(example_batch):
    # Process each image in the batch
    processed_images = [custom_image_processor(img) for img in example_batch['image']]
    
    # Convert to a batch tensor
    inputs = torch.stack(processed_images)
    
    # Include labels (assuming they are present and you want to keep them)
    return {'pixel_values': inputs, 'filename': example_batch['filename']}

# Define the function to resize and pad the image
def transform(example_batch):
    # Define the desired image size
    desired_size = (224, 224)

    # Function to resize and pad an image to the desired size while keeping aspect ratio
    def resize_and_pad(image, desired_size):
        # Resize the image to ensure the longest side is 224 pixels
        #add own resize function
        transform_resize = transforms.Resize(224)
        resized_image = transform_resize(image)

        # Calculate padding to make the image 224x224
        width, height = resized_image.size
        left = (desired_size[0] - width) // 2
        right = desired_size[0] - width - left
        top = (desired_size[1] - height) // 2
        bottom = desired_size[1] - height - top

        # Apply padding
        padding = (left, top, right, bottom)
        padded_image = transforms.Pad(padding, fill=0)(resized_image)  # Assuming padding color is black (0)

        return padded_image.convert("RGB")

    # Apply resizing and padding to each image in the batch
    resized_padded_images = [resize_and_pad(x, desired_size) for x in example_batch['image']]
    
    # Convert resized and padded images to pixel values using the processor
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)
    inputs = processor(resized_padded_images, return_tensors='pt')

    # Include filenames in the inputs
    inputs['filename'] = example_batch['filename']

    return inputs

# Define the function to perform inference on a dataset in batches
def get_predictions_on_dataset_in_batches(dataset, save_dir, batch_size=16):
    # Initialize model
    vit = ViTForImageClassification.from_pretrained(save_dir)
    
    # Check if multiple GPUs are available and set the device accordingly
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     vit = torch.nn.DataParallel(vit)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)
    
    predictions = []
    filenames = []
    probabilities = []
    
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over the DataLoader to collect batches
    for batch in tqdm(dataloader, desc="Processing dataset"):
        #print(batch.keys())
        # Move inputs to the device
        inputs = batch['pixel_values'].to(device)
        #batch_filenames = batch['filename']

        # Perform inference
        with torch.no_grad():
            outputs = vit(pixel_values=inputs)

        # Convert logits to probabilities and get top-5 predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5, dim=-1)
        
        # Convert top-5 predictions to list of labels
        top_labels = [
            [vit.config.id2label[idx.item()] for idx in top_indices[i]]
            for i in range(top_indices.size(0))
        ]
        
        # Collect top-5 predictions
        predictions.extend(top_labels)
        
        # Collect filenames
        filenames.extend(batch['filename'])

        # Collect top-5 probabilities
        probabilities.extend(top_probs.cpu().numpy())
    
    return filenames, predictions, probabilities

if __name__ == "__main__":
    # Load the dataset and include filenames
    root_dir = '/home/fanny/segmentation_output'
    img_dir = os.path.join(root_dir, 'Deconv_crops')
    
    ds_pisco = load_unclassified_images(img_dir)
    print()

    classlist = ['acantharia_protist',
        'acantharia_protist_big_center',
        'acantharia_protist_halo',
        'amphipods',
        'appendicularian_fritillaridae',
        'appendicularian_s_shape',
        'appendicularian_slight_curve',
        'appendicularian_straight',
        'artifacts',
        'artifacts_edge',
        'chaetognath_non_sagitta',
        'chaetognath_other',
        'chaetognath_sagitta',
        'chordate_type1',
        'copepod_calanoid',
        'copepod_calanoid_eggs',
        'copepod_calanoid_eucalanus',
        'copepod_calanoid_flatheads',
        'copepod_calanoid_frillyAntennae',
        'copepod_calanoid_large',
        'copepod_calanoid_large_side_antennatucked',
        'copepod_calanoid_octomoms',
        'copepod_calanoid_small_longantennae',
        'copepod_cyclopoid_copilia',
        'copepod_cyclopoid_oithona',
        'copepod_cyclopoid_oithona_eggs',
        'copepod_other',
        'crustacean_other',
        'ctenophore_cestid',
        'ctenophore_cydippid_no_tentacles',
        'ctenophore_cydippid_tentacles',
        'ctenophore_lobate',
        'decapods',
        'detritus_blob',
        'detritus_filamentous',
        'detritus_other',
        'diatom_chain_string',
        'diatom_chain_tube',
        'echinoderm_larva_pluteus_brittlestar',
        'echinoderm_larva_pluteus_early',
        'echinoderm_larva_pluteus_typeC',
        'echinoderm_larva_pluteus_urchin',
        'echinoderm_larva_seastar_bipinnaria',
        'echinoderm_larva_seastar_brachiolaria',
        'echinoderm_seacucumber_auricularia_larva',
        'echinopluteus',
        'ephyra',
        'euphausiids',
        'euphausiids_young',
        'fecal_pellet',
        'fish_larvae_deep_body',
        'fish_larvae_leptocephali',
        'fish_larvae_medium_body',
        'fish_larvae_myctophids',
        'fish_larvae_thin_body',
        'fish_larvae_very_thin_body',
        'heteropod',
        'hydromedusae_aglaura',
        'hydromedusae_bell_and_tentacles',
        'hydromedusae_h15',
        'hydromedusae_haliscera',
        'hydromedusae_haliscera_small_sideview',
        'hydromedusae_liriope',
        'hydromedusae_narco_dark',
        'hydromedusae_narco_young',
        'hydromedusae_narcomedusae',
        'hydromedusae_other',
        'hydromedusae_partial_dark',
        'hydromedusae_shapeA',
        'hydromedusae_shapeA_sideview_small',
        'hydromedusae_shapeB',
        'hydromedusae_sideview_big',
        'hydromedusae_solmaris',
        'hydromedusae_solmundella',
        'hydromedusae_typeD',
        'hydromedusae_typeD_bell_and_tentacles',
        'hydromedusae_typeE',
        'hydromedusae_typeF',
        'invertebrate_larvae_other_A',
        'invertebrate_larvae_other_B',
        'jellies_tentacles',
        'polychaete',
        'protist_dark_center',
        'protist_fuzzy_olive',
        'protist_noctiluca',
        'protist_other',
        'protist_star',
        'pteropod_butterfly',
        'pteropod_theco_dev_seq',
        'pteropod_triangle',
        'radiolarian_chain',
        'radiolarian_colony',
        'shrimp-like_other',
        'shrimp_caridean',
        'shrimp_sergestidae',
        'shrimp_zoea',
        'siphonophore_calycophoran_abylidae',
        'siphonophore_calycophoran_rocketship_adult',
        'siphonophore_calycophoran_rocketship_young',
        'siphonophore_calycophoran_sphaeronectes',
        'siphonophore_calycophoran_sphaeronectes_stem',
        'siphonophore_calycophoran_sphaeronectes_young',
        'siphonophore_other_parts',
        'siphonophore_partial',
        'siphonophore_physonect',
        'siphonophore_physonect_young',
        'stomatopod',
        'tornaria_acorn_worm_larvae',
        'trichodesmium_bowtie',
        'trichodesmium_multiple',
        'trichodesmium_puff',
        'trichodesmium_tuft',
        'trochophore_larvae',
        'tunicate_doliolid',
        'tunicate_doliolid_nurse',
        'tunicate_partial',
        'tunicate_salp',
        'tunicate_salp_chains',
        'unknown_blobs_and_smudges',
        'unknown_sticks',
        'unknown_unclassified']

    # Apply the transform to the dataset
    ds_pisco_trans = ds_pisco.with_transform(process_batch)
    #print(ds_pisco_trans[0])

    # Define model name and path to the saved model
    model_dir = '/home/fanny/250129_ViT_custom_size_sensitive/best_model'

    # Get predictions on the dataset in batches
    filenames, predictions, probabilities = get_predictions_on_dataset_in_batches(ds_pisco_trans, model_dir, batch_size=128)

    # Create a DataFrame and store it on disk
    df = pd.DataFrame({
        'filename': filenames,
        'top1': [pred[0] for pred in predictions],
        'top2': [pred[1] for pred in predictions],
        'top3': [pred[2] for pred in predictions],
        'top4': [pred[3] for pred in predictions],
        'top5': [pred[4] for pred in predictions],
        'prob1': [prob[0] for prob in probabilities],
        'prob2': [prob[1] for prob in probabilities],
        'prob3': [prob[2] for prob in probabilities],
        'prob4': [prob[3] for prob in probabilities],
        'prob5': [prob[4] for prob in probabilities]
    })    
    result_path = os.path.join(root_dir,'ViT_predictions.csv')
    df.to_csv(result_path, index=False)
    print("Predictions saved!")

    # Print the first 5 predictions
    print(df.head())

    # Define the path to save the histogram
    histogram_path = os.path.join(root_dir, 'top5_prediction_histogram.png')

    # Generate histogram and save it
    generate_histogram(df, histogram_path, classlist)