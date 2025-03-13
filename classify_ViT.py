import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

def classify_images(input_dir, output_csv):
    """
    Classify images using a Vision Transformer (ViT) model.

    Args:
        input_dir (str): Directory containing images to classify.
        output_csv (str): Path to save the classification results as a CSV file.
    """
    # Load the pre-trained ViT model
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # List to store classification results
    results = []

    # Iterate over images in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            # Load and preprocess the image
            input_image = Image.open(img_path)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

            # Classify the image
            with torch.no_grad():
                output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get the top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            results.append([img_name] + top5_catid.tolist() + top5_prob.tolist())

    # Save the results to a CSV file
    columns = ['filename', 'top1', 'top2', 'top3', 'top4', 'top5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(output_csv, index=False)
