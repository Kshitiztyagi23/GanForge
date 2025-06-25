# Required Libraries
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from annoy import AnnoyIndex
import numpy as np

# Load Pretrained ResNet Model (Remove Final Classification Layer)
resnet_model = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
feature_extractor.eval()

# Image Transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to Extract Features from an Image
def get_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = feature_extractor(input_tensor).squeeze().numpy()  # Shape: [512]
    return features

# Function to Build Annoy Index from a Folder of Images
def create_annoy_index(folder_path, output_index_file='image_index.ann', feature_dim=512, trees=10):
    index = AnnoyIndex(feature_dim, 'euclidean')
    filenames = []

    for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, file_name)
        vector = get_image_features(file_path)
        index.add_item(idx, vector)
        filenames.append(file_name)

    index.build(trees)
    index.save(output_index_file)
    return filenames

# Function to Find Similar Images Given a Query Image
def find_similar_images(query_path, saved_index_file='image_index.ann', file_list=[], top_k=5):
    feature_dim = 512
    index = AnnoyIndex(feature_dim, 'euclidean')
    index.load(saved_index_file)

    query_vector = get_image_features(query_path)
    nearest_indices = index.get_nns_by_vector(query_vector, top_k)
    return [file_list[i] for i in nearest_indices]

# -------------------------------
# Example Usage

# Step 1: Build the Annoy Index
dataset_folder = 'images_dataset'  # Folder with images
image_filenames = create_annoy_index(dataset_folder)

# Step 2: Search for Similar Images
query_img = 'query.jpg'  # Input query image path
top_matches = find_similar_images(query_img, file_list=image_filenames)

print("Top similar images:", top_matches)
