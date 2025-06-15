# Required Libraries
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from annoy import AnnoyIndex
import numpy as np

# Load Pretrained ResNet Model (remove the final classification layer)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Extract feature vector from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_t).squeeze().numpy()
    return features


# Build the index
def build_annoy_index(image_folder, index_file='image_index.ann', vector_dim=512, n_trees=10):
    index = AnnoyIndex(vector_dim, 'euclidean')
    image_list = []

    for i, filename in enumerate(os.listdir(image_folder)):
        path = os.path.join(image_folder, filename)
        features = extract_features(path)
        index.add_item(i, features)
        image_list.append(filename)

    index.build(n_trees)
    index.save(index_file)
    return image_list


# Search for similar images
def search_similar(image_path, index_file='image_index.ann', image_list=[], top_k=5):
    vector_dim = 512
    index = AnnoyIndex(vector_dim, 'euclidean')
    index.load(index_file)

    query_vector = extract_features(image_path)
    idxs = index.get_nns_by_vector(query_vector, top_k)
    return [image_list[i] for i in idxs]


# Example Usage
# 1. Build the index
image_folder = 'images_dataset'  # folder with images
image_list = build_annoy_index(image_folder)

# 2. Search similar images
query_image = 'query.jpg'  # your input query image
results = search_similar(query_image, image_list=image_list)

print("Top similar images:", results)
