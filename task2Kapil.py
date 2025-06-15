# -------------------- Libraries Required --------------------
import os
import torch
from PIL import Image
import numpy as np
from annoy import AnnoyIndex
import torchvision.models as vision_models
import torchvision.transforms as T

# -------------------- Model Setup --------------------

# Load pretrained ResNet18 and remove the classifier head
feature_extractor = vision_models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()

# Define preprocessing pipeline for input images
image_preprocessor = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# -------------------- Feature Vector Extraction --------------------

def get_image_embedding(image_file):
    """Convert an image to a numerical feature vector."""
    img = Image.open(image_file).convert('RGB')
    tensor_input = image_preprocessor(img).unsqueeze(0)
    with torch.no_grad():
        vector = feature_extractor(tensor_input).squeeze().numpy()
    return vector

# -------------------- ANN Index Creation --------------------

def create_image_index(folder_path, index_name='img_index.ann', dim=512, trees=10):
    """Create and save an Annoy index for all images in a folder."""
    annoy_idx = AnnoyIndex(dim, 'euclidean')
    filenames = []

    for idx, file in enumerate(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, file)
        vec = get_image_embedding(full_path)
        annoy_idx.add_item(idx, vec)
        filenames.append(file)

    annoy_idx.build(trees)
    annoy_idx.save(index_name)
    return filenames

# -------------------- Image Similarity Search --------------------

def retrieve_similar_images(query_img, index_name='img_index.ann', filenames=[], k=5):
    """Search top-k similar images to the query image."""
    dim = 512
    index = AnnoyIndex(dim, 'euclidean')
    index.load(index_name)

    query_vec = get_image_embedding(query_img)
    nearest = index.get_nns_by_vector(query_vec, k)
    return [filenames[i] for i in nearest]

# -------------------- Example Run --------------------

# Step 1: Index all dataset images
dataset_path = 'images_dataset'  # directory of reference images
file_names = create_image_index(dataset_path)

# Step 2: Query a similar image
query_img_path = 'query.jpg'  # image to search for
similar_images = retrieve_similar_images(query_img_path, filenames=file_names)

print("Top matches found:", similar_images)
