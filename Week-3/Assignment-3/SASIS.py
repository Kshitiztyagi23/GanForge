import os
import torch
from torchvision import models,transforms
from PIL import Image
import torch.nn as nn
from annoy import AnnoyIndex

images_folder="D:/GANFORGE/training_set/training_set/dogs"
images=os.listdir(images_folder)
weights=models.ResNet18_Weights.IMAGENET1K_V1
model=models.resnet18(weights=weights)

model.fc = nn.Identity()
print(model)

model.eval()


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

annoy_index=AnnoyIndex(512,'angular')   

for i in range(len(images)):
    image=Image.open(os.path.join(images_folder,images[i]))
    input_tensor=transform(image).unsqueeze(0)
          
    if input_tensor.size()[1]==3:
          output_tensor=model(input_tensor)
          annoy_index.add_item(i,output_tensor[0])
          
          if i%100 ==0:
               print(f'Processed {i} images.')
        #   print(f'{images[i]} predicted as {weights.meta["categories"][torch.argmax(output_tensor)]}')
        #   input()

        
annoy_index.build(10)
annoy_index.save('dog_index.ann')