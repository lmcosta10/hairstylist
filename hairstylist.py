import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms

# TODO: Replace this with hair detection model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
imagePath = 'person.jpg'
img = cv2.imread(imagePath)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply the transformation and prepare for the model
input_tensor = transform(img_rgb).unsqueeze(0)

# Perform segmentation
with torch.no_grad():
    output = model(input_tensor)['out'][0]

# Generate the segmentation map
segmentation_map = torch.argmax(output, dim=0).byte().cpu().numpy()

# TODO: this is assuming '15' is the class index for hair
HAIR_CLASS_INDEX = 15
hair_mask = (segmentation_map == HAIR_CLASS_INDEX).astype(np.uint8) * 255

# Resize the mask back to the original image size
hair_mask_resized = cv2.resize(hair_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Overlay the mask on the original image
overlay = img_rgb.copy()
overlay[hair_mask_resized == 255] = [255, 0, 0]  # Highlight hair in red

# Display the results
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Hair Detection (Highlighted)")
plt.imshow(overlay)
plt.axis('off')

plt.show()
