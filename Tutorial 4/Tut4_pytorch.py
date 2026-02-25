import os
from PIL import Image
import torch
from torchvision import transforms

# folder
output_dir=r"D:\PHD PROGRAMME\Courses\Sem2\DeepLearning\Tuts\T4\Augmented Images_Pytorch"
os.makedirs(output_dir,exist_ok=True)


# transforms 
augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=30),          # rotation
    transforms.RandomAffine(
        degrees=0,
        shear=20,                                   # shear
        scale=(0.8, 1.2)                            # zoom +/- 20 percent
    ),
    transforms.RandomHorizontalFlip(p=0.5),         # flip
    transforms.ColorJitter(brightness=0.3),         # brightness
])

image_path = "images.jpg"   # image path
image = Image.open(image_path).convert("RGB")

num_augmented_images = 40

for i in range(num_augmented_images):
    augmented_image = augmentation(image)
    save_path = os.path.join(output_dir, f"augmented_{i+1}.jpg")
    augmented_image.save(save_path)

print(f"{num_augmented_images} augmented images were saved in'{output_dir}'")