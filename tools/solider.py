import torch
from torchvision import transforms
from PIL import Image
import glob
import os

def preprocess_image(img_path, heigth,width):
    transform = transforms.Compose([
        transforms.Resize((heigth, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    return image

def extract_images_from_subfolders(folder_paths):
    # If the input is a string (single folder path), convert it into a list
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    all_images = []
    
    for folder_path in folder_paths:
        # Walk through each main folder and its subfolders
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # For each subfolder, find all .png images
            images = glob.glob(os.path.join(dirpath, '*.png'))
            all_images.extend(images)
    return all_images

def solider_result(folder_path=""):
    model_path = '/home/diego/Documents/detectron2/solider_model.pth'
    loaded_model = torch.load(model_path)
    loaded_model.eval()  # Set the model to evaluation mode
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    

    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,384,128)], dim=0) for img in images]
    with torch.no_grad():
        features_list, _ = loaded_model(torch.cat(total_batch,dim=0).to(device))
    
    features_array = features_list.cpu().numpy()
    return features_array, image_names
