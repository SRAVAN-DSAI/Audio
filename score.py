import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import logging

def init():
    global model
    global device
    # Load the registered model
    model_path = 'model.pth'  # Model file in the deployment environment
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    device = torch.device('cpu')

    # Define transformation
    global data_transform
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logging.info("Model initialized successfully.")

def run(raw_data):
    try:
        # Expecting JSON with base64 encoded image or file path
        data = json.loads(raw_data)
        img_path = data.get("image_path")  # Adjust based on input format
        if img_path:
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img).unsqueeze(0).to(device)  # Add batch dimension
        else:
            raise ValueError("No image_path provided in input data.")

        # Get prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        # Map prediction to class (assuming 0-9 map to dataset.classes)
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                       'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
                       'street_music']
        result = {"class": class_names[prediction], "confidence": output[0][prediction].item()}

        logging.info(f"Prediction: {result}")
        return json.dumps(result)
    except Exception as e:
        error = str(e)
        logging.error(f"Prediction error: {error}")
        return json.dumps({"error": error})