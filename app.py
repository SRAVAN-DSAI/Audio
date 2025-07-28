import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import streamlit as st

# Define model architecture
model = models.resnet18(pretrained=False)  # Load without pretrained weights
model.fc = nn.Linear(model.fc.in_features, 10)  # Match your 10 classes
device = torch.device('cpu')

# Load state dictionary
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()  # Set to evaluation mode

# Define transformation
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
               'street_music']

# Streamlit app
st.title("UrbanSound8K Classifier")
uploaded_file = st.file_uploader("Choose a spectrogram image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = data_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    st.image(uploaded_file, caption="Uploaded Spectrogram", use_column_width=True)
    st.write(f"Prediction: {class_names[predicted.item()]}")
    st.write(f"Confidence: {confidence:.4f}")
