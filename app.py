import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import librosa
import librosa.display
import numpy as np
import io

# Define model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes
device = torch.device('cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Define transformation for spectrogram
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
               'street_music']

# Function to generate spectrogram from audio
def audio_to_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    n_fft = min(2048, len(y))  # Adjust for short files
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Streamlit app
st.title("UrbanSound8K Audio Classifier")
uploaded_file = st.file_uploader("Choose an audio file (.wav)...", type="wav")

if uploaded_file is not None:
    # Convert audio to spectrogram
    spectrogram = audio_to_spectrogram(uploaded_file)
    spectrogram_img = Image.fromarray(255 * (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())).convert('RGB')

    # Apply transformation
    input_tensor = data_transform(spectrogram_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    # Display results
    st.audio(uploaded_file, format="audio/wav")
    st.image(spectrogram_img, caption="Generated Spectrogram", use_column_width=True)
    st.write(f"Prediction: {class_names[predicted.item()]}")
    st.write(f"Confidence: {confidence:.4f}")
