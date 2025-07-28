import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import librosa
import numpy as np
import io
import time
import plotly.graph_objects as go
import pandas as pd
import os

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="UrbanSound8K Audio Classifier | Sravan Kodari",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SRAVAN-DSAI/Sound-Classifier',
        'Report a Bug': 'mailto:sravankodari4@gmail.com',
        'About': 'UrbanSound8K Audio Classifier by Sravan Kodari'
    }
)

# --- Custom CSS for a Polished UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --primary-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        --secondary-bg: #f0f2f6;
        --text-color: #2c3e50;
        --accent-color: #27ae60; /* Green for audio theme */
        --button-bg: linear-gradient(45deg, #27ae60, #219653);
        --button-hover-bg: linear-gradient(45deg, #219653, #27ae60);
        --button-text: #ffffff;
        --border-color: #d1d5db;
        --shadow-color: rgba(0,0,0,0.1);
        --alert-success-bg: #e6f4ea;
        --alert-success-text: #2e7d32;
        --alert-warning-bg: #fff3e0;
        --alert-warning-text: #ef6c00;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--primary-bg);
        color: var(--text-color);
    }

    .stSidebar {
        background-color: var(--secondary-bg);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 5px var(--shadow-color);
    }
    
    .stButton>button {
        background: var(--button-bg); color: var(--button-text); border: none;
        border-radius: 12px; padding: 12px 24px; font-size: 1.1rem;
        font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 4px var(--shadow-color);
    }
    .stButton>button:hover {
        background: var(--button-hover-bg); transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }

    .stFileUploader label {
        border-radius: 12px; border: 2px solid var(--border-color); padding: 12px;
        font-size: 1rem; background-color: var(--secondary-bg);
        color: var(--text-color);
    }
    .stFileUploader label:focus {
        border-color: var(--accent-color);
    }

    h1, h2, h3 {
        color: var(--text-color); font-weight: 700;
    }

    a {
        color: var(--accent-color); text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div style='background-color:var(--secondary-bg); padding:1rem; border-radius:12px; margin-bottom:1rem;
                display:flex; justify-content:space-between; align-items:center; box-shadow: 0 2px 4px var(--shadow-color);'>
        <div>
            <h2 style='margin:0;'>🎵 UrbanSound8K Classifier</h2>
            <span style='color:var(--text-color); opacity:0.7;'>by YourName</span>
        </div>
        <div>
            <a href='https://github.com/SRAVAN-DSAI/Sound-Classifier' target='_blank' style='margin-right:1.5rem;'>GitHub</a>
            <a href='https://www.linkedin.com/in/sravan-kodari' target='_blank'>LinkedIn</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes
    device = torch.device('cpu')
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# --- Core Functions ---
def audio_to_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    n_fft = min(2048, len(y))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def predict_category(audio_file, model, device):
    spectrogram = audio_to_spectrogram(audio_file)
    spectrogram_img = Image.fromarray(255 * (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())).convert('RGB')
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = data_transform(spectrogram_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
                   'street_music']
    return {
        "predicted_category": class_names[predicted.item()],
        "confidence": probabilities[predicted.item()],
        "raw_probabilities": dict(zip(class_names, probabilities))
    }

def clear_inputs_and_results():
    st.session_state.uploader_key_counter += 1
    st.session_state.pop('results_df', None)

# --- Initialize session state ---
if 'uploader_key_counter' not in st.session_state:
    st.session_state.uploader_key_counter = 0

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.75, 0.05,
        help="Filter predictions by minimum confidence level."
    )
    max_batch_size = st.number_input(
        "Max Batch Size", 1, 1000, 100, 10,
        help="Limit audio files processed from a batch."
    )
    st.button(
        "🧹 Clear Inputs & Results",
        on_click=clear_inputs_and_results,
        use_container_width=True
    )

# --- Main App ---
tab1, tab2, tab3 = st.tabs(["🎙️ **Single Audio**", "📈 **Batch Analysis**", "ℹ️ **Model & About**"])

with tab1:
    st.header("Analyze a Single Audio File")
    st.markdown("Upload a `.wav` file to classify the urban sound.")
    uploaded_file = st.file_uploader(
        "Upload Audio File", type=["wav"],
        key=f"single_file_uploader_{st.session_state.uploader_key_counter}"
    )

    if uploaded_file is not None:
        if st.button("🚀 Classify Audio", type="primary", use_container_width=True):
            with st.spinner("Classifying..."):
                result = predict_category(uploaded_file, model, device)
            st.subheader("Classification Result")
            col_result, col_probs = st.columns(2)
            with col_result:
                category = result['predicted_category']
                confidence = result['confidence']
                if confidence >= confidence_threshold:
                    st.success(f"**Category: {category}**")
                else:
                    st.warning(f"**Category: {category}** (Confidence below threshold)")
                st.metric("Confidence Score", f"{confidence:.2%}")
            with col_probs:
                probs_df = pd.DataFrame(list(result['raw_probabilities'].items()), columns=['Category', 'Probability'])
                probs_df = probs_df.sort_values(by='Probability', ascending=True)
                fig = go.Figure(go.Bar(
                    x=probs_df['Probability'], y=probs_df['Category'], orientation='h', marker_color='#27ae60'
                ))
                fig.update_layout(
                    title="Probability Distribution", xaxis_title="Probability", yaxis_title="",
                    height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)'), xaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig, use_container_width=True)
            st.audio(uploaded_file, format="audio/wav")

with tab2:
    st.header("Analyze a Batch of Audio Files")
    st.markdown("Upload a `.zip` file containing multiple `.wav` files.")
    uploaded_zip = st.file_uploader(
        "Upload Zip File", type=["zip"],
        key=f"batch_file_uploader_{st.session_state.uploader_key_counter}"
    )

    if uploaded_zip is not None:
        import zipfile
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            audio_files = [f for f in zip_ref.namelist() if f.endswith('.wav')]
            if not audio_files:
                st.error("No .wav files found in the zip.")
                st.stop()
            st.info(f"Found **{len(audio_files)}** audio files in the zip.")
            if len(audio_files) > max_batch_size:
                st.warning(f"Zip contains {len(audio_files)} files. Processing the first **{max_batch_size}**.")
                audio_files = audio_files[:max_batch_size]
            results = []
            progress_bar = st.progress(0, text="Starting batch analysis...")
            for i, audio_file in enumerate(audio_files):
                with zip_ref.open(audio_file) as f:
                    result = predict_category(io.BytesIO(f.read()), model, device)
                results.append(result)
                progress_bar.progress((i + 1) / len(audio_files), f"Processing... {i+1}/{len(audio_files)} files classified.")
            progress_bar.empty()
            st.success("Batch analysis complete!")
            results_df = pd.DataFrame(results)
            st.session_state.results_df = results_df

    if 'results_df' in st.session_state:
        st.subheader("Analysis Results")
        results_df = st.session_state.results_df
        filtered_df = results_df[results_df['confidence'] >= confidence_threshold]
        category_counts = filtered_df['predicted_category'].value_counts()
        if not category_counts.empty:
            col_chart, col_summary = st.columns(2)
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=category_counts.index, y=category_counts.values, marker_color=['#27ae60', '#e74c3c', '#2ecc71']
                ))
                fig.update_layout(
                    title=f"Category Distribution (Confidence ≥ {confidence_threshold:.0%})",
                    xaxis_title="Category", yaxis_title="File Count", paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)')
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_summary:
                st.dataframe(category_counts.reset_index().rename(columns={'index': 'Category', 'predicted_category': 'Count'}), use_container_width=True)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Results (CSV)", csv, 'audio_classification_results.csv', 'text/csv', use_container_width=True
            )
        else:
            st.warning("No files met the confidence threshold. Try lowering it in the sidebar settings.")

with tab3:
    st.header("ℹ️ Model & App Details")
    st.markdown(f"""
    This app uses a **ResNet18** model fine-tuned for multi-class audio classification on the **UrbanSound8K** dataset.
    It classifies audio into one of ten categories: **{', '.join(['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])}**.
    - **Device In Use:** `{device.type.upper()}`
    - **Model Source:** Trained on Azure Machine Learning
    - **Frontend:** Streamlit

    ### Performance Metrics (on validation set)
    """)
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "96.37%")
    col2.metric("F1-Score (Macro)", "95.80%")  # Estimated based on confusion matrix
    st.header("👋 About the Creator")
    st.markdown("""
    This app was built by **KODARI SRAVAN**.
    - **Connect:** [GitHub](https://github.com/SRAVAN-DSAI/Sound-Classifier) | [LinkedIn](https://www.linkedin.com/in/sravan-kodari)
    - **Contact:** sravankodari4@gmail.com

    <small>_Last updated: July 2025_</small>
    """, unsafe_allow_html=True)
