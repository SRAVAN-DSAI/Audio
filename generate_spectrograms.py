import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram(audio_path, output_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    # Set n_fft to the length of the signal if shorter than 2048
    n_fft = min(2048, len(y))
    if len(y) < 2048:
        print(f"Short file detected: {audio_path}, using n_fft={n_fft}")
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.imsave(output_path, spectrogram_db, cmap='viridis')

if __name__ == '__main__':
    input_base_dir = 'UrbanSound8K_structured/'
    output_base_dir = 'spectrograms/'
    os.makedirs(output_base_dir, exist_ok=True)

    # Process each class folder
    for class_name in os.listdir(input_base_dir):
        class_input_dir = os.path.join(input_base_dir, class_name)
        if os.path.isdir(class_input_dir):
            class_output_dir = os.path.join(output_base_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            for file in os.listdir(class_input_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(class_input_dir, file)
                    output_path = os.path.join(class_output_dir, f'{file.replace(".wav", ".png")}')
                    generate_spectrogram(audio_path, output_path)