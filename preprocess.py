# preprocess.py
import os
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # A library to show a progress bar

def create_spectrogram(audio_path, output_path):
    """
    Takes the path to an audio file, creates a spectrogram image,
    and saves it to the specified output path. This version is
    optimized for speed in a cloud job.
    """
    try:
        # 1. Load the audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=None) # sr=None preserves original sample rate

        # 2. Create a Mel spectrogram
        # For speed, we use parameters that create a smaller, but still useful, image.
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        
        # 3. Convert to decibels (log scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 4. Save the image
        # We create a simple figure without axes or colorbars for speed and to save space.
        fig = plt.figure(figsize=(1, 1)) # Create a small 1x1 inch figure
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Create axes that fill the entire figure
        ax.set_axis_off() # Turn off the axis labels
        fig.add_axes(ax) # Add the axes to the figure
        
        # Display the spectrogram on the axes
        librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, ax=ax)
        
        # Save the figure to the specified path
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        # 5. Clean up to free memory
        plt.close(fig)
        
    except Exception as e:
        # If a file is corrupt or causes an error, we print it and continue.
        print(f"Could not process {audio_path}. Error: {e}")


def main(args):
    """
    The main function that orchestrates the whole process.
    It walks through the input directory and calls create_spectrogram for each file.
    """
    print(f"Starting preprocessing...")
    print(f"Input data path: {args.input_data}")
    print(f"Output data path: {args.output_data}")

    # Walk through the input directory. os.walk is a Python generator that
    # yields the root directory, a list of subdirectories, and a list of files
    # for each level of the directory tree.
    for root, dirs, files in os.walk(args.input_data):
        for filename in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            # We only care about .wav files
            if filename.endswith('.wav'):
                # Full path to the original audio file
                input_audio_path = os.path.join(root, filename)
                
                # Create a corresponding folder structure in the output directory.
                # This preserves our labels (e.g., 'dog_bark', 'siren').
                # We replace the input base path with the output base path.
                relative_path = os.path.relpath(root, args.input_data)
                output_dir = os.path.join(args.output_data, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # The final output path for our new spectrogram image.
                # We change the extension from .wav to .png.
                output_image_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
                
                # Call our function to do the conversion
                create_spectrogram(input_audio_path, output_image_path)

    print("Preprocessing complete.")


if __name__ == '__main__':
    # This block sets up how the script receives arguments from the Azure ML Job.
    parser = argparse.ArgumentParser(description="Preprocess audio files to spectrograms.")
    
    # Define the command-line argument for the input data path.
    # Azure ML will automatically map our Data Asset to this path.
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input audio data.')
    
    # Define the command-line argument for the output data path.
    # Azure ML provides this path for us to save our results.
    parser.add_argument('--output_data', type=str, required=True, help='Path to save the output spectrograms.')
    
    # Parse the arguments provided by the Azure ML Job controller.
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments.
    main(args)
