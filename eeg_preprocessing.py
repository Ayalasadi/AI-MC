import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import joblib #for a pre-trained model (come back to this)
import simpleaudio as sa #for playing audio files
import os
import wave

#EEG Data Preprocessing Functions
def load_eeg_data(file_path):
    """
    Load EEG data from a CSV file.
    This function assumes the EEG data is stored in a CSV file with columns representing different electrodes.
    """
    #adding a try-except block to catch file not found errors
    try:
        eeg_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} could not be found.")
        raise
    return eeg_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.
    lowcut: Low frequency bound for the filter.
    highcut: High frequency bound for the filter.
    fs: Sampling rate of the EEG data.
    order: Order of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_eeg_data(eeg_data, lowcut, highcut, fs, order=5):
    """
    Apply bandpass filtering to the EEG data.
    eeg_data: DataFrame with EEG signals.
    lowcut, highcut: Frequency range for the filter.
    fs: Sampling rate.
    order: Order of the filter.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    eeg_filtered = eeg_data.apply(lambda x: lfilter(b, a, x))
    return eeg_filtered

def normalize_eeg_data(eeg_data):
    """
    Normalize EEG data.
    """
    normalized_data = (eeg_data - eeg_data.mean()) / eeg_data.std()
    return normalized_data

def extract_features(eeg_data):
    """
    Extract basic features from EEG data.
    Features: mean, standard deviation, and maximum value for each channel.
    """
    features = pd.DataFrame()
    features['mean'] = eeg_data.mean()
    features['std'] = eeg_data.std()
    features['max'] = eeg_data.max()
    return features

#Functions for Emotion Prediction and Music Generation
def predict_emotion(features):
    """
    Predict the emotional state based on the features extracted from EEG data.
    """
    #adding a try-except block to catch file not found errors
    try:
        model = joblib.load('emotion_classification_model.pkl')
        emotion_label = model.predict(features.values.reshape(1, -1))
    except FileNotFoundError:
        print("Error: The model file 'emotion_classification_model.pkl' was not found.")
        raise
    return emotion_label[0]

def select_music(emotion_label):
    """
    Map the predicted emotion to a specific type of music.
    """
    base_path = '/Users/aya/Desktop/AI_MC/AI-MC/Music/'  
    music_mapping = {
        'calm': os.path.join(base_path, 'calm_instrumental.wav'),
        'excited': os.path.join(base_path, 'upbeat_pop.wav'),
        'sad': os.path.join(base_path, 'soft_piano.wav')
    }
    return music_mapping.get(emotion_label, os.path.join(base_path, 'default_music.wav'))

def play_music(music_file):
    """
    Play the selected music file.
    """
    #adding a try-except block to catch file not found errors
    try:
        wave_obj = sa.WaveObject. from_wave_file(music_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except FileNotFoundError:
        print(f"Error: The music file at {music_file} could not be found.")
        raise

def preprocess_and_generate_music(file_path, lowcut=1, highcut=50, fs=256):
    """
    Preprocess the EEG data by loading, filtering, normalizing, and extracting features,
    then predict the emotion and play the corresponding/appropriate music.
    file_path: path to the EEG data file.
    lowcut, highcut: Frequency range for the bandpass filter.
    fs: Sampling rate of the EEG data.
    """
    try:
        print("Loading EEG data...")
        eeg_data = load_eeg_data(file_path)
        print("EEG data loaded successfully.")

        print("Filtering EEG data...")
        eeg_filtered = filter_eeg_data(eeg_data, lowcut, highcut, fs)
        print("EEG data filtered successfully.")

        print("Normalizing EEG data...")
        eeg_normalized = normalize_eeg_data(eeg_filtered)
        print("EEG data normalized successfully.")

        print("Extracting features...")
        features = extract_features(eeg_normalized)
        print("Features extracted successfully.")

        print("Predicting emotion...")
        emotion_label = predict_emotion(features)
        print(f"Predicted emotion: {emotion_label}")

        print("Selecting music...")
        music_file = select_music(emotion_label)
        print(f"Playing music file: {music_file}")

        play_music(music_file)
        print("Music played successfully.")
    except Exception as e:
        print(f"An error occurred during preprocessing or music generation: {e}")

#Function to generate dummy .wav files for testing
def generate_dummy_wav(filename):
    sample_rate = 44100  #44.1 kHz
    duration = 1.0       #1 second
    frequency = 440.0    #Frequency of the generated tone (A4)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  #mono
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

#Generate the necessary dummy files
generate_dummy_wav('calm_instrumental.wav')
generate_dummy_wav('upbeat_pop.wav')
generate_dummy_wav('soft_piano.wav')

#Testing
preprocess_and_generate_music('sample_eeg_data.csv')

