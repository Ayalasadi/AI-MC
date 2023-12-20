import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

def load_eeg_data(file_path):
    """
    Load EEG data from a CSV file.
    This function assumes the EEG data is stored in a CSV file with columns representing different electrodes.
    """
    eeg_data = pd.read_csv(file_path)
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

def preprocess_eeg(file_path, lowcut=1, highcut=50, fs=256):
    """
    Preprocess the EEG data by loading, filtering, normalizing, and extracting features.
    file_path: Path to the EEG data file.
    lowcut, highcut: Frequency range for the bandpass filter.
    fs: Sampling rate of the EEG data.
    """
    eeg_data = load_eeg_data(file_path)
    eeg_filtered = filter_eeg_data(eeg_data, lowcut, highcut, fs)
    eeg_normalized = normalize_eeg_data(eeg_filtered)
    features = extract_features(eeg_normalized)
    return features

# Example usage
# features = preprocess_eeg('path_to_your_eeg_data_file.csv')
