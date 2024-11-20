## EEG Music Therapy Project

This project aims to use EEG data to generate music based on a patient's emotional state. The generated music is intended to help patients relax, feel better, and support their healing journey. The core idea is to analyze EEG data, predict emotional states, and generate a suitable auditory experience.

## Features

- EEG Data Preprocessing: Includes filtering, normalization, and feature extraction from EEG data.
- Emotion Prediction: Uses a pre-trained model (`emotion_classification_model.pkl`) to predict the emotional state based on EEG data.
- Music Selection and Playback: Maps the predicted emotional state to an appropriate music track and plays it using `simpleaudio`.
- Dummy WAV Generation: Creates dummy `.wav` files to use for testing music playback.

## Files in the Repository

- `eeg_preprocessing.py`: The main script for preprocessing EEG data, predicting emotional states, and generating music. It also includes functions to generate dummy `.wav` files.
- `sample_eeg_data.csv`: A sample CSV file containing EEG data with columns for different channels (`channel_1`, `channel_2`, `channel_3`), used for testing.
- `emotion_classification_model.pkl`: A pre-trained dummy model for emotion prediction.
- **Dummy Music Files**: Three `.wav` files are generated by the script to simulate different emotions (`calm_instrumental.wav`, `upbeat_pop.wav`, `soft_piano.wav`).

## How to Use

1. Setup: Make sure you have Python installed along with the following libraries:

   - `numpy`
   - `pandas`
   - `scipy`
   - `simpleaudio`
   - `scikit-learn`
   - `joblib`

   You can install the required packages using:
   pip install numpy pandas scipy simpleaudio scikit-learn joblib
   

2. Run the Script:

   - To preprocess EEG data and generate music, run the main script:
     ```sh
     python3 eeg_preprocessing_extended.py
     ```
   - This will process the EEG data from `sample_eeg_data.csv`, predict the emotional state, and play the appropriate music.

3. Generate Dummy Data:

   - The script will automatically generate three dummy `.wav` files if they do not exist, which are used for testing music playback.

## How It Works

1. Load EEG Data: The script reads EEG data from a CSV file.
2. Preprocess the Data: Filters, normalizes, and extracts key features (mean, standard deviation, and maximum value).
3. Predict Emotion: Uses a pre-trained model to predict the emotional state based on EEG features.
4. Select and Play Music: Plays a music file that corresponds to the predicted emotion (`calm`, `excited`, `sad`).

## Notes

- The current implementation includes a dummy emotion classification model and dummy `.wav` files. You can replace these with real data and trained models for production use.
- Make sure all files (CSV, `.pkl`, and `.wav`) are in the same directory as the script for it to run properly.

## License

 Feel free to use this project for learning and personal use. Contributions are welcome!

