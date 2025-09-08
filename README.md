### Emotion Recognition from Speech
A deep learning-based system for recognizing human emotions from speech audio using MFCC features and convolutional neural networks.

## Features
* Audio Processing: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files

* Deep Learning Model: CNN architecture with batch normalization and dropout for regularization

* Multi-Emotion Classification: Recognizes 8 different emotions:

`Neutral`

`Calm`

`Happy`

`Sad`

`Angry`

`Fearful`

`Disgust`

`Surprised`

* Comprehensive Evaluation: Includes accuracy metrics, confusion matrix, and classification report

* Visualization: Training history plots and confusion matrix visualization

## Usage
 1. Prepare Your Dataset
Download one of the supported datasets:

`RAVDESS`: Download here

`TESS`: Download here

`EMO-DB`: Download here  

2. Update the Data Path
Edit the `data_dir` variable in the `main()` function to point to your dataset location:

``data_dir = r"path/to/your/dataset" ``

3. Run the Script
     
``python emotion_recognition.py``

 4. Output
The script will:

Load and preprocess audio files

Extract MFCC features

Train a CNN model

Evaluate model performance

Generate visualizations

Save the trained model as `emotion_recognition_model.h5`

