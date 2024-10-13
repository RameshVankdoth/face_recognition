# Face Recognition System
## Overview
This Face Recognition System is a Python-based application designed to identify and recognize individuals using their facial features. It leverages deep learning techniques and computer vision to process images and provide real-time face recognition capabilities.

## Key Features
Real-Time Recognition: The system captures video from a webcam and recognizes faces in real-time.
Customizable Dataset: Easily add new individuals by placing their images in designated folders within the dataset.
Accurate Identification: Utilizes the face_recognition library for high-accuracy face detection and recognition.
Logging: Comprehensive logging of the face encoding process for easy debugging and tracking.

This script processes the images in the dataset to generate facial encodings, which are saved in face_encodings.pkl.
Run this script before attempting recognition to ensure the system is trained with the latest images.
Recognition Script (recognize_faces.py):

This script captures live video from the webcam and performs face recognition using the encodings stored in face_encodings.pkl.
Recognized faces are displayed with bounding boxes and names in the video feed.

## Dependencies
Python 3.x
OpenCV
face_recognition
NumPy

This project implements a face recognition system using Python, leveraging the `face_recognition` library for detecting and recognizing faces in real-time. The system is capable of training on a custom dataset of images and identifying individuals via webcam input.

## Table of Contents

- [Face Recognition System](#face-recognition-system)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Dependencies](#dependencies)
  - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
    - [Training:](#training)
    - [Recognition:](#recognition)
    - [Notes](#notes)
    - [Troubleshooting](#troubleshooting)

### Installation

1. Clone the repository or download the project files.
2. Navigate to the project folder in your terminal or command prompt.
3. Install the required packages using the following command:

## Requirements

Before running the scripts, ensure you have the following packages installed:

- **opencv-python**: Library for computer vision tasks.
- **face_recognition**: Library for face recognition.
- **numpy**: Library for numerical operations.

You can use the provided `requirements.txt` to install them:

```bash
pip install -r requirements.txt
```
or you can use

```bash
pip install opencv-python face_recognition numpy
```

## Usage
Step 1: Train Face Encodings
The first step is to generate face encodings from the images stored in the dataset directory. The images should be organized in subdirectories named after the individuals they represent. For example:

```markdown
dataset/
├── Ramesh/
│   ├── Ramesh.1.jpeg
│   └── Ramesh.2.jpeg
├── Asad/
│   ├── Asad.1.jpeg
│   └── Asad.2.jpeg
```

Run the following command to execute the training script:

```bash
python train_faces.py
```
This script will:

Process the images in the dataset directory.
Extract face encodings for each individual.
Save the encodings and corresponding names in a file named face_encodings.pkl.

Step 2: Recognize Faces
After generating the face encodings, you can recognize faces in real-time. Run the following command to execute the recognition script:

```bash
python recognize_faces.py
```
This script will:

Open your webcam and start capturing video.
Detect faces in each frame and match them against the known encodings.
Display the name of the recognized individual on the screen.
Press 'q' to quit the webcam feed.

## How It Works
### Training:

The train_faces.py script reads images from the dataset directory.
It extracts face encodings using the face_recognition library.
The encodings are stored in a pickle file for later use.

### Recognition:
The recognize_faces.py script captures frames from the webcam.
It resizes and processes the frames to detect faces.
Detected faces are compared against the encodings from the training step.
Recognized names are displayed on the screen along with bounding boxes around the faces.

### Notes
Ensure your webcam is functioning correctly before running the recognition script.
The dataset should contain a minimum of a few images for each individual to improve recognition accuracy.
You can add more individuals by creating a new subdirectory in the dataset directory and placing their images in it.

### Troubleshooting
Error Loading Images: If you encounter errors loading images, check that all files in the dataset are valid image files and organized correctly.
No Faces Found: If the recognition script reports that no faces were found, ensure that your webcam is positioned properly and that there are faces visible in the frame.
Low Recognition Accuracy: If the system struggles to recognize faces, consider adding more images for each individual in the dataset.