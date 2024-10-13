import logging
import os
import pickle

import cv2
import face_recognition
import numpy as np

# Set up logging configuration
logging.basicConfig(
    filename='face_encoding.log',  # Log file name
    level=logging.DEBUG,  # Set logging level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format in log messages
)

# Log the start of the face encoding process
logging.info("Starting the face encoding process.")

# Directory where your images are stored
DATASET_DIR = r'C:\Users\vanky\OneDrive\Desktop\Face\dataset'  # Path to dataset directory

# Lists to store known face encodings and names
known_face_encodings = []  # Will hold the face encodings
known_face_names = []      # Will hold the names corresponding to the encodings

# Check if the dataset directory exists
if not os.path.exists(DATASET_DIR):
    logging.error(f"Dataset directory {DATASET_DIR} does not exist.")  # Log an error if the directory is missing
    exit()  # Exit the program if the directory does not exist

# Loop over each person in the dataset directory
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)  # Create full path to the person's directory
    
    # Ensure it's a directory
    if not os.path.isdir(person_dir):
        logging.warning(f"Skipping non-directory {person_dir}.")  # Log a warning for non-directory items
        continue  # Skip to the next person if the current is not a directory
    
    # Loop over each image file in the person's directory
    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)  # Full path to the image file
        logging.info(f'Processing image: {image_path}')  # Log the image being processed
        
        # Load the image using face_recognition
        try:
            image = face_recognition.load_image_file(image_path)  # Load the image
            
            # Get face encodings for the image
            face_encodings = face_recognition.face_encodings(image)  # Extract face encodings
            
            # Check if any encodings were found
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])  # Add the first encoding to the list
                logging.info(f"Encoding for {person_name} added with shape: {face_encodings[0].shape}.")  # Log the encoding shape

            # If a face was found in the image, add it to the list
            if face_encodings:
                known_face_encodings.append(face_encodings[0])  # Add the face encoding
                known_face_names.append(person_name)  # Add the corresponding person's name
                logging.info(f"Encoding for {person_name} added.")  # Log that the encoding was added
            else:
                logging.debug(f"No faces found in image {image_path}.")  # Log debug info if no faces found
        
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")  # Log any error that occurs during processing

# Save the encodings and names to a file using pickle
try:
    with open('face_encodings.pkl', 'wb') as f:  # Open a file in binary write mode
        pickle.dump((known_face_encodings, known_face_names), f)  # Save the encodings and names
    logging.info("Encodings saved to face_encodings.pkl.")  # Log that encodings were successfully saved
except Exception as e:
    logging.error(f"Error saving encodings to file: {e}")  # Log error if saving fails

# Print completion message indicating how many face encodings were saved
print(f"Training complete. Encodings for {len(known_face_names)} faces saved.")
