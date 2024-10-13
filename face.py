import os

import cv2
import face_recognition
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Function to load known face encodings from a specified folder
def load_face_encodings_from_folder(folder_path):
    """
    Loads face encodings and their corresponding names from a folder containing subdirectories
    for each person.

    Parameters:
        folder_path (str): The path to the directory containing subdirectories of images for each person.

    Returns:
        tuple: A tuple containing:
            - encodings (list): A list of face encodings for the known individuals.
            - names (list): A list of names corresponding to each face encoding.
    """
    encodings = []  # Initialize an empty list to hold face encodings
    names = []      # Initialize an empty list to hold names of individuals

    # Loop through each person in the dataset directory
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)  # Create full path to the person's folder
        if os.path.isdir(person_folder):  # Ensure it is a directory
            # Loop through each image file in the person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)  # Full path to the image
                image = face_recognition.load_image_file(image_path)  # Load the image using face_recognition
                person_encodings = face_recognition.face_encodings(image)  # Extract face encodings from the image
                
                # Check if any encodings were found
                if person_encodings:
                    encodings.append(person_encodings[0])  # Add the first encoding to the list
                    names.append(person_name)  # Add the corresponding name to the list
                    
    return encodings, names  # Return the lists of encodings and names

# Load known face encodings and names from the 'dataset' directory
known_face_encodings, known_face_names = load_face_encodings_from_folder('dataset')

# Check if any encodings were loaded
if not known_face_encodings:
    print("Error loading face encodings. Please check your images.")  # Print error message
    video_capture.release()  # Release the webcam
    cv2.destroyAllWindows()   # Close any OpenCV windows
    exit()  # Exit the program

process_this_frame = True  # Flag to control frame processing

# Main loop for real-time face recognition
while True:
    ret, frame = video_capture.read()  # Capture a single frame from the webcam

    # Check if the frame was captured
    if not ret:
        print("Error: Failed to capture image.")  # Print error if frame capture fails
        break  # Exit the loop if image capture fails

    # Process only every other frame to improve performance
    if process_this_frame:
        # Resize frame to a smaller size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  
        # Convert the image from BGR color (OpenCV format) to RGB color
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)  
        # Get face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []  # List to hold names of recognized faces
        # Check if no faces were found in the current frame
        if not face_locations:
            print("No faces found in the current frame.")  # Print message if no faces are detected
        else:
            # Loop through each detected face encoding
            for face_encoding in face_encodings:
                # Compare detected face with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"  # Default name if no match is found

                # Check if any matches were found
                if any(matches):
                    # Compute distances to known face encodings
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    # Get the index of the closest match
                    best_match_index = np.argmin(face_distances)
                    # If the best match is valid, retrieve the corresponding name
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)  # Append the name of the recognized face

    process_this_frame = not process_this_frame  # Toggle frame processing flag

    # Display results on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4  # Scale the face location coordinates back to the original frame size
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a filled rectangle for the name background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX  # Set font for the text
        # Put the name text on the frame
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show the frame in a window titled 'Real-Time Face Recognition'
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
