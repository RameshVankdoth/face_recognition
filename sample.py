import os
import threading

import cv2
import face_recognition
import numpy as np


# Function to load known face encodings and names from a specified folder
def load_face_encodings_from_folder(folder_path):
    encodings = []  # List to hold the face encodings
    names = []      # List to hold the corresponding names
    
    # Iterate through each person in the specified folder
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)  # Create path for each person's folder
        
        # Check if the path is a directory
        if os.path.isdir(person_folder):
            # Iterate through each image file in the person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)  # Create full image path
                try:
                    # Load the image and get face encodings
                    image = face_recognition.load_image_file(image_path)
                    person_encodings = face_recognition.face_encodings(image)
                    
                    # If encodings are found, append them to the lists
                    if person_encodings:
                        encodings.append(person_encodings[0])  # Add the first encoding
                        names.append(person_name)  # Add the corresponding name
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")  # Print error if loading fails
    
    return encodings, names  # Return the lists of encodings and names

# Function to recognize faces in a given frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Convert to RGB format

    # Find face locations and encodings in the resized frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []  # List to hold recognized face names
    for face_encoding in face_encodings:
        # Compare the detected face encoding with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match is found

        # If there are matches, find the best match
        if any(matches):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)  # Index of the closest match
            if matches[best_match_index]:
                name = known_face_names[best_match_index]  # Get the name of the matched encoding

        face_names.append(name)  # Append the name (or "Unknown") to the list
    
    return face_locations, face_names  # Return the locations and names of detected faces

# Function to draw rectangles around detected faces and label them
def draw_face_boxes(frame, face_locations, face_names):
    # Iterate through each detected face location and name
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back the face box coordinates to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label box beneath the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.LINE_4)
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX  # Choose a font for the text
        # Put the name text on the frame
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Class to handle video capture in a separate thread
class VideoCaptureThread(threading.Thread):
    def __init__(self, video_source=0):
        super().__init__()  # Initialize the threading superclass
        self.video_source = video_source  # Set the video source
        self.capture = cv2.VideoCapture(video_source)  # Open the video source
        self.frame = None  # Placeholder for the captured frame
        self.running = True  # Flag to control the thread's execution

    # Method to continuously capture frames
    def run(self):
        while self.running:
            ret, frame = self.capture.read()  # Read a frame from the video source
            if ret:
                self.frame = frame  # Store the captured frame

    # Method to stop the video capture
    def stop(self):
        self.running = False  # Set the running flag to False
        self.capture.release()  # Release the video capture resource

# Main function to coordinate the face recognition process
def main():
    # Load known face encodings and names from the specified folder
    known_face_encodings, known_face_names = load_face_encodings_from_folder('dataset')
    if not known_face_encodings:
        print("Error loading face encodings. Please check your images.")
        return  # Exit if no encodings were loaded

    # Start the video capture thread
    video_thread = VideoCaptureThread()
    video_thread.start()

    while True:
        frame = video_thread.frame  # Get the latest frame from the thread
        if frame is not None:
            # Recognize faces in the frame
            face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)
            # Draw boxes around recognized faces
            draw_face_boxes(frame, face_locations, face_names)
            # Display the resulting frame
            cv2.imshow('Real-Time Face Recognition', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_thread.stop()  # Stop the video capture thread
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Entry point of the program
if __name__ == "__main__":
    main()  # Call the main function to start the program
