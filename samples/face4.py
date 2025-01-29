import cv2
import os
import numpy as np

# Initialize the face cascade and ORB feature extractor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
orb = cv2.ORB_create()

# Brute Force Matcher for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Function to extract ORB keypoints and descriptors
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Load known faces from directory
known_faces = []
known_names = []

# Directory containing known faces
known_faces_dir = "known_faces/"  # Replace with your directory containing face images

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(image_path)
        
        # Detect faces in the known face image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            keypoints, descriptors = extract_features(face)
            if descriptors is not None:
                known_faces.append(descriptors)
                known_names.append(filename.split(".")[0])  # Store name without extension

# Open the video stream (use 0 for webcam or a video path)
video_capture = cv2.VideoCapture(r"C:\Users\saduv\Downloads\adhavan2.mp4")  # Replace with your video file or 0 for webcam

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through all detected faces in the frame
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Extract features from the detected face
        keypoints, descriptors = extract_features(face)
        
        if descriptors is not None:
            # Compare the detected face's descriptors with the known faces
            best_match_index = -1
            best_match_score = float('inf')
            
            for i, known_descriptor in enumerate(known_faces):
                # Match descriptors using Brute Force Matcher
                matches = bf.match(descriptors, known_descriptor)
                
                # Calculate a matching score (the lower, the better)
                match_score = sum([m.distance for m in matches])
                
                if match_score < best_match_score:
                    best_match_score = match_score
                    best_match_index = i
            
            # If a good match is found, label the face
            if best_match_index != -1 and best_match_score < 500:
                name = known_names[best_match_index]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the video feed with the matched faces
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
