# face recignition
import os
import cv2
import face_recognition
import numpy as np
from deepface import DeepFace

# Path to the folder containing known faces
KNOWN_FACES_DIR = "known_faces/"
TOLERANCE = 0.6  # Lower tolerance means stricter matching
FRAME_RESIZE_FACTOR = 0.25  # Resize factor for faster processing
MODEL = "hog"  # Or use "cnn" for better accuracy (requires GPU)

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    image = cv2.imread(filepath)
    encoding = DeepFace.represent(img_path=filepath, model_name='Facenet')[0]["embedding"]
    known_face_encodings.append(np.array(encoding))
    known_face_names.append(os.path.splitext(filename)[0])  # Use filename as label

print(f"Loaded {len(known_face_encodings)} known faces.")
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]  # Generate encoding
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])  # Use filename as label

print(f"Loaded {len(known_face_encodings)} known faces.")

# Open the video file (or webcam with `0`)
video_capture = cv2.VideoCapture(r"C:\Users\saduv\Videos\Camera_13_UB_ENTRANCE_UB_ENTRANCE_20250127155000_20250127160558_198544198.mp4")  # Replace with 0 for live webcam

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces and compute encodings for detected faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if any(matches):  # Check if a match is found
            best_match_index = face_distances.argmin()  # Find the closest match
            name = known_face_names[best_match_index]

        # Draw rectangle and label around the detected face
        top, right, bottom, left = [int(coord * (1 / FRAME_RESIZE_FACTOR)) for coord in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the video frame with detections
    cv2.imshow("Face Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
