import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8
def load_yolo():
    model = YOLO('yolov8n.pt')  # Use a YOLO model with face detection trained weights if available
    return model

# Detect faces using YOLOv8
def detect_faces(frame, model):
    results = model(frame)
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            confidence = detection.conf[0]
            class_id = detection.cls[0]
            if confidence > 0.5:  # Confidence threshold
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes

# Extract ORB features from the image
def extract_features(image):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Load known faces from the directory
def load_known_faces(known_faces_dir):
    known_face_descriptors = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(known_faces_dir, filename))
            keypoints, descriptors = extract_features(image)
            if descriptors is not None:
                known_face_descriptors.append(descriptors)
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_descriptors, known_face_names

# Match the detected face with known faces
def match_faces(detected_descriptors, known_face_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_score = float('inf')
    best_match_index = -1
    
    for idx, known_descriptors in enumerate(known_face_descriptors):
        matches = bf.match(detected_descriptors, known_descriptors)
        match_score = sum([match.distance for match in matches])
        
        if match_score < best_match_score:
            best_match_score = match_score
            best_match_index = idx

    return best_match_index

def main():
    # Initialize YOLO
    model = load_yolo()
    
    # Load known faces from the local directory
    known_faces_dir = "known_faces"  # Provide the path to your directory with known faces
    known_face_descriptors, known_face_names = load_known_faces(known_faces_dir)
    
    # Open video stream
    cap = cv2.VideoCapture(r"C:\Users\saduv\Videos\Camera_13_UB_ENTRANCE_UB_ENTRANCE_20250127155000_20250127160558_198544198.mp4")  # Provide your video file or 0 for webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the video frame
        boxes, indexes = detect_faces(frame, model)

        # Process each detected face
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                face = frame[y:y+h, x:x+w]
                
                # Extract features from the detected face
                keypoints, descriptors = extract_features(face)
                
                if descriptors is not None:
                    # Match the detected face with known faces
                    best_match_index = match_faces(descriptors, known_face_descriptors)
                    
                    if best_match_index != -1:
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = known_face_names[best_match_index]
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the video frame with detected faces and matched names
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()