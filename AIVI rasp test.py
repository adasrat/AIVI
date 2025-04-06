import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
import numpy as np

#can switch between best.pt and yolov8n.pt
model = YOLO("yolov8n.pt")

# Use eSpeak for better performance on Raspberry Pi
engine = pyttsx3.init()
engine.setProperty('rate', 150)  

# Set camera index (0 for built-in, 1 for USB)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_announce_time = time.time()

def speak(text):
   
    threading.Thread(target=lambda: engine.say(text) and engine.runAndWait()).start()

def get_direction(center_x, frame_width):
    
    if center_x < frame_width / 3:
        return "to your left"
    elif center_x > 2 * frame_width / 3:
        return "to your right"
    else:
        return "in front of you"

def get_distance(area, frame_area):
    
    if area > frame_area * 0.5:
        return "very close"
    elif area > frame_area * 0.25:
        return "close"
    elif area > frame_area * 0.1:
        return "moderately close"
    else:
        return "far"

# List of objects to prioritize
obstacle_objects = ["Chair", "Desk", "Person", "Trashcan", "Computer", "Window", "Door"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, verbose=False)  # Disable verbose for performance

    closest_object = None
    max_area = 0
    closest_direction = ""
    closest_distance = ""

    frame_area = frame.shape[0] * frame.shape[1]

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = r.names[int(box.cls[0])]

            area = (x2 - x1) * (y2 - y1)

            if label in obstacle_objects and area > max_area:
                max_area = area
                closest_object = label
                center_x = (x1 + x2) // 2
                closest_direction = get_direction(center_x, frame.shape[1])
                closest_distance = get_distance(area, frame_area)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display results
    cv2.imshow("Object Detection", frame)

    # Announce the closest object every 2 seconds
    current_time = time.time()
    if closest_object and (current_time - last_announce_time) >= 2:
        speak(f"There is a {closest_object} {closest_direction}, {closest_distance}")
        last_announce_time = current_time  

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
