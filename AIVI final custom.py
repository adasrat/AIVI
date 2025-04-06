import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

model = YOLO("best.pt")
engine = pyttsx3.init()

# Set the voice to a female voice 
voices = engine.getProperty('voices')
for voice in voices:
    engine.setProperty('voice', voices[1].id)
    break

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Set to 1 for built in, 0 for usb, 2-onwards for anything else
camera_index = 0
cap = cv2.VideoCapture(camera_index)

last_announce_time = time.time()

cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 1280, 720) 

def get_direction(center_x, frame_width):
    #sets the direction relative to the cam
    if center_x < frame_width / 3:
        return "to your left"
    elif center_x > 2 * frame_width / 3:
        return "to your right"
    else:
        return "in front of you"

def get_distance(area, frame_area):
    # set the distance from the cam
    if area > frame_area * 0.5:
        return "very close"
    elif area > frame_area * 0.25:
        return "close"
    elif area > frame_area * 0.1:
        return "moderately close"
    else:
        return "far"

# List of objects to prioritize as obstacles
obstacle_objects = ["Person", "Desk", "Chair", "Trashcan", "Computer", "Window", "Door"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

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

            # Prioritize obstacle objects
            if label in obstacle_objects and area > max_area:
                max_area = area
                closest_object = label
                center_x = (x1 + x2) // 2
                closest_direction = get_direction(center_x, frame.shape[1])
                closest_distance = get_distance(area, frame_area)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            text_x = center_x - (text_width // 2)
            text_y = center_y + (text_height // 2)

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    resized_frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Object Detection", resized_frame)

    # Set the timer for speaking every 2 seconds
    current_time = time.time()
    if closest_object and (current_time - last_announce_time) >= 2:
        threading.Thread(target=speak, args=(f"There is a {closest_object} {closest_direction}, {closest_distance}",)).start()
        last_announce_time = current_time  

    # press q to quit, maybe voice activate turn on/off?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()