import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

model = YOLO("yolov8n.pt")
engine = pyttsx3.init()

# Set the voice to a female voice 
voices = engine.getProperty('voices')
for voice in voices:
    engine.setProperty('voice', voices[1].id)
    break

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Set to 0 for built in, 1 for usb, 2-onwards for anything else
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    closest_object = None
    max_area = 0
    closest_direction = ""

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = r.names[int(box.cls[0])]

            area = (x2 - x1) * (y2 - y1)

            # Set the closest object
            if area > max_area:
                max_area = area
                closest_object = label
                center_x = (x1 + x2) // 2
                closest_direction = get_direction(center_x, frame.shape[1])

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

    # Set the timer for speaking every 5 seconds
    current_time = time.time()
    if closest_object and (current_time - last_announce_time) >= 5:
        # Create a thread to avoid lag
        threading.Thread(target=speak, args=(f"There is a {closest_object} {closest_direction}",)).start()
        last_announce_time = current_time  

    # This is to change between cameras because I was lazy
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): # Quit
        break
    elif key == ord("s"):  # Switch
        cap.release()
        camera_index = 2 if camera_index == 0 else 0
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            break

cap.release()
cv2.destroyAllWindows()