import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Show results
    for r in results:
        im_array = r.plot()  # Plot results
        cv2.imshow("YOLOv8 Detection", im_array)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
