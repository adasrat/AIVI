CURRENT BUILD: AIVI final custom / AIVI rasp test (for raspberry pi)

STEPS TO RUN:

1) Install pip

2) Run: pip install ultralytics, pip install pyttsx3

3) Download AIVI camera test.py

4) Change cap = cv2.VideoCapture(1) to the desired index (0 is interanl, 1 is external, 2 is anything else)

5) Run 

If you are getting errors with OpenCV run the following:

1) pip uninstall opencv-python opencv-python-headless
2) pip install opencv-python
