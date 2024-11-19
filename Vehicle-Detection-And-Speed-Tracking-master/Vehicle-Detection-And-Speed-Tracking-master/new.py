import cv2
import dlib
import time
import math
import numpy as np

# Verify Haar Cascade File Path
haar_file = "C:\\Vehicle-Detection-And-Speed-Tracking-master\\Vehicle-Detection-And-Speed-Tracking-master\\myhaar.xml"
if not os.path.exists(haar_file):
    print(f"Error: Haar cascade file {haar_file} not found.")
    exit()

# Load Haar Cascade Classifier
carCascade = cv2.CascadeClassifier(haar_file)
if carCascade.empty():
    print(f"Error loading Haar cascade from {haar_file}.")
    exit()

# Video File Path
video_file = "C:\\Vehicle-Detection-And-Speed-Tracking-master\\Vehicle-Detection-And-Speed-Tracking-master\\cars.mp4"
video = cv2.VideoCapture(video_file)
if not video.isOpened():
    print(f"Error: Could not open video {video_file}.")
    exit()

WIDTH = 1280
HEIGHT = 720
fps = video.get(cv2.CAP_PROP_FPS)

# Speed Estimation Function
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # Adjust pixels per meter
    d_meters = d_pixels / ppm
    speed = d_meters * fps * 3.6  # Convert to km/h
    return speed

# Function to track multiple objects (vehicles)
def trackMultipleObjects():
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    # Video output
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (WIDTH, HEIGHT))

    frameCounter = 0
    currentCarID = 0

    while True:
        rc, image = video.read()
        if not rc:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        frameCounter += 1
        
        carIDtoDelete = []

        # Update car trackers
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image_rgb)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        # Detect cars every 10 frames
        if frameCounter % 10 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar, y_bar = x + 0.5 * w, y + 0.5 * h
                matchCarID = None

                # Check if detected car matches existing tracker
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x, t_y, t_w, t_h = int(trackedPosition.left()), int(trackedPosition.top()), int(trackedPosition.width()), int(trackedPosition.height())
                    t_x_bar, t_y_bar = t_x + 0.5 * t_w, t_y + 0.5 * t_h

                    if (t_x <= x_bar <= t_x + t_w) and (t_y <= y_bar <= t_y + t_h):
                        matchCarID = carID

                # If no match, create a new tracker
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image_rgb, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        # Update car positions and estimate speed
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y, t_w, t_h = int(trackedPosition.left()), int(trackedPosition.top()), int(trackedPosition.width()), int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

            if carLocation1.get(carID) and carLocation2.get(carID):
                x1, y1, w1, h1 = carLocation1[carID]
                x2, y2, w2, h2 = carLocation2[carID]

                # Estimate speed if car moves within a specific vertical range
                if (speed[carID] is None or speed[carID] == 0) and y1 >= 275 and y1 <= 285:
                    speed[carID] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                # Display speed on the image
                if speed[carID] is not None and y1 >= 180:
                    cv2.putText(resultImage, f"{int(speed[carID])} km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(resultImage)

        # Optionally display the processed image (useful for debugging)
        # cv2.imshow('Result', resultImage)

        # Exit if 'ESC' is pressed
        if cv2.waitKey(33) == 27:  # 33ms delay for each frame (approx. 30 FPS)
            break
    
    # Release video resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()
