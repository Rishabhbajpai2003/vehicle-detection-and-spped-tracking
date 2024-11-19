import cv2

# Video file path for the existing output
existing_output_video = r"C:\Vehicle-Detection-And-Speed-Tracking-master\Vehicle-Detection-And-Speed-Tracking-master\output.mp4"

# Open the existing video file
existing_video = cv2.VideoCapture(existing_output_video)

# Check if the video is opened correctly
if not existing_video.isOpened():
    print(f"Error: Could not open existing video {existing_output_video}.")
    exit()

# Get the video's properties (resolution, fps)
WIDTH = int(existing_video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(existing_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = existing_video.get(cv2.CAP_PROP_FPS)

# Display the video frames
while True:
    ret, frame = existing_video.read()
    if not ret:
        print("End of video stream.")
        break
    
    # Display the current frame
    cv2.imshow('Existing Output Video', frame)
    
    # Wait for the 'ESC' key to exit
    if cv2.waitKey(33) == 27:  # 33ms delay for each frame (approx. 30 FPS)
        break

# Release the video capture and close the window
existing_video.release()
cv2.destroyAllWindows()
