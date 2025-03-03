import cv2
import mediapipe as mp

# Initialize MediaPipe Pose (for human detection)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open Camera
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Create a resizable window
cv2.namedWindow("Human Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Human Detection", 1280, 720)  # Set to desired size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Pose detection
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Get bounding box around the detected person
        h, w, c = frame.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0
        
        for landmark in result.pose_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min, y_min = min(x, x_min), min(y, y_min)
            x_max, y_max = max(x, x_max), max(y, y_max)

        # Draw green boundary box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Human Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
