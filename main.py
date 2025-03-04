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
        # Get image dimensions
        h, w, c = frame.shape

        # Get nose coordinates (landmark 0 is the nose)
        nose = result.pose_landmarks.landmark[0]
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)

        # Adjust the y-coordinate to place the dot just above the nose
        dot_y = int(nose_y - 10)  # Adjust 40 pixels above the nose

        # Draw a red dot just above the nose position
        cv2.circle(frame, (nose_x, dot_y), 7, (0, 0, 255), -1)

        # Draw a green rectangle around the dot (bullseye area)
        rect_width, rect_height = 110, 150  # Width and height of the rectangle
        top_left = (nose_x - rect_width // 2, dot_y - rect_height // 2)
        bottom_right = (nose_x + rect_width // 2, dot_y + rect_height // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

        # Display coordinates near the bullseye (the red dot above the nose)
        cv2.putText(frame, f"Head: ({nose_x}, {dot_y})", 
                    (nose_x - 50, dot_y - 85),  # Positioning near the bullseye
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Print coordinates (for debugging or motor control)
        print(f"Head Position: X = {nose_x}, Y = {dot_y}")

    # Show the frame
    cv2.imshow("Human Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
