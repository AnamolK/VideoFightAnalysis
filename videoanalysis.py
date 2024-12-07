import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Initialize YOLOv8 model (small version)
model = YOLO('yolov8s.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Load the video file
video_path = 'C:/Users/kaspa/PycharmProjects/UFCData/test.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Set the round you want to analyze (e.g., Round 1)
round_number = 1
round_duration = 5 * 60  # Each UFC round is 5 minutes (in seconds)
round_start_time = (round_number - 1) * round_duration  # Start time of the round
round_end_time = round_number * round_duration  # End time of the round

# Frame rate (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
round_start_frame = int(round_start_time * fps)  # Calculate starting frame for the round
round_end_frame = int(round_end_time * fps)  # Calculate ending frame for the round

# Initialize variables for fight statistics for the current round
total_strikes_F1 = 0
total_strikes_F2 = 0
knockdowns_F1 = 0
knockdowns_F2 = 0
takedowns_F1 = 0
takedowns_F2 = 0


# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to infer strikes based on arm angles
def detect_strike(landmarks, side='right'):
    # Define keypoints indices based on MediaPipe Pose
    # Right side: Shoulder=12, Elbow=14, Wrist=16
    # Left side: Shoulder=11, Elbow=13, Wrist=15
    if side == 'right':
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
    else:
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]

    # Calculate the angle at the elbow
    angle = calculate_angle(shoulder, elbow, wrist)

    # If the arm is nearly straight, consider it as a potential strike
    if angle > 160:
        return True
    return False


# Enhanced Takedown Detection Function
def detect_takedown(landmarks_prev, landmarks_current, side='left', orientation_threshold=5, vertical_threshold=0.1):
    """
    Detects a takedown based on downward hip movement and torso angle approaching horizontal.

    Parameters:
    - landmarks_prev: List of previous frame's landmarks [(x, y, z), ...]
    - landmarks_current: List of current frame's landmarks [(x, y, z), ...]
    - side: 'left' or 'right' indicating which fighter
    - orientation_threshold: Degrees threshold to consider torso horizontal
    - vertical_threshold: Minimum normalized y-axis movement to consider significant

    Returns:
    - True if takedown is detected, False otherwise
    """
    # Define keypoints indices based on MediaPipe Pose
    # Shoulders: Left=11, Right=12
    # Hips: Left=23, Right=24
    if side == 'left':
        shoulder_prev = landmarks_prev[11]
        shoulder_curr = landmarks_current[11]
        hip_prev = landmarks_prev[23]
        hip_curr = landmarks_current[23]
    else:
        shoulder_prev = landmarks_prev[12]
        shoulder_curr = landmarks_current[12]
        hip_prev = landmarks_prev[24]
        hip_curr = landmarks_current[24]

    # Calculate torso angle relative to the horizontal axis
    delta_x = shoulder_curr[0] - hip_curr[0]
    delta_y = shoulder_curr[1] - hip_curr[1]
    if delta_x == 0:
        torso_angle = 90.0  # Vertical torso
    else:
        angle = np.degrees(np.arctan(delta_y / delta_x))
        torso_angle = abs(angle)

    # Check if torso is approximately horizontal
    is_horizontal = torso_angle < orientation_threshold or torso_angle > (180 - orientation_threshold)

    # Calculate vertical movement of hips (normalized coordinates)
    vertical_movement = hip_curr[1] > hip_prev[1] + vertical_threshold  # Hip moved downward

    # Takedown is detected if there is downward hip movement and torso is horizontal
    if vertical_movement and is_horizontal:
        return True
    return False


# Initialize previous landmarks for takedown detection
prev_landmarks_F1 = None
prev_landmarks_F2 = None

# Set desired frame width and height to resize
desired_width = 1280
desired_height = 720

# Loop through frames in the video, focusing on the selected round
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # End of video

    if frame_count > round_end_frame:
        break  # End of round

    # Resize the frame to the desired size
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Process frames within the round's timeframe
    if round_start_frame <= frame_count <= round_end_frame:
        if frame_count % int(fps) == 0:  # Process every second
            # YOLOv8: Detect objects in the frame
            results = model(frame)

            # Extract detections from the results (use results[0] since results is a list)
            detections = results[0].boxes  # Get bounding boxes

            # Filter out only "person" detections
            person_detections = [d for d in detections if int(d.cls[0]) == 0]  # Class 0 is 'person'

            if len(person_detections) >= 2:
                # Sort persons by their size (area of the bounding box) to identify the two largest (fighters)
                person_detections.sort(key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]),
                                       reverse=True)
                fighter_1 = person_detections[0]  # Largest person detected
                fighter_2 = person_detections[1]  # Second largest person detected

                # Extract bounding box coordinates
                x1_min, y1_min, x1_max, y1_max = fighter_1.xyxy[0].tolist()
                x2_min, y2_min, x2_max, y2_max = fighter_2.xyxy[0].tolist()

                # Crop the fighter regions for pose estimation
                fighter_1_crop = frame[int(y1_min):int(y1_max), int(x1_min):int(x1_max)]
                fighter_2_crop = frame[int(y2_min):int(y2_max), int(x2_min):int(x2_max)]

                # Perform pose estimation
                results_f1 = pose.process(cv2.cvtColor(fighter_1_crop, cv2.COLOR_BGR2RGB))
                results_f2 = pose.process(cv2.cvtColor(fighter_2_crop, cv2.COLOR_BGR2RGB))

                # Extract landmarks
                if results_f1.pose_landmarks:
                    landmarks_f1 = [(lm.x, lm.y, lm.z) for lm in results_f1.pose_landmarks.landmark]

                    # Detect strikes for Fighter 1 (both arms)
                    strike_detected_F1 = False
                    for side in ['right', 'left']:
                        if detect_strike(landmarks_f1, side=side):
                            strike_detected_F1 = True
                            print(f"Strike detected for Fighter 1 ({side} arm) at frame {frame_count}")

                    if strike_detected_F1:
                        total_strikes_F1 += 1

                    # Detect takedown for Fighter 1
                    if prev_landmarks_F1 is not None:
                        if detect_takedown(prev_landmarks_F1, landmarks_f1, side='left'):
                            takedowns_F1 += 1
                            print(f"Takedown detected for Fighter 1 at frame {frame_count}")
                    prev_landmarks_F1 = landmarks_f1

                if results_f2.pose_landmarks:
                    landmarks_f2 = [(lm.x, lm.y, lm.z) for lm in results_f2.pose_landmarks.landmark]

                    # Detect strikes for Fighter 2 (both arms)
                    strike_detected_F2 = False
                    for side in ['right', 'left']:
                        if detect_strike(landmarks_f2, side=side):
                            strike_detected_F2 = True
                            print(f"Strike detected for Fighter 2 ({side} arm) at frame {frame_count}")

                    if strike_detected_F2:
                        total_strikes_F2 += 1

                    # Detect takedown for Fighter 2
                    if prev_landmarks_F2 is not None:
                        if detect_takedown(prev_landmarks_F2, landmarks_f2, side='right'):
                            takedowns_F2 += 1
                            print(f"Takedown detected for Fighter 2 at frame {frame_count}")
                    prev_landmarks_F2 = landmarks_f2


                # Improved Pose Visualization
                # Function to map landmarks from cropped to main frame coordinates
                def map_landmarks(landmarks, bbox):
                    """
                    Maps normalized landmarks to the main frame coordinates based on the bounding box.

                    Parameters:
                    - landmarks: List of (x, y, z) tuples with normalized coordinates.
                    - bbox: Tuple of (x_min, y_min, x_max, y_max) in the main frame.

                    Returns:
                    - List of (x, y) tuples in the main frame coordinates.
                    """
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    mapped = []
                    for lm in landmarks:
                        x = int(lm[0] * width) + int(x_min)
                        y = int(lm[1] * height) + int(y_min)
                        mapped.append((x, y))
                    return mapped


                # Map landmarks to main frame coordinates
                if results_f1.pose_landmarks:
                    mapped_landmarks_f1 = map_landmarks(landmarks_f1, (x1_min, y1_min, x1_max, y1_max))
                else:
                    mapped_landmarks_f1 = None

                if results_f2.pose_landmarks:
                    mapped_landmarks_f2 = map_landmarks(landmarks_f2, (x2_min, y2_min, x2_max, y2_max))
                else:
                    mapped_landmarks_f2 = None

                # Define drawing specifications
                drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

                # Draw landmarks and connections for Fighter 1
                if mapped_landmarks_f1:
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(mapped_landmarks_f1) and end_idx < len(mapped_landmarks_f1):
                            start_point = mapped_landmarks_f1[start_idx]
                            end_point = mapped_landmarks_f1[end_idx]
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                    for idx, lm in enumerate(mapped_landmarks_f1):
                        cv2.circle(frame, lm, 4, (0, 255, 0), -1)

                # Draw landmarks and connections for Fighter 2
                if mapped_landmarks_f2:
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(mapped_landmarks_f2) and end_idx < len(mapped_landmarks_f2):
                            start_point = mapped_landmarks_f2[start_idx]
                            end_point = mapped_landmarks_f2[end_idx]
                            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

                    for idx, lm in enumerate(mapped_landmarks_f2):
                        cv2.circle(frame, lm, 4, (255, 0, 0), -1)

                # Draw bounding boxes
                # Fighter 1
                cv2.rectangle(frame, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (0, 255, 0), 2)
                # Fighter 2
                cv2.rectangle(frame, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, 0), 2)

                # Add Real-Time Strike Tracking Display
                # Define the position and font for the text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 255, 255)  # Yellow color
                thickness = 2
                line_type = cv2.LINE_AA

                # Prepare the strike count texts
                text_F1 = f"Fighter 1 Strikes: {total_strikes_F1}"
                text_F2 = f"Fighter 2 Strikes: {total_strikes_F2}"

                # Put the texts on the frame
                cv2.putText(frame, text_F1, (10, 30), font, font_scale, font_color, thickness, line_type)
                cv2.putText(frame, text_F2, (10, 70), font, font_scale, font_color, thickness, line_type)
                # End of Strike Tracking Display

                # Display the frame with annotations (for debugging)
                cv2.imshow('Fight Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    frame_count += 1

cap.release()
pose.close()
cv2.destroyAllWindows()

# After processing the round, print the round's fight statistics
print(f"Round {round_number} statistics:")
print(f"Total strikes - Fighter 1: {total_strikes_F1}, Fighter 2: {total_strikes_F2}")
print(f"Takedowns - Fighter 1: {takedowns_F1}, Fighter 2: {takedowns_F2}")
