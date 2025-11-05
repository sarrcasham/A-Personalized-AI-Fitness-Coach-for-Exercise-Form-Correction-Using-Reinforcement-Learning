# perception.py - Fixed version for barbell curl analysis

import cv2
import mediapipe as mp
import numpy as np

# Define the relevant keypoints for barbell curl analysis
BARBELL_CURL_KEYPOINTS = [
    11, 12,  # Left and right shoulders
    13, 14,  # Left and right elbows  
    15, 16,  # Left and right wrists
    23, 24,  # Left and right hips (for stability)
    25, 26,  # Left and right knees (for stance)
    0,        # Nose (for head position)
    7, 8      # Left and right ears (for head stability)
]

def extract_landmarks(video_path: str) -> np.ndarray:
    """Processes a video file to extract 3D pose landmarks for each frame."""

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f" Error: Could not open video file {video_path}")
        return np.array([])

    all_landmarks = []
    frame_count = 0
    successful_detections = 0

    print(f"📹 Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_landmarks = []

            # Extract coordinates for specific keypoints we need for barbell curl
            for landmark_id in BARBELL_CURL_KEYPOINTS:
                if landmark_id < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[landmark_id]
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    # Fill with zeros if landmark is missing
                    frame_landmarks.extend([0.0, 0.0, 0.0, 0.0])

            all_landmarks.append(frame_landmarks)
            successful_detections += 1
        else:
            # If no pose detected, add zero frame to maintain sequence length
            zero_landmarks = [0.0] * (len(BARBELL_CURL_KEYPOINTS) * 4)
            all_landmarks.append(zero_landmarks)

    cap.release()
    pose.close()

    if all_landmarks:
        landmarks_array = np.array(all_landmarks)
        print(f" Successfully processed {frame_count} frames")
        print(f"   Pose detected in {successful_detections} frames ({successful_detections/frame_count*100:.1f}%)")
        print(f"   Output shape: {landmarks_array.shape}")
        return landmarks_array
    else:
        print(f" No landmarks detected in video")
        return np.array([])

def visualize_landmarks(image, landmarks, connections=None):
    """Draw landmarks and connections on image for visualization."""

    if landmarks is None:
        return image

    h, w, _ = image.shape

    # Draw landmarks
    for i, (landmark_id) in enumerate(BARBELL_CURL_KEYPOINTS):
        if i * 4 < len(landmarks):
            x = int(landmarks[i * 4] * w)
            y = int(landmarks[i * 4 + 1] * h)
            visibility = landmarks[i * 4 + 3]

            if visibility > 0.5:  # Only draw if visible
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, str(landmark_id), (x + 5, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return image

def create_annotated_video(input_path: str, output_path: str, landmarks_sequence: np.ndarray):
    """Create annotated video with pose landmarks overlaid."""

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open input video {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    print(f"🎥 Creating annotated video: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add landmark overlay if available
        if frame_idx < len(landmarks_sequence):
            frame = visualize_landmarks(frame, landmarks_sequence[frame_idx])

        # Add frame number and info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"✅ Annotated video saved: {output_path}")

def extract_curl_specific_features(landmarks_sequence: np.ndarray) -> dict:
    """Extract barbell curl specific features from landmark sequence."""

    if landmarks_sequence.size == 0:
        return {}

    features = {
        'total_frames': len(landmarks_sequence),
        'curl_phases': [],
        'range_of_motion': {},
        'symmetry_scores': [],
        'stability_scores': []
    }

    # Analyze each frame
    elbow_angles_left = []
    elbow_angles_right = []

    for frame_landmarks in landmarks_sequence:
        if np.any(frame_landmarks):  # Check if frame has data
            from biomechanics import get_angles_from_frame
            angles = get_angles_from_frame(frame_landmarks)

            if angles:
                left_elbow = angles.get('left_elbow', 90)
                right_elbow = angles.get('right_elbow', 90)

                elbow_angles_left.append(left_elbow)
                elbow_angles_right.append(right_elbow)

                # Calculate symmetry score for this frame
                symmetry = 100 - abs(left_elbow - right_elbow) * 2
                features['symmetry_scores'].append(max(0, symmetry))

    # Calculate range of motion
    if elbow_angles_left and elbow_angles_right:
        features['range_of_motion'] = {
            'left_elbow': {
                'min': min(elbow_angles_left),
                'max': max(elbow_angles_left),
                'range': max(elbow_angles_left) - min(elbow_angles_left)
            },
            'right_elbow': {
                'min': min(elbow_angles_right),
                'max': max(elbow_angles_right),
                'range': max(elbow_angles_right) - min(elbow_angles_right)
            }
        }

        # Average range of motion
        avg_range = (features['range_of_motion']['left_elbow']['range'] + 
                    features['range_of_motion']['right_elbow']['range']) / 2
        features['average_range_of_motion'] = avg_range

    # Calculate average scores
    if features['symmetry_scores']:
        features['average_symmetry'] = np.mean(features['symmetry_scores'])

    return features

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Barbell Curl Perception System...")

    # Test file path (replace with your actual video)
    test_video_path = "test_curl_video.mp4"

    if not os.path.exists(test_video_path):
        print(f" Test video not found: {test_video_path}")
        print("Please provide a barbell curl video to test the system")
    else:
        # Extract landmarks
        landmarks = extract_landmarks(test_video_path)

        if landmarks.size > 0:
            print(" Landmark extraction successful!")

            # Extract curl-specific features
            features = extract_curl_specific_features(landmarks)
            print(f" Extracted features: {features}")

            # Create annotated video
            output_path = "annotated_" + os.path.basename(test_video_path)
            create_annotated_video(test_video_path, output_path, landmarks)

        else:
            print(" No landmarks detected in test video")

    print(" Perception system ready for barbell curl analysis!")
