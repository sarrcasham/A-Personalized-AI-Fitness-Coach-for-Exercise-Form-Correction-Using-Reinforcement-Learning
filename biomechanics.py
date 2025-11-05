import numpy as np

def get_angles_from_frame(landmarks_flat):
    try:
        if landmarks_flat is None or len(landmarks_flat) < 52:
            return {}
        
        landmarks = landmarks_flat.reshape(13, 4)
        
        angles = {}
        
        left_shoulder = landmarks[1]
        right_shoulder = landmarks[2]
        left_elbow = landmarks[3]
        right_elbow = landmarks[4]
        left_wrist = landmarks[5]
        right_wrist = landmarks[6]
        left_hip = landmarks[7]
        right_hip = landmarks[8]
        left_knee = landmarks[9]
        right_knee = landmarks[10]
        
        left_elbow_angle = calculate_angle(
            left_shoulder[:3], left_elbow[:3], left_wrist[:3]
        )
        right_elbow_angle = calculate_angle(
            right_shoulder[:3], right_elbow[:3], right_wrist[:3]
        )
        
        angles['left_elbow'] = left_elbow_angle
        angles['right_elbow'] = right_elbow_angle
        
        left_shoulder_angle = calculate_angle(
            left_hip[:3], left_shoulder[:3], left_elbow[:3]
        )
        right_shoulder_angle = calculate_angle(
            right_hip[:3], right_shoulder[:3], right_elbow[:3]
        )
        
        angles['left_shoulder'] = left_shoulder_angle
        angles['right_shoulder'] = right_shoulder_angle
        
        left_hip_pos = left_hip[:3]
        right_hip_pos = right_hip[:3]
        mid_hip = (left_hip_pos + right_hip_pos) / 2
        
        left_knee_pos = left_knee[:3]
        core_sway = calculate_angle(
            left_hip_pos, mid_hip, left_knee_pos
        )
        
        angles['core_stability'] = abs(90 - core_sway)
        
        left_wrist_angle = calculate_angle(
            left_elbow[:3], left_wrist[:3], np.array([left_wrist[0] + 0.1, left_wrist[1], left_wrist[2]])
        )
        right_wrist_angle = calculate_angle(
            right_elbow[:3], right_wrist[:3], np.array([right_wrist[0] + 0.1, right_wrist[1], right_wrist[2]])
        )
        
        angles['left_wrist'] = left_wrist_angle
        angles['right_wrist'] = right_wrist_angle
        
        return angles
        
    except Exception as e:
        return {}

def calculate_angle(point_a, point_b, point_c):
    try:
        point_a = np.array(point_a)
        point_b = np.array(point_b)
        point_c = np.array(point_c)
        
        ba = point_a - point_b
        bc = point_c - point_b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        angle_deg = np.degrees(angle)
        
        return angle_deg
        
    except Exception as e:
        return 0.0

KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 0, 7, 8]
