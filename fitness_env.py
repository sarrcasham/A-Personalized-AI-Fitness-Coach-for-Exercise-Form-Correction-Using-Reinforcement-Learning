# In fitness_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

# This will now import the *corrected* version of your function
from biomechanics import get_angles_from_frame

class FitnessEnv(gym.Env):
    """
    A custom offline Reinforcement Learning environment for fitness coaching.
    """
    
    def __init__(self, landmark_sequence: np.ndarray, user_biometrics: np.ndarray):
        super(FitnessEnv, self).__init__()
        
        self.landmark_data = landmark_sequence
        self.user_biometrics = user_biometrics
        self.current_frame_index = 0
        
        # Load the ideal form benchmark
        try:
            with open("data/processed_data/ideal_form_benchmark.json", "r") as f:
                self.ideal_angles = json.load(f)
        except FileNotFoundError:
            print("FATAL ERROR: ideal_form_benchmark.json not found. Run process_ideal_form.py first.")
            self.ideal_angles = {'bottom_of_press': {}, 'top_of_press': {}}
            
        # --- DEFINE ACTION SPACE ---
        # 0: No Feedback, 1: Correct Elbows, 2: Correct Shoulders, 3: Control Body Sway, 4: Praise
        self.action_space = spaces.Discrete(5)
        
        # --- DEFINE OBSERVATION SPACE ---
        # Landmark data shape: 13 keypoints * 4 values = 52
        # Biometric data shape: 4 (age, height, weight, gender)
        observation_shape = 52 + 4 # Total shape is 56
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(observation_shape,), 
                                            dtype=np.float32)
        
        # Used for reward calculation
        self.last_error = 0.0

    def _get_state(self) -> np.ndarray:
        """Constructs the state vector for the current frame."""
        current_landmarks = self.landmark_data[self.current_frame_index]
        state = np.concatenate([current_landmarks, self.user_biometrics]).astype(np.float32)
        return state

    def _calculate_form_error(self, frame_landmarks_52_values: np.ndarray) -> float:
        """Calculates the deviation from the ideal form benchmark."""
        
        # Pass the 52-value array directly to the new get_angles_from_frame
        current_angles = get_angles_from_frame(frame_landmarks_52_values)
        
        if not current_angles: # If angle calculation failed
            return self.last_error # Return the last known error to avoid bad rewards

        # Determine if the arm is in the 'up' or 'down' phase
        avg_elbow_angle = (current_angles.get('left_elbow', 180) + current_angles.get('right_elbow', 180)) / 2
        
        # Compare to the closer of the two ideal positions
        if avg_elbow_angle < 120: # Arbitrary threshold for "up" phase
            ideal_pose = self.ideal_angles.get('bottom_of_press', {}) # Corresponds to 'top of curl'
        else:
            ideal_pose = self.ideal_angles.get('top_of_press', {}) # Corresponds to 'bottom of curl'
        
        error = 0.0
        joint_count = 0
        for joint, ideal_angle in ideal_pose.items():
            if joint in current_angles:
                error += (current_angles[joint] - ideal_angle) ** 2
                joint_count += 1
        
        if joint_count == 0:
            return self.last_error # No valid joints found for comparison

        final_error = np.sqrt(error / joint_count) # Return Root Mean Squared Error
        self.last_error = final_error
        return final_error

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_frame_index = 0
        
        # Calculate initial error
        self.last_error = self._calculate_form_error(self.landmark_data[self.current_frame_index])
        
        initial_observation = self._get_state()
        info = {}
        return initial_observation, info

    def step(self, action):
        # old_error was calculated in the previous step (or in reset)
        old_error = self.last_error
        
        # Move to the next frame in the video
        self.current_frame_index += 1
        
        # Check if the episode (video) is over
        done = self.current_frame_index >= len(self.landmark_data) - 1
        
        if not done:
            new_error = self._calculate_form_error(self.landmark_data[self.current_frame_index])
            # REWARD: The reward is the *improvement* in form.
            reward = old_error - new_error
        else:
            reward = 0
            
        # Get the next state
        observation = self._get_state()
        
        terminated = done
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info