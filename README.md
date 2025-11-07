# A-Personalized-AI-Fitness-Coach-for-Exercise-Form-Correction-Using-Reinforcement-Learning
This project introduces an AI Fitness Coach that learns an optimal and personalized coaching policy through Reinforcement Learning (RL). By analyzing a user's exercise video and biometric data (age, weight, height, gender)

---

## 📁 Repository Structure

### `app.py`
- The main **Streamlit application** file.  
- Provides the user interface for uploading exercise videos and entering biometric data (age, height, weight, gender).  
- Runs the full inference pipeline:
  - Extracts pose landmarks using `perception.py`.
  - Computes biomechanical angles with `biomechanics.py`.
  - Loads the trained PPO agent from `train_rl_agent.py`.
  - Generates real-time, LLM-based textual feedback through the **Groq API**.
  - Displays annotated video results and a personalized 3-week training plan.

---

### `perception.py`
- Implements the **perception pipeline** using **MediaPipe Pose**.  
- Extracts **33 3D skeletal keypoints** (x, y, z, visibility) from each frame in uploaded videos.  
- Handles video preprocessing, frame-by-frame landmark serialization, and saving as `.npy` files for RL training.  
- Serves as the foundational data extraction module for the entire system.

---

### `biomechanics.py`
- Calculates **biomechanical joint angles** (elbow, shoulder, wrist, and core) using the landmarks from `perception.py`.  
- Compares user form with the **ideal form benchmark** (defined in `ideal_form_benchmark.json`).  
- Generates quantitative error metrics used by the reward function in `fitness_env.py`.  
- Provides biomechanical analysis reports for user feedback.

---

### `fitness_env.py`
- Defines the **custom Gymnasium environment** (`FitnessEnv`) used for reinforcement learning.  
- Models the exercise as a **Markov Decision Process (MDP)**:
  - **State (s):** 52 landmark values + 4 biometric parameters.  
  - **Actions (A):** 5 discrete coaching cues (no feedback, fix elbow, fix shoulder, control sway, praise).  
  - **Reward (R):** Calculated from the reduction in form error (RMSE between current and ideal joint angles).  
- Enables offline training of the RL agent on pre-processed video data.

---

### `train_rl_agent.py`
- Trains the **Actor–Critic PPO agent** using the Stable Baselines3 library.  
- Iterates through all preprocessed `.npy` landmark files and fits the model on each “episode” (video).  
- Saves the trained model (`ppo_bicep_curl_coach.zip`) for deployment in `app.py`.  
- Tracks training progress through cumulative reward and performance metrics.

---

### `process_training_data/`
- A preprocessing utility directory containing scripts for handling raw video data.  
- Includes:
  - Dataset organization and cleaning scripts.  
  - Functions to convert raw `.mp4` exercise videos into structured landmark arrays.  
  - Generation of the `ideal_form_benchmark.json` reference file.  
- Outputs ready-to-train datasets used by the RL environment.

---

## 🧠 Key Technologies
- **MediaPipe Pose** – Pose estimation and skeletal keypoint extraction  
- **PyTorch & Stable Baselines3** – PPO-based RL agent  
- **Gymnasium** – Custom RL environment  
- **OpenCV & NumPy** – Video and data handling  
- **Streamlit** – Frontend for real-time interaction  
- **Groq API (LLM)** – Converts RL actions into natural language coaching feedback  

---

## 🧩 Usage Overview
1. Preprocess videos with `process_training_data/` and `perception.py`.  
2. Compute joint angles using `biomechanics.py`.  
3. Train the PPO model with `train_rl_agent.py`.  
4. Launch the app using:
   ```bash
   streamlit run app.py
   ```
5. Upload a workout video → receive annotated output, coaching feedback, and training recommendations.
