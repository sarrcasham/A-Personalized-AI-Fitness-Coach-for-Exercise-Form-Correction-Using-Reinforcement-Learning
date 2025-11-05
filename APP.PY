import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import tempfile
from stable_baselines3 import PPO
from groq import Groq

try:
    from enhanced_feedback import BarbellCurlAnalyzer
    analyzer = BarbellCurlAnalyzer()
except ImportError:
    st.error("enhanced_feedback.py not found in project directory")
    st.stop()

try:
    from biomechanics import get_angles_from_frame
    KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 0, 7, 8]
except ImportError:
    st.error("biomechanics.py not found")
    st.stop()

try:
    from train_rl_agent import normalize_biometrics
except ImportError:
    st.error("train_rl_agent.py not found")
    st.stop()

st.set_page_config(
    page_title="AI Fitness Coach - Barbell Curl",
    page_icon="💪",
    layout="wide"
)

MODEL_PATH = "models/ppo_bicep_curl_coach.zip"

GROQ_MODELS = [
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "llama-3.1-70b-versatile",
]

@st.cache_resource
def load_ppo_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return PPO.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None, None
    try:
        client = Groq(api_key=api_key)
        return client, api_key
    except Exception:
        return None, api_key

model = load_ppo_model()
client, groq_key = get_groq_client()

def create_annotated_video_with_bounding_boxes(input_path, output_path, detected_issues):
    """Create annotated video with MediaPipe pose + bounding boxes for detected issues"""
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Map issues to joint indices for bounding boxes
    issue_to_joints = {
        'Elbow Range Issue': [13, 14],
        'Shoulder Instability': [11, 12],
        'Body Sway / Excess Momentum': [23, 24, 25, 26],
        'Arm Asymmetry': [13, 14, 15, 16],
    }
    
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            h, w, c = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Draw pose skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Draw bounding boxes for each detected issue
                for issue in detected_issues:
                    issue_title = issue.get('title', '')
                    joints = issue_to_joints.get(issue_title, [])
                    
                    if joints:
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0
                        found_any = False
                        
                        for joint_idx in joints:
                            if joint_idx < len(results.pose_landmarks.landmark):
                                lm = results.pose_landmarks.landmark[joint_idx]
                                
                                if lm.visibility > 0.3:
                                    x = int(lm.x * w)
                                    y = int(lm.y * h)
                                    x_min = min(x_min, x)
                                    x_max = max(x_max, x)
                                    y_min = min(y_min, y)
                                    y_max = max(y_max, y)
                                    found_any = True
                        
                        if found_any and x_max > x_min and y_max > y_min:
                            padding = 30
                            # Draw red bounding box
                            cv2.rectangle(
                                frame,
                                (max(0, x_min - padding), max(0, y_min - padding)),
                                (min(w, x_max + padding), min(h, y_max + padding)),
                                (0, 0, 255),  # Red color
                                4
                            )
                            
                            # Add issue label
                            label = issue_title[:20]
                            cv2.putText(
                                frame,
                                label,
                                (max(5, x_min - padding), max(25, y_min - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )
                
                # Add issue count overlay
                if detected_issues:
                    overlay_text = f"Issues Detected: {len(detected_issues)}"
                    cv2.putText(
                        frame,
                        overlay_text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
            
            out.write(frame)
            frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return frame_count

def extract_angles_from_video(video_path):
    all_angles = []
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks_flat = []
                
                for keypoint_index in KEYPOINTS:
                    if keypoint_index < len(results.pose_landmarks.landmark):
                        lm = results.pose_landmarks.landmark[keypoint_index]
                        landmarks_flat.extend([lm.x, lm.y, lm.z, lm.visibility])
                    else:
                        landmarks_flat.extend([0.0, 0.0, 0.0, 0.0])
                
                try:
                    angles = get_angles_from_frame(np.array(landmarks_flat, dtype=np.float32))
                    if angles:
                        all_angles.append(angles)
                except:
                    pass
    
    cap.release()
    return all_angles

def calculate_average_angles(angles_list):
    if not angles_list:
        return {}
    
    avg_angles = {}
    all_keys = set()
    
    for angles in angles_list:
        all_keys.update(angles.keys())
    
    for key in all_keys:
        values = [angles.get(key, 0) for angles in angles_list if key in angles]
        if values:
            avg_angles[key] = np.mean(values)
    
    return avg_angles

def get_llm_feedback(prompt):
    if client is None or not groq_key:
        return ("Warning: Groq API not configured. To enable AI coaching, set your Groq API key. "
                "Here is your analysis prompt to use with ChatGPT manually: " + prompt)
    
    for model_name in GROQ_MODELS:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an elite strength coach and biomechanics expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_name,
                max_tokens=800,
                temperature=0.7
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            if "decommissioned" in str(e) or "not available" in str(e):
                continue
            else:
                break
    
    return ("Groq models currently unavailable. Please paste this into ChatGPT or Claude manually: " + prompt)

def main():
    if model is None:
        st.error("Trained PPO model not found!")
        st.info("Run: python train_rl_agent_fixed.py first")
        return
    
    st.title("💪 AI Fitness Coach - Barbell Curl Analysis")
    st.markdown("**Detailed biomechanics analysis with pose detection and bounding boxes**")
    
    with st.sidebar:
        st.header("👤 Your Profile")
        
        age = st.slider("Age", 16, 80, 25)
        height = st.slider("Height (cm)", 140, 220, 170)
        weight = st.slider("Weight (kg)", 40, 150, 60)
        
        gender_str = st.radio("Gender", ["Male", "Female"], index=0)
        gender_numeric = 0 if gender_str == "Male" else 1
        
        biometrics = {
            "age": age,
            "height": height,
            "weight": weight,
            "gender": gender_str
        }
        
        biometrics_array = normalize_biometrics(age, height, weight, gender_numeric)
        
        st.markdown("---")
        st.header("📹 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a barbell curl video",
            type=["mp4", "mov", "avi"],
            help="MP4, MOV, AVI. 30 sec - 2 min recommended"
        )
    
    if uploaded_file is not None:
        st.success(f"✅ Video loaded: **{uploaded_file.name}**")
        
        if st.button("🚀 Analyze My Form", type="primary", use_container_width=True):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_input_path = tfile.name
            
            output_video_path = "data/output/annotated_video.mp4"
            os.makedirs("data/output", exist_ok=True)
            
            try:
                # STEP 1: Extract angles for biomechanics analysis
                with st.spinner("📊 Extracting joint angles..."):
                    angles_list = extract_angles_from_video(temp_input_path)
                    avg_angles = calculate_average_angles(angles_list)
                
                # STEP 2: Analyze biomechanics and detect issues
                with st.spinner("🔍 Analyzing biomechanics..."):
                    detected_issues, _ = analyzer.analyze_angles(avg_angles)
                
                # STEP 3: Create annotated video with bounding boxes
                with st.spinner("🎬 Creating annotated video with bounding boxes..."):
                    frame_count = create_annotated_video_with_bounding_boxes(
                        temp_input_path, 
                        output_video_path, 
                        detected_issues
                    )
                    st.info(f"✅ Processed {frame_count} frames with pose detection + bounding boxes")
                
                # Verify video was created
                if os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                    st.success(f"✅ Video saved: {output_video_path} ({file_size:.2f} MB)")
                else:
                    st.error("❌ Video file was not created!")
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # DISPLAY RESULTS
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📽️ Annotated Video")
                    
                    if os.path.exists(output_video_path):
                        try:
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                                st.caption(f"✅ Saved at: {os.path.abspath(output_video_path)}")
                        except Exception as e:
                            st.error(f"❌ Error displaying video: {e}")
                            
                            # Provide download button
                            with open(output_video_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=f.read(),
                                    file_name="annotated_barbell_curl.mp4",
                                    mime="video/mp4"
                                )
                    else:
                        st.error("Video file not found!")
                
                with col2:
                    st.subheader("📊 Joint Angles")
                    if avg_angles:
                        for joint, angle in sorted(avg_angles.items()):
                            st.metric(joint, f"{angle:.1f}°")
                    else:
                        st.warning("No angles detected")
                
                # BIOMECHANICS REPORT
                st.markdown("---")
                st.subheader("📋 Biomechanics Report")
                
                biomech_report = analyzer.generate_comprehensive_report(
                    avg_angles, biometrics, detected_issues
                )
                
                st.markdown(biomech_report)
                
                # LLM COACHING FEEDBACK
                st.markdown("---")
                st.subheader("🤖 Personalized Coach Feedback")
                
                prompt = (
                    f"{biomech_report}\n\n"
                    f"Based on the above biomechanics analysis, provide a detailed 3-week "
                    f"progressive training plan to fix these issues. Include:\n"
                    f"1. Week-by-week progression\n"
                    f"2. Specific weight recommendations\n"
                    f"3. Supplementary exercises\n"
                    f"4. Form cues to focus on each session"
                )
                
                llm_feedback = get_llm_feedback(prompt)
                st.markdown(llm_feedback)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.write(traceback.format_exc())
            
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
    
    else:
        st.header("🏋️‍♀️ Welcome to Advanced Form Analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**How It Works**")
            st.write("1. Enter Profile - Age, height, weight, gender")
            st.write("2. Upload Video - Clear barbell curl video")
            st.write("3. Analyze - AI processes video with MediaPipe")
            st.write("4. Review - Get detailed feedback with annotated video, biomechanics analysis, and 3-week training plan")
        
        with col2:
            st.markdown("**What You Get**")
            st.write("📽️ Annotated Video - Pose detection with RED bounding boxes on problem areas")
            st.write("📊 Detailed Reports - Exact angle measurements, issue severity ratings, step-by-step fixes")
            st.write("🤖 Coaching Plan - Personalized progressions, weekly schedules, form cues and tips")

if __name__ == "__main__":
    main()
