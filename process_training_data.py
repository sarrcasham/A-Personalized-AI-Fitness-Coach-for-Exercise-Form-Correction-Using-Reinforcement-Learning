# process_training_data.py - Fixed version for barbell curl exercise

import os
import numpy as np
from perception import extract_landmarks # We reuse the function you already have

RAW_VIDEOS_DIR = "data/raw_videos/training_form"
PROCESSED_DATA_DIR = "data/processed_data/training_sequences"

def main():
    # Ensure the output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # FIX: Properly get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"Error: Directory {RAW_VIDEOS_DIR} does not exist")
        print(f"Please create the directory and add your barbell curl training videos")
        return

    video_files = [f for f in os.listdir(RAW_VIDEOS_DIR) 
                   if any(f.lower().endswith(ext) for ext in video_extensions)]

    if not video_files:
        print(f"Error: No training videos found in {RAW_VIDEOS_DIR}")
        print(f"Looking for files with extensions: {video_extensions}")
        return

    print(f"Found {len(video_files)} training videos to process.")

    for video_file in video_files:
        video_path = os.path.join(RAW_VIDEOS_DIR, video_file)
        print(f"Processing {video_path}...")

        try:
            landmarks = extract_landmarks(video_path)

            if landmarks.size > 0:
                # Save the processed landmark sequence as a NumPy file
                output_filename = os.path.splitext(video_file)[0] + ".npy"  # FIX: Added [0] for proper filename
                output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)

                np.save(output_path, landmarks)
                print(f" Saved processed landmark sequence to {output_path}")
                print(f"   Shape: {landmarks.shape}")

            else:
                print(f"⚠️ Warning: No landmarks detected in {video_path}")

        except Exception as e:
            print(f" Error processing {video_path}: {e}")

    print(f"\n Processing completed! Check {PROCESSED_DATA_DIR} for output files.")

if __name__ == "__main__":
    main()
