import cv2
import torch
import numpy as np
import os

class YOLOv5Detector:
    def __init__(self, model_name='yolov5s'):
        """Initialize YOLOv5 model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.conf = 0.5
        self.model.iou = 0.45
    
    def detect_and_annotate_video(self, input_video_path, output_video_path):
        """Process video with YOLOv5 and save annotated version with bounding boxes"""
        
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # YOLOv5 inference
            results = self.model(frame)
            
            # Get annotated frame from YOLOv5
            annotated_frame = results.render()[0]
            
            # Convert BGR back if needed
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            out.write(annotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return frame_count

def process_video_yolov5(input_path, output_path):
    """Wrapper function to process video with YOLOv5"""
    detector = YOLOv5Detector(model_name='yolov5s')
    frame_count = detector.detect_and_annotate_video(input_path, output_path)
    return frame_count
