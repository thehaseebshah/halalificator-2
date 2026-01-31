import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import urllib.request
import sys

# Model Configurations
FACE_MODEL_URL = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
FACE_MODEL_NAME = "yolov8n-face.pt"

GENDER_PROTO_URL = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_deploy.prototxt"
GENDER_MODEL_URL = "https://huggingface.co/AjaySharma/genderDetection/resolve/main/gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)
    else:
        print(f"Found {filename}.")

def setup_models():
    download_file(FACE_MODEL_URL, FACE_MODEL_NAME)
    download_file(GENDER_PROTO_URL, GENDER_PROTO)
    download_file(GENDER_MODEL_URL, GENDER_MODEL)

def blur_region(image, box):
    """Applies a Gaussian blur to the specified bounding box region."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0: return image
    
    # Adaptive kernel size
    ksize = int(max(roi.shape[:2]) // 5) | 1 # Odd number
    blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def process_video(input_path, output_path, conf_threshold=0.4):
    setup_models()
    
    print("Loading models...")
    try:
        face_model = YOLO(FACE_MODEL_NAME)
        gender_net = cv2.dnn.readNet(GENDER_PROTO, GENDER_MODEL)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    gender_list = ['Male', 'Female']
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    print(f"Processing {input_path} -> {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Detect Faces
        results = face_model(frame, verbose=False, conf=conf_threshold)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get Face Box
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                
                # Crop Face for Gender Config
                face_img = frame[max(0,by1):min(height,by2), max(0,bx1):min(width,bx2)]
                
                if face_img.size > 0:
                    # Preprocess for Caffe Model
                    # Mean values from standard gender_net docs
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    gender_net.setInput(blob)
                    preds = gender_net.forward()
                    
                    gender = gender_list[preds[0].argmax()]
                    conf = preds[0].max()
                    
                    if gender == 'Female':
                        # Blur the face
                        frame = blur_region(frame, (bx1, by1, bx2, by2))
                        # Optional: Draw label for debug? No, user wants blur.
        
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames", end='\r')
            
    cap.release()
    out.release()
    print("\nProcessing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output", default="output_fixed.mp4")
    args = parser.parse_args()
    
    if os.path.exists(args.input_video):
        process_video(args.input_video, args.output)
    else:
        print("Input file not found.")
