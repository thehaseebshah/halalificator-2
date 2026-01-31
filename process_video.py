
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import urllib.request
import sys
import random
import shutil
import subprocess
from process_audio import remove_music
from static_ffmpeg import add_paths
try:
    import torch
    import open_clip
    from PIL import Image
except ImportError:
    torch = None
    open_clip = None

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

def setup_models(mode='caffe'):
    download_file(FACE_MODEL_URL, FACE_MODEL_NAME)
    if mode == 'caffe':
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
    if ksize <= 1: return image

    blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def draw_debug(image, face_box, person_box, gender, conf, is_tracked_female=False, track_id=None):
    """Draws debug bounding boxes and labels."""
    # Draw Face if present
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        color = (255, 0, 0) if gender == 'Male' else (255, 0, 255) 
        cv2.rectangle(image, (fx1, fy1), (fx2, fy2), color, 2)
        label = f"{gender} {conf:.2f}"
        cv2.putText(image, label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw Person
    if person_box is not None:
        px1, py1, px2, py2 = person_box
        p_color = (0, 0, 255) if is_tracked_female else (0, 255, 0) # Red if flagged as female
        cv2.rectangle(image, (px1, py1), (px2, py2), p_color, 2)
        
        id_label = f"ID:{track_id}" if track_id is not None else "ID:?"
        if is_tracked_female:
            cv2.putText(image, f"{id_label} FEMALE", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
        else:
            cv2.putText(image, id_label, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

class GlobalTrackManager:
    def __init__(self, sensitivity=0.15):
        self.track_gender_votes = {} # id -> {'Male': count, 'Female': count}
        self.final_decisions = {} # id -> 'Male' or 'Female'
        self.sensitivity = sensitivity

    def add_vote(self, track_id, gender):
        if track_id is None: return
        if track_id not in self.track_gender_votes:
            self.track_gender_votes[track_id] = {'Male': 0, 'Female': 0}
        self.track_gender_votes[track_id][gender] += 1

    def finalize(self):
        print("\nFinalizing Gender Tracks:")
        for tid, votes in self.track_gender_votes.items():
            # If significant female votes detected, mark as female
            total = votes['Male'] + votes['Female']
            if total == 0: continue
            
            female_ratio = votes['Female'] / total
            # Heuristic: If > sensitivity ratio of valid face detections say Female, treat as Female
            # Lowering this makes the system more "halal" (aggressive blurring).
            if female_ratio >= self.sensitivity:
                self.final_decisions[tid] = 'Female'
            else:
                self.final_decisions[tid] = 'Male'
            print(f"Track {tid}: {votes} -> {self.final_decisions[tid]} ({female_ratio:.2f} >= {self.sensitivity})")

def process_video(input_path, output_path, conf_threshold=0.25, start_seconds=0, duration_seconds=None, debug=False, mode='clip', **kwargs):
    setup_models(mode=mode)
    
    print(f"Loading models (mode={mode})...")
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    try:
        face_model = YOLO(FACE_MODEL_NAME)
        person_model = YOLO("yolov8n.pt") 
        
        gender_net = None
        clip_model = None
        clip_preprocess = None
        clip_tokenizer = None
        
        if mode == 'caffe':
            gender_net = cv2.dnn.readNet(GENDER_PROTO, GENDER_MODEL)
        elif mode == 'clip':
            if open_clip is None:
                raise ImportError("open_clip not installed.")
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
            clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    gender_list = ['Male', 'Female']
    global_tracker = GlobalTrackManager(sensitivity=kwargs.get('sensitivity', 0.15))
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    if start_seconds < 0: start_seconds = max(0, video_duration + start_seconds)
    start_frame = int(start_seconds * fps)
    
    max_frames = int(duration_seconds * fps) if duration_seconds else (total_frames - start_frame)
    if start_frame + max_frames > total_frames: max_frames = total_frames - start_frame
    
    print(f"Processing range: {start_seconds:.2f}s to {start_seconds + (max_frames/fps):.2f}s ({max_frames} frames)")
    
    # --- PASS 1: Analysis ---
    print(f"\n--- PASS 1: Analyzing Tracks ---")
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        if max_frames and frame_count >= max_frames: break
        ret, frame = cap.read()
        if not ret: break
        
        # Track
        results = person_model.track(frame, verbose=False, classes=[0], persist=True, conf=conf_threshold)
        
        current_boxes = []
        current_ids = []
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, ids):
                current_boxes.append(box)
                current_ids.append(track_id)
                
            if mode == 'clip':
                # --- CLIP PERSON CLASSIFICATION ---
                text_prompts = clip_tokenizer(["a photo of a man", "a photo of a woman"]).to(device)
                
                for i, (px1, py1, px2, py2) in enumerate(current_boxes):
                    # Crop person
                    person_img = frame[max(0,py1):min(height,py2), max(0,px1):min(width,px2)]
                    if person_img.size > 0:
                        # Convert to PIL for CLIP
                        person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                        image_input = clip_preprocess(person_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image_input)
                            text_features = clip_model.encode_text(text_prompts)
                            
                            logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            probs = logits_per_image.cpu().numpy()[0]
                            
                            gender = 'Male' if probs[0] > probs[1] else 'Female'
                            global_tracker.add_vote(current_ids[i], gender)
            else:
                # --- FACE-BASED DETECTION (Caffe) ---
                face_results = face_model(frame, verbose=False, conf=conf_threshold)
                for result in face_results:
                    for box in result.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        face_cx, face_cy = (bx1+bx2)/2, (by1+by2)/2
                        
                        face_img = frame[max(0,by1):min(height,by2), max(0,bx1):min(width,bx2)]
                        if face_img.size > 0:
                            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                            gender_net.setInput(blob)
                            preds = gender_net.forward()
                            gender = gender_list[preds[0].argmax()]
                            
                            for i, (px1, py1, px2, py2) in enumerate(current_boxes):
                                h_margin = (py2 - py1) * 0.2
                                if px1 <= face_cx <= px2 and (py1 - h_margin) <= face_cy <= py2:
                                    global_tracker.add_vote(current_ids[i], gender)
                                    break
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Analyzing {frame_count}/{max_frames}...", end='\r')

    global_tracker.finalize()

    # --- PASS 2: Rendering ---
    print(f"\n--- PASS 2: Rendering Output ---")
    
    # We need to RESET tracking for the second pass to ensure IDs match up 
    # (YOLO tracking is deterministic if inputs are identical)
    # However, to be safe, standard practice for YOLO track is difficult in 2-pass 
    # unless we saved the tracks. 
    # Optimally, we re-run tracking exactly as before.
    
    # Reset Model internal state for tracking strictly? 
    # YOLO(predictor) resets on new stream usually.
    person_model = YOLO("yolov8n.pt") # Re-init to clear internal tracker state
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    while True:
        if max_frames and frame_count >= max_frames: break
        ret, frame = cap.read()
        if not ret: break
        
        # Track (Must be identical call to Pass 1)
        results = person_model.track(frame, verbose=False, classes=[0], persist=True, conf=conf_threshold)
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                # Lookup decision
                gender_decision = global_tracker.final_decisions.get(track_id, 'Male') # Default to male if unknown
                
                if gender_decision == 'Female':
                    if debug:
                        draw_debug(frame, None, box, None, 1.0, True, track_id=track_id)
                    else:
                        frame = blur_region(frame, box)
                elif debug:
                     draw_debug(frame, None, box, None, 0.0, False, track_id=track_id)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Rendering {frame_count}/{max_frames}...", end='\r')
            
    cap.release()
    out.release()
    print("\nVideo Rendering Complete.")

def combine_video_audio(video_path, audio_path, output_path):
    """Combines video from one file and audio from another using ffmpeg."""
    print(f"Combining blurred video and isolated vocals into {output_path}...")
    add_paths() # Ensure static-ffmpeg is in PATH
    
    # Use ffmpeg to merge
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",    # Stream copy video (no re-encoding)
        "-c:a", "aac",     # Encode audio to AAC
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",       # Match the shortest stream (usually video if it was trimmed)
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("Final merge complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error merging video and audio: {e.stderr.decode()}")
        # If merge fails, at least we have the blurred video
        shutil.move(video_path, output_path)
        print(f"Merge failed, but blurred video saved to {output_path}")

def halalify(input_path, output_path, audio_path=None, **kwargs):
    """Full pipeline: Blur females AND remove music."""
    temp_blurred = "temp_blurred_video.mp4"
    temp_vocals = "temp_vocals.wav"
    
    # 1. Process Video
    process_video(input_path, temp_blurred, 
                  conf_threshold=kwargs.get('conf', 0.25),
                  start_seconds=kwargs.get('start', 0),
                  duration_seconds=kwargs.get('duration'),
                  debug=kwargs.get('debug', False),
                  sensitivity=kwargs.get('sensitivity', 0.15),
                  mode=kwargs.get('mode', 'clip'))
    
    # 2. Process Audio (Remove Music)
    print("\n--- Audio Processing: Removing Music ---")
    try:
        audio_source = audio_path if audio_path else input_path
        print(f"Extracting vocals from: {audio_source}")
        remove_music(audio_source, output_path=temp_vocals)
        
        # 3. Combine
        combine_video_audio(temp_blurred, temp_vocals, output_path)
    except Exception as e:
        print(f"Audio processing failed: {e}")
        print("Falling back to blurred video only (no audio or original audio from video file).")
        # If input_path had some audio, it might have been lost in cv2 processing,
        # so we really want that merge to work.
        shutil.move(temp_blurred, output_path)
    
    # Cleanup
    if os.path.exists(temp_blurred): os.remove(temp_blurred)
    if os.path.exists(temp_vocals): os.remove(temp_vocals)
    print(f"\nHalalification complete: {output_path}")


def extract_random_frames(video_path, output_dir="outputs/preview_frames", num_frames=10):
    """Extracts random frames from the video for preview."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path} for preview extraction.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Could not determine total frames for preview.")
        return

    frame_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))
    
    print(f"Extracting {len(frame_indices)} random frames to '{output_dir}'...")
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(output_dir, f"frame_{idx}.jpg")
            cv2.imwrite(frame_name, frame)
            
    cap.release()
    print("Preview extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output", default="outputs/output_fixed.mp4")
    parser.add_argument("--test", action="store_true", help="Process only the first 5 seconds for testing")
    parser.add_argument("--test-end", action="store_true", help="Process only the last 5 seconds for testing")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process in seconds")
    parser.add_argument("--preview", action="store_true", help="Extract 10 random frames from the output for verification")
    parser.add_argument("--debug", action="store_true", help="Draw bounding boxes instead of blurring for debugging")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--sensitivity", type=float, default=0.15, help="Gender sensitivity (lower = more blurring, default 0.15)")
    parser.add_argument("--mode", choices=['caffe', 'clip'], default='clip', help="Gender detection engine")
    parser.add_argument("--audio", help="Optional separate audio file to use (e.g. m4a from youtube)")
    args = parser.parse_args()
    
    if os.path.exists(args.input_video):
        start = 0
        duration = args.duration
        
        if args.test:
            duration = 5
        elif args.test_end:
            start = -5 # Negative value handled in process_video
            duration = 5
            
        halalify(args.input_video, args.output, 
                 audio_path=args.audio,
                 conf=args.conf, 
                 sensitivity=args.sensitivity,
                 mode=args.mode,
                 start=start, 
                 duration=duration, 
                 debug=args.debug)
        
        if args.preview:
            extract_random_frames(args.output)
    else:
        print("Input file not found.")
