# Halalificator-2 🎥⚖️

**Halalificator-2** is an advanced AI-powered video processing pipeline designed to automate two primary "halalification" tasks: blurring female individuals and removing background music while preserving vocals.

It combines state-of-the-art computer vision (YOLOv8, CLIP) with high-fidelity audio source separation (Demucs) to provide a robust, automated solution for content filtering.

---

## ✨ Key Features

- **🛡️ Robust Gender Detection**: 
    - **CLIP Mode (Recommended)**: Uses OpenAI's CLIP model for person-level analysis, allowing for accurate gender classification even when faces are obscured or at difficult angles.
    - **Caffe Mode**: Traditional face-based detection using Caffe models.
- **🔄 Two-Pass Analysis**: 
    - **Pass 1 (Analysis)**: Tracks individuals across the entire video and aggregates "gender votes" to ensure stable, flicker-free classification.
    - **Pass 2 (Rendering)**: Applies Gaussian blur to individuals locked as female based on Pass 1 data.
- **🎵 AI Music Removal**:
    - Leverages **Demucs (Hybrid Transformer)** to separate audio into stems.
    - Isolates vocals and speech while removing background music and accompaniment.
- **⏱️ Precision Trimming**: Process specific segments of a video using start time and duration offsets.
- **🛠️ Debug & Preview**: 
    - Visualization mode to see bounding boxes and classification confidence.
    - Automatic extraction of random preview frames for quick verification.

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (highly recommended for CLIP and Demucs performance)
- FFmpeg (automatically handled via `static-ffmpeg`)

### Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd halalificator-2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # Or use the pyproject.toml
   ```

3. The models (YOLO, CLIP, Caffe) will be automatically downloaded on the first run.

---

## 📖 Usage Guide

The main entry point is `process_video.py`. It handles the full pipeline from video analysis to audio merging.

### Basic Command
```bash
python process_video.py input_video.mp4 --output outputs/final_result.mp4
```

### Advanced Options
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--mode` | Gender detection engine (`clip` or `caffe`). | `clip` |
| `--duration` | Process only $N$ seconds of the video. | Full Video |
| `--test` | Process only the first 5 seconds. | - |
| `--test-end` | Process only the last 5 seconds. | - |
| `--sensitivity` | Threshold for female classification (lower = more aggressive blurring). | `0.15` |
| `--debug` | Draw bounding boxes instead of blurring. | `False` |
| `--preview` | Extract 10 random frames from output for verification. | `False` |

### Example: Testing a YouTube Segment
```bash
python process_video.py "path/to/downloaded_video.mp4" --duration 30 --output outputs/test_30s.mp4 --mode clip --preview
```

---

## 🧠 The Pipeline (How it Works)

1. **Audio Isolation**: The video is passed to the `process_audio` module where Demucs splits the audio track. The `vocals` stem is extracted and saved as a temporary wav file.
2. **Track Identification**: YOLOv8 detects and tracks "person" objects across frames.
3. **Global Voting**:
   - For every frame an ID is visible, CLIP (or Caffe) predicts the gender.
   - These predictions are aggregated. If a track's "Female Ratio" exceeds the `sensitivity` threshold, that ID is globally flagged as female.
4. **Rendering**: The video is re-processed. Any person with a flagged ID is blurred using an adaptive Gaussian kernel.
5. **Final Merge**: `ffmpeg` combines the blurred video and the isolated vocals into the final MP4 container.

---

## 📦 Dependencies
- `ultralytics` (YOLOv8)
- `open-clip-torch` (Person-level gender analysis)
- `demucs` (Audio stem separation)
- `opencv-python` (Image processing)
- `static-ffmpeg` (Media handling)

---

## 🌐 Web Interface & Public Deployment

Halalificator now includes a modern web interface for easy uploading and real-time processing feedback.

### 1. Start the Web Server
To run the server in production mode (using Gunicorn):
```bash
chmod +x start_server.sh
./start_server.sh
```
The app will be available locally at `http://localhost:5000`.

### 2. Make it Public (Ngrok)
To share the app with others without configuring complex router settings, use the included ngrok script:
1. Get a free auth token from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
2. Start the tunnel:
   ```bash
   python start_tunnel.py YOUR_NGROK_AUTH_TOKEN
   ```
3. Copy the `https://...` link provided in the console.

---

## ⚠️ Disclaimer
This tool is intended for personal and educational use in automating content preferences. Accuracy of AI models is not 100%; always use the `--preview` or `--debug` modes to verify results before final deployment.
