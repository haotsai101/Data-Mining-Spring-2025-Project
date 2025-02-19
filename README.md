# Data-Mining-Spring-2025-Project

# Face and Emotion Detection from Video

## Overview
This project processes a video file to detect faces and recognize emotions in real-time. It utilizes OpenCV for video handling and the `facelib` library for face and emotion detection.

## Features
- Detects faces in a given video.
- Recognizes emotions (e.g., happy, sad, angry, etc.).
- Displays the processed frames with bounding boxes and emotion labels in a Jupyter Notebook.
- Processes a specified portion of the video (from a given timestamp).

## Dependencies
Make sure you have the facelib dependency installed:
```bash
pip install git+https://github.com/sajjjadayobi/FaceLib.git
```

## Usage
### 1. Set Video Path
Modify the `VIDEO_PATH` variable to point to your input video file.

```python
VIDEO_PATH = 'data/input.mp4'
```

### 2. Initialize Face and Emotion Detectors
```python
face_detector = FaceDetector()
emotion_detector = EmotionDetector()
```

### 3. Process Video
- The script starts processing from `offset_min` (default: 10 minutes).
- It processes the next 5 minutes of frames.
- It detects faces and overlays emotion labels.
- The processed frames are displayed inline in a Jupyter Notebook.

## Configuration
- Change `offset_min` to start processing from a different timestamp.
- Modify `frames_to_process` to adjust how many frames are analyzed.
- Adjust `cv2.putText()` settings for different font sizes and colors.

## Output
- Bounding boxes and emotion labels are drawn on detected faces.
- Processed frames are displayed in real-time.
- Prints logs of processing progress.

## Notes
- Ensure that `facelib` is correctly installed and supports your system.
- For large videos, processing may take some time.
- To save the processed video, modify the script to write frames to an output file using `cv2.VideoWriter()`.
