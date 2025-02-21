import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from facelib import FaceDetector, EmotionDetector

def get_args():
  parser = argparse.ArgumentParser(description='Face and Emotion Detection in Video')
  parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
  parser.add_argument('--output_dest', type=str, default='.', help='Directory to save detected face images')
  return parser.parse_args()

def main():
  args = get_args()
  VIDEO_PATH = args.video_path
  OUTPUT_DIR = args.output_dest

  # Ensure output directory exists
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  face_detector = FaceDetector()
  emotion_detector = EmotionDetector()

  cap = cv2.VideoCapture(VIDEO_PATH)
  if not cap.isOpened():
      print(f"Error: Could not open video {VIDEO_PATH}")
      exit()

  # Frames per second
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames_per_minute = int(60 * fps)

  # Start processing at 10 minutes in
  offset_min = 10
  offset_in_frames = int(offset_min * frames_per_minute)
  cap.set(cv2.CAP_PROP_POS_FRAMES, offset_in_frames)

  # Process the next 5 minutes of video
  frames_to_process = 20 * frames_per_minute
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  upper_frame_limit = min(offset_in_frames + frames_to_process, total_frames)

  print(f"Starting from {offset_min} min (frame {offset_in_frames}), processing up to frame {upper_frame_limit}...")

  for frame_idx in tqdm(range(offset_in_frames, upper_frame_limit), desc='Processing Frames'):
    ret, frame = cap.read()
    if not ret:
      print("Error: Could not read frame.")
      break

    # Save all faces once every minute of processed time
    # (i.e., if it's an exact multiple of 'frames_per_minute' from the offset)
    if (frame_idx - offset_in_frames) % frames_per_minute == 0:
      # Detect faces
      faces, boxes, scores, landmarks = face_detector.detect_align(frame)
      for i, box in enumerate(boxes):
          # box should be [x1, y1, x2, y2]
          x1, y1, x2, y2 = [int(coord.item()) for coord in box]

          # Crop the face region
          face_crop = frame[y1:y2, x1:x2]

          # Construct filename and save
          face_filename = os.path.join(
              OUTPUT_DIR,
              f"face_frame_{frame_idx}_det_{i}.jpg"
          )
          cv2.imwrite(face_filename, face_crop)

  cap.release()
  print("Processing complete!")

if __name__ == "__main__":
  main()
