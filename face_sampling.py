import threading
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from facelib import FaceDetector, EmotionDetector
import queue

def get_args():
    parser = argparse.ArgumentParser(description='Face and Emotion Detection in Video')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    return parser.parse_args()

def worker(q, output_path):
    """Worker function to process videos from the queue with a progress bar."""
    while True:
        video_path = q.get()
        if video_path is None:
            break
        extract_frames(video_path, output_path)
        q.task_done()

def extract_frames(video_path, output_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_path, f"{video_name}-Sample")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    num_minutes = int(duration // 60) + 1

    face_detector = FaceDetector(name='resnet')

    print(f"Processing {video_path}: {duration:.2f} seconds, {fps:.2f} FPS")

    # Initialize progress bar for video processing
    progress_bar = tqdm(total=num_minutes, desc=f"Processing {video_name}", position=0, leave=True)

    for minute in range(num_minutes):
        frame_idx = int(minute * 60 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at minute {minute} in {video_name}.")
            continue

        faces, boxes, scores, landmarks = face_detector.detect_align(frame)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord.item()) for coord in box]
            face_crop = frame[y1:y2, x1:x2]

            face_filename = os.path.join(output_folder, f"face_frame_{frame_idx}_det_{i}.jpg")
            cv2.imwrite(face_filename, face_crop)

        progress_bar.update(1)  # Update progress bar after each minute is processed

    cap.release()
    progress_bar.close()
    print(f"Processing complete for {video_name}!")

def main():
    args = get_args()
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path

    video_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.endswith('.mp4')]

    q = queue.Queue()
    threads = []
    num_threads = 10

    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(q, OUTPUT_PATH))
        t.start()
        threads.append(t)

    for video_file in video_files:
        q.put(video_file)

    q.join()

    for _ in range(num_threads):
        q.put(None)

    for t in threads:
        t.join()

    print("Processing complete!")

if __name__ == "__main__":
    main()
