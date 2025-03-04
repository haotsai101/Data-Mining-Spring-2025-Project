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
    parser.add_argument('--step_size', type=str, required=True, help='Step size for frame extraction')
    return parser.parse_args()

def worker(q, output_path, step_size, progress_bar):
    """Worker function to process videos from the queue."""
    while True:
        video_path = q.get()
        if video_path is None:
            break
        extract_frames(video_path, output_path, step_size)
        q.task_done()
        progress_bar.update(1)  # Update progress bar when a video is processed

def extract_frames(video_path, output_path, step_size):
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

    face_detector = FaceDetector(name='resnet')

    print(f"Processing {video_path}: {duration:.2f} seconds, {fps:.2f} FPS")

    for step in range(int(duration // step_size) + 1):
        frame_idx = int(step * step_size * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        faces, boxes, scores, landmarks = face_detector.detect_align(frame)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord.item()) for coord in box]

            # Crop the face region
            face_crop = frame[y1:y2, x1:x2]

            # Construct filename and save
            face_filename = os.path.join(
                output_folder,
                f"face_frame_{frame_idx}_det_{i}.jpg"
            )
            cv2.imwrite(face_filename, face_crop)

    cap.release()
    print(f"Processing complete for {video_name}!")

def main():
    args = get_args()
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    STEP_SIZE = args.step_size

    video_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.endswith('.mp4')]

    q = queue.Queue()
    threads = []
    num_threads = 10

    # Initialize the progress bar
    with tqdm(total=len(video_files), desc="Processing Videos", unit="file") as progress_bar:
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(q, OUTPUT_PATH, int(STEP_SIZE), progress_bar))
            t.start()
            threads.append(t)

        for video_file in video_files:
            q.put(video_file)

        q.join()

        for _ in range(num_threads):
            q.put(None)

        for t in threads:
            t.join()

    print("All videos processed successfully!")

if __name__ == "__main__":
    main()
