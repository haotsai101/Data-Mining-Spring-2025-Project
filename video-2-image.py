import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Process video to images.')
    parser.add_argument('--filename', type=str, required=True, help='MP4 file in /data folder')
    parser.add_argument('--resolution', type=int, default=0, help='Target height to resize; 0 means no resize')
    return parser.parse_args()

def main():
    args = parse_args()
    source = os.path.join('data', args.filename)
    dest_folder = 'data/frames'

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    cap = cv2.VideoCapture(source)
    path_to_save = os.path.abspath(dest_folder)
    
    if not cap.isOpened():
        print('Error: Unable to open video file.')
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 1
    
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if args.resolution > 0:
                    h, w = frame.shape[:2]
                    aspect_ratio = w / h
                    target_w = int(aspect_ratio * args.resolution)
                    frame = cv2.resize(frame, (target_w, args.resolution), interpolation=cv2.INTER_AREA)

                name = f'frame{current_frame}.jpg'
                cv2.imwrite(os.path.join(path_to_save, name), frame)
                current_frame += 1
                pbar.update(1)
            else:
                break

    cap.release()
    print('Processing complete!')

if __name__ == '__main__':
    main()
