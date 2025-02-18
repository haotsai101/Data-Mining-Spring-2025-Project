import os
import numpy as np
import argparse
import cv2

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
    
    current_frame = 1

    if not cap.isOpened():
        print('Cap is not open')

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
        else:
            break

    cap.release()
    print('done')

if __name__ == '__main__':
    main()