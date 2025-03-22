import os
import sys
import shutil
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import queue
import threading

from facelib import FaceRecognizer, FaceDetector
from facelib import get_config

def encode_faces_from_folder(folder_path):
    """
    Loads images from the given folder, detects faces using InsightFace,
    and returns the face embeddings along with their image paths.
    """
    embeddings = []
    image_paths = []
    no_face_count = 0
    
    conf = get_config()
    face_regonizer = FaceRecognizer(conf)
    face_detector = FaceDetector(name='resnet')

    filenames = os.listdir(folder_path)
    for filename in tqdm(filenames, desc=f"Encoding faces in {os.path.basename(folder_path)}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            faces, boxes, scores, landmarks = face_detector.detect_align(image)
            if faces.numel() > 0:
                embs = face_regonizer.feature_extractor(faces)
                embeddings += embs.tolist()
                image_paths.append(image_path)
            else:
                no_face_count += 1  # Increment the counter
                # print(f"No face detected in {filename}")
                
    print(f"{folder_path}. All files_count {len(filenames)}. No faces detected in {no_face_count} images.")

    return np.array(embeddings), image_paths

def cluster_faces_dbscan(embeddings, eps=0.3, min_samples=2):
    """
    Clusters face embeddings using DBSCAN with cosine distance.
    """
    cosine_similarities = cosine_similarity(embeddings)
    print(f"Min cosine similarity: {np.min(cosine_similarities)}")
    print(f"Max cosine similarity: {np.max(cosine_similarities)}")

    cosine_distances = 1 - cosine_similarities
    cosine_distances = np.maximum(0, cosine_distances)
    cosine_distances = np.nan_to_num(cosine_distances, nan=1.0)

    dbscan = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(cosine_distances)
    return labels

def save_clustered_images(image_paths, labels, output_folder):
    """
    Copies images into subfolders based on their cluster labels.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_path, cluster in zip(image_paths, labels):
        cluster_folder = os.path.join(output_folder, f"cluster_{cluster}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        shutil.copy(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))

def process_folder(folder_path):
    """
    Processes a single subfolder.
    """
    print(f"\nProcessing folder: {folder_path}")
    embeddings, image_paths = encode_faces_from_folder(folder_path)
    if len(embeddings) < 2:
        print(f"Not enough faces to perform clustering in {folder_path}.")
        return

    labels = cluster_faces_dbscan(embeddings)
    output_folder = os.path.join(folder_path, "clustered_faces")
    save_clustered_images(image_paths, labels, output_folder)
    print(f"Clustered images saved in: {output_folder}")

def worker(q, progress_bar):
    """Worker function to process folders from the queue."""
    while True:
        folder_path = q.get()
        if folder_path is None:
            break
        process_folder(folder_path)
        q.task_done()
        progress_bar.update(1)

def main():
    """
    Main function that takes one command-line argument: the main folder path.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <main_folder_path>")
        sys.exit(1)

    main_folder = sys.argv[1]
    subfolders = [os.path.join(main_folder, item) for item in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, item))]

    q = queue.Queue()
    threads = []
    num_threads = 10

    with tqdm(total=len(subfolders), desc="Processing subfolders") as progress_bar:
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(q, progress_bar))
            t.start()
            threads.append(t)

        for subfolder_path in subfolders:
            q.put(subfolder_path)

        q.join()

        for _ in range(num_threads):
            q.put(None)

        for t in threads:
            t.join()

    print(f"All subfolders processed successfully!")

if __name__ == "__main__":
    main()