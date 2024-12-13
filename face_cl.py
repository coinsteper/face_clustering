import os
import face_recognition
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from shutil import copy2
import dlib

# 입력 이미지 폴더와 출력 폴더 설정
INPUT_FOLDER = r"C:\Users\quim\downloaded_faces\copy2"
OUTPUT_FOLDER = r"C:\Users\quim\output"

MODEL_PATH =r"C:\Users\quim\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

#face_recognition.api.pose_predictor_68_point = face_recognition.dlib.shape_predictor(MODEL_PATH)
#face_recognition.api.pose_predictor_68_point = dlib.shape_predictor(MODEL_PATH)

def load_images_from_folder(folder):
    """폴더에서 이미지 불러오기"""
    images = []
    image_paths = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            images.append(image)
            image_paths.append(path)
    return images, image_paths

def extract_face_encodings(images):
    """얼굴 특징(임베딩) 추출"""
    encodings = []
    image_indices = []  # 얼굴이 감지된 이미지의 인덱스를 저장
    
    for idx, img in enumerate(images):
        # 이미지 크기 조정으로 정확도 향상
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            img = cv2.resize(img, (800, int(height * scale)))
            
        face_locations = face_recognition.face_locations(img, model="cnn")
        face_encs = face_recognition.face_encodings(
            img, 
            face_locations, 
            num_jitters=100,
            model='large'
        )
        
        if face_encs:
            for enc in face_encs:
                enc = enc / np.linalg.norm(enc)
                encodings.append(enc)
                image_indices.append(idx)  # 얼굴이 감지된 이미지의 인덱스 저장
            
    return encodings, image_indices

def cluster_faces(encodings, eps=0.35, min_samples=3):
    """DBSCAN을 사용해 얼군집화"""
    valid_encodings = [enc for enc in encodings if enc is not None]
    if not valid_encodings:
        return []
    
    X = np.array(valid_encodings)
    # cosine 거리 메트릭 사용
    clustering = DBSCAN(
        eps=eps, 
        min_samples=min_samples, 
        metric='cosine',  # euclidean 대신 cosine 거리 사용
        n_jobs=-1  # 모든 CPU 코어 사용
    ).fit(X)
    labels = clustering.labels_
    return labels

def save_clustered_faces(image_paths, labels):
    """군집화된 얼굴 이미지를 폴더에 저장"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 두 리스트의 길이가 일치하는지 확인
    if len(image_paths) != len(labels):
        raise ValueError("The length of image_paths and labels must be the same.")

    for idx, label in enumerate(labels):
        if label == -1:
            cluster_folder = os.path.join(OUTPUT_FOLDER, "unknown")
        else:
            cluster_folder = os.path.join(OUTPUT_FOLDER, f"cluster_{label}")
        
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        if os.path.exists(image_paths[idx]):
            copy2(image_paths[idx], cluster_folder)

def main():
    print("Loading images...")
    images, image_paths = load_images_from_folder(INPUT_FOLDER)

    print("Extracting face encodings...")
    encodings, image_indices = extract_face_encodings(images)  # image_indices로 변경

    print("Clustering faces...")
    labels = cluster_faces(encodings, eps=0.35, min_samples=3)

    print("Saving clustered faces...")
    # 얼굴이 감지된 이미지의 경로만 선택
    valid_image_paths = [image_paths[idx] for idx in image_indices]
    save_clustered_faces(valid_image_paths, labels)

    print("Done! Clustered faces are saved in 'grouped_faces' folder.")

if __name__ == "__main__":
    main()
