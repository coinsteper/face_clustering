import os
import cv2
import numpy as np
import shutil
from mtcnn import MTCNN
from sklearn.cluster import DBSCAN
import onnxruntime as ort
import unicodedata  # 한글 파일명 정리용


# 파일명 정리 함수 (유효하지 않은 문자 제거)
def sanitize_filename(filename):
    return unicodedata.normalize('NFC', filename)


# ArcFace ONNX 모델 로드
class ArcFace:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image, size=(112, 112)):
        image = cv2.resize(image, size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = (image - 127.5) / 127.5  # Normalize
        return np.expand_dims(image, axis=0)

    def get_embedding(self, face):
        preprocessed_face = self.preprocess(face)
        embedding = self.session.run([self.output_name], {self.input_name: preprocessed_face})[0]
        return embedding.flatten()


# 얼굴 클러스터링 함수
def face_clustering(input_folder, output_folder, model_path, eps=0.7, min_samples=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    detector = MTCNN()  # 얼굴 감지기
    arcface = ArcFace(model_path)  # ArcFace 모델 로드

    embeddings = []
    image_paths = []

    # 얼굴 감지 및 임베딩 추출
    for file in os.listdir(input_folder):
        sanitized_file = sanitize_filename(file)
        image_path = os.path.join(input_folder, sanitized_file)

        # 확장자 검사
        if not sanitized_file.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        # 이미지 유효성 검사
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Unable to read image: {sanitized_file}")
                continue
        except Exception as e:
            print(f"Error reading image {sanitized_file}: {e}")
            continue

        # 얼굴 감지
        results = detector.detect_faces(image)
        if not results:
            print(f"No faces detected in image: {sanitized_file}")
            continue

        # 얼굴 영역에서 임베딩 추출
        x, y, w, h = results[0]['box']
        face = image[y:y+h, x:x+w]
        embedding = arcface.get_embedding(face)

        # 임베딩 유효성 검사
        if embedding is None or embedding.size == 0:
            print(f"Failed to extract embedding for: {sanitized_file}")
            continue

        embeddings.append(embedding)
        image_paths.append(image_path)
        print(f"Processed: {sanitized_file}")

    # DBSCAN 클러스터링
    if len(embeddings) == 0:
        print("No valid embeddings found. Clustering cannot proceed.")
        return

    embeddings = np.array(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # 그룹 폴더 생성 및 이미지 복사
    for label, img_path in zip(labels, image_paths):
        group_folder = os.path.join(output_folder, f"group_{label}" if label != -1 else "unclassified")
        os.makedirs(group_folder, exist_ok=True)
        shutil.copy(img_path, os.path.join(group_folder, os.path.basename(img_path)))

    print(f"Clustering complete. Results saved to {output_folder}")


# 메인 실행
if __name__ == "__main__":
    input_folder = r"C:\repo\python\downloaded_faces"  # 입력 이미지 폴더 경로
    output_folder = r"C:\repo\python\out"   # 출력 폴더 경로
    model_path = r"C:\repo\python\arcface.onnx"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 얼굴 클러스터링 실행
    face_clustering(input_folder, output_folder, model_path)
