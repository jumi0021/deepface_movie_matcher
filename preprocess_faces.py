import os
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from pathlib import Path
from PIL import Image



# 경로 설정 및 디렉토리 생성
CHARACTER_DIR = Path("characters")                  # 원본 캐릭터 폴더
PREPROCESSED_DIR = Path("characters_preprocessed")  # 전처리된 캐릭터 폴더
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True) # 전처리 디렉터리 생성

# DeepFac 얼굴 검출 모델 선택
DETECTOR_BACKEND = "mtcnn"

# DeepFace 표준 입력 크기 
IMAGE_SIZE = (224, 224)      



# 전처리
def preprocess_faces():
    print("preprocess_faces.py : 전처리 시작")

    # 하위 모든 폴더 포함해서 jpg/jpeg/png 확장자 이미지 검색
    image_paths = list(CHARACTER_DIR.glob("**/*.jpg")) \
                + list(CHARACTER_DIR.glob("**/*.jpeg")) \
                + list(CHARACTER_DIR.glob("**/*.png"))

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # 이미지 읽기
            img = cv2.imread(str(img_path))
            if img is None:
                tqdm.write(f"이미지 읽기 실패: {img_path}")
                continue

            # 얼굴 검출
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            
            if not faces:
                tqdm.write(f"얼굴 감지 실패: {img_path}")
                continue


            # DeepFace 반환: float형 [0~1] → uint8 [0~255]
            face_array = (faces[0]["face"] * 255).astype("uint8")

            #PIL Image로 변환 후 224x224 리사이즈
            face_pil = Image.fromarray(face_array)
            face_pil = face_pil.resize(IMAGE_SIZE)
            
            
            # 전처리 폴더에 같은 구조로 저장
            relative_path = img_path.relative_to(CHARACTER_DIR)
            save_path = PREPROCESSED_DIR / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            face_pil.save(save_path)

        except Exception as e:
            tqdm.write(f"처리 실패: {img_path} - {e}")

    print("\n preprocess_faces.py : 전처리 완료 (모든 캐릭터 얼굴 'characters_preprocessed'에 224*224 형태로 저장)")



# 실행
if __name__ == "__main__":
    preprocess_faces()


