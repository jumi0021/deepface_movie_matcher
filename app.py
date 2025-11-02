import os
import cv2
import shutil
import logging
import sqlite3
import unicodedata
import numpy as np
from flask import Flask, request, render_template, send_file
from deepface import DeepFace
from werkzeug.utils import secure_filename
from PIL import Image
import uuid

# 기본 설정
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARACTER_DIR = os.path.join(BASE_DIR, "characters")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "characters_preprocessed")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
MATCH_DIR = os.path.join(STATIC_DIR, "matches")
DB_PATH = os.path.join(BASE_DIR, "characters.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MATCH_DIR, exist_ok=True)

app = Flask(__name__)


# DB 데이터 가져오기
def load_character_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT movie, character, actor, path, img FROM characters")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "movie": r[0],
            "character": r[1],
            "actor": r[2],
            "path": r[3],
            "img": r[4],
        })
    logging.info(f"DB 캐릭터 데이터 {len(data)}개 로드 완료")
    return data

CHARACTER_DATA = load_character_data()


# 메인 페이지
@app.route("/")
def index():
    return render_template("index.html")


# 업로드 처리
@app.route("/upload", methods=["POST"])
def upload():
    
    # 파일이 업로드 되지 않은 경우
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("result.html", error="파일이 선택되지 않았습니다.", match_data=None)

    file = request.files["file"]
    unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, unique_filename)
    file.save(upload_path)
    relative_upload_path = os.path.join("uploads", unique_filename)

    try:
        # 사용자 얼굴 검출 및 전처리
        faces = DeepFace.extract_faces(
            img_path=upload_path,
            detector_backend="mtcnn",
            enforce_detection=True
        )
        if not faces:
            return render_template("result.html", error="얼굴 감지 실패", match_data=None)

        # 얼굴 부분 추출 후 224x224로 변환
        face_array = (faces[0]["face"] * 255).astype("uint8")
        resized_face = cv2.resize(face_array, (224,224))
        face_pil = Image.fromarray(resized_face) # NumPy → PIL 변환
        
        # 전처리된 업로드 파일 저장
        upload_preprocessed_path = os.path.join(UPLOAD_DIR, f"preprocessed_{unique_filename}")
        face_pil.save(upload_preprocessed_path)

        # 전처리된 캐릭터 폴더를 기반으로 유사도 탐색
        
       #1. 단일 모델 유사도 계산
        dfs = DeepFace.find(
            img_path=upload_preprocessed_path, db_path=PREPROCESSED_DIR, #전처리된 캐릭터 폴더
            model_name="Facenet512", detector_backend="skip", enforce_detection=False, 
            threshold=1.0, silent=True
        )

        if not dfs or dfs[0].empty:
            return render_template("result.html", error="유사한 캐릭터를 찾지 못했습니다.", match_data=None)

        # dfs[0] : 딥페이스가 반환하는 결과 데이터 프레임
            # identity: 매칭된 이미지의 파일 경로(문자열)
            # distance: 업로드 이미지와 DB 이미지 사이의 거리(숫자, 작을수록 유사)
        dfs[0]["identity"] = dfs[0]["identity"].apply(lambda x: unicodedata.normalize('NFKC', str(x))) # identity를 유니코드 정규화(NFKC)
        dfs[0] = dfs[0].reset_index(drop=True)
        best_match_path = dfs[0].iloc[0]["identity"] # 가장 위(가장 유사한 결과) 행의 경로
        distance = dfs[0].iloc[0]["distance"] # 가장 위(가장 유사한 결과) 행의 거리
        similarity = max(0, (1 - distance) * 100) # 거리 -> 유사도


        # 2. 여러 모델을 이용한 Ensemble 유사도 계산
        # models = ["Facenet512", "VGG-Face", "ArcFace"]
        # similarity_scores = []
        # best_match_candidates = []

        # for model in models:
        #     dfs = DeepFace.find(
        #         img_path=upload_preprocessed_path,
        #         db_path=PREPROCESSED_DIR,
        #         model_name=model,
        #         detector_backend="skip",
        #         enforce_detection=False,
        #         threshold=1.0,
        #         silent=True
        #     )

        #     if dfs and not dfs[0].empty:
        #         dfs[0]["identity"] = dfs[0]["identity"].apply(lambda x: unicodedata.normalize('NFKC', str(x)))
        #         dfs[0] = dfs[0].reset_index(drop=True)
        #         best_match_path = dfs[0].iloc[0]["identity"]
        #         distance = dfs[0].iloc[0]["distance"]
        #         similarity = max(0, (1 - distance) * 100)
        #         similarity_scores.append(similarity)
        #         best_match_candidates.append(best_match_path)

        # if not similarity_scores:
        #     return render_template("result.html", error="유사한 캐릭터를 찾지 못했습니다.", match_data=None)

        # # 평균 유사도로 최종 결과 결정
        # avg_similarity = np.mean(similarity_scores)
        # best_match_path = max(set(best_match_candidates), key=best_match_candidates.count)


        # DB 데이터에서 해당 path와 img 찾기
        match_info = None
        
        for row in CHARACTER_DATA:
            candidate = os.path.normpath(os.path.join(PREPROCESSED_DIR, row["path"], row["img"]))
            if candidate == best_match_path:
                match_info = row
                break

        if not match_info:
            return render_template("result.html", error="DB에서 일치하는 캐릭터를 찾을 수 없습니다.", match_data=None)

        movie = match_info["movie"]
        character = match_info["character"]
        actor = match_info["actor"]
        path = match_info["path"]
        img = match_info["img"]

        # 원본 이미지 복사 후 결과 렌더링
        original_img_path = os.path.join(CHARACTER_DIR, path, img)
        match_image_name = os.path.basename(original_img_path)
        final_match_path = os.path.join(MATCH_DIR, match_image_name)
        shutil.copy2(original_img_path, final_match_path)

        match_data = {
            "movie": movie,
            "character_actor": f"{character} ({actor})",
            "similarity": f"{similarity:.2f}%",      #1. 단일 모델인 경우
            #"similarity": f"{avg_similarity:.2f}%", #2. 앙상블 모델인 경우
            "match_image": os.path.join("matches", match_image_name),
            "user_image": relative_upload_path
        }

        return render_template("result.html", match_data=match_data, error=None)

    except Exception as e:
        logging.error(f"오류 발생: {e}")
        return render_template("result.html", error=f"처리 중 오류 발생: {e}", match_data=None)


# 정적 파일 제공
@app.route("/static/<path:filename>")
def static_file(filename):
    return send_file(os.path.join(STATIC_DIR, filename))


# 실행
if __name__ == "__main__":
    app.run(debug=True)
