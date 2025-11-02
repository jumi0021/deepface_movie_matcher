# DeepFace Movie Matcher

DeepFace Movie Matcher는 DeepFace를 활용하여 사용자가 업로드한 얼굴 사진을 영화 속 캐릭터 이미지들과 비교해 **가장 닮은 캐릭터**를 찾아주는 Flask 웹 애플리케이션입니다.

---

## 🚀 주요 기능
- 얼굴 인식: DeepFace를 이용한 얼굴 임베딩 추출  
- 유사도 비교: 캐릭터 데이터베이스와의 임베딩 거리 계산  
- 결과 표시: 닮은 캐릭터 이미지, 이름, 유사도 점수 출력  

---

## 🧱 기술 스택
- **AI Model**: DeepFace
- **Backend**: Flask
- **DataBase**: SQLite 
- **Frontend**: HTML, CSS
- **이미지 처리**: OpenCV, PIL

---

## ⚙️ 실행 방법
```bash
git clone https://github.com/jumi0021/deepface_movie_matcher.git
cd deepface_movie_matcher

python3 -m venv venv
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

python3 preprocess_faces.py (이미 실행 완료, 생략)
python3 initialize_db.py (이미 실행 완료, 생략)
python3 app.py
