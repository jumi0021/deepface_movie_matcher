import os
import sqlite3
import csv
from datetime import datetime

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARACTER_DIR = os.path.join(BASE_DIR, "characters")                  # 원본 캐릭터 이미지 폴더
PREPROCESSED_DIR = os.path.join(BASE_DIR, "characters_preprocessed")  # 전처리된 캐릭터 이미지 폴더
DB_PATH = os.path.join(BASE_DIR, "characters.db")   # DB 파일
CSV_PATH = os.path.join(BASE_DIR, "mapping.csv")    # 매핑 CSV 파일

# DB 테이블 생성
def initialize_db():
    
    # 만약 이미 존재한다면 재생성
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("기존 DB 삭제 후 재생성")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            movie TEXT,
            character TEXT,
            actor TEXT,
            path TEXT,
            img TEXT,
            added_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# DB 데이터 저장 (CSV 파일 활용)
def build_character_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # CSV 읽기
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            
            # characters + /movie/character(actor)
            folder_path = os.path.join(CHARACTER_DIR, *row["path"].split("/"))
            
            # 해당 폴더가 있는 지 확인
            if not os.path.isdir(folder_path):
                print(f"initialize_db.py : 폴더 없음 {folder_path}")
                continue

            # 해당 폴더의 이미지 확인
            img_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if not img_files:
                print(f"initialize_db.py : 이미지 없음 {folder_path}")
                continue

            # 원본 이미지 경로 (이미지 파일 1장만 선택, 이후 확장 가능)
            image_name = sorted(img_files)[0]

            # DB에 데이터 삽입
            cursor.execute("""
            INSERT 
            INTO characters (movie, character, actor, path, img, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["movie"],
                row["character"],
                row["actor"],
                row["path"],
                image_name,
                datetime.now().isoformat()
            ))

    conn.commit()
    conn.close()
    print("initialize_db.py : CSV 기반 DB 생성 완료")


# 실행
if __name__ == "__main__":
    initialize_db()
    build_character_db()


