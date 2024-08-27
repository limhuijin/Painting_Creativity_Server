import os
import requests
import base64
import hashlib
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from config import MODEL_PATH, UPLOAD_DIRECTORY, GITHUB_TOKEN, GITHUB_REPO

app = FastAPI()

# GitHub API URL 구성
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/uploaded_images/"

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처를 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = load_model(MODEL_PATH)

def find_lowest_available_filename(directory, extension=".png"):
    """디렉토리 내에서 중복되지 않는 가장 낮은 숫자의 파일명을 찾습니다."""
    existing_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    existing_numbers = sorted(int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit())
    
    lowest_number = 0
    for number in existing_numbers:
        if lowest_number < number:
            break
        lowest_number += 1
    
    return f"{lowest_number:04d}{extension}"

def upload_to_github(file_path, file_content):
    github_headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }

    # 파일을 base64로 인코딩
    encoded_content = base64.b64encode(file_content).decode('utf-8')

    # GitHub에 업로드할 데이터 구성
    data = {
        "message": "Add new image",
        "content": encoded_content,
        "branch": "main"  # 업로드할 브랜치
    }

    response = requests.put(f"{GITHUB_API_URL}{file_path}", json=data, headers=github_headers)

    if response.status_code in [201, 200]:
        return response.json()["content"]["download_url"]
    else:
        raise Exception(f"Failed to upload to GitHub: {response.status_code}, {response.text}")

@app.post("/upload/")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # 파일 크기 확인 (10MB 이상인 경우 업로드 금지)
        file_size_mb = len(await file.read()) / (1024 * 1024)
        if file_size_mb > 10:
            raise HTTPException(status_code=413, detail="파일 용량이 너무 큽니다. (최대 10MB)")

        # 파일 내용을 다시 읽음
        await file.seek(0)
        file_content = await file.read()

        # 중복되지 않는 가장 낮은 숫자로 파일 이름 설정
        file_name = find_lowest_available_filename(UPLOAD_DIRECTORY)
        file_path = os.path.join(UPLOAD_DIRECTORY, file_name)

        # 서버에 임시로 파일 저장
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 5분 후 파일 삭제 작업 예약
        background_tasks.add_task(delete_file_after_delay, file_path)

        # GitHub에 파일 업로드
        github_url = upload_to_github(f"uploaded_images/{file_name}", file_content)

        return {"file_url": github_url}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/")
async def analyze_image(image: str):
    img_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(image))

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    for attempt in range(3):  # 총 3회 시도
        try:
            pil_image = Image.open(img_path).convert('RGB')
            img_array = prepare_image(pil_image, target_size=(224, 224))

            # 모델 예측
            predictions = model.predict(img_array)
            predictions = predictions[0].tolist()

            predictions = [max(1, min(5, value)) for value in predictions]

            total_score = float(np.sum(predictions))
            score_category = categorize_score(total_score)

            result_data = {
                "predictions": predictions,
                "total_score": total_score,
                "category": score_category
            }

            return JSONResponse(content=result_data)

        except Exception as e:
            if attempt == 2:  # 3회 시도 후 실패 시 예외 발생
                raise HTTPException(status_code=500, detail="이미지 분석에 실패했습니다. 서버에 문제가 발생했습니다.")
            else:
                continue  # 다음 시도로 넘어감

def prepare_image(img, target_size):
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def categorize_score(score):
    if score < 7:
        return 'Low'
    elif score < 17:
        return 'Medium'
    else:
        return 'High'

async def delete_file_after_delay(file_path: str, delay: int = 300):
    """일정 시간 후에 파일을 삭제하는 함수"""
    await asyncio.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
