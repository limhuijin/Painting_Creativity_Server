import os
import requests
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from config import FRONTEND_URL, BASE_URL, MODEL_PATH, UPLOAD_DIRECTORY, GITHUB_TOKEN, GITHUB_REPO

app = FastAPI()

# GitHub API URL 구성
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/"

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처를 허용하려면 "*" 사용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = load_model(MODEL_PATH)

def upload_to_github(file_path, file_content):
    github_headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }

    # 파일을 base64로 인코딩
    encoded_content = base64.b64encode(file_content).decode('utf-8')

    data = {
        "message": "Add new image",
        "content": encoded_content,
        "branch": "main"  # 업로드할 브랜치
    }

    response = requests.put(f"{GITHUB_API_URL}{file_path}", json=data, headers=github_headers)

    if response.status_code == 201:
        return response.json()["content"]["download_url"]
    else:
        raise Exception(f"Failed to upload to GitHub: {response.status_code}, {response.text}")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 파일 내용을 읽음
    file_content = await file.read()

    # 서버에 임시로 파일 저장
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # GitHub에 파일 업로드
    try:
        github_url = upload_to_github(file.filename, file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 서버에서 임시 파일 삭제
    os.remove(file_path)

    return {"file_url": github_url}
    
@app.get("/analyze/")
async def analyze_image(image: str):
    img_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(image))
    
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

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

    # 분석이 끝난 후 서버에 저장된 이미지 삭제
    os.remove(img_path)

    return JSONResponse(content=result_data)

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
