import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# 상수 정의
FRONTEND_URL = "https://b3cb-106-255-245-242.ngrok-free.app"
BASE_URL = "https://a236-106-255-245-242.ngrok-free.app"
MODEL_PATH = "C:/Users/gabri/Desktop/coding/Painting_Creativity_Tester/model/model15.keras"
UPLOAD_DIRECTORY = "C:/Users/gabri/Desktop/coding/Painting_Creativity_Server/uploaded_images"

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

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 파일 저장 경로 설정
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
    # 파일을 경로에 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 서버에서 접근 가능한 URL 생성
    file_url = f"{BASE_URL}/uploads/{file.filename}"

    return {"file_url": file_url}
    
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
