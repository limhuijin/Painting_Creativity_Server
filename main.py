from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np  
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["202"],  # React 애플리케이션의 URL
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 모델 로드
model = load_model('C:/Users/gabri/Desktop/coding/Painting_Creativity_Tester/model/model15.keras')

# 이미지 저장 경로
UPLOAD_DIRECTORY = "C:/Users/gabri/Desktop/coding/Painting_Creativity_Server/uploaded_images/"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 파일 저장 경로 설정
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
        # 파일을 경로에 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 서버에서 접근 가능한 URL 생성
        file_url = f"b20/uploads/{file.filename}"

        return {"file_url": file_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
@app.get("/analyze/")
async def analyze_image(image: str):
    # 이미지 경로 확인
    img_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(image))
    
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    pil_image = Image.open(img_path).convert('RGB')
    img_array = prepare_image(pil_image, target_size=(224, 224))

    # 모델 예측
    predictions = model.predict(img_array)
    predictions = predictions[0].tolist()  # NumPy 배열을 Python 리스트로 변환

    total_score = float(np.sum(predictions))  # NumPy float를 Python float로 변환
    score_category = categorize_score(total_score)  # 점수 범주 결정

    # 결과 데이터 구성
    result_data = {
        "predictions": predictions,
        "total_score": total_score,
        "category": score_category
    }

    return JSONResponse(content=result_data)

def prepare_image(img, target_size):
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Image.ANTIALIAS 대신 Image.Resampling.LANCZOS 사용
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 이미지 배열의 차원을 확장
    img_array /= 255.0  # 이미지 데이터를 정규화
    return img_array

def categorize_score(score):
    if score < 7:
        return 'Low'
    elif score < 17:
        return 'Medium'
    else:
        return 'High'
