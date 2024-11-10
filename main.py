from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]  
)

MODEL_PATH = "C:/Users/gabri/Desktop/coding/Painting_Creativity_Tester/model/model15.keras"
UPLOAD_DIRECTORY = "C:/Users/gabri/Desktop/coding/Painting_Creativity_Server/uploaded_images"

model = load_model(MODEL_PATH)

def prepare_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = img_to_array(img)
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

@app.post("/analyze-image")
async def upload_image(file: UploadFile = File(...)):
    if file is None or file.filename == "":
        raise HTTPException(status_code=422, detail="No file uploaded")
    
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        pil_image = Image.open(file_path).convert("RGB")
        img_array = prepare_image(pil_image)

        predictions = model.predict(img_array)[0].tolist()
        total_score = float(np.sum(predictions))
        category = categorize_score(total_score)

        return JSONResponse(content={
            "predictions": predictions,
            "total_score": total_score,
            "category": category
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
