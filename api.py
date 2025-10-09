from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
import uvicorn
import pickle
import os
from keras.preprocessing import image
from keras.saving import load_model
from Training.data_tools import get_prediction, visualize_model_prediction
from fastapi.responses import Response
# from PIL import Image
import io
import numpy as np

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Training')))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

latest_model = None
try:
    # model_path = r'.\Deployment\Models\best_model.pkl'
    model_path = os.path.join('.', 'Training', 'models-dr', 'model.keras')
    # model_path = r"C:\Users\d0t\OC\Projet7\OC7\Training\mlartifacts\372886525690534527\models\m-6224c5eef06a49c5b1ea33879ec8922f\artifacts\data\model.keras"
    latest_model = load_model(model_path, custom_objects=None, compile=False, safe_mode=True)
    # visualize_model_prediction(latest_model)
    # latest_model_in = open(model_path, "rb")
    # latest_model = pickle.load(latest_model_in)

except Exception as e:
    print(f"Error loading model: {e}")
    # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load the model")


@app.get("/")
def read_root():
    return {"Hello": "World"}




@app.post("/predict/", response_class=Response)#seems to require python_multipart
async def get_prediction_mask(file: UploadFile = File(...)):
    # Validate image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read and process input image
        contents = await file.read()
        image_raw = image.load_img(io.BytesIO(contents))

        prediction_dic = get_prediction(latest_model,image_raw)
        
        
        prediction_array = prediction_dic['mask_resized']
        prediction_image = image.array_to_img(prediction_array)
        
        # Save image to bytes
        img_bytes = io.BytesIO()
        prediction_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(content=img_bytes.getvalue(), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run('api:app', host='0.0.0.0', port=8000)


    