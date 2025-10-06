from fastapi import HTTPException
import requests
import streamlit as st
import io
from keras.preprocessing import image as Image
import json


API_ENDPOINT = "http://localhost:8000"

def get_prediction(image_bytes):
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(API_ENDPOINT + '/predict/', files=files)
    
    if response.status_code == 200:
        # Return the image bytes from response
        return response.content
    else:
        raise HTTPException(status_code=response.status_code, detail="API error")

# Streamlit app
st.title("Cityscape clustering app")
st.write("Upload an image and get the prediction result")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to bytes for API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    
    # Prediction button
    if st.button("Get Prediction"):
        with st.spinner("Processing image..."):
            try:
                # Get prediction from API
                prediction_bytes = get_prediction(img_bytes)
                
                # Display prediction image
                prediction_image = Image.load_img(io.BytesIO(prediction_bytes))
                st.image(prediction_image, caption="Prediction Result", use_container_width=True)
                
                # Optional: Add download button for the result
                st.download_button(
                    label="Download Prediction",
                    data=prediction_bytes,
                    file_name="prediction.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error getting prediction: {str(e)}")