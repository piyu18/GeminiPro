## Invoice Extractor

from dotenv import load_dotenv
import google.generativeai as genai
import os
import streamlit as st
from PIL import Image

load_dotenv()

## Configure API Key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

## Load Gemini-Pro vision Model

def get_gemini_vision_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image(upload_image):
    if upload_image is not None:
        bytes_data = upload_image.getvalue()

        image_parts = [{
            'mime_type': upload_image.type,
            'data': bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError('No file uploaded')
    
## Streamlit app
st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Application")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Tell me about the image")

input_prompt = """
               You are an expert in understanding invoices.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """

## If ask button is clicked

if submit:
    image_data = input_image(uploaded_file)
    response=get_gemini_vision_response(input_prompt,image_data,input)
     # Check if response is tabular and create download button
  
    st.subheader("The Response is")
    st.write(response)
    
    
        
