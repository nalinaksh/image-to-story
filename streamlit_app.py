from transformers import BlipForConditionalGeneration, BlipProcessor
from openai import OpenAI
import streamlit as st
from PIL import Image
import os
from authenticate import *

#Authenticate user
if not check_password():
    st.stop()

st.title("Image to short story generator")
st.write("Upload an image to generate a story")

@st.cache_resource
def get_model():
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
  return model

@st.cache_resource
def get_processor():
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  return processor

@st.cache_resource
def get_client():
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  return client

client = get_client()

def caption_to_story(image_caption):
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      max_tokens=256,
      messages=[
            {"role": "system", "content": "Create a very short fictional story from the given prompt. It should not exceed 100 words."},
            {"role": "user", "content": image_caption}
        ]
    )

  return response.choices[0].message.content

model = get_model()
processor = get_processor()

def image_to_text(image):
    inputs = processor(image,return_tensors="pt")
    out = model.generate(**inputs)
    image_caption = processor.decode(out[0], skip_special_tokens=True)
    story = caption_to_story(image_caption)
    return story

#upload image 
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['png', 'jpg'])

# If user attempts to upload a file.
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image")
  with st.spinner('Generating short story...'):
    story = image_to_text(image)
    st.write(story)
