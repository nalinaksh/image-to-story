from transformers import BlipForConditionalGeneration, AutoProcessor
from openai import OpenAI
import streamlit as st
from PIL import Image
import os

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

def image_to_text(image):
    inputs = processor(image,return_tensors="pt")
    out = model.generate(**inputs)
    image_caption = processor.decode(out[0], skip_special_tokens=True)
    story = caption_to_story(image_caption)
    return story

#upload image 
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['png', 'jpg'])

# If user attempts to upload a file.
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image")
  with st.spinner('Generating short story...'):
    story = image_to_text(image)
    st.write(story)
