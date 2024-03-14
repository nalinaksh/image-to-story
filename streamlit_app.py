from transformers import BlipForConditionalGeneration, AutoProcessor
# from transformers import pipeline
# from huggingface_hub import snapshot_download
from openai import OpenAI
import streamlit as st
import os

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# snapshot_download(repo_id="Salesforce/blip-image-captioning-base", local_dir="Salesforce")

# pipe = pipeline("image-to-text", model="Salesforce")

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
    # pipeline_output = pipe(image)
    # story = caption_to_story(pipeline_output[0]['generated_text'])
    story = caption_to_story(image_caption)
    return story

#upload image 
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

# If user attempts to upload a file.
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # Show the image filename and image.
    st.write(f'filename: {uploaded_file.name}')
    st.image(bytes_data)
    story = image_to_text(bytes_data)
    st.write(story)
