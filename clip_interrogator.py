import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode
import requests


CLIP_ENDPOINT = "https://clip-interrogator-4jkxk521l3v1.octoai.cloud"

def run_clip_interrogator(upload, mode):
    # Input image that the user is uploading
    input_img = Image.open(upload)

    # Apply cropping and resizing to work on a square image
    st.write("Input Image :camera:")
    st.image(input_img)

    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    input_img.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)
    model_request = {
        "image": image_out_b64.decode("utf8"),
        "mode": mode
    }

    # Send the model request!
    reply = requests.post(
        "{}/predict".format(CLIP_ENDPOINT),
        headers={"Content-Type": "application/json"},
        json=model_request
    )

    # Print the labels returned by the inference endpoint
    labels = reply.json()["completion"]["labels"]
    st.write("Labels from CLIP-Interrogator: {}".format(labels))


st.set_page_config(layout="wide", page_title="CLIP Interrogator")

st.write("## CLIP Interrogator - Powered by OctoAI")
st.markdown(
    "Upload a photo and let the CLIP Interrogator model determine what it is!"
)

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

mode = st.radio(
        'CLIP mode',
        ("default", "classic", "fast", "negative"))

if my_upload is not None:
    run_clip_interrogator(my_upload, mode)