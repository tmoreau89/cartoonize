import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode
import requests

st.set_page_config(layout="wide", page_title="Pixarizer")

st.write("## Pixarizer - Powered by OctoAI")
st.write(
    "Upload an image to turn it into a scene from a Pixar movie! Full quality images can be downloaded from the sidebar. This application is powered by OctoML's OctoAI compute service. The image to image transfer is achieved via the Pixar Cartoon Type B Stable Diffusion checkpoint available on CivitAI: https://civitai.com/models/75650/disney-pixar-cartoon-type-b."
)

st.write(
    "     :camera_with_flash: Tip #1: works best with a square-ish image."
)
st.write(
    "     :blush: Tip #2: works best on close ups (e.g. portraits, profile pics)."
)
st.write(
    "     :woman-getting-haircut: Tip #3: avoid cropping the heads of the person in the photo."
)

st.sidebar.write("## Upload and download :gear:")

# PIL helper
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


# PIL helper
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def pixarize_image(upload):
    image = Image.open(upload)
    image = crop_max_square(image)
    image = image.resize((512, 512))
    col1.write("Original Image :camera:")
    col1.image(image)

    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    image.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)
    model_request = {
        "image": image_out_b64.decode("utf8"),
        "prompt": "masterpiece, best quality",
        "negative_prompt": "EasyNegative, drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry",
        "model": "DisneyPixarCartoon_v10",
        "vae": "YOZORA.vae.pt",
        "sampler": "K_EULER_ANCESTRAL",
        "cfg_scale": 7,
        "strength": 0.4,
        "num_images": 1,
        "width": 512,
        "height": 512,
        "steps": 20
    }
    reply = requests.post(
        f"https://pixarify-4jkxk521l3v1.octoai.cloud/predict",
        headers={"Content-Type": "application/json"},
        json=model_request
    )

    img_bytes = b64decode(reply.json()["completion"]["image_0"])
    pixarized = Image.open(BytesIO(img_bytes), formats=("png",))

    col2.write("Pixarized Image (takes about 20s to generate) :magic_wand:")
    col2.image(pixarized)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download pixarized image", convert_image(pixarized), "pixarized.png", "pixarized/png")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image (works best on square photos)", type=["png", "jpg", "jpeg"])


if my_upload is not None:
    pixarize_image(upload=my_upload)
else:
    image = Image.open("./thierry.png")
    col1.write("Original Image :camera:")
    col1.image(image)
    pixarized = Image.open("./pixarized.png")
    col2.write("Pixarized Image (takes about 20s to generate) :magic_wand:")
    col2.image(pixarized)
