import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode
import requests
import random

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

def pixarize_image(upload, strength, seed):
    input_img = Image.open(upload)
    cropped_img = crop_max_square(input_img)
    resized_img = cropped_img.resize((512, 512))
    col1.write("Original Image :camera:")
    col1.image(resized_img)

    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    resized_img.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)
    model_request = {
        "image": image_out_b64.decode("utf8"),
        "prompt": "masterpiece, best quality",
        "negative_prompt": "EasyNegative, drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry, nsfw",
        "model": "DisneyPixarCartoon_v10",
        "vae": "YOZORA.vae.pt",
        "sampler": "K_EULER_ANCESTRAL",
        "cfg_scale": 7,
        "strength": float(strength)/10,
        "num_images": 1,
        "seed": seed,
        "width": 512,
        "height": 512,
        "steps": 20
    }
    reply = requests.post(
        f"https://cartoonizer-4jkxk521l3v1.octoai.cloud/predict",
        headers={"Content-Type": "application/json"},
        json=model_request
    )

    img_bytes = b64decode(reply.json()["completion"]["image_0"])
    cartoonized = Image.open(BytesIO(img_bytes), formats=("png",))

    col2.write("Transformed Image :magic_wand:")
    col2.image(cartoonized)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download transformed image", convert_image(cartoonized), "cartoonized.png", "cartoonized/png")

st.set_page_config(layout="wide", page_title="Cartoonizer")

st.write("## Cartoonizer - Powered by OctoAI")
st.markdown(
    "Upload a photo and turn yourself into a CGI character in about 3s! Full quality images can be downloaded from the sidebar. This application is powered by OctoML's OctoAI compute service. The image to image transfer is achieved via the [Pixar Cartoon Type B](https://civitai.com/models/75650/disney-pixar-cartoon-type-b) checkpoint on CivitAI."
)

st.markdown(
    " * :camera_with_flash: Tip #1: works best on a square-ish image."
)
st.markdown(
    " * :blush: Tip #2: works best on close ups (e.g. portraits, profile pics), rather than full body pictures."
)
st.markdown(
    " * :woman-getting-haircut: Tip #3: for best results, avoid cropping heads/faces."
)

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image (works best on square photos)", type=["png", "jpg", "jpeg"])

strength = st.slider(
    "Select the imagination level (lower: closer to original, higher: more imaginative result, 5 being a good sweet spot)",
    2, 8, 5)

seed = 0
if st.button('I\'m feeling lucky'):
    seed = random.randint(0, 1024)

st.sidebar.write("## Upload and download :gear:")

st.markdown(
    "**Disclaimer**: Cartoonizer is built on the foundation of CLIP and Stable Diffusion models, and is therefore likely to carry forward the potential dangers inherent in these base models. ***It's capable of generating unintended, unsuitable, offensive, and/or incorrect outputs. We therefore strongly recommend exercising caution and conducting comprehensive assessments before deploying this model into any practical applications.***"
)

st.markdown(
    "By releasing this model, we acknowledge the possibility of it being misused. However, we believe that by making such models publicly available, we can encourage the commercial and research communities to delve into the potential risks of generative AI and subsequently, devise improved strategies to lessen these risks in upcoming models. If you are researcher and would like to study this subject further, contact us and we’d love to work with you!"
)

st.markdown(
    "Report any issues, bugs, unexpected behaviors [here](https://github.com/tmoreau89/cartoonize/issues)"
)

if my_upload is not None:
    pixarize_image(my_upload, strength, seed)
else:
    image = Image.open("./thierry.png")
    col1.write("Original Image :camera:")
    col1.image(image)
    cartoonized = Image.open("./cartoonized.png")
    col2.write("Cartoonized Image (preview):magic_wand:")
    col2.image(cartoonized)
