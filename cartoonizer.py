import streamlit as st
from PIL import Image, ExifTags
from io import BytesIO
from base64 import b64decode, b64encode
import requests
import random



CLIP_ENDPOINT = "https://cartoonizer-clip-test-4jkxk521l3v1.octoai.cloud"
SD_CGI_ENDPOINT = "https://cartoonizer-sd-demo-cgi-4jkxk521l3v1.octoai.cloud"
SD_DRAWING_ENDPOINT = "https://cartoonizer-sd-demo-handdrawn-4jkxk521l3v1.octoai.cloud"
SD_COMIC_ENDPOINT = "https://cartoonizer-sd-demo-comic-4jkxk521l3v1.octoai.cloud"

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

def cartoonize_image(upload, mode, strength, seed):
    input_img = Image.open(upload)
    try:
        # Rotate based on Exif Data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = input_img._getexif()
        if exif[orientation] == 3:
            input_img=input_img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            input_img=input_img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            input_img=input_img.rotate(90, expand=True)
    except:
        # Do nothing
        print("No rotation to perform based on Exif data")
    # Apply cropping and resizing to work on a square image
    cropped_img = crop_max_square(input_img)
    resized_img = cropped_img.resize((512, 512))
    col1.write("Original Image :camera:")
    col1.image(resized_img)

    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    resized_img.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)

    # Prepare CLIP request
    clip_request = {
        "mode": "fast",
        "image": image_out_b64.decode("utf8"),
    }
    # Send to CLIP endpoint
    reply = requests.post(
        "{}/predict".format(CLIP_ENDPOINT),
        headers={"Content-Type": "application/json"},
        json=clip_request
    )
    # Retrieve prompt
    clip_reply = reply.json()["completion"]["labels"]

    # Uncomment if you want to edit the results of the CLIP model
    # Editable CLIP interrogator output
    # prompt = st.text_area("AI-generated, human editable label:", value=clip_reply)

    if mode == "Hand Drawn":
        sd_endpoint = SD_DRAWING_ENDPOINT
        model = "hand_drawn"
    elif mode == "CGI":
        sd_endpoint = SD_CGI_ENDPOINT
        model = "cgi"
    elif mode == "Comic Strip":
        sd_endpoint = SD_COMIC_ENDPOINT
        model = "comic"

    # Prepare SD request for img2img
    sd_request = {
        "image": image_out_b64.decode("utf8"),
        "prompt": clip_reply,
        "negative_prompt": "EasyNegative, drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry, nsfw",
        "model": model,
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
        "{}/predict".format(sd_endpoint),
        headers={"Content-Type": "application/json"},
        json=sd_request
    )

    img_bytes = b64decode(reply.json()["completion"]["image_0"])
    cartoonized = Image.open(BytesIO(img_bytes), formats=("png",))

    col2.write("Transformed Image :star2:")
    col2.image(cartoonized)
    st.markdown("\n")
    st.download_button("Download transformed image", convert_image(cartoonized), "cartoonized.png", "cartoonized/png")

st.set_page_config(layout="wide", page_title="Cartoonizer")

st.write("## Cartoonizer - Powered by OctoAI")

st.markdown(
    "The fastest version of Stable Diffusion in the world is now available on OctoAI, where devs run, tune, and scale generative AI models. [Try it for free here.](http://octoml.ai/)"
)

st.markdown(
    "### Upload a photo and turn yourself into a cartoon character!"
)


st.markdown(
    " :camera_with_flash: Tip #1: works best on a square image."
)
st.markdown(
    " :blush: Tip #2: works best on close ups (e.g. portraits), rather than full body or group photos."
)
st.markdown(
    " :woman-getting-haircut: Tip #3: for best results, avoid cropping heads/faces."
)

mode = st.radio(
    'Visual Effect',
    ("CGI", "Hand Drawn", "Comic Strip"))

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

strength = st.slider(
    ":brain: Imagination Slider (lower: closer to original, higher: more imaginative result)",
    3, 10, 5)

seed = 0
if st.button('Regenerate'):
    seed = random.randint(0, 1024)

st.sidebar.image("octoml-octo-ai-logo-color.png")
st.sidebar.markdown("The image to image generation is achieved via the [following checkpoint](https://civitai.com/models/75650/disney-pixar-cartoon-type-b) on CivitAI.")

st.sidebar.markdown(
    ":warning: **Disclaimer** :warning:: Cartoonizer is built on the foundation of [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator) and [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and is therefore likely to carry forward the potential dangers inherent in these base models. ***It's capable of generating unintended, unsuitable, offensive, and/or incorrect outputs. We therefore strongly recommend exercising caution and conducting comprehensive assessments before deploying this model into any practical applications.***"
)

st.sidebar.markdown(
    "By releasing this model, we acknowledge the possibility of it being misused. However, we believe that by making such models publicly available, we can encourage the commercial and research communities to delve into the potential risks of generative AI and subsequently, devise improved strategies to lessen these risks in upcoming models. If you are researcher and would like to study this subject further, contact us and we’d love to work with you!"
)

st.sidebar.markdown(
    "Report any issues, bugs, unexpected behaviors [here](https://github.com/tmoreau89/cartoonize/issues)"
)

if my_upload is not None:
    cartoonize_image(my_upload, mode, strength, seed)
