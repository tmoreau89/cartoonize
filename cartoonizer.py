import streamlit as st
from PIL import Image, ExifTags
from io import BytesIO
from base64 import b64decode, b64encode
import requests
import random



CLIP_ENDPOINT = "https://cartoonizer-clip-test-4jkxk521l3v1.octoai.cloud"
SD_ENDPOINT = "https://sd-demo-gcsv8y11zs17.octoai.cloud"

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

def cartoonize_image(upload, strength, seed, extra_desc):
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

    # Add the extra desc
    prompt = extra_desc + ", " + clip_reply

    # Prepare SD request for img2img
    sd_request = {
        "init_image": image_out_b64.decode("utf8"),
        "prompt": prompt,
        "strength": float(strength)/10,
        # The rest below is hard coded
        "negative_prompt": "EasyNegative, drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry, nsfw",
        "model_name": "cartoon_v2",
        "scheduler": "DPM++2MKarras",
        "guidance_scale": 7,
        "num_images_per_prompt": 1,
        "seed": seed,
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "clip_skip": 2,
        "loras": None,
        "text_inversions": None
    }
    reply = requests.post(
        "{}/predict".format(SD_ENDPOINT),
        headers={"Content-Type": "application/json"},
        json=sd_request
    )

    img_bytes = b64decode(reply.json()["image_0"])
    cartoonized = Image.open(BytesIO(img_bytes), formats=("png",))

    col2.write("Transformed Image :star2:")
    col2.image(cartoonized)
    st.markdown("\n")
    st.download_button("Download transformed image", convert_image(cartoonized), "cartoonized.png", "cartoonized/png")

st.set_page_config(layout="wide", page_title="Cartoonizer")

st.write("## Cartoonizer - Powered by OctoAI")

st.markdown(
    "This demo app was build with the fastest Stable Diffusion in the world, now available for free on OctoAI, where devs run, tune, and scale super-fast generative AI models. [Try it for free here.](http://octoml.ai/)"
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

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

extra_desc = st.text_input("Add more context to customize the output")
# extra_desc_strength = st.slider("Strength of extra context. The higher this is the more your text matters", 1.0, 5.0, value=1.0)
# if extra_desc:
#     extra_desc = f"({extra_desc}: {extra_desc_strength})"

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
    cartoonize_image(my_upload, strength, seed, extra_desc)
