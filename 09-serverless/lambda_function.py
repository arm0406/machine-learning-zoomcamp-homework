import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort

session = ort.InferenceSession("hair_classifier_empty.onnx")

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
        stream = BytesIO(buffer)
        img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    x = np.array(img).astype("float32")
    x = x / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, 0)        # add batch
    return x

def lambda_handler(event, context=None):
    url = event["url"]
    img = download_image(url)
    img = prepare_image(img)
    x = preprocess(img)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preds = session.run([output_name], {input_name: x})[0]
    return float(preds[0])