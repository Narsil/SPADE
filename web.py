import torch
from models.pix2pix_model import Pix2PixModel
from torchvision.transforms import ToTensor
from util.util import labelcolormap
from util.coco import id2label
import numpy as np
from flask import Flask, send_from_directory, request
from PIL import Image
import string
from random import sample
import json
import re
import io
import base64

app = Flask(__name__)


class Options:
    gpu_ids = []
    checkpoints_dir = "./checkpoints"
    model = "pix2pix"
    norm_G = "spectralspadesyncbatch3x3"
    norm_D = "spectralinstance"
    norm_E = "spectralinstance"
    phase = "test"
    batchSize = 1
    preprocess_mode = "resize_and_crop"
    load_size = 256
    crop_size = 256
    aspect_ratio = 1.0
    label_nc = 182
    contain_dontcare_label = True
    output_nc = 3
    dataroot = "datasets/coco_stuff/"
    dataset_mode = "coco"
    serial_batches = True
    no_flip = True
    nThreads = 0
    max_dataset_size = 9_223_372_036_854_775_807
    load_from_opt_file = False
    cache_filelist_write = True
    cache_filelist_read = True
    display_winsize = 256
    netG = "spade"
    ngf = 64
    init_type = "xavier"
    init_variance = 0.02
    z_dim = 256
    no_instance = False
    nef = 16
    use_vae = False
    results_dir = "./results/"
    which_epoch = "latest"
    num_upsampling_layers = "normal"
    no_pairing_check = False
    coco_no_portraits = False
    isTrain = False
    semantic_nc = 184
    name = "coco_pretrained"
    dataset_mode = "coco"


opts = Options()
MODEL = None


def model():
    global MODEL
    if MODEL is not None:
        return MODEL
    model = Pix2PixModel(opts)
    model.eval()
    MODEL = model
    return model


label = torch.zeros(1, 1, 256, 256)
label[:, :, :, :] = 171
instance = torch.zeros(1, 1, 256, 256)
instance[:, :, :, :] = 171
data = {"label": label, "instance": instance, "image": torch.Tensor()}
colors = labelcolormap(opts.label_nc)


@app.route("/")
def index():
    colors_html = ["#%02x%02x%02x" % tuple(color.tolist()) for color in colors]
    colors_buttons = "".join(
        [
            f'<button type="button" value="{color}" title="{id2label(i)}"></button>'
            for i, color in enumerate(colors_html)
        ]
    )
    return f"""
    <head>
        <link href="static/web.css" rel="stylesheet">
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
        <script type="text/javascript" src="static/web.js"></script>
    </head>
    <body>

 <main>
      <div class="left-block">
        <div class="colors">{colors_buttons}</div>
        <div class="brushes">
          <button type="button" value="10"></button>
          <button type="button" value="20"></button>
          <button type="button" value="30"></button>
          <button type="button" value="40"></button>
          <button type="button" value="50"></button>
        </div>
        <div class="buttons">
          <button id="clear" type="button">Clear</button>
          <button id="generate" type="button">Generate</button>
        </div>
      </div>
      <div class="right-block">
        <canvas id="paint-canvas" width="256" height="256"></canvas>
      </div>
    </main>
        </body>
    """


def pil2rgb(piltensor, normalized=True):
    image_numpy = np.transpose(piltensor, (1, 2, 0))
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
    return image_numpy


def generate_img(label, name, instance=None):
    if instance is None:
        instance = label
    data = {"label": label, "instance": instance, "image": torch.Tensor()}
    generated = model()(data, mode="inference")

    image_numpy = pil2rgb((generated[0].numpy() + 1) / 2.0 * 255)
    # pilimage = ToPILImage()(generated[0, :, :, :])
    pilimage = Image.fromarray(image_numpy)
    name = f"static/{name}"
    image_url = f"{name}.png"
    pilimage.save(image_url)
    return image_url


def color2label(image):
    # PIL to numpy RGBA -> RGB
    tensor = np.array(image)[:, :, :3]
    label = np.zeros(tensor.shape)
    total = opts.crop_size ** 2
    for l, color in enumerate(colors):
        locations = np.where((tensor == color.tolist()).all(axis=2))
        n = len(locations[0])
        if n:
            p = n / total * 100
            print(f"{id2label(l)}: {p}%")
        label[locations] = [l, l, l]
    return ToTensor()(label[:, :, :1])


def random_name(n=6):
    return "".join(sample(string.ascii_letters, n))


@app.route("/generate", methods=["POST"])
def generate():
    data_url = request.data
    imgstr = re.search(rb"base64,(.*)", data_url).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    label_img = color2label(im)
    label_img = label_img.unsqueeze(0)

    h = random_name()
    image_url = generate_img(label_img, f"test_{h}")
    return json.dumps({"url": image_url})


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run()
