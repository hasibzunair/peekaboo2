import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import codecs
import numpy as np
import cv2

from PIL import Image, ImageDraw
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T

from misc import get_bbox_from_segmentation_labels

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def inference(img_path):
    # Load the image
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        original_size = img.size  # (w, h)

        # Preprocess
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

    # Forward step
    print(f"Start Peekaboo prediction.")
    with torch.no_grad():
        preds = model(inputs, for_eval=True)
    print(f"Done Peekaboo prediction.")

    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    orig_h, orig_w = original_size[1], original_size[0]
    preds_up = F.interpolate(
        preds, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    print(f"Shape of output after interpolation is {preds_up.shape}")
    preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()

    # Get segmentation mask
    pred_bin_mask = preds_up.cpu().squeeze().numpy().astype(np.uint8)
    initial_image_size = img.size[::-1]
    scales = [
        initial_image_size[0] / pred_bin_mask.shape[0],
        initial_image_size[1] / pred_bin_mask.shape[1],
    ]

    # Get bounding box for single object discovery
    pred_bbox = get_bbox_from_segmentation_labels(
        pred_bin_mask, initial_image_size, scales
    )
    print(f"Predicted bounding box: {pred_bbox}")

    # Plot mask and box in the image
    img_draw = img.convert("RGBA")

    # Create mask overlay with proper alpha channel
    alpha = (pred_bin_mask * 100).astype(np.uint8)
    color = (255, 0, 0, 0)

    color_mask = Image.fromarray(
        np.stack(
            [
                np.full_like(alpha, color[0]),
                np.full_like(alpha, color[1]),
                np.full_like(alpha, color[2]),
                alpha,
            ],
            axis=-1,
        ),
        mode="RGBA",
    )

    # Composite red mask over the image
    img_draw = Image.alpha_composite(img_draw, color_mask)

    # Draw bounding box
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(
        [(pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3])],
        outline=(255, 0, 0, 255),
        width=2,
    )
    img_draw = np.array(img_draw.convert("RGB"))
    return img_draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img-path",
        type=str,
        default="data/examples/VOC_000030.jpg",
        help="Image path.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/peekaboo_decoder_weights_niter500.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )
    args = parser.parse_args()

    # Configuration
    config, _ = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # App
    title = "PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization"
    description = codecs.open("./media/description.html", "r", "utf-8").read()
    article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2407.17628' target='_blank'>PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization</a> | <a href='https://github.com/hasibzunair/peekaboo' target='_blank'>Github</a></p>"

    gr.Interface(
        fn=inference,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=gr.Image(type="numpy", label="Predicted Output"),
        examples=[
            "./data/examples/octopus.jpeg",
            "./data/examples/godzillaxkong.jpeg",
            "./data/examples/avengers.jpeg",
            "./data/examples/dinosaur.jpeg",
            "./data/examples/chitauri.jpeg",
        ],
        title=title,
        description=description,
        article=article,
    ).launch()
