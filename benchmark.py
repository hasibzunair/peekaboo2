# Code for Peekaboo
# Author: Hasib Zunair

"""Benchmark code for Peekaboo"""

import time
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from model import PeekabooModel
from misc import load_config, get_bbox_from_segmentation_labels

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

"""
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python benchmark.py \
    --img-path data/examples/doggo.jpg \
    --model-weights data/weights/peekaboo_decoder_weights_niter500.pt \
    --config configs/peekaboo_DUTS-TR.yaml \
    --batch-size 16
"""


def load_model(args, device):
    config, _ = load_config(args.config)
    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    model.decoder_load_weights(args.model_weights)
    model.eval()
    model.to(device)
    print(f"Model loaded from {args.model_weights}")
    return model


def preprocess_images(image_paths, device):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            NORMALIZE,
        ]
    )
    img_tensors = []
    orig_sizes = []

    for im_path in image_paths:
        img = Image.open(im_path).convert("RGB")
        orig_sizes.append(img.size)  # (w, h)
        img_tensors.append(transform(img))

    img_batch = torch.stack(img_tensors).to(device)
    return img_batch, orig_sizes


def run_segmentation_mask_inference(model, inputs, orig_sizes):
    sigmoid = nn.Sigmoid()
    masks = []

    with torch.no_grad():
        preds = model(inputs, for_eval=True)  # (B, 1, H', W')
        for i in range(inputs.size(0)):
            orig_w, orig_h = orig_sizes[i]
            preds_up = F.interpolate(
                preds[i : i + 1],
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            mask = (sigmoid(preds_up) > 0.5).squeeze(0).float()
            masks.append(mask.cpu())

    return masks


def run_bbox_inference(model, inputs, orig_sizes):
    sigmoid = nn.Sigmoid()
    bboxes = []

    with torch.no_grad():
        preds = model(inputs, for_eval=True)  # (B, 1, H', W')
        for i in range(inputs.size(0)):
            orig_w, orig_h = orig_sizes[i]
            preds_up = F.interpolate(
                preds[i : i + 1],
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            mask = (sigmoid(preds_up) > 0.5).squeeze(0).float()
            pred_bin_mask = mask.cpu().squeeze().numpy().astype(np.uint8)
            initial_image_size = (orig_h, orig_w)
            scales = [
                initial_image_size[0] / pred_bin_mask.shape[0],
                initial_image_size[1] / pred_bin_mask.shape[1],
            ]
            pred_bbox = get_bbox_from_segmentation_labels(
                pred_bin_mask, initial_image_size, scales
            )
            bboxes.append(pred_bbox)

    return bboxes


def benchmark_fn(fn, *fn_args, device="cuda", n_warmup=10, n_iters=50):
    for _ in range(n_warmup):
        _ = fn(*fn_args)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(n_iters)):
        _ = fn(*fn_args)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    return n_iters / total_time  # iterations per second


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    image_paths = [args.img_path] * args.batch_size
    inputs, orig_sizes = preprocess_images(image_paths, device)

    n_iters = 50
    n_warmup = 10

    mask_ips = (
        benchmark_fn(
            run_segmentation_mask_inference,
            model,
            inputs,
            orig_sizes,
            device=device.type,
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        * args.batch_size
    )

    bbox_ips = (
        benchmark_fn(
            run_bbox_inference,
            model,
            inputs,
            orig_sizes,
            device=device.type,
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        * args.batch_size
    )

    print(f"\nBatch size: {args.batch_size}")
    print(f"Image size: {orig_sizes[0]}")
    print(f"Segmentation mask throughput: {mask_ips:.2f} images/sec")
    print(f"Bounding box throughput (end-to-end): {bbox_ips:.2f} images/sec\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Peekaboo inference throughput"
    )

    parser.add_argument(
        "--img-path", type=str, required=True, help="Path to image file"
    )
    parser.add_argument(
        "--model-weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference"
    )
    args = parser.parse_args()
    main(args)
