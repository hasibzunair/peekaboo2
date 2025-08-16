# Code for Peekaboo
# Author: Hasib Zunair

"""Visualize model predictions on video.

Usage
CUDA_VISIBLE_DEVICES=1 python video_demo.py --video-path ./data/examples/videos/person_2.mp4 --output-dir ./outputs/
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T
from torchinfo import summary

from misc import get_bbox_from_segmentation_labels

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Demo of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video-path",
        type=str,
        default="data/examples/video.mp4",
        help="Video path.",
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

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    # Print params
    summary(model, input_size=(1, 3, 224, 224))
    print(f"\n")

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Prepare output video
    video_name = os.path.basename(args.video_path).split(".")[0]
    output_path = os.path.join(args.output_dir, f"{video_name}-peekaboo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    sigmoid = nn.Sigmoid()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end="\r")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            original_size = img.size  # (w, h)

            # Preprocess
            t = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
            img_t = t(img)[None, :, :, :]
            inputs = img_t.to(device)

            # Forward step
            with torch.no_grad():
                preds = model(inputs, for_eval=True)

            orig_h, orig_w = original_size[1], original_size[0]
            preds_up = F.interpolate(
                preds, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()

            # Get segmentation mask
            pred_bin_mask = preds_up.cpu().squeeze().numpy().astype(np.uint8)

            # Check if there is any predicted foreground
            if pred_bin_mask.sum() == 0:
                # No foreground detected, just save the original frame
                out.write(frame)
                continue

            initial_image_size = img.size[::-1]
            scales = [
                initial_image_size[0] / pred_bin_mask.shape[0],
                initial_image_size[1] / pred_bin_mask.shape[1],
            ]

            # Get bounding box for single object discovery
            pred_bbox = get_bbox_from_segmentation_labels(
                pred_bin_mask, initial_image_size, scales
            )

            # Start with the original image
            img_array = np.array(img)
            result_image = img_array.copy()

            # Create overlay for the entire image
            overlay = np.zeros_like(img_array, dtype=np.uint8)

            # Apply red mask where prediction is positive
            mask_indices = pred_bin_mask > 0
            if np.any(mask_indices):
                overlay[mask_indices] = [255, 0, 0]  # Red color

                # Blend the entire image with alpha blending
                alpha = 0.4
                result_image = cv2.addWeighted(img_array, 1, overlay, alpha, 0)

                # Draw bounding box around the mask (calculate from actual mask pixels)
                y_indices, x_indices = np.where(pred_bin_mask)
                if y_indices.size > 0 and x_indices.size > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    cv2.rectangle(
                        result_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2
                    )

            # Convert back to BGR
            result_frame = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            # Write frame to output video
            out.write(result_frame)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nVideo processing completed. Saved to: {output_path}")
