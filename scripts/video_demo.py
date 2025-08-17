# Code for Peekaboo 2
# Author: Hasib Zunair

"""PeekabooSAM2 demo on videos."""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gc
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T
from misc import get_bbox_from_segmentation_labels


NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

### Setup Device ###

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


### Main function ###


def main(args):

    # Detection model configuration
    config, _ = load_config(args.det_model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the detection model
    detection_model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    # Load weights
    detection_model.decoder_load_weights(args.det_model_weights)
    detection_model.eval()
    print(f"Detection model {args.det_model_weights} loaded correctly.")

    # Load tracker predictor
    predictor = build_sam2_video_predictor(
        args.track_model_config, args.track_model_weights, device=device
    )

    # Open input video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video_path}")

    # Get first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    # Init frame rate, w, h, total frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {width}x{height} at {frame_rate} FPS, {total_frames} frames")

    with torch.inference_mode():

        # Convert to PIL for the detection model
        img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        original_size = img.size  # (w, h)

        # Preprocess
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

        # Detection model forward step
        with torch.no_grad():
            preds = detection_model(inputs, for_eval=True)

        sigmoid = nn.Sigmoid()
        orig_h, orig_w = original_size[1], original_size[0]
        preds_up = F.interpolate(
            preds, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
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

        # Init predictor state with the video path
        inference_state = predictor.init_state(video_path=args.video_path)

        # Get box from Peekaboo in (x_min, y_min, x_max, y_max)
        ann_frame_idx = 0
        ann_obj_id = 0
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=pred_bbox,
        )

        # Prepare output video writer
        (
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            if os.path.dirname(args.output_path)
            else None
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_path, fourcc, frame_rate, (width, height))

        # Run propagation throughout the video
        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Reset capture to read frames again for overlay
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Write frames
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = np.zeros_like(frame, dtype=np.uint8)

            if frame_idx in video_segments:
                for _, mask in video_segments[frame_idx].items():
                    mask_2d = np.squeeze(mask)
                    overlay[mask_2d] = (0, 0, 255)

                # Draw bounding box around the mask
                y_indices, x_indices = np.where(mask_2d)
                if y_indices.size > 0 and x_indices.size > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            blended = cv2.addWeighted(frame, 1, overlay, 0.4, 0)
            out.write(blended)

        cap.release()
        out.release()
        print(f"Output saved to {args.output_path}")

    # Cleanup
    del predictor, inference_state
    gc.collect()
    if device.type == "cuda":
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo of Peekaboo 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-path", default="../data/examples/person.mp4", help="Input video path (.mp4)")
    parser.add_argument(
        "--det-model-config",
        type=str,
        default="../configs/peekaboo_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--det-model-weights",
        type=str,
        default="../data/weights/peekaboo_decoder_weights_niter500.pt",
    )
    parser.add_argument(
        "--track-model-weights",
        default="../sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--track-model-config",
        default="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to model config",
    )
    parser.add_argument("--output-path", default="../outputs/person-output.mp4", help="Output video path")
    args = parser.parse_args()

    main(args)
