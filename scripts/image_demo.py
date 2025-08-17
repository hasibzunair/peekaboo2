# Code for Peekaboo. 2
# Author: Hasib Zunair

"""PeekabooSAM2 demo on an image."""

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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

    # Load SAM2 predictor (for image inference)
    predictor = SAM2ImagePredictor(build_sam2(args.track_model_config, args.track_model_weights, device=device))

    # Load input image
    if not os.path.exists(args.image_path):
        raise ValueError(f"Image not found: {args.image_path}")
    
    # Read image with OpenCV
    input_image = cv2.imread(args.image_path)
    if input_image is None:
        raise ValueError(f"Could not read image: {args.image_path}")
    
    height, width = input_image.shape[:2]
    print(f"Image loaded: {width}x{height}")

    with torch.inference_mode():

        # Convert to PIL for the detection model
        img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
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

        # Convert image to RGB
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Set image in SAM2
        predictor.set_image(image_rgb)
        
        # Use the bounding box from Peekaboo to refine with SAM2
        # pred_bbox is in format (x_min, y_min, x_max, y_max)
        input_box = np.array(pred_bbox)
        
        # Get refined mask from SAM2
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # Get the best mask
        refined_mask = masks[0]
        refined_mask = refined_mask.astype(bool)

        # Create output image with overlay
        output_image = input_image.copy()
        
        # Create overlay for the refined mask
        overlay = np.zeros_like(output_image, dtype=np.uint8)
        overlay[refined_mask] = (0, 0, 255)
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = pred_bbox
        cv2.rectangle(output_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        
        # Blend the overlay with the original image
        blended = cv2.addWeighted(output_image, 1, overlay, 0.4, 0)
        
        # Save
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None
        cv2.imwrite(args.output_path, blended)
        print(f"Output saved to {args.output_path}")
        
        # Optionally save just the mask
        if args.save_mask:
            mask_path = args.output_path.replace('.jpg', '_mask.jpg').replace('.png', '_mask.png')
            mask_vis = (refined_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_vis)
            print(f"Mask saved to {mask_path}")

    # Cleanup
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo of Peekaboo + SAM2 for image inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-path", default="../data/examples/octopus.jpeg", help="Input image path (.jpg, .png, etc.)")
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
        help="Path to SAM2 model checkpoint",
    )
    parser.add_argument(
        "--track-model-config",
        default="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 model config",
    )
    parser.add_argument("--output-path", default="output.jpg", help="Output image path")
    parser.add_argument("--save-mask", action="store_true", help="Save the binary mask separately")
    args = parser.parse_args()

    main(args)