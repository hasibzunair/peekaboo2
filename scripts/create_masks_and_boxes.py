# Code for Peekaboo 2
# Author: Hasib Zunair

"""Create binary masks and bounding boxes using Peekaboo 2 on multiple images in a folder."""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gc
import csv
from pathlib import Path
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


def process_single_image(image_path, detection_model, predictor, device):
    """Process a single image and return mask and bounding box."""
    
    # Read image with OpenCV
    input_image = cv2.imread(str(image_path))
    if input_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = input_image.shape[:2]

    with torch.inference_mode():
        # Convert to PIL for the detection model
        img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        original_size = img.size  # (w, h)

        # Preprocess
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

        # Detection model forward step
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

        # Convert image to RGB
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Set image in SAM2
        predictor.set_image(image_rgb)

        # Use the bounding box from Peekaboo to refine with SAM2
        input_box = np.array(pred_bbox)

        # Get refined mask from SAM2 without negative keypoints
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]

        # Generate random negative keypoints outside the bounding box
        # TODO: need to investiage further if this helps improve segmentation,
        # currently seeing some noisy masks
        # num_neg_points = np.random.randint(5, 11)
        # x_min, y_min, x_max, y_max = input_box
        # neg_points = []

        # for _ in range(num_neg_points * 5):
        #     x_rand = np.random.randint(0, width)
        #     y_rand = np.random.randint(0, height)

        #     # keep only points *outside* the bbox
        #     if not (x_min <= x_rand <= x_max and y_min <= y_rand <= y_max):
        #         neg_points.append([x_rand, y_rand])
            
        #     if len(neg_points) >= num_neg_points:
        #         break

        # if len(neg_points) == 0:
        #     print("Warning: No negative points generated, defaulting to box only.")
        #     input_point = None
        #     input_label = None
        # else:
        #     input_point = np.array(neg_points)
        #     input_label = np.zeros(len(neg_points), dtype=int)

        # # Get refined mask from SAM2 with negative keypoints
        # masks, scores, _ = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     box=input_box[None, :],
        #     multimask_output=True,
        # )
        # sorted_ind = np.argsort(scores)[::-1]
        # masks = masks[sorted_ind]

        # Get the best mask
        refined_mask = masks[0]
        refined_mask = refined_mask.astype(np.uint8)

        return refined_mask, pred_bbox, input_image


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
    predictor = SAM2ImagePredictor(
        build_sam2(args.track_model_config, args.track_model_weights, device=device)
    )

    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create visualizations folder if requested
    vis_folder = output_folder / "visualizations"
    if args.save_vis:
        vis_folder.mkdir(parents=True, exist_ok=True)

    # Get all JPG images from input folder
    input_folder = Path(args.input_folder)
    image_paths = sorted(list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.JPG")))
    
    if not image_paths:
        print(f"No JPG images found in {args.input_folder}")
        return

    # TODO: Remove slice to process all images
    #image_paths = image_paths[:100]
    print(f"Processing {len(image_paths)} images...")

    # Prepare CSV file
    csv_path = output_folder / "results.csv"
    csv_data = []

    # Process each image
    for idx, image_path in enumerate(image_paths):
        try:
            print(f"\nProcessing [{idx+1}/{len(image_paths)}]: {image_path.name}")
            
            # Process image
            mask, bbox, input_image = process_single_image(image_path, detection_model, predictor, device)
            
            # Save mask
            mask_filename = f"{image_path.stem}_mask.png"
            mask_path = output_folder / mask_filename
            mask_vis = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_vis)

            # Save visualization (side by side)
            if args.save_vis:
                combined = np.hstack((input_image, cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)))
                vis_path = vis_folder / f"{image_path.stem}_vis.png"
                cv2.imwrite(str(vis_path), combined)
            
            # Store results for CSV
            x_min, y_min, x_max, y_max = bbox
            csv_data.append({
                'input_filename': image_path.name,
                'output_mask_filename': mask_filename,
                'bbox_x_min': x_min,
                'bbox_y_min': y_min,
                'bbox_x_max': x_max,
                'bbox_y_max': y_max
            })
            
            print(f"Saved mask to {mask_filename}")
            if args.save_vis:
                print(f"Saved visualization to {vis_path.name}")
            print(f"Bounding box: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            csv_data.append({
                'input_filename': image_path.name,
                'output_mask_filename': 'ERROR',
                'bbox_x_min': '',
                'bbox_y_min': '',
                'bbox_x_max': '',
                'bbox_y_max': ''
            })

    # Write CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['input_filename', 'output_mask_filename', 
                     'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n")
    print(f"Processed: {len(image_paths)} images")
    print(f"Results CSV: {csv_path}")
    print(f"Output folder: {output_folder}")
    print(f"Done!")

    # Cleanup
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch processing of images with Peekaboo + SAM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Input folder containing JPG images",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for masks and CSV",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to process (0 for all images)",
    )
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
    parser.add_argument(
        "--save-vis", action="store_true", 
        help="If set, save input and mask side by side"
    )
    args = parser.parse_args()

    main(args)