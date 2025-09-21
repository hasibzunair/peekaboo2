# Code for Peekaboo
# Author: Hasib Zunair

"""Infer Peekaboo model on video feed from webcam"""

import time
import torch
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T
from torchinfo import summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer Peekaboo model on video feed from webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    model.decoder_load_weights(args.model_weights)
    model.eval()
    model.to(device)
    print(f"Model {args.model_weights} loaded correctly.")

    # Print params
    summary(model, input_size=(1, 3, 224, 224))
    print("\n")

    # Predefine transformations and sigmoid
    transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    sigmoid = nn.Sigmoid()

    # Open video feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        exit()
    print("Press 'q' to exit.")

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        img_t = transform(img_t).unsqueeze(0).to(device)

        # Model inference
        with torch.inference_mode():
            preds = model(img_t, for_eval=True)
            print(f"Shape of model output is {preds.shape}")

        # Post process predictions
        h, w = frame.shape[:2]
        # no dynamic adjustments, resize outputs directly to match the frame
        # size instead of scaling with intermediate patch sizes.
        preds_up = F.interpolate(
            preds, size=(h, w), mode="bilinear", align_corners=False
        )
        preds_up = (sigmoid(preds_up) > 0.5).squeeze(0).cpu().numpy()

        # Create mask and blur background
        mask = preds_up.squeeze(0).astype(np.uint8)

        # Blur the background
        blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

        # Create foreground and background
        background = cv2.bitwise_and(
            blurred_frame, blurred_frame, mask=cv2.bitwise_not(mask)
        )
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Combine blurred background with unblurred foreground
        combined = cv2.add(background, foreground)

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Display FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 2
        )

        # Display the output
        cv2.imshow("Peekaboo Predictions from Video Feed", combined)
        print(f"Shape of output frame is {combined.shape}.\n")

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
