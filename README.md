# Peekaboo 2

[[`Project`](https://hasibzunair.github.io/peekaboo2/)]

This is official code for:<br>
PEEKABOO2: Adapting Peekaboo with Segment Anything Model for Unsupervised Object Localization in Images and Videos
<br>

https://github.com/user-attachments/assets/f2db4e19-4dc3-40fa-a18a-e037852fffbb

## Updates

- \[2025.08.19\] Release demo scripts for inference of Peekaboo 2 on images and videos. see [scripts](https://github.com/hasibzunair/peekaboo2/tree/main/scripts), see demos in [project page](https://hasibzunair.github.io/peekaboo2/)
- \[2025.08.02\] Create inference pipeline combining Peekaboo and Segment Anything 2 (SAM2) for videos. built on top of source code of [github.com/hasibzunair/peekaboo](https://github.com/hasibzunair/peekaboo)

## 1. Specification of dependencies

This code requires Python 3.10 and CUDA 12.4. Clone the project repository, then create a fresh environment and install the project requirements inside that environment by:

```bash
cd sam2
conda create -n peekaboo2 python=3.10
conda activate peekaboo2
pip install -e .
pip install -e ".[notebooks]"
cd ..
pip install -r requirements.txt
```

## 2. Demo on video

SAM 2.1 Checkpoint Download

```
cd sam2/checkpoints
./download_ckpts.sh
cd ../..
```

To run the demo with a video,

```bash
cd scripts/
python video_demo.py --video-path ../data/examples/person.mp4 --output-path ../outputs/person-peekaboo2.mp4
```

## 3. Demo on image

This runs the demo with an image,

```bash
cd scripts/
python image_demo.py --image-path ../data/examples/octopus.jpeg --output-path ../outputs/octpous-peekaboo2.png
```

## 4. Automatic Labeling

```bash
cd scripts/

# visualize mask and bounding box for a single image
python create_mask.py --image-path ../data/examples/car.jpg --output-path ../outputs/result.png --save-mask
```

which will save an image [result.png](./data/examples/result.png) with the mask and box as well as [result_mask.png](./data/examples/result_mask.png) with the binary segmentation mask and output like:

```bash
Using device: cuda
hyperparameters: model={'arch': 'vit_small', 'patch_size': 8, 'pre_training': 'dino'}, peekaboo={'feats': 'k'}, training={'dataset': 'DUTS-TR', 'dataset_set': None, 'seed': 0, 'max_iter': 500, 'nb_epochs': 3, 'batch_size': 50, 'lr0': 0.05, 'step_lr_size': 50, 'step_lr_gamma': 0.95, 'crop_size': 224, 'scale_range': [0.1, 3.0], 'photometric_aug': 'gaussian_blur', 'proba_photometric_aug': 0.5, 'cropping_strategy': 'random_scale'}, evaluation={'type': 'saliency', 'datasets': ['DUT-OMRON', 'ECSSD'], 'freq': 50}
Loading model from weights ../data/weights/peekaboo_decoder_weights_niter500.pt.
Detection model ../data/weights/peekaboo_decoder_weights_niter500.pt loaded correctly.
Image loaded: 1920x1080
Predicted bounding box: [ 633  244 1263  959]
Output saved to ../outputs/result.png
[  0 255]
Mask saved to ../outputs/result_mask.png
```

To automatically label a folder of images, run:
```bash
cd scripts/

# save binary masks and bounding boxes
python create_masks_and_boxes.py --input-folder PATH_TO_IMAGES --output-folder OUTPUT_PATH

# save binary masks and bounding boxes and visualize images and masks (do NOT use this when running inference on large datasets)
python create_masks_and_boxes.py --input-folder PATH_TO_IMAGES --output-folder OUTPUT_PATH --save-vis
```

This will create and save the binary masks in a folder as well as create a CSV file which stores the filenames and the bounding boxes.

## 5. Citation

```bibtex
@misc{HasibGitHub,
author = {Zunair, Hasib},
booktitle = {GitHub},
title = {PEEKABOO2: Adapting Peekaboo with Segment Anything Model for Unsupervised Object Localization in Images and Videos},
url = {https://github.com/hasibzunair/peekaboo2},
year = {2025}
}
```

## Acknowledgements

This work was built on top of [Peekaboo](https://github.com/hasibzunair/peekaboo) and [Segment Anything 2](https://github.com/facebookresearch/sam2). Consider acknowledging these projects.
