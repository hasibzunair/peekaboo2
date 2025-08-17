# Peekaboo 2: Unsupervised Object Localization in Videos

This is official code for our work:<br>
Peekaboo 2: Hiding Parts of an Image for Unsupervised Object Localization in Videos
<br>

https://github.com/user-attachments/assets/5ff39c8c-5503-47bd-be17-71183b03d35b

## Updates

- \[2025.08.16\] Create demo for Peekaboo 2 on custom video.

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

## 2. Demo on custom video

SAM 2.1 Checkpoint Download

```
cd sam2/checkpoints
./download_ckpts.sh
cd ../..
```

To run the demo with your custom video, 

```
cd scripts/
python vide_demo.py --video-path ../data/examples/videos/person_2.mp4 --output-path ../outputs/output.mp4

python image_demo.py --image-path ../data/examples/octopus.jpeg --output-path ../outputs/octpous-peekaboo2.png
```

## 3. Demo in image

This bascially run an image through the Peekaboo model to discover the object in the image, and then refines the output mask using SAM2.

```
python image_demo.py --image-path ../data/examples/octopus.jpeg --output-path ../outputs/octpous-peekaboo2.png
```

## Acknowledgements

This repository was built on top of [Peekaboo](https://github.com/hasibzunair/peekaboo) and [SAM](https://github.com/facebookresearch/sam2). Consider acknowledging these projects.
