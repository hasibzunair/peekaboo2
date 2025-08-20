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

## 4. Citation

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
