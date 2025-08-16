# Peekaboo 2: Unsupervised Object Localization in Videos

This is official code for our work:<br>
Peekaboo 2: Hiding Parts of an Image for Unsupervised Object Localization in Videos
<br>

ADD_WRITEUP

ADD_VIDEO

## Updates

- \[2024.07.19\] Create demo for Peekaboo 2 on custom video.

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
python demo.py --video-path ../data/examples/videos/person_2.mp4 --output-path ../outputs/output.mp4
```

## Acknowledgements

This repository was built on top of [Peekaboo](https://github.com/hasibzunair/peekaboo) and [SAM](https://github.com/facebookresearch/sam2). Consider acknowledging these projects.
