# DualCrackNet

Code for **Robust and Real-time Road Crack Detection through Collaborative Dual-Branch Learning on Robotic Sensing Platform**, IEEE RCAR 2025.

## Environment

Training and evaluation can be run on Linux with Python 3.8, PyTorch 1.11.0+cu113, Albumentations 1.4.18, and an NVIDIA RTX 3090.

```bash
git clone https://github.com/zhengfeisong/DualCrackNet.git
cd DualCrackNet

conda create -n dualcracknet python=3.8 -y
conda activate dualcracknet

python -m pip install "pip<25.1"
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install \
  numpy==1.24.4 Pillow==10.4.0 \
  scipy==1.10.1 scikit-image==0.21.0 scikit-learn==1.3.2 \
  opencv-python-headless==4.10.0.84 \
  albumentations==1.4.18 albucore==0.0.17 \
  pandas==2.0.3 matplotlib==3.7.5

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available())"
python train.py --help
python evaluate/evaluate.py --help
```

The PyTorch/CUDA package pairing follows the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). A CUDA-capable NVIDIA driver is required for GPU training.

## Datasets

### RCD900

Road Crack with Disturbance 900 (RCD900) contains 900 road images at 320 × 320 resolution with pixel-level crack masks. Images were collected at the Jiading Campus of Tongji University and in urban environments. Disturbances include zebra crossings, traffic markings, road boundaries, occlusions, and varying illumination and weather conditions.

The dataset is included in `RCD900/`, with 630 training images, 135 validation images, and 135 test images.

### DeepCrack

[DeepCrack dataset](https://github.com/yhlleo/DeepCrack/tree/master/dataset)

### Directory structure and bounding-box annotations

```text
RCD900/                       # Or Deepcrack/
├── train/
│   ├── image/
│   ├── label/
│   └── box/
├── val/
│   ├── image/
│   ├── label/
│   └── box/
└── test/
    ├── image/
    ├── label/
    └── box/
```

**Generate bounding boxes before training or evaluation.**

```bash
python - <<'PYBOX'
from pathlib import Path
from bounding_box.make_box import process_folder

root = Path("RCD900")  # Change to Path("Deepcrack") for DeepCrack.
for split in ("train", "val", "test"):
    process_folder(str(root / split / "label"), str(root / split / "box"))
PYBOX
```

## Training

Run from the repository root. Set `datapath` and `savepath` to your dataset and output directories.

```bash
datapath=/path/to/dataset
savepath=/path/to/output

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data_root "$datapath" \
  --savepath "$savepath" \
  --device cuda:0 \
  --n_epochs 200 \
  --batch_size 8 \
  --n_workers 4 \
  --lr 0.0005 \
  --print_freq 10 \
  --save_freq 50
```

## Evaluation

Set `checkpoint` to a weight file or a directory containing multiple weights.

```bash
datapath=/path/to/dataset
checkpoint=/path/to/checkpoint_or_directory

CUDA_VISIBLE_DEVICES=0 python -u evaluate/evaluate.py \
  --data-root "$datapath" \
  --checkpoint "$checkpoint" \
  --device cuda:0
```
