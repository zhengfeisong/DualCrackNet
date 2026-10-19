# DualCrackNet

Code for **Robust and Real-time Road Crack Detection through Collaborative Dual-Branch Learning on Robotic Sensing Platform**, IEEE RCAR 2025.

DualCrackNet performs road crack segmentation with collaborative dual-branch learning. The implementation is adapted from HrSegNet; see the attribution in `model/dualcracknet.py`.

## Environment

Recent training and evaluation were run on Linux with Python 3.8, PyTorch 1.11.0+cu113, Albumentations 1.4.18, and an NVIDIA RTX 3090. The commands below provide a pinned installation recipe; the supporting package versions are not a complete export of the original experiment environment.

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

The code also supports a locally prepared DeepCrack dataset. The split used in recent experiments contains 210 training, 90 validation, and 237 test images. DeepCrack is not included in this repository; prepare it separately in the same directory structure below.

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

Use RGB images and binary masks with background 0 and crack 255. Images and masks must have matching filenames/stems and sort order. Each bounding-box text file must share the image stem and contain one line: `x_min y_min width height`, in pixel coordinates.

**Generate bounding boxes before training or evaluation.** The repository includes RCD900 images and masks; the current data loader requires `box/` annotations in all three splits, including the test split. The following command creates one union bounding box enclosing all crack pixels in each mask:

```bash
python - <<'PYBOX'
from pathlib import Path
from bounding_box.make_box import process_folder

root = Path("RCD900")  # Change to Path("Deepcrack") for DeepCrack.
for split in ("train", "val", "test"):
    process_folder(str(root / split / "label"), str(root / split / "box"))
PYBOX
```

Test boxes are required by the shared data loader; evaluation predictions are obtained from the segmentation output without ground-truth box cropping.

## Training

Run commands from the repository root. Train on RCD900:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data_root ./RCD900 \
  --savepath ./runs/rcd900 \
  --device cuda:0 \
  --n_epochs 200 \
  --batch_size 8 \
  --n_workers 4 \
  --lr 0.0005 \
  --print_freq 10 \
  --save_freq 50
```

For DeepCrack, use the same command with `--data_root ./Deepcrack` and `--savepath ./runs/deepcrack`.

Training uses Adam with weight decay 0. The learning rate at zero-based epoch `epoch` is `initial_lr * 0.99**epoch`. The legacy arguments `--weight_decay`, `--gamma`, and `--n_steps` are retained for compatibility but do not affect training.

Each run creates a timestamped directory under `--savepath`:

```text
runs/rcd900/<timestamp>/
├── best_model.pth
├── model_epoch_50.pth
├── model_epoch_100.pth
├── model_epoch_150.pth
└── model_epoch_200.pth
```

Validation runs every five epochs. `best_model.pth` is selected by validation F1. Periodic weights are saved according to `--save_freq`. The model uses the implementation's default `base=64`.

`--resume_model /path/to/model.pth` initializes model weights only; optimizer state and the epoch counter start fresh. Independent runs can produce different results because training does not fix a random seed. Select checkpoints using validation results.

## Evaluation

Evaluate one checkpoint on the test split and export predictions:

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate/evaluate.py \
  --data-root ./RCD900 \
  --checkpoint ./runs/rcd900/<timestamp>/best_model.pth \
  --device cuda:0
```

Replace `<timestamp>` with the actual run directory name. To evaluate every `.pth` or `.pt` file directly inside a run directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate/evaluate.py \
  --data-root ./RCD900 \
  --checkpoint ./runs/rcd900/<timestamp> \
  --device cuda:0
```

For DeepCrack, replace the dataset and checkpoint paths with `./Deepcrack` and the corresponding training output. Optional arguments include `--output-dir ./evaluation_results`, `--workers 0`, and `--threshold 0.5`.

By default, each checkpoint produces a directory next to the weight file:

```text
<run directory>/
├── best_model.pth
├── best_model/
│   ├── predictions/            # Binary PNG masks: 0 or 255
│   └── metrics.json
└── test_summary.json           # Summary for all evaluated checkpoints
```

Evaluation uses batch size 1, the original image size, and a default probability threshold of 0.5. Precision, recall, accuracy, F1, crack IoU, and mean crack/background IoU are calculated from the accumulated test-set confusion matrix. JSON files contain both fractional metrics and percentage metrics. Existing results at the same output paths are overwritten.
