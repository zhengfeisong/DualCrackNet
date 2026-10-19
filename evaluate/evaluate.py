"""Evaluate checkpoints and export binary predictions without changing inference."""
import argparse
import json
from pathlib import Path
import sys

# Support both direct script execution and python -m evaluate.evaluate.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from evaluate.binary_confusion_matrix import get_binary_confusion_matrix
from evaluate.calculate_metric import calculate_metrics_from_counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, help='Dataset containing test/image, test/label and test/box')
    parser.add_argument('--checkpoint', required=True, help='A .pth/.pt file or a directory of checkpoints')
    parser.add_argument('--output-dir', help='Default: <checkpoint directory>/<checkpoint stem>/')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    return args


def evaluate(data_root, checkpoint, output_dir=None, device='cpu', workers=0, threshold=0.5):
    from dataset import CrackDataset
    from model.dualcracknet import DualCrackNet

    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.is_file():
        weights, weight_dir = [checkpoint], checkpoint.parent
    elif checkpoint.is_dir():
        weights = sorted(p for p in checkpoint.iterdir() if p.is_file() and p.suffix.lower() in ('.pth', '.pt'))
        weight_dir = checkpoint
    else:
        raise FileNotFoundError(f'Checkpoint path does not exist: {checkpoint}')
    if not weights:
        raise ValueError(f'No .pth or .pt checkpoints found in {checkpoint}')
    if len({p.stem for p in weights}) != len(weights):
        raise ValueError('Checkpoint stems must be unique to avoid overwriting outputs')

    output_root = Path(output_dir).expanduser().resolve() if output_dir else weight_dir
    device = torch.device(device)
    dataset = CrackDataset(str(Path(data_root).expanduser().resolve()), 'test')
    if not len(dataset):
        raise ValueError('The test split is empty')
    names = [Path(dataset.dataset.get_image_filename(i)).stem for i in range(len(dataset))]
    if len(set(names)) != len(names):
        raise ValueError('Test image stems must be unique to avoid overwriting PNG files')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=device.type == 'cuda')
    print(f'Device: {device}; test images: {len(dataset)}; checkpoints: {len(weights)}', flush=True)

    summary = {}
    for path in weights:
        # Original training saves raw state dictionaries; also accept a wrapped model.
        try:
            saved = torch.load(str(path), map_location='cpu', weights_only=True)
        except TypeError:
            saved = torch.load(str(path), map_location='cpu')  # Older PyTorch releases.
        state = saved['model'] if 'model' in saved else saved
        base = int(state['stage2.0.weight'].shape[0])
        model = DualCrackNet(base=base).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        destination = output_root / path.stem
        predictions = destination / 'predictions'
        predictions.mkdir(parents=True, exist_ok=True)
        counts = torch.zeros(4, dtype=torch.int64)  # TP, FP, TN, FN
        with torch.no_grad():
            for index, batch in enumerate(loader):
                image = batch['image'].to(device=device, dtype=torch.float32)
                target = batch['mask'].to(device=device, dtype=torch.long)
                probability = model(image)[0]  # Original eval path; no box or boundary crop.
                if probability.shape != target.shape:
                    raise ValueError(f'Prediction/label shape mismatch: {names[index]}')
                current = get_binary_confusion_matrix(
                    probability, target, device=device, threshold=threshold,
                    reduction='sum', a=1)
                counts += torch.stack(current).detach().to(device='cpu', dtype=torch.int64)
                mask = (probability[0, 0].cpu().numpy() >= threshold).astype(np.uint8) * 255
                Image.fromarray(mask).save(str(predictions / f'{names[index]}.png'))
                if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                    print(f'{path.name}: {index + 1}/{len(dataset)}', flush=True)

        values = calculate_metrics_from_counts(*counts.unbind())
        metrics = dict(zip(('accuracy', 'recall', 'precision', 'f1', 'iou', 'mean_iou'), values))
        tp, fp, tn, fn = counts.tolist()
        result = {
            'checkpoint': str(path), 'data_root': str(Path(data_root).expanduser().resolve()),
            'test_images': len(dataset), 'base': base, 'threshold': threshold,
            'aggregation': 'global test-set confusion matrix',
            'metrics': metrics, 'metrics_percent': {k: v * 100 for k, v in metrics.items()},
            'confusion_matrix': [[tn, fp], [fn, tp]],
        }
        (destination / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        summary[path.name] = result
        print(f'\n{path.name}\n{json.dumps(result["metrics_percent"], indent=2)}\nOutput: {destination}', flush=True)
        del model

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'test_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.data_root, args.checkpoint, args.output_dir,
             args.device, args.workers, args.threshold)
