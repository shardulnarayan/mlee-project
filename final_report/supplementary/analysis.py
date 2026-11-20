"""New camera-fault and quantitative Grad-CAM evaluations."""

import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .cnn import ManifestImages, load_model, val_test_transforms
from .evidence import CLASSES, manifest, scores, sha256, write_csv, write_json
from .pixel_faults import corruption_cases, corrupt_pixels


def camera_faults(args):
    output = Path(args.output)
    rows = [r for r in manifest() if r['split'] == 'test']
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    result, prediction_rows = [], []
    for variant in args.variants:
        dataset = ManifestImages(rows, variant, output)
        for seed in args.seeds:
            model, meta = load_model(output, variant, seed, device)
            for kind, severity in corruption_cases():
                noise_seeds = args.noise_seeds if kind == 'normalized_gaussian_sigma' else [None]
                for noise_seed in noise_seeds:
                    truth, predicted = [], []
                    for start in range(0, len(rows), args.batch_size):
                        tensors = []
                        for index in range(start, min(start + args.batch_size, len(rows))):
                            image = corrupt_pixels(dataset.image(index), kind, severity)
                            tensor = val_test_transforms(image)
                            if kind == 'normalized_gaussian_sigma':
                                # Same standard-normal field for matching image indices across variants/models.
                                generator = torch.Generator().manual_seed(1_000_003 * noise_seed + index)
                                tensor = tensor + float(severity) * torch.randn(tensor.shape, generator=generator)
                            tensors.append(tensor)
                        with torch.inference_mode():
                            indices = model(torch.stack(tensors).to(device)).argmax(1).cpu().tolist()
                        predicted.extend(CLASSES[i] for i in indices)
                        truth.extend(r['class'] for r in rows[start:start + len(indices)])
                    summary, per_class = scores(truth, predicted)
                    item = dict(variant=variant, model_seed=seed, fault=kind, severity=severity,
                                noise_seed='' if noise_seed is None else noise_seed,
                                corruption_stage='after input representation, at 224x224; normalized noise after normalization',
                                checkpoint_sha256=meta['checkpoint_sha256'], **summary)
                    item['leaf_miner_recall'] = next(r['recall'] for r in per_class if r['class_name'] == 'Leaf_Miner')
                    item['healthy_recall'] = next(r['recall'] for r in per_class if r['class_name'] == 'Healthy')
                    result.append(item)
                    for row, y in zip(rows, predicted):
                        prediction_rows.append(dict(variant=variant, model_seed=seed, fault=kind,
                                                    severity=severity, noise_seed=item['noise_seed'],
                                                    filename=row['filename'], group=row['group'],
                                                    true_class=row['class'], predicted_class=y))
                    print(f'{variant}, seed {seed}, {kind}={severity}, noise seed {noise_seed}: {summary}', flush=True)
                    write_csv(output / 'camera_faults.csv', result)
                    write_csv(output / 'camera_fault_predictions.csv', prediction_rows)
    pooled = []
    keys = sorted({(r['variant'], r['fault'], r['severity']) for r in result})
    for variant, kind, severity in keys:
        selected = [r for r in result if (r['variant'], r['fault'], r['severity']) == (variant, kind, severity)]
        row = dict(variant=variant, fault=kind, severity=severity, evaluation_records=len(selected),
                   pooling='first average noise seeds per model, then mean/sample SD across models')
        model_seeds = sorted({r['model_seed'] for r in selected})
        row['model_runs'] = len(model_seeds)
        for metric in ('accuracy', 'macro_f1', 'leaf_miner_recall', 'healthy_recall'):
            values = [statistics.mean(r[metric] for r in selected if r['model_seed'] == seed)
                      for seed in model_seeds]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_sample_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        pooled.append(row)
    write_csv(output / 'camera_faults_summary.csv', pooled)
    write_json(output / 'camera_protocol.json', dict(
        implementation='Annex C supplementary camera-fault evaluation',
        cases=corruption_cases(), noise_seeds=args.noise_seeds,
        jpeg_encoder=f'Pillow {Image.__version__}, subsampling=2', blur='Pillow GaussianBlur radius',
        exposure_proxy='Pillow brightness multiplication, not a physical exposure model',
        noise='additive, unclipped, normalized tensor; not sensor-space noise',
        corruption_order='clean representation -> resize -> pixel fault -> normalize -> optional normalized noise',
        script_sha256=sha256(__file__)))


def foreground_mask(image, external=None):
    """Estimate foreground in model coordinates without consulting the class."""
    rgb = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
    if external is not None:
        with Image.open(external) as handle:
            # Manual masks must already use the same model-input coordinates.
            if handle.size != (224, 224):
                raise ValueError('External attention masks must be 224x224 model-coordinate masks')
            return np.asarray(handle.convert('L')) > 127, 'external'
    from src.dip.pipeline import hsv_leaf_mask
    mask = hsv_leaf_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), np.full((224, 224), 255, np.uint8)) > 0
    return mask, 'HSV foreground proxy (not ground truth)'


def gradcam_map(model, tensor, class_index):
    captured = []
    def save_activation(_module, _inputs, output):
        captured.append(output)
        output.retain_grad()
    hook = model.features[-1].register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.unsqueeze(0).requires_grad_(True))
        logits[0, class_index].backward()
        activation = captured[0]
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0].detach().cpu().numpy()
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    finally:
        hook.remove()


def attention(args):
    output = Path(args.output)
    rows = [r for r in manifest() if r['split'] == 'test']
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    result = []
    for variant in args.variants:
        dataset = ManifestImages(rows, variant, output)
        for seed in args.seeds:
            model, meta = load_model(output, variant, seed, device)
            for index, row in enumerate(rows):
                image = dataset.image(index)
                external = None
                if args.mask_dir:
                    external = Path(args.mask_dir) / variant / row['class'] / (row['filename'] + '.png')
                    if not external.is_file():
                        raise FileNotFoundError(f'Missing externally supplied mask: {external}')
                mask, method = foreground_mask(image, external)
                fraction = float(mask.mean())
                item = dict(variant=variant, seed=seed, filename=row['filename'], group=row['group'],
                            class_name=row['class'], target='true class', mask_method=method,
                            mask_area=fraction, attention_share='', status='ok',
                            checkpoint_sha256=meta['checkpoint_sha256'])
                if not 0.10 <= fraction <= 0.85:
                    item['status'] = 'excluded: mask coverage outside [0.10, 0.85]'
                else:
                    cam = gradcam_map(model, val_test_transforms(image).to(device), CLASSES.index(row['class']))
                    total = float(cam.sum())
                    if total <= 0 or not np.isfinite(cam).all():
                        item['status'] = 'excluded: zero/nonfinite Grad-CAM mass'
                    else:
                        item['attention_share'] = float(cam[mask].sum() / total)
                result.append(item)
            print(f'Attention complete: {variant}, seed {seed}', flush=True)
            write_csv(output / 'attention_per_image.csv', result)
    write_json(output / 'attention_protocol.json', dict(
        target='true class', layer='MobileNetV2 features[-1]',
        formula='sum(ReLU Grad-CAM inside foreground mask) / sum(ReLU Grad-CAM)',
        mask='external 224x224 masks' if args.mask_dir else 'HSV foreground proxy from model input, not manual annotation',
        mask_dir=str(args.mask_dir) if args.mask_dir else None,
        accepted_mask_coverage=[0.10, 0.85], zero_mass_policy='exclude and count, never impute',
        comparability='compare attention shares only with matching mask definitions and eligibility criteria',
        script_sha256=sha256(__file__)))
