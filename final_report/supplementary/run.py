"""CLI: python -m final_report.supplementary.run --help."""

import argparse
import importlib.util
from pathlib import Path

from .evidence import ROOT, VARIANTS, assemble_tables, check_inputs


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'train', 'attention', 'corruptions', 'tables', 'all'])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--noise-seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=positive, default=10)
    p.add_argument('--batch-size', type=positive, default=16)
    p.add_argument('--threads', type=positive, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', type=Path, default=ROOT / 'results/supplementary/annex_c_v1')
    p.add_argument('--mask-dir', type=Path, help='Optional masks: VARIANT/CLASS/FILENAME.png, including original extension')
    p.add_argument('--resume', action='store_true', help='Reuse only hash-verified checkpoints with identical recipe')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if any(s < 0 or s >= 2**32 for s in args.seeds + args.noise_seeds):
        raise SystemExit('Seeds must be in [0, 2**32).')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise SystemExit('Duplicate seeds/variants would bias aggregation.')
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise SystemExit('Duplicate noise seeds would bias aggregation.')
    if args.command == 'tables':
        for item in assemble_tables(args.output):
            print(item)
        return
    report = check_inputs(args.variants)
    print(report, flush=True)
    packages = ['torch', 'torchvision', 'cv2', 'numpy', 'matplotlib']
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    print('Missing ML dependencies:', missing or 'none', flush=True)
    if args.command == 'check':
        if report['missing_image_count'] or missing:
            raise SystemExit(1)
        return
    if report['missing_image_count'] or missing:
        raise SystemExit('Missing inputs/dependencies. Activate the project environment and run check first.')
    from .cnn import train_runs
    from .analysis import attention, camera_faults
    if args.command in ('train', 'all'):
        train_runs(args)
    if args.command in ('attention', 'all'):
        attention(args)
    if args.command in ('corruptions', 'all'):
        camera_faults(args)
    for item in assemble_tables(args.output):
        print(item)


if __name__ == '__main__':
    main()
