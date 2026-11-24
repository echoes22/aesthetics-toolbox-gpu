"""Command-line interface: ``qip-machine IMAGE_FOLDER [-o qips.csv]``."""

import argparse
import os
import sys

from . import __version__

GPU_MEMORY_PER_WORKER = 1.5e9  # bytes; a worker typically uses ~0.7 GB, more for very large images
MAX_AUTO_WORKERS = 32


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))  # respects cgroups / cluster allocations
    except AttributeError:
        return os.cpu_count() or 1


def build_parser():
    parser = argparse.ArgumentParser(
        prog='qip-machine',
        description="Compute quantitative image properties (QIPs) of the Aesthetics Toolbox "
                    "for every image in a folder and write them to a CSV file.",
        epilog="Example: qip-machine paintings/ -o paintings_qips.csv",
    )
    parser.add_argument('input', nargs='?', help="folder containing the images (searched recursively)")
    parser.add_argument('-o', '--output', metavar='CSV',
                        help="output CSV file (default: <folder name>_qips.csv in the current directory)")
    parser.add_argument('--qips', nargs='+', metavar='NAME', help="compute only these QIPs (see --list-qips)")
    parser.add_argument('--exclude', nargs='+', metavar='NAME', help="compute all QIPs except these")
    parser.add_argument('--short-side', type=int, metavar='PIXELS',
                        help="first downscale images whose shorter side is longer than PIXELS "
                             "(much faster for large images, but resolution-dependent QIPs change; "
                             "default: full resolution, as in the Aesthetics Toolbox)")
    parser.add_argument('--device', default='auto',
                        help="'auto' (default: CUDA GPU if available), 'cuda', 'cuda:N' or 'cpu'")
    parser.add_argument('--workers', default='auto', metavar='N',
                        help="number of images processed in parallel (default: auto)")
    parser.add_argument('--shard', metavar='I/N',
                        help="process only part I of N of the image list (I = 0..N-1), e.g. for cluster job arrays")
    parser.add_argument('--toolbox-names', action='store_true',
                        help="use the column names of the Aesthetics Toolbox QIP machine")
    parser.add_argument('--digits', type=int, default=6, metavar='N',
                        help="significant digits written to the CSV (default: 6)")
    parser.add_argument('--overwrite', action='store_true',
                        help="replace an existing output file instead of resuming it")
    parser.add_argument('--list-qips', action='store_true', help="list the available QIPs and exit")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return parser


def _parse_shard(text, parser):
    try:
        index, total = (int(x) for x in text.split('/'))
    except ValueError:
        parser.error(f"--shard must look like I/N (e.g. 0/4), got '{text}'")
    if not 0 <= index < total:
        parser.error(f"--shard {text}: I must be between 0 and N-1")
    return index, total


def _auto_workers(device, n_cpus):
    import torch

    """One worker per CPU core, as long as the GPU has memory for them."""
    if device.type == 'cpu':
        return min(n_cpus, MAX_AUTO_WORKERS)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    return max(1, min(n_cpus, MAX_AUTO_WORKERS, int(gpu_memory // GPU_MEMORY_PER_WORKER)))


def list_qips():
    from .pipeline import QIPS

    width = max(len(q.name) for q in QIPS)
    print(f"{'NAME':<{width}}  DESCRIPTION")
    for q in QIPS:
        columns = f" [{', '.join(q.columns)}]" if len(q.columns) > 1 else ''
        print(f"{q.name:<{width}}  {q.description}{columns}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_qips:
        list_qips()
        return 0
    if not args.input:
        parser.error("the image folder is required")
    if args.short_side is not None and args.short_side < 1:
        parser.error("--short-side must be a positive number of pixels")
    shard = _parse_shard(args.shard, parser) if args.shard else None

    output = args.output
    if output is None:
        name = os.path.basename(os.path.normpath(os.path.abspath(args.input))) or 'images'
        output = f"{name}_qips" + (f"_shard{shard[0]}of{shard[1]}" if shard else '') + '.csv'

    from .pipeline import QIP_BY_NAME, resolve_device

    for name in (args.qips or []) + (args.exclude or []):
        if name not in QIP_BY_NAME:
            parser.error(f"unknown QIP '{name}' (see --list-qips)")
    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    n_cpus = available_cpus()
    if args.workers == 'auto':
        workers = _auto_workers(device, n_cpus)
    else:
        try:
            workers = int(args.workers)
        except ValueError:
            parser.error("--workers must be a number or 'auto'")
        if workers < 1:
            parser.error("--workers must be at least 1")

    if workers > 1:
        # Images are processed in parallel, so each worker process (which imports
        # NumPy after being spawned) uses a single thread.
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(var, '1')

    from .runner import run

    failed = run(args.input, output, qips=args.qips, exclude=args.exclude, device=str(device),
                 workers=workers, short_side=args.short_side, shard=shard,
                 toolbox_names=args.toolbox_names, digits=args.digits, overwrite=args.overwrite)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
