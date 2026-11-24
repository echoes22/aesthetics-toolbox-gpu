"""Batch processing of an image folder into a CSV file."""

import csv
import logging
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import torch
from tqdm import tqdm

from .image_io import find_images
from .pipeline import QIPExtractor, resolve_device, select_qips

log = logging.getLogger('qipmachine')

_extractor = None  # one per worker process


def _init_worker(extractor_kwargs):
    global _extractor
    torch.set_num_threads(1)
    _extractor = QIPExtractor(**extractor_kwargs)


def _process(task, extractor=None):
    rel_path, path = task
    try:
        values, errors = (extractor or _extractor).compute_with_errors(path)
        return rel_path, values, errors, None
    except Exception as exc:
        return rel_path, None, [], f"{type(exc).__name__}: {exc}"


def format_value(value, digits):
    """Number as CSV text: empty for NaN/inf, integers without decimals, else ``digits`` significant digits."""
    if not math.isfinite(value):
        return ''
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f'{value:.{digits}g}'


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


def _setup_logging(log_path):
    log.setLevel(logging.DEBUG)
    log.propagate = False
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S'))
    console = _TqdmHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    log.handlers = [file_handler, console]


def _read_existing(path, header):
    """Rows already in ``path``. Incomplete rows (e.g. from an interrupted run) are dropped."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    if rows[0] != header:
        raise SystemExit(
            f"error: {path} exists but has different columns (other QIPs or column names).\n"
            "Use --overwrite to replace it, or choose another output file with -o.")
    return [r for r in rows[1:] if len(r) == len(header)]


def _write_csv(path, header, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def _format_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s'


def run(input_dir, output, qips=None, exclude=None, device='auto', workers=1, short_side=None,
        shard=None, toolbox_names=False, digits=6, overwrite=False):
    """Compute QIPs for every image below ``input_dir`` and write them to the CSV file ``output``.

    Rows are written as soon as each image is done; rerunning the same command
    resumes an interrupted run. ``shard=(i, n)`` restricts the run to every n-th image
    starting at i. Returns the number of images that could not be read.
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"error: {input_dir} is not a folder")

    selected = [q.name for q in select_qips(qips, exclude)]
    device = resolve_device(device)
    extractor_kwargs = dict(qips=selected, device=str(device), short_side=short_side)
    probe = QIPExtractor(**extractor_kwargs)
    header = (['img_file'] + probe.toolbox_columns) if toolbox_names else (['file'] + probe.columns)

    files = find_images(input_dir)
    if shard:
        files = files[shard[0]::shard[1]]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    rows = []
    if os.path.exists(output) and os.path.getsize(output) > 0 and not overwrite:
        rows = _read_existing(output, header)
    _write_csv(output, header, rows)
    done = {r[0] for r in rows}
    todo = [f for f in files if f not in done]
    workers = max(1, min(workers, len(todo)))

    _setup_logging(os.path.splitext(output)[0] + '.log')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    log.info(f"Input    : {input_dir} ({len(files)} images"
             + (f", shard {shard[0]}/{shard[1]}" if shard else '') + ")")
    log.info(f"Output   : {output}" + (f" (resuming, {len(files) - len(todo)} already done)" if done else ''))
    log.info(f"QIPs     : {len(selected)} ({len(header) - 1} columns)")
    log.info(f"Device   : {device} ({device_name}), {workers} worker(s)")
    log.info("Resizing : " + (f"shorter side to at most {short_side} px" if short_side else "none (full resolution)"))
    if not todo:
        log.info("Nothing to do.")
        return 0

    tasks = [(rel, os.path.join(input_dir, rel)) for rel in todo]
    n_failed_images = n_failed_qips = 0
    start = time.perf_counter()
    interrupted = False

    with open(output, 'a', newline='', encoding='utf-8') as f, \
            tqdm(total=len(tasks), unit='img', dynamic_ncols=True,
                 mininterval=0.5 if sys.stderr.isatty() else 60) as progress:  # sparse updates in log files
        writer = csv.writer(f)

        def handle(result):
            nonlocal n_failed_images, n_failed_qips
            rel_path, values, errors, load_error = result
            if load_error:
                n_failed_images += 1
                log.warning(f"{rel_path}: could not read image ({load_error})")
            else:
                writer.writerow([rel_path] + [format_value(v, digits) for v in values.values()])
                f.flush()
                for group, message, tb in errors:
                    n_failed_qips += 1
                    log.warning(f"{rel_path}: {group} QIPs failed ({message})")
                    log.debug(tb)
            progress.update(1)

        pool = None
        try:
            if workers == 1:
                for task in tasks:
                    handle(_process(task, probe))
            else:
                # 'spawn': CUDA cannot be used in forked processes
                pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                           initializer=_init_worker, initargs=(extractor_kwargs,))
                pending = iter(tasks)
                running = {pool.submit(_process, t) for t in _take(pending, 4 * workers)}
                while running:
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        handle(future.result())
                        running |= {pool.submit(_process, t) for t in _take(pending, 1)}
        except KeyboardInterrupt:
            interrupted = True
        except BrokenProcessPool:
            interrupted = True
            log.error("A worker process died (possibly out of memory). "
                      "Rerun the same command to resume, with fewer --workers.")
        finally:
            if pool is not None:
                pool.shutdown(wait=not interrupted, cancel_futures=True)

    elapsed = time.perf_counter() - start
    n_done = progress.n
    if interrupted:
        log.info(f"Stopped after {n_done} images. Rerun the same command to resume.")
        return n_failed_images

    # Rewrite the file sorted by image path
    _write_csv(output, header, sorted(_read_existing(output, header), key=lambda r: r[0]))

    log.info(f"Done: {n_done} images in {_format_duration(elapsed)} ({n_done / max(elapsed, 1e-9):.2f} img/s). "
             f"Results: {output}")
    if n_failed_images or n_failed_qips:
        log.info(f"{n_failed_images} image(s) could not be read and {n_failed_qips} QIP group(s) failed "
                 f"(left empty); see {os.path.splitext(output)[0] + '.log'}")
    return n_failed_images


def _take(iterator, n):
    out = []
    for _ in range(n):
        try:
            out.append(next(iterator))
        except StopIteration:
            break
    return out
