"""End-to-end tests of the command-line tool and the Python API on synthetic images."""

import csv
import math

import numpy as np
import pytest
import torch
from PIL import Image

from qipmachine import QIPS, QIPExtractor
from qipmachine.cli import main

N_COLUMNS = sum(len(q.columns) for q in QIPS)
FAST_QIPS = ['image_size', 'aspect_ratio', 'mean_rgb', 'balance', 'cnn_symmetry_lr', 'fourier_slope_mather']


def natural_image(height, width, seed=0):
    """RGB 1/f noise, which has natural-image-like statistics."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    falloff = 1.0 / np.maximum(np.sqrt(fx ** 2 + fy ** 2), 1.0 / max(height, width))
    channels = []
    for _ in range(3):
        noise = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(height, width))) * falloff))
        channels.append((noise - noise.min()) / (noise.max() - noise.min()) * 255)
    return np.stack(channels, axis=2).astype(np.uint8)


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / 'images'
    (folder / 'sub').mkdir(parents=True)
    Image.fromarray(natural_image(240, 320, 0)).save(folder / 'a.png')
    Image.fromarray(natural_image(260, 200, 1)).convert('L').save(folder / 'sub' / 'b.jpg')
    Image.fromarray(natural_image(256, 256, 2)).convert('RGBA').save(folder / 'c.png')
    (folder / 'notes.txt').write_text('not an image')
    (folder / '.hidden.jpg').write_text('ignored')
    return folder


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_all_qips(image_folder, tmp_path):
    out = tmp_path / 'qips.csv'
    assert main([str(image_folder), '-o', str(out), '--device', 'cpu', '--workers', '1']) == 0
    header, rows = read_csv(out)
    assert header == ['file'] + [c for q in QIPS for c in q.columns]
    assert len(header) == N_COLUMNS + 1 == 48
    assert [r[0] for r in rows] == ['a.png', 'c.png', 'sub/b.jpg']
    for row in rows:
        values = [float(v) for v in row[1:]]
        assert all(math.isfinite(v) for v in values), row
    a = dict(zip(header, rows[0]))
    assert a['image_size'] == '560'
    assert float(a['aspect_ratio']) == pytest.approx(320 / 240, rel=1e-5)
    assert (tmp_path / 'qips.log').exists()


def test_resume_subset_and_toolbox_names(image_folder, tmp_path):
    out = tmp_path / 'subset.csv'
    args = [str(image_folder), '-o', str(out), '--device', 'cpu', '--workers', '1', '--qips', *FAST_QIPS,
            '--toolbox-names']
    assert main(args) == 0
    header, rows = read_csv(out)
    assert header == ['img_file', 'Image size (pixels)', 'Aspect ratio', 'mean R channel', 'mean G channel',
                      'mean B channel (RGB)', 'Balance', 'CNN symmetry left-right', 'Slope Mather']
    assert len(rows) == 3

    # An interrupted run leaves a partial line; resuming drops it and recomputes that image
    with open(out, 'a') as f:
        f.write('d.png,123')
    Image.fromarray(natural_image(200, 300, 3)).save(image_folder / 'd.png')
    assert main(args) == 0
    _, rows_after = read_csv(out)
    assert [r[0] for r in rows_after] == ['a.png', 'c.png', 'd.png', 'sub/b.jpg']
    assert [r for r in rows_after if r[0] != 'd.png'] == rows  # earlier results kept as they were

    # Different QIPs cannot be appended to the same file
    with pytest.raises(SystemExit):
        main([str(image_folder), '-o', str(out), '--device', 'cpu', '--qips', 'balance'])


def test_unreadable_image_and_workers(image_folder, tmp_path):
    (image_folder / 'broken.jpg').write_bytes(b'not a jpeg')
    out = tmp_path / 'qips.csv'
    assert main([str(image_folder), '-o', str(out), '--device', 'cpu', '--workers', '2',
                 '--qips', *FAST_QIPS]) == 1
    _, rows = read_csv(out)
    assert [r[0] for r in rows] == ['a.png', 'c.png', 'sub/b.jpg']
    assert 'broken.jpg' in (tmp_path / 'qips.log').read_text()


def test_short_side_keeps_original_size(tmp_path):
    path = tmp_path / 'big.png'
    Image.fromarray(natural_image(600, 900)).save(path)
    values = QIPExtractor(qips=['image_size', 'aspect_ratio'], device='cpu', short_side=300).compute(path)
    assert values == {'image_size': 1500.0, 'aspect_ratio': 1.5}


def test_shards_partition_the_images(image_folder, tmp_path):
    files = []
    for i in range(2):
        out = tmp_path / f'shard{i}.csv'
        main([str(image_folder), '-o', str(out), '--device', 'cpu', '--workers', '1', '--qips', 'image_size',
              '--shard', f'{i}/2'])
        files += [r[0] for r in read_csv(out)[1]]
    assert sorted(files) == ['a.png', 'c.png', 'sub/b.jpg']


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_matches_cpu(tmp_path):
    path = tmp_path / 'img.png'
    Image.fromarray(natural_image(300, 400, 4)).save(path)
    cpu = QIPExtractor(device='cpu').compute(path)
    gpu = QIPExtractor(device='cuda').compute(path)
    for column in cpu:
        assert gpu[column] == pytest.approx(cpu[column], rel=1e-9, abs=1e-12), column
