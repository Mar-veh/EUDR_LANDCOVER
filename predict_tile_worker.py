#!/usr/bin/env python3
"""Per-GPU inference worker for Predict_TabPFN35_PolygonLevel.ipynb.

Launched as a subprocess (one per GPU/worker slot) by the notebook's Section 6.
CUDA_VISIBLE_DEVICES / GDAL_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS are
set by the caller via the environment before this process starts.

For each assigned tile: reads it row-block by row-block (a background thread
prefetches the next block while the current one is classified on GPU), runs
the fitted TabPFN-3.5 classifier on every valid pixel, and writes two
single-band GeoTIFFs (class code, confidence) to --temp-dir. Writes go to a
.partial path first and are atomically renamed on completion, so a
killed/crashed worker never leaves a temp file that looks finished.
"""
import argparse
import json
import os
import queue
import threading
import traceback

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

EMBEDDING_COLS = [f'A{i:02d}' for i in range(64)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-tag', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--tiles-file', required=True)
    p.add_argument('--temp-dir', required=True)
    p.add_argument('--block-rows', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--nodata-class', type=int, required=True)
    p.add_argument('--nodata-conf', type=float, required=True)
    return p.parse_args()


def resolve_band_order(src):
    assert src.count == 64, f'Expected 64 embedding bands, found {src.count} in {src.name}'
    band_names = list(src.descriptions)
    if band_names == EMBEDDING_COLS:
        return list(range(1, 65))
    name_to_band = {name: i + 1 for i, name in enumerate(band_names)}
    missing = [c for c in EMBEDDING_COLS if c not in name_to_band]
    assert not missing, f'Image {src.name} is missing expected bands: {missing}'
    return [name_to_band[c] for c in EMBEDDING_COLS]


def tile_temp_paths(image_path, temp_dir):
    tile_name = os.path.splitext(os.path.basename(image_path))[0]
    return (
        os.path.join(temp_dir, f'{tile_name}_classcode.tif'),
        os.path.join(temp_dir, f'{tile_name}_confidence.tif'),
    )


def tile_is_done(image_path, temp_dir):
    class_path, conf_path = tile_temp_paths(image_path, temp_dir)
    return os.path.exists(class_path) and os.path.exists(conf_path)


def read_blocks(image_path, band_order, block_rows, out_queue):
    """Background-thread target: pushes (row0, row1, block) tuples, then None.
    Pushes the exception instead of a sentinel if the read itself fails, so the
    consumer can raise it in the main thread rather than hanging forever."""
    try:
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            for row0 in range(0, height, block_rows):
                row1 = min(row0 + block_rows, height)
                window = Window(0, row0, width, row1 - row0)
                block = src.read(band_order, window=window)  # (64, rows, cols)
                out_queue.put((row0, row1, block))
    except Exception as exc:  # noqa: BLE001 - deliberately forwarded to main thread
        out_queue.put(exc)
        return
    out_queue.put(None)


def classify_tile(image_path, clf, classes_arr, args):
    class_path, conf_path = tile_temp_paths(image_path, args.temp_dir)
    class_partial, conf_partial = class_path + '.partial', conf_path + '.partial'

    with rasterio.open(image_path) as src:
        band_order = resolve_band_order(src)
        width = src.width
        profile_class = src.profile.copy()
        profile_class.update(count=1, dtype='uint8', nodata=args.nodata_class, compress='lzw')
        profile_conf = src.profile.copy()
        profile_conf.update(count=1, dtype='float32', nodata=args.nodata_conf, compress='lzw')

    prefetch_q = queue.Queue(maxsize=2)
    reader_thread = threading.Thread(
        target=read_blocks, args=(image_path, band_order, args.block_rows, prefetch_q), daemon=True,
    )
    reader_thread.start()

    with rasterio.open(class_partial, 'w', **profile_class) as class_dst, \
            rasterio.open(conf_partial, 'w', **profile_conf) as conf_dst:
        while True:
            item = prefetch_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            row0, row1, block = item  # block: (64, rows, cols)
            rows, cols = block.shape[1], block.shape[2]
            flat = block.reshape(64, rows * cols).T  # (rows*cols, 64)

            valid_mask = ~np.isnan(flat).any(axis=1)
            class_out = np.full(rows * cols, args.nodata_class, dtype=np.uint8)
            conf_out = np.full(rows * cols, args.nodata_conf, dtype=np.float32)

            if valid_mask.any():
                X_valid = flat[valid_mask]
                n_valid = X_valid.shape[0]
                pred_codes = np.empty(n_valid, dtype=np.uint8)
                pred_confs = np.empty(n_valid, dtype=np.float32)

                for b0 in range(0, n_valid, args.batch_size):
                    b1 = min(b0 + args.batch_size, n_valid)
                    X_batch = pd.DataFrame(X_valid[b0:b1], columns=EMBEDDING_COLS)
                    proba = clf.predict_proba(X_batch)
                    top_idx = np.argmax(proba, axis=1)
                    pred_codes[b0:b1] = classes_arr[top_idx].astype(np.uint8)
                    pred_confs[b0:b1] = proba[np.arange(len(top_idx)), top_idx].astype(np.float32)

                class_out[valid_mask] = pred_codes
                conf_out[valid_mask] = pred_confs

            window = Window(0, row0, width, row1 - row0)
            class_dst.write(class_out.reshape(rows, cols), 1, window=window)
            conf_dst.write(conf_out.reshape(rows, cols), 1, window=window)

    reader_thread.join()
    os.replace(class_partial, class_path)
    os.replace(conf_partial, conf_path)


def main():
    args = parse_args()

    with open(args.tiles_file) as f:
        tiles = json.load(f)

    print(f'[{args.gpu_tag}] loading model: {args.model_path}', flush=True)
    clf = joblib.load(args.model_path)
    classes_arr = np.asarray(clf.classes_)
    print(f'[{args.gpu_tag}] model loaded, {len(tiles)} tile(s) assigned', flush=True)

    for i, image_path in enumerate(tiles):
        tag = f'[{args.gpu_tag}] ({i + 1}/{len(tiles)})'
        if tile_is_done(image_path, args.temp_dir):
            print(f'{tag} already done, skipping: {os.path.basename(image_path)}', flush=True)
            continue
        print(f'{tag} classifying: {os.path.basename(image_path)}', flush=True)
        try:
            classify_tile(image_path, clf, classes_arr, args)
        except Exception:
            print(f'{tag} FAILED on {os.path.basename(image_path)}:', flush=True)
            traceback.print_exc()
            raise
        print(f'{tag} done: {os.path.basename(image_path)}', flush=True)

    print(f'[{args.gpu_tag}] all {len(tiles)} tile(s) complete.', flush=True)


if __name__ == '__main__':
    main()
