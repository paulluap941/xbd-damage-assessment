"""
Finding the xBD splits on disk and turning the label polygons into targets.

No torch in here, so notebook 1 runs without a GPU environment.
"""

import os
import json

import numpy as np
from PIL import Image
from shapely import wkt
from skimage.draw import polygon2mask

import config


def split_paths(root, split):
    """First candidate folder that holds images, with labels and targets next to it."""
    for folder in config.SPLIT_FOLDERS[split]:
        base = root / folder
        if (base / "images").is_dir():
            return {"images": base / "images", "targets": base / "targets", "labels": base / "labels"}
    return None


def resolve_splits(root):
    return {split: found for split in config.SPLIT_FOLDERS
            if (found := split_paths(root, split)) is not None}


def list_pre_images(image_dir):
    return sorted(f for f in os.listdir(image_dir) if f.endswith("_pre_disaster.png"))


def json_to_target(label_path, image_shape=(1024, 1024)):
    with open(label_path) as f:
        labels = json.load(f)

    target = np.zeros(image_shape, dtype=np.uint8)
    for feature in labels["features"]["xy"]:
        coords = np.array(wkt.loads(feature["wkt"]).exterior.coords)
        subtype = feature["properties"].get("subtype", "no-damage")
        target[polygon2mask(image_shape, coords[:, ::-1])] = config.DAMAGE_TO_CLASS.get(subtype, 1)

    return target


def nodata_fraction(image_path):
    image = np.array(Image.open(image_path))
    return (image.sum(axis=-1) == 0).mean()


def stems_for_events(file_lists, split, events):
    return [(split, f.replace("_pre_disaster.png", ""))
            for f in file_lists[split] if any(f.startswith(e) for e in events)]


def event_of(stem):
    return stem.split("_")[0]
