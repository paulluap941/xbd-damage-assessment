"""
Cutting 256x256 crops out of the full 1024x1024 images.

Crops are drawn live on every access instead of being written to disk. A crop
is accepted once it holds enough building pixels, and after max_attempts the
first valid crop is returned and flagged as a negative example.
"""

import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def is_valid_nodata(tile_image, max_nodata_fraction=0.10):
    return (tile_image.sum(axis=-1) == 0).mean() <= max_nodata_fraction


def sample_tile(image, target, tile_size=256, min_tile_fraction=0.03,
                max_nodata_fraction=0.10, max_attempts=30):

    height, width = target.shape
    fallback_image, fallback_target = None, None
    attempts, draws = 0, 0
    max_draws = 10 * max_attempts   # guard against images that are almost all nodata

    while attempts < max_attempts and draws < max_draws:
        draws += 1
        row = np.random.randint(0, height - tile_size)
        col = np.random.randint(0, width - tile_size)

        tile_image = image[row:row + tile_size, col:col + tile_size]
        if not is_valid_nodata(tile_image, max_nodata_fraction):
            continue

        attempts += 1
        tile_target = target[row:row + tile_size, col:col + tile_size]

        if fallback_image is None:
            fallback_image, fallback_target = tile_image, tile_target

        if (tile_target > 0).mean() >= min_tile_fraction:
            return tile_image, tile_target, False

    if fallback_image is None:
        fallback_image = image[:tile_size, :tile_size]
        fallback_target = target[:tile_size, :tile_size]

    return fallback_image, fallback_target, True


def sample_damage_tile(pre_image, post_image, post_target, tile_size=256,
                       min_building_fraction=0.03, max_nodata_fraction=0.10,
                       max_attempts=30, damage_probability=0.0,
                       min_damage_fraction=0.005):
    """Random pre/post crop pair that preferably contains buildings.

    With probability `damage_probability` the crop additionally has to contain
    damaged buildings (classes 2 to 4). Since over 80 percent of all building
    pixels are undamaged, plain building sampling barely shows the model any
    minor or major damage.
    """
    height, width = post_target.shape
    require_damage = (damage_probability > 0
                      and np.random.random() < damage_probability)
    fallback = None
    attempts, draws = 0, 0
    max_draws = 10 * max_attempts

    while attempts < max_attempts and draws < max_draws:
        draws += 1
        row = np.random.randint(0, height - tile_size)
        col = np.random.randint(0, width - tile_size)

        pre_tile = pre_image[row:row + tile_size, col:col + tile_size]
        post_tile = post_image[row:row + tile_size, col:col + tile_size]

        if (pre_tile.sum(axis=-1) == 0).mean() > max_nodata_fraction:
            continue
        if (post_tile.sum(axis=-1) == 0).mean() > max_nodata_fraction:
            continue

        attempts += 1
        target_tile = post_target[row:row + tile_size, col:col + tile_size]

        if fallback is None:
            fallback = (pre_tile, post_tile, target_tile)

        if require_damage:
            accepted = (target_tile >= 2).mean() >= min_damage_fraction
        else:
            accepted = (target_tile > 0).mean() >= min_building_fraction

        if accepted:
            return pre_tile, post_tile, target_tile, False

    if fallback is None:
        fallback = (pre_image[:tile_size, :tile_size],
                    post_image[:tile_size, :tile_size],
                    post_target[:tile_size, :tile_size])

    pre_tile, post_tile, target_tile = fallback
    return pre_tile, post_tile, target_tile, True


def to_tensor(tile_image, tile_target):
    image = torch.from_numpy(tile_image.copy()).permute(2, 0, 1).float() / 255.0
    target = torch.from_numpy((tile_target > 0).astype(np.int64).copy())
    return image, target


def to_chw(tile, normalize=False):
    """Scale to [0, 1]. ImageNet pretrained encoders additionally expect the
    ImageNet mean and standard deviation."""
    tensor = torch.from_numpy(tile.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD if normalize else tensor


def normalize_batch(batch, normalize):
    if not normalize:
        return batch
    return (batch - IMAGENET_MEAN.to(batch.device)) / IMAGENET_STD.to(batch.device)


def stack_inputs(pre_tile, post_tile, building_model=None, device="cpu"):
    pre_t, post_t = to_chw(pre_tile), to_chw(post_tile)
    if building_model is None:
        return torch.cat([pre_t, post_t], dim=0)

    with torch.no_grad():
        logits = building_model(pre_t.unsqueeze(0).to(device))
        building_prob = torch.softmax(logits, dim=1)[0, 1:2].cpu()
    return torch.cat([pre_t, post_t, building_prob], dim=0)


def jitter_colors(tile, brightness=0.15, contrast=0.15, max_tint=10):
    """Brightness, contrast and a per channel tint. Nodata stays exactly black."""
    nodata = tile.sum(axis=-1) == 0
    result = tile.astype(np.float32)

    result *= 1.0 + np.random.uniform(-brightness, brightness)
    mean = result.mean()
    result = (result - mean) * (1.0 + np.random.uniform(-contrast, contrast)) + mean
    result += np.random.uniform(-max_tint, max_tint, size=3)

    result = np.clip(result, 0, 255).astype(np.uint8)
    result[nodata] = 0
    return result


def augment_pair(pre_tile, post_tile, target_tile, rotate=True, color_jitter=True):
    """Geometry identically on all three, colour independently on pre and post.

    Horizontal and vertical flips always apply. rotate adds the 90 degree
    rotations, color_jitter the independent colour shift of pre and post.
    """
    if rotate:
        k = np.random.randint(4)
        if k:
            pre_tile, post_tile, target_tile = (np.rot90(t, k) for t in (pre_tile, post_tile, target_tile))
    if np.random.random() < 0.5:
        pre_tile, post_tile, target_tile = (np.flip(t, axis=1) for t in (pre_tile, post_tile, target_tile))
    if np.random.random() < 0.5:
        pre_tile, post_tile, target_tile = (np.flip(t, axis=0) for t in (pre_tile, post_tile, target_tile))

    if color_jitter:
        pre_tile = jitter_colors(np.ascontiguousarray(pre_tile))
        post_tile = jitter_colors(np.ascontiguousarray(post_tile))

    return pre_tile, post_tile, np.ascontiguousarray(target_tile)


def resize_tiles(pre_tile, post_tile, target_tile, tile_size):
    """Back to the network input size after a scaled crop. Nearest for the target."""
    pre = np.array(Image.fromarray(np.ascontiguousarray(pre_tile)).resize(
        (tile_size, tile_size), Image.BILINEAR))
    post = np.array(Image.fromarray(np.ascontiguousarray(post_tile)).resize(
        (tile_size, tile_size), Image.BILINEAR))
    target = np.array(Image.fromarray(np.ascontiguousarray(target_tile)).resize(
        (tile_size, tile_size), Image.NEAREST))
    return pre, post, target


def damage_to_three_class(target_tile):
    """0 background, 1 intact building, 2 damaged building."""
    result = np.zeros_like(target_tile)
    result[target_tile == 1] = 1
    result[target_tile >= 2] = 2
    return result


def damage_identity(target_tile):
    """Full target: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed."""
    return target_tile.astype(np.uint8)


class BuildingDataset(Dataset):
    """Live pre-disaster crops with a binary building target."""

    def __init__(self, files, paths, tile_size=256, n_crops=8, min_tile_fraction=0.03,
                 max_nodata_fraction=0.10, max_attempts=30, augment=False):
        self.files = files
        self.paths = paths
        self.tile_size = tile_size
        self.n_crops = n_crops
        self.min_tile_fraction = min_tile_fraction
        self.max_nodata_fraction = max_nodata_fraction
        self.max_attempts = max_attempts
        self.augment = augment

    def __len__(self):
        return len(self.files) * self.n_crops

    def __getitem__(self, index):
        split, stem = self.files[index // self.n_crops]
        image = np.array(Image.open(
            os.path.join(self.paths[split]["images"], f"{stem}_pre_disaster.png")))
        target = np.array(Image.open(
            os.path.join(self.paths[split]["targets"], f"{stem}_pre_disaster_target.png")))

        tile_image, tile_target, is_negative = sample_tile(
            image, target, self.tile_size, self.min_tile_fraction,
            self.max_nodata_fraction, self.max_attempts,
        )

        if self.augment:
            if np.random.random() < 0.5:
                tile_image, tile_target = np.flip(tile_image, 1), np.flip(tile_target, 1)
            if np.random.random() < 0.5:
                tile_image, tile_target = np.flip(tile_image, 0), np.flip(tile_target, 0)

        image_t, target_t = to_tensor(tile_image, tile_target)
        return image_t, target_t, int(is_negative)


class FixedTileDataset(Dataset):
    def __init__(self, tiles):
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        return self.tiles[index]


class DamageDataset(Dataset):
    """Live random crops from the full images, n_crops per image and epoch.

    building_model optionally appends the building probability channel from the
    localization U-Net.
    """

    def __init__(self, files, paths, target_fn, tile_size=256, n_crops=12,
                 min_building_fraction=0.03, max_nodata_fraction=0.10,
                 max_attempts=30, augment=False, building_model=None, device="cpu",
                 damage_probability=0.0):
        self.files = files
        self.paths = paths
        self.target_fn = target_fn
        self.tile_size = tile_size
        self.n_crops = n_crops
        self.min_building_fraction = min_building_fraction
        self.max_nodata_fraction = max_nodata_fraction
        self.max_attempts = max_attempts
        self.augment = augment
        self.building_model = building_model
        self.device = device
        self.damage_probability = damage_probability

    def __len__(self):
        return len(self.files) * self.n_crops

    def __getitem__(self, index):
        split, stem = self.files[index // self.n_crops]
        image_dir, target_dir = self.paths[split]["images"], self.paths[split]["targets"]

        pre_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_pre_disaster.png")))
        post_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_post_disaster.png")))
        post_target = np.array(Image.open(os.path.join(target_dir, f"{stem}_post_disaster_target.png")))

        pre_tile, post_tile, target_tile, is_negative = sample_damage_tile(
            pre_image, post_image, post_target, self.tile_size,
            self.min_building_fraction, self.max_nodata_fraction, self.max_attempts,
            self.damage_probability,
        )

        if self.augment:
            if np.random.random() < 0.5:
                pre_tile, post_tile, target_tile = (np.flip(t, axis=1) for t in (pre_tile, post_tile, target_tile))
            if np.random.random() < 0.5:
                pre_tile, post_tile, target_tile = (np.flip(t, axis=0) for t in (pre_tile, post_tile, target_tile))

        inputs = stack_inputs(pre_tile, post_tile, self.building_model, self.device)
        target = torch.from_numpy(self.target_fn(target_tile).copy()).long()
        return inputs, target, is_negative


class FixedDamageDataset(Dataset):
    """Wrapper around the stored raw tiles, so every variant sees the same crops."""

    def __init__(self, raw_tiles, target_fn, building_model=None, device="cpu"):
        self.raw_tiles = raw_tiles
        self.target_fn = target_fn
        self.building_model = building_model
        self.device = device

    def __len__(self):
        return len(self.raw_tiles)

    def __getitem__(self, index):
        pre_tile, post_tile, target_tile, is_negative = self.raw_tiles[index]
        inputs = stack_inputs(pre_tile, post_tile, self.building_model, self.device)
        target = torch.from_numpy(self.target_fn(target_tile).copy()).long()
        return inputs, target, is_negative


class SiameseDamageDataset(Dataset):
    """Live random crops with pre and post as separate inputs.

    augment switches augmentation on at all; rotate, color_jitter and
    scale_range add the stronger variants on top of the plain flips.
    damage_probability is the share of crops that must contain damage.
    """

    def __init__(self, files, paths, target_fn, tile_size=256, n_crops=12,
                 min_building_fraction=0.03, max_nodata_fraction=0.10,
                 max_attempts=30, augment=False, damage_probability=0.0,
                 normalize=False, scale_range=None, rotate=False, color_jitter=False):
        self.files = files
        self.paths = paths
        self.target_fn = target_fn
        self.tile_size = tile_size
        self.n_crops = n_crops
        self.min_building_fraction = min_building_fraction
        self.max_nodata_fraction = max_nodata_fraction
        self.max_attempts = max_attempts
        self.augment = augment
        self.damage_probability = damage_probability
        self.normalize = normalize
        self.scale_range = scale_range
        self.rotate = rotate
        self.color_jitter = color_jitter

    def __len__(self):
        return len(self.files) * self.n_crops

    def __getitem__(self, index):
        split, stem = self.files[index // self.n_crops]
        image_dir, target_dir = self.paths[split]["images"], self.paths[split]["targets"]

        pre_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_pre_disaster.png")))
        post_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_post_disaster.png")))
        post_target = np.array(Image.open(os.path.join(target_dir, f"{stem}_post_disaster_target.png")))

        crop_size = self.tile_size
        if self.augment and self.scale_range is not None:
            crop_size = int(round(self.tile_size * np.random.uniform(*self.scale_range)))

        pre_tile, post_tile, target_tile, is_negative = sample_damage_tile(
            pre_image, post_image, post_target, crop_size,
            self.min_building_fraction, self.max_nodata_fraction, self.max_attempts,
            self.damage_probability,
        )

        if crop_size != self.tile_size:
            pre_tile, post_tile, target_tile = resize_tiles(
                pre_tile, post_tile, target_tile, self.tile_size)

        if self.augment:
            pre_tile, post_tile, target_tile = augment_pair(
                pre_tile, post_tile, target_tile, self.rotate, self.color_jitter)

        target = torch.from_numpy(self.target_fn(target_tile).copy()).long()
        return (to_chw(pre_tile, self.normalize), to_chw(post_tile, self.normalize),
                target, is_negative)


class FixedSiameseDataset(Dataset):
    """The stored raw tiles with pre and post kept apart."""

    def __init__(self, raw_tiles, target_fn, normalize=False):
        self.raw_tiles = raw_tiles
        self.target_fn = target_fn
        self.normalize = normalize

    def __len__(self):
        return len(self.raw_tiles)

    def __getitem__(self, index):
        pre_tile, post_tile, target_tile, is_negative = self.raw_tiles[index]
        target = torch.from_numpy(self.target_fn(target_tile).copy()).long()
        return (to_chw(pre_tile, self.normalize), to_chw(post_tile, self.normalize),
                target, is_negative)


def draw_raw_tiles(files, paths, n_crops, seed=357):
    """Raw (pre, post, 0-4 target) crops. The target format is decided on access."""
    np.random.seed(seed)
    tiles = []
    for split, stem in files:
        image_dir, target_dir = paths[split]["images"], paths[split]["targets"]
        pre_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_pre_disaster.png")))
        post_image = np.array(Image.open(os.path.join(image_dir, f"{stem}_post_disaster.png")))
        post_target = np.array(Image.open(os.path.join(target_dir, f"{stem}_post_disaster_target.png")))
        for _ in range(n_crops):
            tiles.append(sample_damage_tile(pre_image, post_image, post_target))
    return tiles


def describe_tiles(tiles, name):
    n_total = len(tiles)
    n_negative = sum(t[3] for t in tiles)
    damage_fraction = np.mean([(t[2] >= 2).mean() for t in tiles])
    print(f"{name}: {n_total} tiles, "
          f"{100 * (n_total - n_negative) / n_total:.1f}% positive, "
          f"{100 * damage_fraction:.2f}% damaged building pixels")
