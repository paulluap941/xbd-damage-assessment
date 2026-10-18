"""
The figures that appear in more than one notebook.
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
import torch

import config
from common.crops import to_chw, stack_inputs

THREE_CLASS_CMAP = ListedColormap(["#20304a", "#2ca02c", "#d62728"])
FIVE_CLASS_CMAP = ListedColormap(["#20304a", "#2ca02c", "#ffdd57", "#ff7f0e", "#d62728"])


def plot_history(history, title, val_labels=("val",)):
    """Left: loss curves. Right: share of positive vs negative training crops."""
    if history is None:
        return

    n_val = len(val_labels)
    fig, ax = pl.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(history[:, 0], label="train")
    for i, label in enumerate(val_labels):
        ax[0].plot(history[:, 1 + i], label=label)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title(title)
    ax[0].legend()

    ax[1].plot(history[:, 1 + n_val], label="positive")
    ax[1].plot(history[:, 2 + n_val], label="negative")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("share of training crops [%]")
    ax[1].set_title("crop composition")
    ax[1].legend()

    pl.tight_layout()
    pl.show()


def show_examples(predict, raw_tiles, target_fn, cmap, n_classes, n=4):
    """Post image, ground truth and prediction for a few validation tiles."""
    indices = np.random.choice(len(raw_tiles), n, replace=False)
    fig, ax = pl.subplots(3, n, figsize=(3.2 * n, 9.6))

    for col, index in enumerate(indices):
        pre_tile, post_tile, target_tile, _ = raw_tiles[index]
        prediction = predict(pre_tile, post_tile)

        ax[0, col].imshow(post_tile)
        ax[1, col].imshow(target_fn(target_tile), cmap=cmap, vmin=0, vmax=n_classes - 1)
        ax[2, col].imshow(prediction, cmap=cmap, vmin=0, vmax=n_classes - 1)
        for row in range(3):
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

    for row, label in enumerate(["post image", "ground truth", "prediction"]):
        ax[row, 0].set_ylabel(label)

    pl.tight_layout()
    pl.show()


def make_predictor(model, device, building_model=None, siamese=False, normalize=False):
    def predict(pre_tile, post_tile):
        model.eval()
        with torch.no_grad():
            if siamese:
                pre_t = to_chw(pre_tile, normalize).unsqueeze(0).to(device)
                post_t = to_chw(post_tile, normalize).unsqueeze(0).to(device)
                logits = model(pre_t, post_t)
            else:
                inputs = stack_inputs(pre_tile, post_tile, building_model, device)
                logits = model(inputs.unsqueeze(0).to(device))
        return logits.argmax(dim=1)[0].cpu().numpy()
    return predict
