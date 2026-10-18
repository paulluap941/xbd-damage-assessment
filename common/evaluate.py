"""
Pixel based metrics.

The official xView2 score works per building polygon, these numbers are
computed over pixels, so they are comparable among themselves but not one to
one against the leaderboard.
"""

import json

import numpy as np
import torch

import config


def per_class_metrics(model, loader, n_classes, device, siamese=False):
    """Pixel wise precision, recall and F1 for every class."""
    model.eval()
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            if siamese:
                pre, post, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                preds = model(pre, post).argmax(dim=1)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                preds = model(inputs).argmax(dim=1)

            for c in range(n_classes):
                pred_c, target_c = (preds == c), (targets == c)
                tp[c] += (pred_c & target_c).sum().item()
                fp[c] += (pred_c & ~target_c).sum().item()
                fn[c] += (~pred_c & target_c).sum().item()

    results = {}
    for c in range(n_classes):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        results[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return results


def print_metrics(results, class_names, title):
    print(title)
    print(f"{'class':>12s}{'precision':>12s}{'recall':>10s}{'F1':>8s}")
    for c, (p, r, f1) in results.items():
        print(f"{class_names[c]:>12s}{p:>12.3f}{r:>10.3f}{f1:>8.3f}")


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def harmonic_mean(values):
    """A single weak class pulls the result down hard, which is the point."""
    if any(v == 0 for v in values):
        return 0.0
    return len(values) / sum(1 / v for v in values)


class ConfusionCounter:
    """Accumulates pixel wise TP, FP and FN per class over any number of images."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.tp = np.zeros(n_classes, dtype=np.int64)
        self.fp = np.zeros(n_classes, dtype=np.int64)
        self.fn = np.zeros(n_classes, dtype=np.int64)

    def update(self, prediction, target):
        for c in range(self.n_classes):
            pred_c, target_c = (prediction == c), (target == c)
            self.tp[c] += int((pred_c & target_c).sum())
            self.fp[c] += int((pred_c & ~target_c).sum())
            self.fn[c] += int((~pred_c & target_c).sum())

    def per_class(self):
        return {c: precision_recall_f1(self.tp[c], self.fp[c], self.fn[c])
                for c in range(self.n_classes)}


def tile_positions(height, width, tile_size=config.TILE_SIZE,
                   stride=config.STRIDE):
    return [(r, c)
            for r in range(0, height - tile_size + 1, stride)
            for c in range(0, width - tile_size + 1, stride)]


def xview2_score(per_class_results, f1_localization):
    f1_damage = harmonic_mean([per_class_results[c][2] for c in config.DAMAGE_CLASSES])
    return f1_damage, 0.3 * f1_localization + 0.7 * f1_damage


def save_counters(path, counters):
    """Write a nesting of ConfusionCounter and count arrays as plain numbers."""
    def dump(obj):
        if isinstance(obj, ConfusionCounter):
            return {"n_classes": obj.n_classes, "tp": obj.tp.tolist(),
                    "fp": obj.fp.tolist(), "fn": obj.fn.tolist()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return {str(k): dump(v) for k, v in obj.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump(counters), indent=1), encoding="utf-8")


def load_counters(path):
    def build(obj):
        if isinstance(obj, list):
            return np.array(obj, dtype=np.int64)
        if "n_classes" in obj:
            counter = ConfusionCounter(obj["n_classes"])
            counter.tp = np.array(obj["tp"], dtype=np.int64)
            counter.fp = np.array(obj["fp"], dtype=np.int64)
            counter.fn = np.array(obj["fn"], dtype=np.int64)
            return counter
        return {k: build(v) for k, v in obj.items()}

    return build(json.loads(path.read_text(encoding="utf-8")))
