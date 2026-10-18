# xBD building damage assessment

Course project for GEW-DAP05, Advanced Topics of Data Analysis and Programming,
University of Potsdam.

Building localization and damage classification on satellite image pairs from
the [xBD dataset](https://xview2.org/), from a plain U-Net to Siamese
architectures and an ensemble, scored with the official xView2 formula.

---

## Task

For every location xBD provides an image pair, one taken **before** and one
**after** a disaster. Two things are asked:

1. **Localization**, where are the buildings, from the pre image
2. **Damage classification**, how badly is each building damaged
   (`no-damage`, `minor`, `major`, `destroyed`), from the image pair

The dataset is heavily imbalanced. 94 % of all pixels are background, and of the
building pixels 74 % are undamaged, 9 % minor, 12 % major and 5 % destroyed.
That runs through the whole project as the central difficulty.

---

## Result

All 933 images of the holdout split, at full size, with a sliding window:

| model | no-damage | minor | major | destroyed | F1 damage | F1 loc | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble | 0.829 | 0.513 | 0.663 | 0.759 | **0.669** | 0.862 | **0.727** |
| Siamese ResNet50 | 0.818 | 0.497 | 0.642 | 0.742 | 0.652 | 0.862 | 0.715 |
| Siamese U-Net | 0.815 | 0.464 | 0.609 | 0.744 | 0.628 | 0.862 | 0.698 |
| U-Net, six stacked channels | 0.801 | 0.452 | 0.629 | 0.717 | 0.621 | 0.862 | 0.693 |

```
score = 0.3 * F1_localization + 0.7 * F1_damage
```

where `F1_damage` is the harmonic mean of the four damage classes. The baseline
published with the dataset reaches a weighted damage F1 of 0.27, with 0.14 for
minor and 0.01 for major (Gupta et al., 2019). `minor` and `major` stay the
hardest classes here as well, but they are recognised.

Train, test and holdout all contain the same ten events, so even the holdout
number describes disasters the models have seen. Notebook 6 removes three events
from training completely and evaluates on exactly those:

| images | without the events | with the events |
|---|---:|---:|
| the three held out events | 0.541 | 0.696 |
| a sample of the other seven | 0.697 | 0.712 |

Building detection barely moves (F1 0.845 against 0.868), the damage classes
carry the whole loss.

---

## The six notebooks

They build on each other and are meant to be read in order.

| # | notebook | content |
|---|---|---|
| 1 | [`01_data_check.ipynb`](01_data_check.ipynb) | class encoding, Tier3 target generation, nodata filtering, class distribution per event |
| 2 | [`02_localization_and_first_damage_models.ipynb`](02_localization_and_first_damage_models.ipynb) | localization U-Net and the first two damage models, with and without a building mask, on hurricanes and tornadoes |
| 3 | [`03_scale_to_all_events.ipynb`](03_scale_to_all_events.ipynb) | all events with a wider validation set, and the full five class target |
| 4 | [`04_advanced_models.ipynb`](04_advanced_models.ipynb) | Siamese U-Net with a shared encoder, and a Siamese variant with an ImageNet pretrained ResNet50 |
| 5 | [`05_ensemble_and_final_evaluation.ipynb`](05_ensemble_and_final_evaluation.ipynb) | probability averaging ensemble and the final evaluation on full holdout images |
| 6 | [`06_cross_event_generalisation.ipynb`](06_cross_event_generalisation.ipynb) | three events removed from training, evaluated on exactly those |

**Notebook 1 has to run once before the others.** It generates the target
images for the Tier3 split, which xBD ships only as label polygons. After that,
notebooks 2 to 6 can run in any order.

---

## What a run does by default

Nothing is trained and nothing is recomputed that is already on disk. Every
expensive step checks for its own result first:

```python
if not (config.CKPT_DIR / "damage_siamese_best.pt").exists():
    ... train ...
```

The same holds for the fixed validation tiles and for the two evaluations that
run the models over whole images, which write their counts to `results/`. A
second run of notebook 5 therefore takes seconds instead of half an hour.

Training is thirty epochs with Adam at a learning rate of 0.001, batch size 16,
256x256 crops, horizontal and vertical flips, cross entropy plus 0.5 times Dice,
keeping the epoch with the lowest validation loss. About one hour per model on a
GPU.

---

## Layout

```
config.py          paths, splits, class encoding, all constants
common/
  data.py          finding the splits, file lists, targets from label polygons
  crops.py         256x256 crops and the datasets that cut them live
  models.py        U-Net, Siamese U-Net, Siamese ResNet
  losses.py        cross entropy plus Dice
  train.py         the training loop and the loaders for a finished run
  evaluate.py      pixel metrics, confusion counts, the xView2 score
  plots.py         the figures that appear in more than one notebook
checkpoints/       trained models (state_dict, *.pt)
histories/         loss curves of the training runs (*.pt)
val_tiles/         fixed validation tiles (*.pt)
results/           counts of the full image evaluations (*.json)
predictions/       full image predictions of the ensemble (*.png)
```

A notebook holds what makes that notebook, everything shared is imported.

---

## Setup

```bash
git clone <repo-url>
cd xbd-damage-assessment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

For GPU training install the matching PyTorch build from
[pytorch.org](https://pytorch.org) first, see the comment at the top of
`requirements.txt`.

### Data

The raw xBD data is not part of this repository. It is available after
registration at [xview2.org](https://xview2.org/dataset). Expected layout:

```
data/
├── train_images_labels_targets/train/{images,labels,targets}/
├── test_images_labels_targets/test/{images,labels,targets}/
├── hold_images_labels_targets/hold/{images,labels,targets}/
└── tier3/tier3/{images,labels}/          # targets are generated by notebook 1
```

Set `XBD_DATA_ROOT` to point somewhere else, otherwise `config.py` uses `data/`
next to the notebooks.

### Files that are not in the repository

The `.pt` files are too large for Git, around 120 MB per U-Net and several GB
per tile set. A missing checkpoint stops the notebook with a clear message, a
missing history only skips the corresponding plot.

| `checkpoints/` | model | from |
|---|---|---|
| `unet_localization_best.pt` | U-Net, 3 to 2 channels | notebook 2 |
| `method_a_best.pt` | U-Net, 7 to 3, with building mask | notebook 2 |
| `method_b_best.pt` | U-Net, 6 to 3, without | notebook 2 |
| `damage_full_best.pt` | U-Net, 6 to 3, all events | notebook 3 |
| `damage_5class_best.pt` | U-Net, 6 to 5 | notebook 3 |
| `damage_siamese_best.pt` | Siamese U-Net, 3 to 5 | notebook 4 |
| `damage_siamese_resnet_best.pt` | Siamese ResNet50, 3 to 5 | notebook 4 |
| `damage_siamese_resnet_event_holdout.pt` | the same, three events removed | notebook 6 |

| `val_tiles/` | content | used by |
|---|---|---|
| `val_tiles_fixed.pt` | 8 crops per test image, binary building target | notebook 2 |
| `val_damage_raw_tiles.pt` | 12 crops per hurricane test image, raw 0 to 4 target | notebooks 2, 3 |
| `val_damage_raw_tiles_all_events.pt` | 12 crops per test image, all events | notebooks 3, 4, 5 |
| `val_damage_raw_tiles_event_holdout.pt` | the same without the held out events | notebook 6 |

---

## Method notes

**Crops instead of full images.** Training runs on 256x256 tiles cut from a
random position of the 1024x1024 original on every access. Tiles with more than
10 % nodata are discarded, and a new position is drawn up to 30 times until at
least 3 % of the tile is buildings. Only if that fails does the tile count as a
negative example, which happens for about 22 % of them.

**Fixed validation tiles.** The random draw happens once and is then only
loaded, so every model across every notebook is scored on exactly the same
crops. The stored tiles keep the raw 0 to 4 target and the target format is
decided on access, so three class and five class models see identical crops.

**Pixel based metrics.** All numbers are counted over pixels. The official
xView2 competition scores per building polygon, so these numbers are comparable
among themselves but not one to one against the leaderboard.

---

## Dataset

Gupta et al., *Creating xBD: A Dataset for Assessing Building Damage from
Satellite Imagery*, CVPR Workshops 2019, [xview2.org](https://xview2.org/)
