# xBD building damage assessment

Course project for GEW-DAP05, Advanced Topics of Data Analysis and Programming,
University of Potsdam.

For every location the [xBD dataset](https://xview2.org/) has two satellite
images, one before and one after a disaster. Two questions follow from that:
where are the buildings, and how badly is each one damaged. Scoring uses the
official xView2 formula.

The difficulty is the imbalance. 94 % of all pixels are background, and of the
building pixels three quarters are undamaged, so a model that predicts nothing
is already right most of the time.

## Notebooks

| # | notebook | |
|---|---|---|
| 1 | [`01_data_check.ipynb`](01_data_check.ipynb) | class encoding, Tier3 targets, nodata filtering, class distribution per event |
| 2 | [`02_localization_and_first_damage_models.ipynb`](02_localization_and_first_damage_models.ipynb) | building localization, and damage with and without a building mask |
| 3 | [`03_scale_to_all_events.ipynb`](03_scale_to_all_events.ipynb) | from two event types to all ten, and the full five class target |
| 4 | [`04_advanced_models.ipynb`](04_advanced_models.ipynb) | Siamese U-Net, and a Siamese variant with a pretrained ResNet50 |
| 5 | [`05_ensemble_and_final_evaluation.ipynb`](05_ensemble_and_final_evaluation.ipynb) | the models combined, scored on full holdout images |
| 6 | [`06_cross_event_generalisation.ipynb`](06_cross_event_generalisation.ipynb) | three events removed from training, evaluated on exactly those |

Notebook 1 has to run once, it writes the target images for the Tier3 split,
which xBD ships only as label polygons. After that the order does not matter.
Nothing is retrained and nothing is recomputed that is already on disk, so a
second run takes seconds.

## Results

All 933 images of the holdout split, at full size, with a sliding window:

| model | no-damage | minor | major | destroyed | F1 damage | F1 loc | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble | 0.829 | 0.513 | 0.663 | 0.759 | **0.669** | 0.862 | **0.727** |
| Siamese ResNet50 | 0.818 | 0.497 | 0.642 | 0.742 | 0.652 | 0.862 | 0.715 |
| Siamese U-Net | 0.815 | 0.464 | 0.609 | 0.744 | 0.628 | 0.862 | 0.698 |
| U-Net, six stacked channels | 0.801 | 0.452 | 0.629 | 0.717 | 0.621 | 0.862 | 0.693 |

`score = 0.3 * F1_localization + 0.7 * F1_damage`, where `F1_damage` is the
harmonic mean of the four damage classes. The baseline published with the
dataset reaches a weighted damage F1 of 0.27 (Gupta et al., 2019).

Train, test and holdout all contain the same ten events, so even the holdout
number describes disasters the models have seen. Notebook 6 removes three events
from training and evaluates on exactly those:

| images | trained without the events | trained with them |
|---|---:|---:|
| the three held out events | 0.541 | 0.696 |
| a sample of the other seven | 0.697 | 0.712 |

Finding buildings barely suffers, F1 0.845 against 0.868. The damage classes
carry the whole loss.

## Layout

```
config.py          paths, splits, class encoding, all constants
common/
  data.py          splits, file lists, targets from label polygons
  crops.py         256x256 crops and the datasets that cut them live
  models.py        U-Net, Siamese U-Net, Siamese ResNet
  losses.py        cross entropy plus Dice
  train.py         the training loop and the loaders for a finished run
  evaluate.py      pixel metrics, confusion counts, the xView2 score
  plots.py         the figures that appear in more than one notebook
checkpoints/       trained models, not in the repository
val_tiles/         fixed validation tiles, not in the repository
results/           counts of the full image evaluations
predictions/       full image predictions, one example per event
```

A notebook holds what makes that notebook, everything shared is imported.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

For GPU training install the matching PyTorch build from
[pytorch.org](https://pytorch.org) first. The raw data is not in the repository,
it is available after registration at [xview2.org](https://xview2.org/dataset)
and goes into `data/`, or wherever `XBD_DATA_ROOT` points. The `.pt` files are
too large for Git; a missing checkpoint stops the notebook with a clear message.

## Notes

**Crops instead of full images.** Training cuts a 256x256 tile from a random
position of the 1024x1024 original on every access, so the same image gives a
different crop in every epoch. A new position is drawn up to 30 times until at
least 3 % of the tile is buildings, which fails for about 22 % of them.

**Fixed validation tiles.** Drawn once and then only loaded, so every model in
every notebook is scored on exactly the same crops. They keep the raw 0 to 4
target and the format is decided on access, so the three class and five class
models see identical data.

**Pixel metrics.** Everything is counted over pixels. The competition scores per
building polygon, so these numbers compare among themselves but not against the
leaderboard.

**Held out events.** Splitting by image leaves the same disasters on both sides.
Splitting by event is the harder and more honest question, and the gap between
the two tables above is what that costs.

## Dataset

Gupta et al., *Creating xBD: A Dataset for Assessing Building Damage from
Satellite Imagery*, CVPR Workshops 2019, [xview2.org](https://xview2.org/)
