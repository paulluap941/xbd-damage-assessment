"""Paths and parameters. Everything that more than one notebook needs is here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# the raw xBD data is not part of the repository, see README
DATA_ROOT = Path(os.environ.get("XBD_DATA_ROOT", ROOT / "data"))

CKPT_DIR = ROOT / "checkpoints"      # model weights (state_dict, *.pt)
HISTORY_DIR = ROOT / "histories"     # loss curves (*.pt)
VAL_TILES_DIR = ROOT / "val_tiles"   # fixed validation tiles (*.pt)
RESULTS_DIR = ROOT / "results"       # one json per model
PRED_DIR = ROOT / "predictions"      # full image predictions
FIGURE_DIR = ROOT / "figures"

# xBD ships the four splits under different folder names depending on the
# download. The first candidate that holds an images folder wins.
SPLIT_FOLDERS = {
    "train": ["train_images_labels_targets/train", "train"],
    "test":  ["test_images_labels_targets/test", "test"],
    "hold":  ["hold_images_labels_targets/hold", "holdout_images_labels_targets/hold", "hold"],
    "tier3": ["tier3/tier3", "tier3"],
}

# pixel value of every label subtype in the target images
DAMAGE_TO_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}
DAMAGE_CLASSES = [1, 2, 3, 4]

LOCALIZATION_NAMES = {0: "background", 1: "building"}
THREE_CLASS_NAMES = {0: "background", 1: "building", 2: "damaged"}
FIVE_CLASS_NAMES = {0: "background", 1: "no-damage", 2: "minor", 3: "major", 4: "destroyed"}

# crops
TILE_SIZE = 256
STRIDE = 128                    # full image prediction, so the tiles overlap by half
MIN_BUILDING_FRACTION = 0.001   # images below this building area are dropped
N_VAL_CROPS_PER_IMAGE = 8       # localization
N_VAL_CROPS = 12                # damage

# training
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DECODER_CHANNELS = (256, 128, 64, 32, 16)

# the first damage models see one event type at a time
HURRICANE_EVENTS = ["hurricane-florence", "hurricane-harvey",
                    "hurricane-matthew", "hurricane-michael"]
TORNADO_EVENTS = ["joplin-tornado", "moore-tornado", "tuscaloosa-tornado"]

# events kept out of training completely, notebook 6
HELD_OUT_EVENTS = ("hurricane-michael", "santa-rosa-wildfire", "palu-tsunami")

# nodata filtering, notebook 1
NODATA_THRESHOLD = 0.6
PROTECTED_SPLITS = ("hold",)    # never delete from the final evaluation split
