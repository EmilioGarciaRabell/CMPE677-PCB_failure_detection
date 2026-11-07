"""Build train/val/test folders from the raw DsPCBSD+ download.

test      = the dataset's official val set (untouched)
train/val = the official train set, split 85/15 with a fixed seed

Filename lists are saved in splits/ and committed to git. If they already
exist they are reused, so every machine rebuilds the exact same split.
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/dspcbsd_raw/Data_YOLO"
OUT = ROOT / "data/dspcbsd"
SPLITS = ROOT / "splits"
VAL_FRACTION = 0.15
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Order matches the YOLO class IDs (COCO category id - 1).
# SH short circuit, SP spur, SC spurious copper, OP open circuit,
# MB mouse bite, HB hole breakout, CS conductor scratch,
# CFO conductor foreign object, BMFO base material foreign object
NAMES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]

# Which raw folder each of our splits comes from
SOURCE = {"train": "train", "val": "train", "test": "val"}


def list_images(folder):
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def make_lists():
    pool = list_images(RAW / "images/train")
    test = list_images(RAW / "images/val")
    random.Random(SEED).shuffle(pool)
    n_val = round(len(pool) * VAL_FRACTION)
    return {"train": sorted(pool[n_val:]), "val": sorted(pool[:n_val]), "test": test}


def load_or_make_lists():
    files = {s: SPLITS / f"{s}.txt" for s in SOURCE}
    if all(f.exists() for f in files.values()):
        print("Reusing existing split lists in splits/")
        return {s: f.read_text().split() for s, f in files.items()}
    lists = make_lists()
    SPLITS.mkdir(exist_ok=True)
    for s, names in lists.items():
        files[s].write_text("\n".join(names) + "\n")
    print("Created new split lists in splits/")
    return lists


def build(lists):
    # Sanity check: no image may appear in two splits
    sets = {s: set(n) for s, n in lists.items()}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"

    if OUT.exists():
        shutil.rmtree(OUT)  # only removes the generated folder, never the raw data
    for split, names in lists.items():
        for kind in ("images", "labels"):
            (OUT / kind / split).mkdir(parents=True)
        for name in names:
            stem = Path(name).stem
            src = SOURCE[split]
            shutil.copy2(RAW / "images" / src / name, OUT / "images" / split / name)
            shutil.copy2(RAW / "labels" / src / f"{stem}.txt", OUT / "labels" / split / f"{stem}.txt")
        print(f"{split}: {len(names)} images")


def write_yaml():
    lines = [
        f"path: '{OUT.as_posix()}'",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    build(load_or_make_lists())
    write_yaml()
