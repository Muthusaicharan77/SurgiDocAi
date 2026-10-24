"""Download third-party pretrained weights into weights/ (not committed to git).

    pip install gdown
    python scripts/download_weights.py

Trans-SVNet ResNet50 phase classifier: Gao et al., MICCAI 2021,
github.com/xjgaocs/Trans-SVNet. Downloaded from the authors' Google Drive folder
linked in their repo (Model/README.txt). Credit the authors if you use it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

WEIGHTS = {
    "phase/transsvnet_resnet50.pth": {
        "drive_id": "15nuDP_fSoehHhH-sWQ9hB9ybO_FdSZzk",
        "original_name": "resnetfc_ce_epoch_15_length_1_opt_0_mulopt_1_flip_1_crop_1"
                         "_batch_100_train_9946_val_8404_test_7961.pth",
        "sha256": "4214551166bd177bcef70ef499fbaf96c2e582f4daa91fa1419e5994e57c0118",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import gdown

    root = Path("weights")
    for rel, info in WEIGHTS.items():
        target = root / rel
        if target.exists() and sha256(target) == info["sha256"]:
            print(f"ok      {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=info["drive_id"], output=str(target), quiet=False)
        digest = sha256(target)
        if digest != info["sha256"]:
            target.unlink()
            raise SystemExit(f"checksum mismatch for {rel}: {digest}")
        print(f"fetched {target}")


if __name__ == "__main__":
    main()
