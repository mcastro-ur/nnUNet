#!/usr/bin/env python3
"""
Create 5-fold stratified cross-validation splits for Model 2, ensuring each
fold contains cases from all three centres: CEM, Dijon (DIJ), and PROSTATEx (PEX).

Determines centre from filename prefix:
  CEM_  → CEM
  DIJ_  → Dijon
  PEX_  → PROSTATEx

Saves splits_final.json in the nnU-Net preprocessed dataset directory.

Usage:
  python create_stratified_splits.py --preprocessed-dir PATH_TO_PREPROCESSED
"""
import argparse
import json
import os
from pathlib import Path


def get_centre(case_id: str) -> str:
    for prefix in ("CEM_", "DIJ_", "PEX_"):
        if case_id.startswith(prefix):
            return prefix.rstrip("_")
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(
        description="Create 5-fold stratified splits for Model 2."
    )
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Path to the nnU-Net preprocessed dataset directory "
             "(contains dataset.json and the preprocessed files).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345).",
    )
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)

    # Collect case identifiers from the preprocessed directory
    case_ids = sorted(
        f.stem.replace(".nii", "")
        for f in preprocessed_dir.glob("*.npy")
        if not f.name.startswith(".")
    )
    # Also try .npz
    if not case_ids:
        case_ids = sorted(
            f.stem
            for f in preprocessed_dir.glob("*.npz")
            if not f.name.startswith(".")
        )

    if not case_ids:
        # Fall back: read from dataset.json
        ds_json_path = preprocessed_dir.parent / "dataset.json"
        if not ds_json_path.exists():
            ds_json_path = preprocessed_dir / "dataset.json"
        with open(ds_json_path) as f:
            ds = json.load(f)
        case_ids = sorted(
            os.path.basename(p).replace(".nii.gz", "").replace("_0000", "")
            for p in ds.get("training", [])
        )

    if not case_ids:
        raise RuntimeError(
            f"No case identifiers found in {preprocessed_dir}. "
            "Please check the path."
        )

    print(f"Found {len(case_ids)} cases.")

    # Determine centre labels
    centres = [get_centre(c) for c in case_ids]
    unique_centres = sorted(set(centres))
    print("Centre distribution:")
    for ctr in unique_centres:
        n = sum(1 for c in centres if c == ctr)
        print(f"  {ctr}: {n}")

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        raise ImportError(
            "scikit-learn is required: pip install scikit-learn"
        )

    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )

    splits = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(case_ids, centres)):
        tr_keys = [case_ids[i] for i in tr_idx]
        val_keys = [case_ids[i] for i in val_idx]
        splits.append({"train": tr_keys, "val": val_keys})

        # Print per-centre stats
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(tr_keys)}, Val: {len(val_keys)}")
        for ctr in unique_centres:
            n_tr = sum(1 for k in tr_keys if get_centre(k) == ctr)
            n_val = sum(1 for k in val_keys if get_centre(k) == ctr)
            print(f"    {ctr}: train={n_tr}, val={n_val}")

    splits_path = preprocessed_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits to: {splits_path}")


if __name__ == "__main__":
    main()
