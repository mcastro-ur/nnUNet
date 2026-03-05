#!/usr/bin/env python3
"""
Build nnU-Net dataset for Model 2 (prostate + IOU segmentation) from ground-truth labels.

Strategy:
  - CEM cases   : crop around GT prostate bbox (label=2) + 15 mm margin, remap labels
  - Dijon cases : same as CEM
  - PROSTATEx   : crop around GT prostate bbox + 10 mm margin, remap labels

Label remapping after cropping:
  {0 → 0, 2 → 1 (prostate), 3 → 2 (IOU), everything_else → 0}

Output dataset.json:
  channel_names: {"0": "T2"}
  labels: {"background": 0, "prostate": 1, "intraprostaticurethra": 2}

Usage:
  python build_dataset_prostate_iou.py [--output-dir OUTPUT_DIR]
"""
import argparse
import os
import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np

# ── Configurable paths ────────────────────────────────────────────────────────
# Adjust these to your local file system layout before running.
CEM_IMAGE_DIR = "/data/CEM/images"          # T2 NIfTI images, filename: CEM_XXX_0000.nii.gz
CEM_LABEL_DIR = "/data/CEM/labels"          # GT segmentations, filename: CEM_XXX.nii.gz
DIJON_IMAGE_DIR = "/data/Dijon/images"
DIJON_LABEL_DIR = "/data/Dijon/labels"
PROSTX_IMAGE_DIR = "/data/PROSTATEx/images"
PROSTX_LABEL_DIR = "/data/PROSTATEx/labels"

# Cropping margins in mm
CEM_MARGIN_MM = 15.0
DIJON_MARGIN_MM = 15.0
PROSTX_MARGIN_MM = 10.0

# Label values in the GT segmentations
GT_PROSTATE_LABEL = 2
GT_IOU_LABEL = 3
# ── End configurable paths ────────────────────────────────────────────────────


def maybe_mkdir_p(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def compute_prostate_bbox(label_arr: np.ndarray, prostate_label: int, spacing: tuple) -> tuple:
    """Return (lbs, ubs) voxel indices of the bounding box of the prostate label."""
    mask = label_arr == prostate_label
    if not mask.any():
        return None
    coords = np.where(mask)
    lbs = [int(c.min()) for c in coords]
    ubs = [int(c.max()) + 1 for c in coords]
    return lbs, ubs


def expand_bbox(lbs, ubs, shape, margin_mm, spacing):
    """Expand bbox by margin_mm in each direction, clamped to image bounds."""
    expanded_lbs = []
    expanded_ubs = []
    for i in range(len(lbs)):
        margin_vox = int(np.ceil(margin_mm / spacing[i]))
        expanded_lbs.append(max(0, lbs[i] - margin_vox))
        expanded_ubs.append(min(shape[i], ubs[i] + margin_vox))
    return expanded_lbs, expanded_ubs


def remap_labels(label_arr: np.ndarray) -> np.ndarray:
    """Remap GT labels to Model 2 label space: {prostate→1, IOU→2, else→0}."""
    out = np.zeros_like(label_arr, dtype=np.uint8)
    out[label_arr == GT_PROSTATE_LABEL] = 1
    out[label_arr == GT_IOU_LABEL] = 2
    return out


def process_case(
    image_path: str,
    label_path: str,
    out_image_path: str,
    out_label_path: str,
    margin_mm: float,
) -> bool:
    """Crop image and label around prostate bbox, remap labels, save."""
    img_sitk = sitk.ReadImage(image_path)
    lbl_sitk = sitk.ReadImage(label_path)

    img_arr = sitk.GetArrayFromImage(img_sitk)  # (Z, Y, X)
    lbl_arr = sitk.GetArrayFromImage(lbl_sitk)

    spacing = lbl_sitk.GetSpacing()[::-1]  # convert (X,Y,Z) → (Z,Y,X)

    bbox = compute_prostate_bbox(lbl_arr, GT_PROSTATE_LABEL, spacing)
    if bbox is None:
        print(f"  WARNING: No prostate label found in {label_path}, skipping.")
        return False

    lbs, ubs = bbox
    lbs, ubs = expand_bbox(lbs, ubs, lbl_arr.shape, margin_mm, spacing)

    # Crop
    slices = tuple(slice(lb, ub) for lb, ub in zip(lbs, ubs))
    img_cropped = img_arr[slices]
    lbl_cropped = lbl_arr[slices]

    # Remap labels
    lbl_remapped = remap_labels(lbl_cropped)

    # Reconstruct SITK images with correct metadata
    def arr_to_sitk(arr, reference):
        out = sitk.GetImageFromArray(arr)
        out.SetSpacing(reference.GetSpacing())
        out.SetDirection(reference.GetDirection())
        # Update origin to reflect cropping
        orig = list(reference.GetOrigin())
        dir_mat = np.array(reference.GetDirection()).reshape(3, 3)
        # lbs in (Z,Y,X) → offset in (X,Y,Z) world coords
        offset_vox = np.array([lbs[2], lbs[1], lbs[0]], dtype=float)
        sp = np.array(reference.GetSpacing())
        offset_world = dir_mat @ (offset_vox * sp)
        new_origin = [orig[i] + offset_world[i] for i in range(3)]
        out.SetOrigin(new_origin)
        return out

    img_out = arr_to_sitk(img_cropped, img_sitk)
    lbl_out = arr_to_sitk(lbl_remapped, lbl_sitk)

    maybe_mkdir_p(os.path.dirname(out_image_path))
    maybe_mkdir_p(os.path.dirname(out_label_path))
    sitk.WriteImage(img_out, out_image_path)
    sitk.WriteImage(lbl_out, out_label_path)
    return True


def collect_cases(image_dir: str, label_dir: str, prefix: str):
    """Return list of (case_id, image_path, label_path) for all cases."""
    cases = []
    if not os.path.isdir(image_dir):
        return cases
    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith("_0000.nii.gz"):
            continue
        case_id = fname.replace("_0000.nii.gz", "")
        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, case_id + ".nii.gz")
        if not os.path.isfile(lbl_path):
            print(f"WARNING: label not found for {case_id}, skipping.")
            continue
        cases.append((case_id, img_path, lbl_path))
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Build nnU-Net dataset for Model 2 (prostate + IOU)."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.expanduser("~"), "nnunet_data", "Dataset_ProstateIOU"),
        help="Root output directory for the nnU-Net dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default="002",
        help="nnU-Net dataset ID (default: 002).",
    )
    args = parser.parse_args()

    dataset_name = f"Dataset{args.dataset_id}_ProstateIOU"
    out_dir = os.path.join(args.output_dir, dataset_name)
    image_tr_dir = os.path.join(out_dir, "imagesTr")
    label_tr_dir = os.path.join(out_dir, "labelsTr")
    maybe_mkdir_p(image_tr_dir)
    maybe_mkdir_p(label_tr_dir)

    sources = [
        ("CEM", CEM_IMAGE_DIR, CEM_LABEL_DIR, CEM_MARGIN_MM),
        ("DIJ", DIJON_IMAGE_DIR, DIJON_LABEL_DIR, DIJON_MARGIN_MM),
        ("PEX", PROSTX_IMAGE_DIR, PROSTX_LABEL_DIR, PROSTX_MARGIN_MM),
    ]

    processed = []
    for prefix, img_dir, lbl_dir, margin in sources:
        cases = collect_cases(img_dir, lbl_dir, prefix)
        print(f"Found {len(cases)} {prefix} cases.")
        for case_id, img_path, lbl_path in cases:
            out_img = os.path.join(image_tr_dir, f"{case_id}_0000.nii.gz")
            out_lbl = os.path.join(label_tr_dir, f"{case_id}.nii.gz")
            ok = process_case(img_path, lbl_path, out_img, out_lbl, margin)
            if ok:
                processed.append(case_id)

    print(f"\nProcessed {len(processed)} cases total.")

    # Generate dataset.json
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {
            "background": 0,
            "prostate": 1,
            "intraprostaticurethra": 2,
        },
        "numTraining": len(processed),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Prostate + intraprostatic urethra segmentation (Model 2).",
        "reference": "",
        "licence": "",
        "release": "0.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    save_json(dataset_json, os.path.join(out_dir, "dataset.json"))
    print(f"Dataset saved to: {out_dir}")


if __name__ == "__main__":
    main()
