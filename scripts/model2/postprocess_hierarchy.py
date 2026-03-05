#!/usr/bin/env python3
"""
Post-processing utility for Model 2 predictions.

Enforces the anatomical constraint: IOU ⊂ prostate.
Voxels predicted as IOU that lie outside the predicted prostate are re-labelled
to prostate (conservative) or background (aggressive).

Also removes small connected components for both prostate and IOU labels.

Usage:
  python postprocess_hierarchy.py \\
    --pred-dir /path/to/predictions \\
    --output-dir /path/to/postprocessed \\
    [--prostate-label 1 --iou-label 2] \\
    [--min-prostate-vox 100 --min-iou-vox 10] \\
    [--outside-iou-strategy prostate]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as nd_label


def maybe_mkdir_p(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_small_components(
    arr: np.ndarray,
    label: int,
    min_vox: int,
    background_label: int = 0,
) -> np.ndarray:
    """Remove connected components of ``label`` with fewer than ``min_vox`` voxels."""
    mask = arr == label
    labeled, n = nd_label(mask)
    for i in range(1, n + 1):
        component = labeled == i
        if component.sum() < min_vox:
            arr[component] = background_label
    return arr


def enforce_iou_in_prostate(
    arr: np.ndarray,
    prostate_label: int,
    iou_label: int,
    outside_strategy: str = "prostate",
) -> np.ndarray:
    """
    Ensure IOU ⊂ prostate.

    IOU voxels outside the prostate are reassigned according to ``outside_strategy``:
      'prostate'   → relabel as prostate (conservative, maintains coverage)
      'background' → relabel as background (aggressive, removes false positives)
    """
    iou_mask = arr == iou_label
    prostate_mask = arr == prostate_label
    outside = iou_mask & ~prostate_mask

    if not outside.any():
        return arr

    if outside_strategy == "prostate":
        arr[outside] = prostate_label
    else:
        arr[outside] = 0

    return arr


def postprocess_case(
    pred_path: str,
    out_path: str,
    prostate_label: int,
    iou_label: int,
    min_prostate_vox: int,
    min_iou_vox: int,
    outside_strategy: str,
) -> None:
    pred_sitk = sitk.ReadImage(pred_path)
    arr = sitk.GetArrayFromImage(pred_sitk).astype(np.int16)

    # 1. Remove small prostate components
    arr = remove_small_components(arr, prostate_label, min_prostate_vox)

    # 2. Remove small IOU components
    arr = remove_small_components(arr, iou_label, min_iou_vox)

    # 3. Enforce IOU ⊂ prostate
    arr = enforce_iou_in_prostate(arr, prostate_label, iou_label, outside_strategy)

    out_sitk = sitk.GetImageFromArray(arr)
    out_sitk.CopyInformation(pred_sitk)
    maybe_mkdir_p(os.path.dirname(out_path))
    sitk.WriteImage(out_sitk, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Model 2 predictions: enforce IOU⊂prostate + remove small CCs."
    )
    parser.add_argument("--pred-dir", required=True, help="Directory with prediction NIfTI files.")
    parser.add_argument("--output-dir", required=True, help="Directory for post-processed outputs.")
    parser.add_argument("--prostate-label", type=int, default=1)
    parser.add_argument("--iou-label", type=int, default=2)
    parser.add_argument("--min-prostate-vox", type=int, default=100,
                        help="Min voxels for a prostate component (default: 100).")
    parser.add_argument("--min-iou-vox", type=int, default=10,
                        help="Min voxels for an IOU component (default: 10).")
    parser.add_argument(
        "--outside-iou-strategy",
        choices=["prostate", "background"],
        default="prostate",
        help="How to handle IOU voxels outside prostate: 'prostate' (conservative) "
             "or 'background' (aggressive). Default: prostate.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    if not pred_files:
        raise FileNotFoundError(f"No NIfTI files found in {pred_dir}")

    print(f"Post-processing {len(pred_files)} predictions...")
    maybe_mkdir_p(args.output_dir)

    for pred_path in pred_files:
        out_path = os.path.join(args.output_dir, pred_path.name)
        postprocess_case(
            str(pred_path),
            out_path,
            args.prostate_label,
            args.iou_label,
            args.min_prostate_vox,
            args.min_iou_vox,
            args.outside_iou_strategy,
        )
        print(f"  {pred_path.name}")

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
