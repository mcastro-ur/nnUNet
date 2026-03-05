#!/usr/bin/env python3
"""
Analyze prostate bounding box sizes and IOU-to-prostate-edge margins across
the dataset to determine the optimal cropping margin for Model 2.

Outputs statistics (mean, std, percentiles) of:
  - Prostate bbox size in mm (Z, Y, X)
  - Distance from IOU voxels to the nearest prostate boundary

Usage:
  python analyze_prostate_bbox.py --label-dir /path/to/labels [--output-csv stats.csv]
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# GT label values to analyse
GT_PROSTATE_LABEL = 2
GT_IOU_LABEL = 3


def compute_bbox_size_mm(lbl_arr: np.ndarray, label: int, spacing: tuple) -> list:
    """Return bbox size in mm for a given label."""
    mask = lbl_arr == label
    if not mask.any():
        return None
    coords = np.where(mask)
    sizes_mm = [
        (int(c.max()) - int(c.min()) + 1) * spacing[i]
        for i, c in enumerate(coords)
    ]
    return sizes_mm


def min_distance_iou_to_prostate_edge_mm(
    lbl_arr: np.ndarray, spacing: tuple
) -> float:
    """
    Compute the minimum distance from any IOU voxel to the nearest
    prostate boundary voxel (in mm). Returns None if IOU or prostate absent.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    prostate_mask = lbl_arr == GT_PROSTATE_LABEL
    iou_mask = lbl_arr == GT_IOU_LABEL

    if not prostate_mask.any() or not iou_mask.any():
        return None

    # Prostate boundary = prostate - eroded prostate
    eroded = binary_erosion(prostate_mask)
    boundary = prostate_mask & ~eroded

    # Distance from every voxel to the nearest boundary voxel
    inv_boundary = ~boundary
    dist = distance_transform_edt(inv_boundary, sampling=spacing)

    # Min distance from IOU voxels to the prostate boundary
    return float(dist[iou_mask].min())


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prostate bbox sizes and IOU-prostate margin."
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        help="Directory containing GT label NIfTI files (*.nii.gz).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV file to save per-case statistics.",
    )
    parser.add_argument(
        "--prostate-label",
        type=int,
        default=GT_PROSTATE_LABEL,
        help=f"Prostate label value (default: {GT_PROSTATE_LABEL}).",
    )
    parser.add_argument(
        "--iou-label",
        type=int,
        default=GT_IOU_LABEL,
        help=f"IOU label value (default: {GT_IOU_LABEL}).",
    )
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    label_files = sorted(label_dir.glob("*.nii.gz"))

    if not label_files:
        raise FileNotFoundError(f"No NIfTI label files found in {label_dir}")

    print(f"Analysing {len(label_files)} cases...")

    rows = []
    bbox_sizes = []
    min_dists = []

    for lbl_path in label_files:
        case_id = lbl_path.stem.replace(".nii", "")
        lbl_sitk = sitk.ReadImage(str(lbl_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk)
        spacing = lbl_sitk.GetSpacing()[::-1]  # (Z,Y,X)

        bbox_mm = compute_bbox_size_mm(lbl_arr, args.prostate_label, spacing)
        min_dist = min_distance_iou_to_prostate_edge_mm(lbl_arr, spacing)

        row = {
            "case_id": case_id,
            "bbox_z_mm": bbox_mm[0] if bbox_mm else "N/A",
            "bbox_y_mm": bbox_mm[1] if bbox_mm else "N/A",
            "bbox_x_mm": bbox_mm[2] if bbox_mm else "N/A",
            "min_iou_to_prostate_edge_mm": min_dist if min_dist is not None else "N/A",
        }
        rows.append(row)

        if bbox_mm:
            bbox_sizes.append(bbox_mm)
        if min_dist is not None:
            min_dists.append(min_dist)

    # Print summary statistics
    if bbox_sizes:
        bbox_arr = np.array(bbox_sizes)
        print("\n=== Prostate bounding box size (mm) ===")
        for dim, name in enumerate(["Z", "Y", "X"]):
            vals = bbox_arr[:, dim]
            print(
                f"  {name}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                f"p50={np.percentile(vals,50):.1f}, "
                f"p95={np.percentile(vals,95):.1f}, max={vals.max():.1f}"
            )

    if min_dists:
        d = np.array(min_dists)
        print("\n=== Min distance IOU → prostate boundary (mm) ===")
        print(
            f"  mean={d.mean():.1f}, std={d.std():.1f}, "
            f"p5={np.percentile(d,5):.1f}, min={d.min():.1f}"
        )
        print(
            f"\n  Recommended margin: max prostate half-size + "
            f"p5 IOU-to-edge distance (~{d.min():.0f} mm safety)."
        )

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-case statistics saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
