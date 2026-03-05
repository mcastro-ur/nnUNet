#!/usr/bin/env python3
"""
Coarse-to-fine inference pipeline for pelvic MRI segmentation.

Pipeline:
  1. Model 1 predicts full-pelvis multi-organ segmentation.
  2. Extract prostate ROI bounding box from Model 1 prediction + margin.
     Falls back to center-crop if no prostate is found.
  3. Model 2 predicts prostate + IOU in the ROI.
  4. Fuse: keep Model 1 for all non-prostate/IOU classes; replace prostate/IOU
     with Model 2 predictions.
  5. Post-process: remove small connected components for prostate and IOU.
  6. Save fused segmentation.

Usage:
  python inference_coarse_to_fine.py \\
    --input-dir /path/to/T2_images \\
    --output-dir /path/to/output \\
    --model1-dir /path/to/model1/results \\
    --model2-dir /path/to/model2/results \\
    [--model1-dataset-id 001 --model2-dataset-id 002]
    [--folds 0 1 2 3 4]
    [--margin-mm 15]
    [--min-prostate-vox 100 --min-iou-vox 10]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# Label indices in Model 1 output (full-pelvis)
M1_PROSTATE_LABEL = 2   # adjust to match your Model 1 label map
M1_IOU_LABEL = 3        # adjust to match your Model 1 label map

# Label indices in Model 2 output (prostate-ROI)
M2_PROSTATE_LABEL = 1
M2_IOU_LABEL = 2


def maybe_mkdir_p(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_prostate_bbox(seg_arr: np.ndarray, prostate_label: int, spacing: tuple, margin_mm: float):
    """Return expanded voxel bbox of the prostate, or None if not found."""
    mask = seg_arr == prostate_label
    if not mask.any():
        return None
    coords = np.where(mask)
    lbs = [int(c.min()) for c in coords]
    ubs = [int(c.max()) + 1 for c in coords]
    # Expand by margin
    expanded_lbs = []
    expanded_ubs = []
    for i in range(len(lbs)):
        margin_vox = int(np.ceil(margin_mm / spacing[i]))
        expanded_lbs.append(max(0, lbs[i] - margin_vox))
        expanded_ubs.append(min(seg_arr.shape[i], ubs[i] + margin_vox))
    return expanded_lbs, expanded_ubs


def center_crop_bbox(shape: tuple, crop_size_mm: tuple, spacing: tuple):
    """Fallback: center crop of fixed size."""
    center = [s // 2 for s in shape]
    lbs = []
    ubs = []
    for i in range(len(shape)):
        half = int(np.ceil(crop_size_mm[i] / spacing[i] / 2))
        lbs.append(max(0, center[i] - half))
        ubs.append(min(shape[i], center[i] + half))
    return lbs, ubs


def remove_small_components(arr: np.ndarray, label: int, min_vox: int) -> np.ndarray:
    """Remove connected components of `label` smaller than min_vox voxels."""
    from scipy.ndimage import label as nd_label
    mask = (arr == label)
    labeled, n = nd_label(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_vox:
            arr[labeled == i] = 0
    return arr


def run_nnunet_predictor(
    input_dir: str,
    output_dir: str,
    dataset_id: str,
    model_dir: str,
    folds: list,
    trainer: str = "nnUNetTrainer",
    configuration: str = "3d_fullres",
):
    """Run nnUNetPredictor for a given dataset."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
    )


def fuse_predictions(
    m1_path: str,
    m2_path: str,
    out_path: str,
    roi_bbox: tuple,
    min_prostate_vox: int,
    min_iou_vox: int,
) -> None:
    """Fuse Model 1 and Model 2 predictions."""
    m1_sitk = sitk.ReadImage(m1_path)
    m2_sitk = sitk.ReadImage(m2_path)

    m1_arr = sitk.GetArrayFromImage(m1_sitk).astype(np.int16)
    m2_arr = sitk.GetArrayFromImage(m2_sitk).astype(np.int16)

    fused = m1_arr.copy()
    lbs, ubs = roi_bbox

    # Clear prostate/IOU in Model 1 within the ROI
    slices = tuple(slice(lb, ub) for lb, ub in zip(lbs, ubs))
    roi_fused = fused[slices]
    roi_fused[roi_fused == M1_PROSTATE_LABEL] = 0
    roi_fused[roi_fused == M1_IOU_LABEL] = 0

    # Insert Model 2 predictions
    roi_m2 = m2_arr[slices] if m2_arr.shape == m1_arr.shape else m2_arr

    # Handle possible shape mismatch (m2 was cropped)
    if m2_arr.shape != m1_arr.shape:
        # m2 was predicted on the cropped ROI
        roi_m2 = m2_arr
    else:
        roi_m2 = m2_arr[slices]

    roi_fused[roi_m2 == M2_PROSTATE_LABEL] = M1_PROSTATE_LABEL
    roi_fused[roi_m2 == M2_IOU_LABEL] = M1_IOU_LABEL
    fused[slices] = roi_fused

    # Post-processing: remove small connected components
    fused = remove_small_components(fused, M1_PROSTATE_LABEL, min_prostate_vox)
    fused = remove_small_components(fused, M1_IOU_LABEL, min_iou_vox)

    out_sitk = sitk.GetImageFromArray(fused)
    out_sitk.CopyInformation(m1_sitk)
    maybe_mkdir_p(os.path.dirname(out_path))
    sitk.WriteImage(out_sitk, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine inference: Model 1 (pelvis) + Model 2 (prostate ROI)."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with T2 NIfTI images.")
    parser.add_argument("--output-dir", required=True, help="Output directory for fused predictions.")
    parser.add_argument("--model1-dir", required=True, help="Model 1 trained model folder.")
    parser.add_argument("--model2-dir", required=True, help="Model 2 trained model folder.")
    parser.add_argument("--model1-dataset-id", default="001")
    parser.add_argument("--model2-dataset-id", default="002")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--margin-mm", type=float, default=15.0,
                        help="Margin in mm around prostate ROI (default: 15).")
    parser.add_argument("--fallback-crop-mm", nargs=3, type=float, default=[80, 80, 80],
                        help="Fallback center crop size in mm if no prostate found.")
    parser.add_argument("--min-prostate-vox", type=int, default=100)
    parser.add_argument("--min-iou-vox", type=int, default=10)
    parser.add_argument("--configuration", default="3d_fullres")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    m1_pred_dir = output_dir / "model1_predictions"
    m2_input_dir = output_dir / "model2_input"
    m2_pred_dir = output_dir / "model2_predictions"
    fused_dir = output_dir / "fused"

    maybe_mkdir_p(str(m1_pred_dir))
    maybe_mkdir_p(str(m2_input_dir))
    maybe_mkdir_p(str(m2_pred_dir))
    maybe_mkdir_p(str(fused_dir))

    # Step 1: Run Model 1
    print("=== Step 1: Model 1 full-pelvis prediction ===")
    run_nnunet_predictor(
        str(input_dir), str(m1_pred_dir),
        args.model1_dataset_id, args.model1_dir,
        args.folds, configuration=args.configuration,
    )

    # Steps 2-3: For each case, extract ROI and run Model 2
    cases = sorted(input_dir.glob("*_0000.nii.gz"))
    for img_path in cases:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        m1_pred_path = m1_pred_dir / f"{case_id}.nii.gz"
        if not m1_pred_path.exists():
            print(f"WARNING: Model 1 prediction not found for {case_id}, skipping.")
            continue

        # Extract prostate ROI bbox
        m1_sitk = sitk.ReadImage(str(m1_pred_path))
        m1_arr = sitk.GetArrayFromImage(m1_sitk)
        spacing = m1_sitk.GetSpacing()[::-1]  # (Z,Y,X)

        bbox = get_prostate_bbox(m1_arr, M1_PROSTATE_LABEL, spacing, args.margin_mm)
        if bbox is None:
            print(f"  {case_id}: No prostate in Model 1 prediction, using center crop fallback.")
            bbox = center_crop_bbox(m1_arr.shape, args.fallback_crop_mm, spacing)

        lbs, ubs = bbox

        # Crop original image to ROI
        img_sitk = sitk.ReadImage(str(img_path))
        img_arr = sitk.GetArrayFromImage(img_sitk)
        slices = tuple(slice(lb, ub) for lb, ub in zip(lbs, ubs))
        img_roi = img_arr[slices]

        roi_sitk = sitk.GetImageFromArray(img_roi)
        roi_sitk.SetSpacing(img_sitk.GetSpacing())
        roi_sitk.SetDirection(img_sitk.GetDirection())
        roi_out = m2_input_dir / case_id
        maybe_mkdir_p(str(roi_out))
        sitk.WriteImage(roi_sitk, str(roi_out / f"{case_id}_0000.nii.gz"))

    # Run Model 2 on all ROIs
    print("=== Step 3: Model 2 prostate+IOU prediction ===")
    run_nnunet_predictor(
        str(m2_input_dir), str(m2_pred_dir),
        args.model2_dataset_id, args.model2_dir,
        args.folds,
        trainer="nnUNetTrainerSkeletonRecall_ProstateIOU",
        configuration=args.configuration,
    )

    # Step 4-6: Fuse predictions
    print("=== Step 4: Fusing predictions ===")
    for img_path in cases:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        m1_pred_path = m1_pred_dir / f"{case_id}.nii.gz"
        m2_pred_path = m2_pred_dir / f"{case_id}.nii.gz"
        if not m1_pred_path.exists() or not m2_pred_path.exists():
            continue

        m1_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(m1_pred_path)))
        spacing = sitk.ReadImage(str(m1_pred_path)).GetSpacing()[::-1]
        bbox = get_prostate_bbox(m1_arr, M1_PROSTATE_LABEL, spacing, args.margin_mm)
        if bbox is None:
            bbox = center_crop_bbox(m1_arr.shape, args.fallback_crop_mm, spacing)

        fuse_predictions(
            str(m1_pred_path),
            str(m2_pred_path),
            str(fused_dir / f"{case_id}.nii.gz"),
            bbox,
            args.min_prostate_vox,
            args.min_iou_vox,
        )
        print(f"  Fused: {case_id}")

    print(f"\nDone. Fused predictions saved to: {fused_dir}")


if __name__ == "__main__":
    main()
