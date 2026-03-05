# Model 2: Prostate + Intraprostatic Urethra Segmentation with Skeleton Recall

## Overview

Model 2 implements a prostate-focused segmentation model for prostate and
intraprostatic urethra (IOU) using **Skeleton Recall Loss** on the IOU class.
This is part of a coarse-to-fine two-model strategy for pelvic MRI segmentation.

- **Model 1** (master): Full-pelvis multi-organ segmentation with BoundaryDiceCE loss.
- **Model 2** (this PR): Prostate + IOU with Skeleton Recall Loss on the IOU class only.

### Loss function (Model 2)

```
Loss = 1.0 × Dice(output, target)
     + 1.0 × CrossEntropy(output, target)
     + 1.0 × SkeletonRecall(output, skel_IOU)
```

The `SkeletonRecall` term is computed only on skeleton voxels of the IOU
ground truth.  The prostate is a blob-like structure, so its skeleton is not
computed; only the tubular IOU is skeletonised.

Based on [MIC-DKFZ/Skeleton-Recall](https://github.com/MIC-DKFZ/Skeleton-Recall) (ECCV 2024, Kirchhoff et al.)

---

## Label mapping

| Class               | GT label | Model 2 label |
|---------------------|----------|---------------|
| Background          | 0        | 0             |
| Prostate            | 2        | 1             |
| Intraprostatic IOU  | 3        | 2             |

---

## Dataset building (from GT labels)

**Important**: Model 2 is trained on crops from GT labels, **not** Model 1
predictions.  nnU-Net handles resampling and normalisation internally.

### 1. Build the cropped dataset

```bash
python scripts/model2/build_dataset_prostate_iou.py \
  --output-dir ~/nnunet_data \
  --dataset-id 002
```

Edit the path constants at the top of `build_dataset_prostate_iou.py` to
point to your CEM, Dijon, and PROSTATEx data directories.

### 2. Analyse prostate bbox sizes (optional)

```bash
python scripts/model2/analyze_prostate_bbox.py \
  --label-dir ~/nnunet_data/Dataset002_ProstateIOU/labelsTr \
  --output-csv /tmp/bbox_stats.csv
```

### 3. Oversample CEM / Dijon to balance with PROSTATEx

```bash
python scripts/model2/oversample_cem_dijon.py \
  --dataset-dir ~/nnunet_data/Dataset002_ProstateIOU \
  --n-copies 3
```

### 4. Run nnU-Net fingerprint + planning + preprocessing

```bash
export nnUNet_raw=~/nnunet_data
export nnUNet_preprocessed=~/nnunet_preprocessed
export nnUNet_results=~/nnunet_results

nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity
```

### 5. Create stratified 5-fold splits

```bash
python scripts/model2/create_stratified_splits.py \
  --preprocessed-dir ~/nnunet_preprocessed/Dataset002_ProstateIOU/nnUNetPlans_3d_fullres
```

---

## Training

```bash
nnUNetv2_train 002 3d_fullres 0 \
  --trainer nnUNetTrainerSkeletonRecall_ProstateIOU \
  --npz

# Train all 5 folds
for fold in 0 1 2 3 4; do
  nnUNetv2_train 002 3d_fullres $fold \
    --trainer nnUNetTrainerSkeletonRecall_ProstateIOU
done
```

---

## Inference (coarse-to-fine)

```bash
python scripts/model2/inference_coarse_to_fine.py \
  --input-dir /data/test_images \
  --output-dir /data/coarse_to_fine_output \
  --model1-dir ~/nnunet_results/Dataset001_Pelvis/nnUNetTrainerV2_BoundaryDiceCE__nnUNetPlans__3d_fullres \
  --model2-dir ~/nnunet_results/Dataset002_ProstateIOU/nnUNetTrainerSkeletonRecall_ProstateIOU__nnUNetPlans__3d_fullres \
  --margin-mm 15
```

---

## Post-processing

Enforce anatomical constraint IOU ⊂ prostate:

```bash
python scripts/model2/postprocess_hierarchy.py \
  --pred-dir /data/coarse_to_fine_output/fused \
  --output-dir /data/postprocessed \
  --outside-iou-strategy prostate
```

---

## Key implementation files

| File | Description |
|------|-------------|
| `nnunetv2/training/loss/skeleton_recall_loss.py` | `SoftSkeletonRecallLoss` |
| `nnunetv2/training/loss/compound_skeleton_losses.py` | `DC_SkelREC_and_CE_loss` |
| `nnunetv2/training/data_augmentation/custom_transforms/skeletonization_iou.py` | `SkeletonTransformIOUOnly`, `DownsampleSkeletonForDSTransform` |
| `nnunetv2/training/dataloading/data_loader_3d_skel.py` | `nnUNetDataLoader3DSkel` |
| `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerSkeletonRecall_ProstateIOU.py` | Main trainer |
