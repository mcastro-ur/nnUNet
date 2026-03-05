"""
Skeleton transform that computes skeleton only for the IOU (intraprostatic urethra) class.
The prostate is a blob-like structure; skeletonization is only meaningful for the tubular IOU.

Used in Model 2 (prostate + IOU segmentation with Skeleton Recall loss).
"""
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize, dilation


class SkeletonTransformIOUOnly:
    """Skeleton transform that only computes skeleton for IOU (label=iou_label).

    Prostate skeleton is meaningless (it is a blob, not tubular).
    IOU skeleton preserves connectivity of the tubular urethra.

    Adds a ``skel`` key to the data dict containing the skeleton of the IOU.
    The skeleton is a label map with value ``iou_label`` at skeleton voxels
    and 0 elsewhere. If ``do_tube=True``, the skeleton is dilated twice to
    create a thin tube around the centreline.

    Compatible with the batchgeneratorsv2 transform pipeline (operates on
    single-sample dicts with keys 'image' / 'segmentation' as torch.Tensors).
    """

    def __init__(self, iou_label: int = 2, do_tube: bool = True):
        self.iou_label = iou_label
        self.do_tube = do_tube

    def __call__(self, **data_dict) -> dict:
        seg = data_dict["segmentation"]

        # Accept torch.Tensor or numpy array
        if isinstance(seg, torch.Tensor):
            seg_np = seg.numpy()
            is_tensor = True
        else:
            seg_np = np.asarray(seg)
            is_tensor = False

        # seg_np shape: (1, Z, Y, X)  or  (Z, Y, X)
        if seg_np.ndim == 3:
            seg_np = seg_np[None]  # -> (1, Z, Y, X)

        skel_out = np.zeros_like(seg_np, dtype=np.int16)

        iou_mask = (seg_np[0] == self.iou_label)
        if iou_mask.sum() > 0:
            skel = skeletonize(iou_mask)  # bool array, same shape as iou_mask
            skel = (skel > 0).astype(np.uint8)
            if self.do_tube:
                skel = dilation(dilation(skel))  # 2-voxel tube around skeleton
            skel_out[0][skel > 0] = self.iou_label  # assign IOU label to skeleton voxels

        if is_tensor:
            data_dict["skel"] = torch.from_numpy(skel_out)
        else:
            data_dict["skel"] = skel_out

        return data_dict


class DownsampleSkeletonForDSTransform:
    """Downsample the 'skel' key for deep supervision, mirroring what
    DownsampleSegForDSTransform does for 'segmentation'.

    Converts data_dict['skel'] (a single tensor) into a list of tensors,
    one per DS scale, using nearest-neighbour interpolation.
    """

    def __init__(self, ds_scales: Union[List, Tuple]):
        self.ds_scales = ds_scales

    def __call__(self, **data_dict) -> dict:
        if "skel" not in data_dict or data_dict["skel"] is None:
            return data_dict

        skel = data_dict["skel"]  # (C, Z, Y, X) tensor
        results = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * (skel.ndim - 1)
            else:
                assert len(s) == skel.ndim - 1

            if all(i == 1 for i in s):
                results.append(skel)
            else:
                new_shape = [round(skel.shape[i + 1] * s[i]) for i in range(len(s))]
                dtype = skel.dtype
                results.append(
                    F.interpolate(
                        skel[None].float(), size=new_shape, mode="nearest-exact"
                    )[0].to(dtype)
                )
        data_dict["skel"] = results
        return data_dict
