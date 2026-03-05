"""
Skeleton-aware 3-D data loader that propagates the ``skel`` key produced by
:class:`SkeletonTransformIOUOnly` through the augmentation pipeline and deep
supervision downsampling.

Inherits from :class:`nnUNetDataLoader` and overrides
``generate_train_batch`` to also yield the ``skel`` key alongside ``data``
and ``target``.
"""
import torch

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from threadpoolctl import threadpool_limits


class nnUNetDataLoader3DSkel(nnUNetDataLoader):
    """Data loader variant that yields 'skel' in addition to 'data' and 'target'.

    The 'skel' key is created by :class:`SkeletonTransformIOUOnly` which must
    be part of the transform pipeline passed to this data loader.  If deep
    supervision is active, 'skel' will be a list of tensors (one per DS scale),
    matching the structure of 'target'.
    """

    def generate_train_batch(self):
        # Delegate the bulk of the work to the parent class, but we need to
        # intercept the transform output to capture the 'skel' key.
        # We duplicate the relevant part of the parent's generate_train_batch.
        import numpy as np
        from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack(
                    (seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None])
                )
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    skels = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(
                            **{"image": data_all[b], "segmentation": seg_all[b]}
                        )
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                        skels.append(tmp.get("skel", None))

                    data_all = torch.stack(images)

                    if isinstance(segs[0], list):
                        seg_all = [
                            torch.stack([s[i] for s in segs])
                            for i in range(len(segs[0]))
                        ]
                    else:
                        seg_all = torch.stack(segs)

                    del segs, images

                    # Handle skel: may be None (no skeleton transform), a
                    # single tensor, or a list of tensors (after DS downsampling)
                    if skels[0] is not None:
                        if isinstance(skels[0], list):
                            skel_all = [
                                torch.stack([s[i] for s in skels])
                                for i in range(len(skels[0]))
                            ]
                        else:
                            skel_all = torch.stack(skels)
                        del skels
                        return {
                            "data": data_all,
                            "target": seg_all,
                            "skel": skel_all,
                            "keys": selected_keys,
                        }

                    del skels
                    return {"data": data_all, "target": seg_all, "keys": selected_keys}

        return {"data": data_all, "target": seg_all, "keys": selected_keys}
