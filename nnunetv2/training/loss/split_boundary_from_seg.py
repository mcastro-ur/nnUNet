# split_boundary_from_seg.py
import numpy as np

class SplitBoundaryFromSeg:
    """
    Après SpatialTransform :
    - Reprend sample['seg'] (1 + C_small, Zp, Yp, Xp)
    - Sépare: seg_reconstruit -> (1, Zp, Yp, Xp), boundary -> (C_small, Zp, Yp, Xp)
    """
    def __init__(self, boundary_key='boundary', seg_key='seg', meta_key='_boundary_nch'):
        self.boundary_key = boundary_key
        self.seg_key = seg_key
        self.meta_key = meta_key

    def __call__(self, sample: dict) -> dict:
        if self.meta_key not in sample:
            return sample  # rien à faire si on n'a pas fusionné avant

        n_bch = int(sample[self.meta_key])

        seg = sample[self.seg_key]  # (1 + C_small, Zp, Yp, Xp)
        assert seg.ndim == 4 and seg.shape[0] >= 1 + n_bch, f"seg shape inattendu: {seg.shape}"

        seg_main = seg[:1]                      # (1, Zp, Yp, Xp)
        bnd_ch  = seg[1:1 + n_bch].astype(np.uint8)  # (C_small, Zp, Yp, Xp) en binaire

        sample[self.seg_key] = seg_main
        sample[self.boundary_key] = bnd_ch
        del sample[self.meta_key]
        return sample
