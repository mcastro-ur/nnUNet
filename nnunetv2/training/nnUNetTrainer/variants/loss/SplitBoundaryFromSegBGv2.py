# SplitBoundaryFromSegBGv2.py
import numpy as np

class SplitBoundaryFromSegBGv2:
    
    # Version compatible batchgeneratorsv2:
    #segmentation -> split en seg et boundary
    
    def __init__(self, boundary_key='boundary', meta_key='_boundary_nch'):
        self.boundary_key = boundary_key
        self.meta_key = meta_key

    def __call__(self, image=None, segmentation=None, **kwargs):
        if segmentation is None or self.meta_key not in kwargs:
            return {
                "image": image,
                "segmentation": segmentation,
                **kwargs
            }

        seg = segmentation    # shape (1 + C_small, Z, Y, X)
        n_bnd = kwargs[self.meta_key]

        seg_main = seg[:1]                        # (1, Z, Y, X)
        bnd = seg[1:1+n_bnd].astype(np.uint8)     # (C_small, Z, Y, X)

        new_kwargs = dict(kwargs)
        del new_kwargs[self.meta_key]
        new_kwargs[self.boundary_key] = bnd

        return {
            "image": image,
            "segmentation": seg_main,
            **new_kwargs
        }