# MergeBoundaryIntoSegBGv2.py
import numpy as np

class MergeBoundaryIntoSegBGv2:
    
    # Version compatible batchgeneratorsv2:
    # reçoit les arguments: image=..., segmentation=...
    # Retourne un dictionnaire dans le format que bgv2 attend.
    
    def __init__(self, boundary_key='boundary', meta_key='_boundary_nch'):
        self.boundary_key = boundary_key
        self.meta_key = meta_key

    def __call__(self, image=None, segmentation=None, **kwargs):
        # bgv2 passe image, segmentation séparément

        if segmentation is None or self.boundary_key not in kwargs:
            return {
                "image": image,
                "segmentation": segmentation,
                **kwargs
            }

        seg = segmentation               # (1, Z, Y, X)
        bnd = kwargs[self.boundary_key]  # (C_small, Z, Y, X)

        # seg may be (Z,Y,X)
        if seg.ndim == 3:
            seg = seg[None, ...]

        assert seg.shape[1:] == bnd.shape[1:], \
            f"seg {seg.shape} vs boundary {bnd.shape} mismatch"

        merged = np.concatenate([seg, bnd], axis=0)

        # On retourne segmentation = merged, boundary supprimé
        new_kwargs = dict(kwargs)
        del new_kwargs[self.boundary_key]

        new_kwargs[self.meta_key] = bnd.shape[0]

        return {
            "image": image,
            "segmentation": merged.astype(seg.dtype),
            **new_kwargs
        }