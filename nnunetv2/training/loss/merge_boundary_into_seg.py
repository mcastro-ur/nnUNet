# merge_boundary_into_seg.py
import numpy as np

class MergeBoundaryIntoSeg:
    """
    Avant SpatialTransform :
    - Prend sample['seg'] de shape (1, Z, Y, X) OU (Z, Y, X)
    - Prend sample['boundary'] de shape (C_small, Z, Y, X)
    - Concatène le tout en une seule 'seg' multi-canaux : (1 + C_small, Z, Y, X)
    - Stocke le nombre de canaux ajoutés pour pouvoir "split" ensuite.
    """
    def __init__(self, boundary_key='boundary', seg_key='segmentation', meta_key='_boundary_nch'):
        self.boundary_key = boundary_key
        self.seg_key = seg_key
        self.meta_key = meta_key

    def __call__(self, **data_dict) -> dict:
        if self.boundary_key not in data_dict:
            return data_dict  # pas de boundary -> rien à faire

        seg = data_dict[self.seg_key]
        bnd = data_dict[self.boundary_key]  # (C_small, Z, Y, X)

        # seg peut être (1, Z, Y, X) ou (Z, Y, X)
        if seg.ndim == 3:
            seg = seg[None, ...]  # -> (1, Z, Y, X)

        assert bnd.ndim == 4, f"boundary doit être (C_small, Z, Y, X), reçu {bnd.shape}"
        assert seg.shape[1:] == bnd.shape[1:], f"Shapes incompatibles: seg {seg.shape} vs boundary {bnd.shape}"

        # Concatène sur l'axe des canaux
        merged = np.concatenate([seg, bnd], axis=0)  # (1 + C_small, Z, Y, X)
        data_dict[self.seg_key] = merged.astype(seg.dtype)

        # Sauvegarder combien de canaux "boundary" on a fusionné
        data_dict[self.meta_key] = bnd.shape[0]

        # On peut supprimer la clé 'boundary' temporairement pour éviter confusions
        del data_dict[self.boundary_key]
        return data_dict
