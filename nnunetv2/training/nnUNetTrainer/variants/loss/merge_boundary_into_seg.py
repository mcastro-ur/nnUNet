# merge_boundary_into_seg.py
import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

#class MergeBoundaryIntoSeg:
#    """
#    Avant SpatialTransform :
#    - Prend sample['seg'] de shape (1, Z, Y, X) OU (Z, Y, X)
#    - Prend sample['boundary'] de shape (C_small, Z, Y, X)
#    - Concatène le tout en une seule 'seg' multi-canaux : (1 + C_small, Z, Y, X)
#    - Stocke le nombre de canaux ajoutés pour pouvoir "split" ensuite.
#    """
#    def __init__(self, boundary_key='boundary', seg_key='seg', meta_key='_boundary_nch'):
#        self.boundary_key = boundary_key
#        self.seg_key = seg_key
#        self.meta_key = meta_key
#
#    def __call__(self, sample: dict) -> dict:
#        if self.boundary_key not in sample:
#            return sample  # pas de boundary -> rien à faire
#
#        seg = sample[self.seg_key]
#        bnd = sample[self.boundary_key]  # (C_small, Z, Y, X)
#
#        # seg peut être (1, Z, Y, X) ou (Z, Y, X)
#        if seg.ndim == 3:
#            seg = seg[None, ...]  # -> (1, Z, Y, X)
#
#        assert bnd.ndim == 4, f"boundary doit être (C_small, Z, Y, X), reçu {bnd.shape}"
#        assert seg.shape[1:] == bnd.shape[1:], f"Shapes incompatibles: seg {seg.shape} vs boundary {bnd.shape}"
#
#        # Concatène sur l'axe des canaux
#        merged = np.concatenate([seg, bnd], axis=0)  # (1 + C_small, Z, Y, X)
#        sample[self.seg_key] = merged.astype(seg.dtype)
#
#        # Sauvegarder combien de canaux "boundary" on a fusionné
#        sample[self.meta_key] = bnd.shape[0]
#
#        # On peut supprimer la clé 'boundary' temporairement pour éviter confusions
#        del sample[self.boundary_key]
#        return sample

# merge_boundary_into_seg.py

class MergeBoundaryIntoSeg(BasicTransform):
    """
    Fusionne 'boundary' dans 'segmentation' pour que SpatialTransform applique
    les mêmes transformations aux deux.
    
    nnUNet utilise les clés 'image' et 'segmentation' dans le dataloader.
    """
    def __init__(self, boundary_key='boundary', seg_key='segmentation'):
        super().__init__()
        self.boundary_key = boundary_key
        self.seg_key = seg_key
    
    def apply(self, data_dict, **params):
        # nnUNet passe 'segmentation' pas 'seg'
        if self.seg_key in data_dict:
            seg = data_dict[self.seg_key]  # (1, Z, Y, X)
            
            # Si boundary existe, on le fusionne
            if self.boundary_key in data_dict:
                boundary = data_dict[self.boundary_key]  # (C_small, Z, Y, X)
                
                # Concatener le long de l'axe des canaux
                merged = np.concatenate([seg, boundary], axis=0)  # (1+C_small, Z, Y, X)
                data_dict[self.seg_key] = merged
                
                # Marquer qu'on a fusionné
                data_dict['_boundary_merged'] = True
                data_dict['_boundary_n_channels'] = boundary.shape[0]
        
        return data_dict