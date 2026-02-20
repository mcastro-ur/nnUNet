# split_boundary_from_seg.py
#import numpy as np
#
#class SplitBoundaryFromSeg:
#    """
#    Après SpatialTransform :
#    - Reprend sample['seg'] (1 + C_small, Zp, Yp, Xp)
#    - Sépare: seg_reconstruit -> (1, Zp, Yp, Xp), boundary -> (C_small, Zp, Yp, Xp)
#    """
#    def __init__(self, boundary_key='boundary', seg_key='seg', meta_key='_boundary_nch'):
#        self.boundary_key = boundary_key
#        self.seg_key = seg_key
#        self.meta_key = meta_key
#
#    def __call__(self, sample: dict) -> dict:
#        if self.meta_key not in sample:
#            return sample  # rien à faire si on n'a pas fusionné avant
#
#        n_bch = int(sample[self.meta_key])
#
#        seg = sample[self.seg_key]  # (1 + C_small, Zp, Yp, Xp)
#        assert seg.ndim == 4 and seg.shape[0] >= 1 + n_bch, f"seg shape inattendu: {seg.shape}"
#
#        seg_main = seg[:1]                      # (1, Zp, Yp, Xp)
#        bnd_ch  = seg[1:1 + n_bch].astype(np.uint8)  # (C_small, Zp, Yp, Xp) en binaire
#
#        sample[self.seg_key] = seg_main
#        sample[self.boundary_key] = bnd_ch
#        del sample[self.meta_key]
#        return sample

# split_boundary_from_seg.py
import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SplitBoundaryFromSeg(BasicTransform):
    """
    Sépare 'boundary' de 'segmentation' après les transformations spatiales.
    """
    def __init__(self, boundary_key='boundary', seg_key='segmentation'):
        super().__init__()
        self.boundary_key = boundary_key
        self.seg_key = seg_key
    
    def apply(self, data_dict, **params):
        # Vérifier si on a fusionné avant
        if data_dict.get('_boundary_merged', False):
            seg_merged = data_dict[self.seg_key]  # (1+C_small, Z, Y, X)
            n_boundary = data_dict.get('_boundary_n_channels', 0)
            
            if n_boundary > 0 and seg_merged.shape[0] > 1:
                # Split
                seg = seg_merged[:1, ...]  # Premier canal = seg
                boundary = seg_merged[1:1+n_boundary, ...]  # Canaux suivants = boundary
                
                data_dict[self.seg_key] = seg
                data_dict[self.boundary_key] = boundary
                
                # Nettoyer les flags temporaires
                del data_dict['_boundary_merged']
                del data_dict['_boundary_n_channels']
        
        return data_dict