# boundary_dataset_wrapper.py
import os
import numpy as np

# Les petites classes d'après ton plan (tu peux importer depuis un fichier de conf commun)
SMALL_CLASSES = [2, 3, 5, 6, 7, 8, 9, 10]

class BoundaryDatasetWrapper:
    """
    Enveloppe un dataset nnU-Net v2 pour y ajouter 'boundary' (C_small, Z, Y, X).
    - Si un NPZ pré-calculé existe -> on le charge.
    - Sinon -> fallback en zeros pour conserver la clé et laisser la pipeline tourner.
    """
    def __init__(self, base_dataset, boundary_root):
        self.base = base_dataset
        self.boundary_root = boundary_root
        self.c_small = len(SMALL_CLASSES)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        # 1) On récupère l'échantillon original
        sample = self.base[i]  # {'data':..., 'seg':..., 'properties':..., 'keys':...}

        # 2) Identifiant de cas (clé pour aller retrouver le NPZ)
        case_id = sample.get('keys', [None])[0] or sample['properties'].get('case_identifier', None)
        assert case_id is not None, "Impossible de déterminer case_id depuis le sample"

        # 3) Calcul de la forme cible pour 'boundary' à partir de 'seg'
        seg = sample['seg']
        # seg est typiquement (1, Z, Y, X) ou (Z, Y, X)
        if seg.ndim == 4 and seg.shape[0] == 1:
            Z, Y, X = seg.shape[1:]
        elif seg.ndim == 3:
            Z, Y, X = seg.shape
        else:
            raise RuntimeError(f"Format de seg inattendu: {seg.shape}")

        # 4) Tentative de chargement du NPZ (B1 offline). Sinon fallback zeros.
        npz_path = os.path.join(self.boundary_root, f"{case_id}_boundary_small.npz")
        if os.path.isfile(npz_path):
            with np.load(npz_path) as npz:
                boundary_full = npz["bands"].astype(np.uint8)  # (C_small, Zfull, Yfull, Xfull)
            # IMPORTANT :
            # À ce stade, boundary_full correspond au volume ENTIER (aligné au label original).
            # C'est la SpatialTransform (avec keys_of_seg=('seg','boundary')) qui appliquera
            # le même crop/rot/scale/elastic pour produire le patch (C_small, Zp, Yp, Xp).
            sample['boundary'] = boundary_full
        else:
            # >>>>>>>>>>>> ICI TON POINT 1 EXACTEMENT <<<<<<<<<<<<
            # Fallback si le fichier n'existe pas : on met un tenseur vide (zeros) déjà au bon "range"
            # NB: On met la même taille spatiale que 'seg' POUR NE PAS CASSER la suite.
            sample['boundary'] = np.zeros((self.c_small, Z, Y, X), dtype=np.uint8)

        # 5) Retourne l'échantillon enrichi
        return sample
