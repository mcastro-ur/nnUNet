# boundary_dataset_wrapper.py

import os
import numpy as np

SMALL_CLASSES = [2,3,5,6,7,8,9,10]

class BoundaryDatasetWrapper:
    """
    Transparent dataset wrapper :
    - Forwarde tous les attributs du dataset original (identifiers, label_manager, etc.)
    - Ajoute 'boundary' dans __getitem__
    """

    def __init__(self, base_dataset, boundary_root):
        self.base = base_dataset
        self.boundary_root = boundary_root
        self.c_small = len(SMALL_CLASSES)

    # ---- PATCH 1: forwarder les attributs manquants ----
    def __getattr__(self, name):
        """
        Si l'attribut n'existe pas dans ce wrapper,
        il est cherché dans le dataset d'origine.
        """
        return getattr(self.base, name)

    # ---- PATCH 2: forwarder length ----
    def __len__(self):
        return len(self.base)

    # ---- PATCH 3: __getitem__ enrichi mais transparent ----
    def __getitem__(self, i):
        sample = self.base[i]

        # Identifiant de case (nnU‑Net v2 en a besoin partout)
        case_id = None
        if "keys" in sample:
            case_id = sample["keys"][0]
        else:
            case_id = sample["properties"].get("case_identifier", None)

        if case_id is None:
            raise RuntimeError("Impossible de trouver case_id dans le sample")

        # Chargement du NPZ boundary
        npz_path = os.path.join(self.boundary_root, f"{case_id}_boundary_small.npz")
        if os.path.isfile(npz_path):
            with np.load(npz_path) as npz:
                boundary_full = npz["bands"].astype(np.uint8)
        else:
            # fallback si un fichier n'existe pas
            seg = sample["seg"]
            if seg.ndim == 4:
                Z, Y, X = seg.shape[1:]
            else:
                Z, Y, X = seg.shape
            boundary_full = np.zeros((self.c_small, Z, Y, X), dtype=np.uint8)

        sample["boundary"] = boundary_full
        return sample