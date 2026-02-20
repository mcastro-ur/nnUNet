# boundary_dataset_wrapper.py

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
        # Utiliser __dict__ directement pour éviter des problèmes de récursion
        self.__dict__['base'] = base_dataset
        self.__dict__['boundary_root'] = boundary_root
        self.__dict__['c_small'] = len(SMALL_CLASSES)

    def __len__(self):
        return len(self.__dict__['base'])
    
    def __getattr__(self, name):
        # Déléguer les attributs non trouvés au dataset de base
        # MAIS éviter la récursion en vérifiant d'abord __dict__
        if name in ('base', 'boundary_root', 'c_small'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.__dict__['base'], name)
    
    def __setattr__(self, name, value):
        # Stocker dans __dict__ pour éviter la récursion
        self.__dict__[name] = value

    def __getitem__(self, i):
        # 1) On récupère l'échantillon original
        base_dataset = self.__dict__['base']
        sample = base_dataset[i]  # {'data':..., 'seg':..., 'properties':..., 'keys':...}

        # 2) Identifiant de cas (clé pour aller retrouver le NPZ)
        case_id = sample.get('keys', [None])[0] or sample['properties'].get('case_identifier', None)
        if case_id is None:
            # Fallback si pas de case_id
            seg = sample['seg']
            if seg.ndim == 4 and seg.shape[0] == 1:
                Z, Y, X = seg.shape[1:]
            elif seg.ndim == 3:
                Z, Y, X = seg.shape
            else:
                raise RuntimeError(f"Format de seg inattendu: {seg.shape}")
            sample['boundary'] = np.zeros((self.__dict__['c_small'], Z, Y, X), dtype=np.uint8)
            return sample

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
        boundary_root = self.__dict__['boundary_root']
        npz_path = os.path.join(boundary_root, f"{case_id}_boundary_small.npz")
        
        if os.path.isfile(npz_path):
            try:
                with np.load(npz_path) as npz:
                    boundary_full = npz["bands"].astype(np.uint8)  # (C_small, Zfull, Yfull, Xfull)
                sample['boundary'] = boundary_full
            except Exception as e:
                # Si erreur de lecture, fallback vers zeros
                print(f"Warning: Could not load {npz_path}: {e}. Using zeros fallback.")
                sample['boundary'] = np.zeros((self.__dict__['c_small'], Z, Y, X), dtype=np.uint8)
        else:
            # Fallback si le fichier n'existe pas : on met un tenseur vide (zeros)
            sample['boundary'] = np.zeros((self.__dict__['c_small'], Z, Y, X), dtype=np.uint8)

        # 5) Retourne l'échantillon enrichi
        return sample
    
    # Ajout des méthodes pour le pickling (multiprocessing)
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)