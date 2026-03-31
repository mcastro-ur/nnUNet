import os
import warnings
from typing import Union, Tuple, List, Optional

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 # ── [AJOUT] Répertoire des dist_maps précalculées ─────────
                 # Ex: /scratch/nnUNet_raw/Dataset092_Prostate26/
                 #         Dataset092_Prostate26/boundaryTr
                 # Mettre None pour désactiver (fallback boundary morphologique)
                 dist_maps_dir: Optional[str] = None,
                 dist_maps_suffix: str = "_boundary_small_distmaps.npy"):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

        # ── [AJOUT] dist_maps ─────────────────────────────────────────────
        self.dist_maps_dir = dist_maps_dir
        self.dist_maps_suffix = dist_maps_suffix
        # Log au démarrage pour confirmer que les dist_maps sont bien trouvées
        if dist_maps_dir is not None:
            if os.path.isdir(dist_maps_dir):
                npy_files = [f for f in os.listdir(dist_maps_dir)
                             if f.endswith(dist_maps_suffix)]
                print(f"[DataLoader] dist_maps_dir={dist_maps_dir}")
                print(f"[DataLoader] {len(npy_files)} fichiers *{dist_maps_suffix} trouvés")
            else:
                print(f"[DataLoader] ATTENTION: dist_maps_dir introuvable: {dist_maps_dir}")
                self.dist_maps_dir = None
        else:
            print("[DataLoader] dist_maps_dir=None → boundary loss sans EDT (fallback morphologique)")

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape

    # ── [AJOUT] Chargement des dist_maps pour un cas ─────────────────────────
    def _try_load_dist_maps(self, case_id: str) -> Optional[np.ndarray]:
        """
        Charge les distance maps précalculées pour le cas case_id.

        Chemin construit : {dist_maps_dir}/{case_id}{dist_maps_suffix}
        Ex: .../boundaryTr/case_001_boundary_small_distmaps.npy

        Retourne un array float32 (C, H, W, D) ou None si non disponible.
        """
        if self.dist_maps_dir is None:
            return None
        dist_path = os.path.join(self.dist_maps_dir,
                                 case_id + self.dist_maps_suffix)
        if os.path.exists(dist_path):
            # mmap_mode='r' : numpy accède le fichier comme une vue mémoire
            # → seul le patch croppé est réellement lu depuis le disque
            # → évite de charger 462 MB en RAM par worker à chaque batch
            arr = np.load(dist_path, allow_pickle=False, mmap_mode='r')
            # Copie explicite float32 APRÈS le crop (fait dans generate_train_batch)
            return arr  # dtype float16 conservé, conversion après crop
        return None

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        # ── [AJOUT] Accumulation des dist_maps pour le batch ─────────────────
        dist_maps_list = []   # rempli au fil du for loop ci-dessous
        # ──────────────────────────────────────────────────────────────────────

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

            # ── [AJOUT] Chargement + crop des dist_maps pour ce cas ─────
            # dist_maps sont sauvegardées à pleine résolution (C, H, W, D)
            # → doit être croppé avec le MÊME bbox que data et seg
            dm = self._try_load_dist_maps(i)
            if dm is not None:
                # dm shape : (C, H, W, D) — crop spatial avec bbox identique
                dm_cropped = crop_and_pad_nd(dm, bbox, 0)  # pad_value=0 (hors champ)
                # Conversion float32 ici — après crop, pas sur le volume entier
                dist_maps_list.append(np.array(dm_cropped, dtype=np.float32))
            # ──────────────────────────────────────────────────────────────

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
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            # ── [AJOUT] Ajout des dist_maps dans le batch retourné ────────
            batch = {'data': data_all, 'target': seg_all, 'keys': selected_keys}
            if len(dist_maps_list) == self.batch_size:
                # Tous les cas ont leurs dist_maps → on les passe au trainer
                batch['dist_maps'] = torch.from_numpy(
                    np.stack(dist_maps_list, axis=0)   # (B, C, H, W, D)
                )
            return batch

        # ── [AJOUT] Même chose pour le return sans transforms ────────────────
        batch = {'data': data_all, 'target': seg_all, 'keys': selected_keys}
        if len(dist_maps_list) == self.batch_size:
            batch['dist_maps'] = torch.from_numpy(
                np.stack(dist_maps_list, axis=0)   # (B, C, H, W, D)
            )
        return batch


if __name__ == '__main__':
    folder = join(nnUNet_preprocessed, 'Dataset002_Heart', 'nnUNetPlans_3d_fullres')
    ds = nnUNetDatasetBlosc2(folder)
    pm = PlansManager(join(folder, os.pardir, 'nnUNetPlans.json'))
    lm = pm.get_label_manager(load_json(join(folder, os.pardir, 'dataset.json')))
    dl = nnUNetDataLoader(ds, 5, (16, 16, 16), (16, 16, 16), lm,
                          0.33, None, None)
    a = next(dl)
