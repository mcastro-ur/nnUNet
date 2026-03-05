#!/usr/bin/env python3
"""
Oversample CEM and Dijon cases 3x using symlinks to balance with PROSTATEx.

Rationale:
  ~86 CEM + ~86 Dijon = ~172 non-PROSTATEx cases.
  ~345 PROSTATEx cases (PEX_*).
  Duplicating CEM/Dijon 3x gives ~516 vs 345 → better balance.

Creates symlinks (not copies) in the same imagesTr / labelsTr directories,
with suffixes _dup1, _dup2, _dup3 appended to the case name.

Usage:
  python oversample_cem_dijon.py --dataset-dir DATASET_ROOT_DIR [--n-copies 3]
"""
import argparse
import os
from pathlib import Path


def create_symlinks_for_case(
    image_tr_dir: Path,
    label_tr_dir: Path,
    case_id: str,
    n_copies: int,
) -> None:
    """Create n_copies symlinks for the given case in imagesTr and labelsTr."""
    src_img = image_tr_dir / f"{case_id}_0000.nii.gz"
    src_lbl = label_tr_dir / f"{case_id}.nii.gz"

    if not src_img.exists():
        print(f"  WARNING: image not found: {src_img}")
        return
    if not src_lbl.exists():
        print(f"  WARNING: label not found: {src_lbl}")
        return

    for i in range(1, n_copies + 1):
        dup_id = f"{case_id}_dup{i}"

        dst_img = image_tr_dir / f"{dup_id}_0000.nii.gz"
        dst_lbl = label_tr_dir / f"{dup_id}.nii.gz"

        for dst, src in [(dst_img, src_img), (dst_lbl, src_lbl)]:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # Use relative symlink so the dataset remains portable
            try:
                rel_src = os.path.relpath(src, dst.parent)
                dst.symlink_to(rel_src)
            except OSError as e:
                print(f"  ERROR creating symlink {dst} -> {src}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Oversample CEM/Dijon cases with symlinks to balance with PROSTATEx."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Root directory of the nnU-Net dataset "
             "(contains imagesTr/ and labelsTr/).",
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=3,
        help="Number of duplicate symlinks to create per CEM/Dijon case "
             "(default: 3).",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["CEM_", "DIJ_"],
        help="Case ID prefixes to oversample (default: CEM_ DIJ_).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    image_tr_dir = dataset_dir / "imagesTr"
    label_tr_dir = dataset_dir / "labelsTr"

    if not image_tr_dir.is_dir():
        raise FileNotFoundError(f"imagesTr not found: {image_tr_dir}")
    if not label_tr_dir.is_dir():
        raise FileNotFoundError(f"labelsTr not found: {label_tr_dir}")

    # Collect all original cases to oversample
    cases_to_dup = []
    for fname in sorted(os.listdir(image_tr_dir)):
        if not fname.endswith("_0000.nii.gz"):
            continue
        case_id = fname.replace("_0000.nii.gz", "")
        # Skip already-duplicated cases
        if any(f"_dup" in case_id for _ in [None]):
            continue
        if any(case_id.startswith(pfx) for pfx in args.prefixes):
            cases_to_dup.append(case_id)

    print(
        f"Oversampling {len(cases_to_dup)} cases "
        f"(prefixes: {args.prefixes}) x{args.n_copies}..."
    )

    for case_id in cases_to_dup:
        create_symlinks_for_case(image_tr_dir, label_tr_dir, case_id, args.n_copies)

    total_new = len(cases_to_dup) * args.n_copies
    print(f"Created {total_new} symlink pairs.")
    print("Done.")


if __name__ == "__main__":
    main()
