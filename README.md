# SIH Lunar Model

Tools and analysis for multi-modal lunar image registration using
Chandrayaan-2 OHRC, TMC-2, IIRS, and external lunar reference imagery.

## OHRC–NAC training pairs

[`src/27_create_ohrc_nac_pairs.py`](src/27_create_ohrc_nac_pairs.py) creates
geometry-supervised moving-OHRC/fixed-NAC tiles from complete, map-projected
rasters. It records exact augmentation transforms, validity masks, provenance,
and spatially separated train/validation/test splits without using LoFTR to
generate labels.

See [`OHRC_NAC_PAIRING.md`](OHRC_NAC_PAIRING.md) for input requirements and a
command-line example.

The current dataset limitations and overlap assessment are documented in
[`audit/DATASET_AUDIT.md`](audit/DATASET_AUDIT.md).
