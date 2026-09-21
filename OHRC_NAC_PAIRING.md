# OHRC–NAC pair generation

`src/27_create_ohrc_nac_pairs.py` creates geometry-supervised image pairs for
LoFTR or another correspondence model. LoFTR is not used to generate labels.

## Required inputs

Both inputs must be complete, north-up, map-projected rasters with a lunar CRS:

- moving image: Chandrayaan-2 OHRC;
- fixed image: LRO LROC NAC.

The raw OHRC IMG/geolocation CSV and an unprojected NAC CDR must first be
processed through their mission geometry. The program deliberately rejects
rasters without a projection or affine geotransform. Resizing two images to the
same dimensions is not a substitute for geolocation.

Use `M1126979192RC` for the first baseline. Reserve `M1129325970RC` for the
larger illumination/viewpoint test.

## Example

```bash
python3 src/27_create_ohrc_nac_pairs.py \
  --moving data/processed/OHRC_2024_map.tif \
  --fixed data/processed/M1126979192RC_map.tif \
  --output data/paired/o24_m1126979192 \
  --tile-size 640 \
  --stride 640 \
  --augmentations 4 \
  --min-valid-fraction 0.8
```

The NAC grid is the fixed grid. OHRC is area-resampled onto it, which is
appropriate when reducing the finer OHRC sampling to approximately one metre.
Each accepted tile produces:

```text
train|validation|test/pair_000000/
├── fixed.png
├── moving.png
├── fixed_mask.png
├── moving_mask.png
├── correspondences.npz
└── metadata.json
```

`correspondences.npz` stores both 3×3 coordinate transforms and the common
validity mask. Augmentation zero is the unperturbed common-grid pair. Later
augmentations apply a deterministic affine transform to the moving image and
record its exact inverse.

The split uses contiguous bands of the fixed image rather than randomly
assigning neighbouring tiles. This reduces geographic leakage between training,
validation, and testing. For a final benchmark, also hold out entire NAC
observations or lunar regions.

Generated datasets belong under `data/paired/`, which is intentionally ignored
by Git. Commit the generator and manifests needed for reproducibility, not large
derived training images.
