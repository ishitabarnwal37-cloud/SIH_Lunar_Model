---
title: "Dataset collection"
anchor: "dataset"
weight: 20
draft: false
---

The dataset was assembled from **Chandrayaan-2 OHRC, TMC-2 and IIRS imagery** collected from ISRO, together with **LRO NAC reference images** covering the same lunar regions.

### Dataset preparation

1. Verify image metadata, dimensions, spatial resolution, acquisition time, Sun angles and file integrity.
2. Compare geographic footprints to identify pairs containing common lunar terrain.
3. Read PDS `.IMG`, `.TIF` and spectral-image formats using GDAL together with their metadata.
4. Convert suitable inputs to GeoTIFF-compatible processing formats while preserving lunar coordinate information.

The resulting pairs span different illumination, scale and viewing conditions so that correspondence matching is tested across sensor and imaging differences.
