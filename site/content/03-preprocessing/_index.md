---
title: "Preprocessing pipeline"
anchor: "preprocessing"
weight: 30
draft: false
---

The preprocessing stage converts the raw observations into spatially consistent inputs for registration.

### Processing steps

- Create validity masks to remove exterior padding and invalid pixels while retaining genuine dark shadow regions.
- Resample images to a common spatial resolution using antialiased downsampling so terrain structure is not unnecessarily lost.
- Project moving and reference images onto a common lunar coordinate grid.
- Apply percentile-based intensity normalization to reduce sensor-dependent brightness and contrast differences.
- Prepare spatially aligned image tiles containing moving images, fixed images, validity masks and correspondence transformations.

### Preprocessing examples

![OHRC preprocessing preview](images/ohrc-preprocessing.png)

![TMC and IIRS preprocessing preview](images/tmc-iirs-preprocessing.png)

![Browse preview](images/archive-browse.png)
