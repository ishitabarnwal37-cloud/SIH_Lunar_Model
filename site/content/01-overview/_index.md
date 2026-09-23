---
title: "Project overview"
anchor: "project-overview"
weight: 10
draft: false
---

This project focuses on **cross-sensor lunar image registration**: establishing reliable image correspondences between lunar observations acquired by different instruments under different imaging conditions.

Our workflow is built around Chandrayaan-2 imagery, preprocessing, dataset preparation, transformer-based correspondence matching and evaluation.

### Scope

The work combines:

- Chandrayaan-2 **OHRC**, **TMC-2** and **IIRS** imagery.
- **LRO NAC** images as reference imagery over the same lunar regions.
- Different illumination, scale and viewing conditions.
- LoFTR-based correspondence detection with targeted fine-tuning.
- KAN-based replacement of the transformer's final layer.
- Confidence filtering and robust geometric-transform estimation.

The objective is to make correspondence detection more reliable across sensors and challenging lunar terrain such as craters and sharp shadows.
