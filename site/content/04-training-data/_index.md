---
title: "Training data and transformations"
anchor: "training-data"
weight: 40
draft: false
---

The aligned data are divided geographically into **training, validation and test sets** so that separate lunar regions prevent geographic leakage between splits.

Controlled transformations are then introduced with exactly known ground truth:

- rotation
- translation
- scale
- illumination changes

Each training example therefore retains a correspondence transformation that can be compared directly with the model's predicted geometry during evaluation.
