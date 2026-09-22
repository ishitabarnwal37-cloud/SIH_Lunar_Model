---
title: "Filtering and evaluation"
anchor: "evaluation"
weight: 70
draft: false
---

Predicted correspondences are filtered with **confidence checks** before estimating a robust geometric transformation.

The evaluation in the project brief uses several complementary measures:

| Metric | What it captures |
| --- | --- |
| Match count | Number of predicted correspondences |
| Inlier ratio | Fraction of correspondences consistent with the estimated geometry |
| Ground-truth error | Difference from the known transformation |
| Corner-transfer error | Geometric transfer accuracy at image corners |
| Spatial match coverage | How widely correspondences are distributed over the image |

These measures are used together rather than relying on a single count of matches, because the quality and spatial distribution of matches matter for image registration.
