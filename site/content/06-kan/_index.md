---
title: "KAN-based transformer output"
anchor: "kan"
weight: 60
draft: false
---

We investigate a **Kolmogorov-Arnold Network (KAN)** as a replacement for the transformer's last layer.

Traditional neural-network layers learn fixed weight matrices. KAN places **learnable one-dimensional spline functions on the edges** of the network instead of using the conventional weight representation of the final layer.

KAN is presented as an alternative neural-network formulation derived from the **Kolmogorov Representation Theorem**.

### Why use KAN here?

LoFTR fine module can produce over-smoothed predictions near craters and sharp shadows. Because the KAN uses continuous B-spline functions, the project investigates whether local information can be interpolated at sub-pixel scale without losing localized structure.

![KAN versus MLP](images/kan-vs-mlp.png)
