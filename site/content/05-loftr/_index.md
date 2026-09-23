---
title: "LoFTR fine-tuning"
anchor: "loftr"
weight: 50
draft: false
---

The matching backbone is **LoFTR**. The project fine-tunes LoFTR's **coarse-attention module** while keeping the pre-trained **ResNet** unchanged.

Our motivation was to adapt correspondence detection to different lunar imaging sensors while keeping the computational cost lower than retraining the complete network.

The pre-trained ResNet is retained because it can capture useful features across bright and shadowed environments, which is important for Sun-angle variation.

![LoFTR flow](images/loftr-flow.png)

{{< block note >}}
Only the attention layers are fine-tuned in this stage; the pre-trained ResNet is kept fixed.
{{< /block >}}
