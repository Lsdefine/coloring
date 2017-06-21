# Auto-coloring for gray images

loss: GAN + cyc-to-origimage + cyc-to-gray + neighbor-pixel-diff

## Better params:

A small lr.

Original GAN is better (no WGAN).

Use InstNorm instead of BatchNorm.

Dropouts in D.

No normalization in G.

