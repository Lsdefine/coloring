# Auto-coloring for gray images

loss: GAN + cyc-to-origimage + cyc-to-gray + neighbor-pixel-diff

## Better params:

A small lr.

WGAN-GP.

Use InstNorm instead of BatchNorm.

Dropouts in D.

No normalization in G.


<img src="https://raw.githubusercontent.com/Lsdefine/coloring/master/example/gray_03442.png" alt="Gray" width=200/>
<img src="https://raw.githubusercontent.com/Lsdefine/coloring/master/example/pre_03442.png" alt="Color" width=200/>
