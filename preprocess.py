
import matplotlib.pyplot as plt
from skimage import io, util



impath = ".\\data\\lfw-deepfunneled\\lfw-deepfunneled\\Bill_Gates\\Bill_Gates_0001.jpg"
image = io.imread(impath)
image_f = util.img_as_float64(image)

print(image_f.shape, image_f.dtype, image_f.min(), image_f.max())

plt.imshow(image_f, cmap= 'gray')
plt.title("Float image")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im1 = axes[0].imshow(image_f[...,0], cmap="Reds", vmin=0, vmax=1)
axes[0].set_title(f"Red Channel ({image_f.dtype})")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(image_f[...,1], cmap ="Greens", vmin=0, vmax=1)
axes[1].set_title(f"Green Channel ({image_f.dtype})")
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(image_f[..., 2], cmap="Blues", vmin=0, vmax=1)
axes[2].set_title(f"Blue Channel ({image_f.dtype})")
fig.colorbar(im3, ax=axes[2])

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

