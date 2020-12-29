import numpy as np
import matplotlib.pyplot as plt

from licplot import lic_internal

# create a 2d vector field
vortex_spacing = 0.5
extra_factor = 2.0
size = 700

a = np.array([1, 0]) * vortex_spacing
b = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]) * vortex_spacing
rnv = int(2 * extra_factor / vortex_spacing)
vortices = [n * a + m * b for n in range(-rnv, rnv) for m in range(-rnv, rnv)]
vortices = [
    (x, y)
    for (x, y) in vortices
    if -extra_factor < x < extra_factor and -extra_factor < y < extra_factor
]


xs = np.linspace(-1, 1, size).astype(np.float32)[None, :]
ys = np.linspace(-1, 1, size).astype(np.float32)[:, None]

u = np.zeros((size, size), dtype=np.float32)
v = np.zeros((size, size), dtype=np.float32)
for (x, y) in vortices:
    rsq = (xs - x) ** 2 + (ys - y) ** 2
    u += (ys - y) / rsq
    v += -(xs - x) / rsq

texture = np.random.rand(size, size).astype(np.float32)

# create a kernel
kernel_length = 31
kernel = np.sin(np.arange(kernel_length) * np.pi / kernel_length).astype(np.float32)


image = lic_internal.line_integral_convolution(u, v, texture, kernel)

plt.clf()
plt.axis("off")
plt.imshow(image, cmap="hot")
plt.show()
