import numpy as np
from scipy import signal

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

print gkern(40, 2)
import matplotlib.pyplot as plt
plt.imshow(gkern(40, 2), cmap='hot', interpolation='nearest')
plt.savefig("test_gaussian" + ".png")