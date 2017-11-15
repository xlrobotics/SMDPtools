"""
========================================
An animated image using a list of images
========================================

This examples demonstrates how to animate an image from a list of images (or
Artists).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

fig = plt.figure()


# def f(x, y):
#     return np.sin(x) + np.cos(y)
#
# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(1, 25):
    print i
    # x += np.pi / 15.
    # y += np.pi / 20.

    f = mpimg.imread("Action_result" + str(i) + ".png")

    # im = plt.imshow(f(x, y), animated=True)
    im = plt.imshow(f, animated=True)

            # print im, f
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=10000)
#
ani.save('dynamic_action_VI_images.mp4')

# plt.show()