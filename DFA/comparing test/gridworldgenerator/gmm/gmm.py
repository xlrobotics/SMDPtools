import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X_train = np.load('data.npy')

"""
plt.plot(X[:,0], X[:,1], 'bx')
plt.axis('equal')
plt.show()
"""

gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)

print(gmm.means_)
print('\n')
print(gmm.covariances_)

X, Y = np.meshgrid(np.linspace(-1, 6), np.linspace(-1,6))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape((50,50))

plt.contour(X, Y, Z)
plt.scatter(X_train[:, 0], X_train[:, 1])

plt.show()

plt.savefig("gmm" + ".png")