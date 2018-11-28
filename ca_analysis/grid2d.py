from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


image = np.random.uniform(size=(500, 500))
xx, yy = np.meshgrid(np.linspace(-30, 30, 120), np.linspace(-100, 100, 120))

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
offsets = np.ones((120, 120))
offsets[0, :] = 0
offsets = offsets * np.arange(120)
offsets -= 50

offset_z = Z + offsets
print(Z.shape)
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, offset_z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.plot_surface(xx, 40 * np.ones_like(xx), yy, rstride=4, cstride=4, facecolors=plt.cm.Greys(image), shade=False)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()