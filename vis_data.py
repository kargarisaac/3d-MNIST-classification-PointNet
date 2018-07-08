
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import animation

import h5py
import os, sys

from voxelgrid import VoxelGrid
from plot3D import *


plt.rcParams['image.cmap'] = 'gray'

with h5py.File('train_point_clouds.h5', 'r') as f:
  # Reading digit at zeroth index    
  a = f["0"]   
  # Storing group contents of digit a
  digit = (a["img"][:], a["points"][:], a.attrs["label"])

digits = []

with h5py.File("train_point_clouds.h5", 'r') as h5:
    for i in range(15):
        d = h5[str(i)]
        digits.append((d["img"][:],d["points"][:],d.attrs["label"]))

# Plot some examples from original 2D-MNIST
fig, axs = plt.subplots(3,5, figsize=(12, 12), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)

for ax, d in zip(axs.ravel(), digits):
  ax.imshow(d[0][:])
  ax.set_title("Digit: " + str(d[2]))
  
voxel_grid = VoxelGrid(digit, x_y_z = [16, 16, 16])

def count_plot(array_):
  cm = plt.cm.get_cmap('gist_rainbow')
  n, bins, patches = plt.hist(array_, bins=64)

  bin_centers = 0.5 * (bins[:-1] + bins[1:])

  # scale values to interval [0,1]
  col = bin_centers - min(bin_centers)
  col /= max(col)

  for c, p in zip(col, patches):
      plt.setp(p, 'facecolor', cm(c))
  plt.show()
   
# Get the count of points within each voxel.
plt.title("DIGIT: " + str(digits[0][-1]))
plt.xlabel("VOXEL")
plt.ylabel("POINTS INSIDE THE VOXEL")
count_plot(voxel_grid.structure[:,-1])    

voxels = []
for d in digits:
    voxels.append(VoxelGrid(d[1], x_y_z=[16,16,16]))
    
# Visualizing the Voxel Grid sliced around the z-axis.
voxels[0].plot()
plt.show()

# Save Voxel Grid Structure as the scalar field of Point Cloud.
cloud_vis = np.concatenate((digit[1], voxel_grid.structure), axis=1)
np.savetxt('Cloud Visualization - ' + str(digit[2]) + '.txt', cloud_vis)


with h5py.File("data/full_dataset_vectors.h5", 'r') as h5:
    X_train, y_train = h5["X_train"][:], h5["y_train"][:]
    X_test, y_test = h5["X_test"][:], h5["y_test"][:]
    
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

reg = LogisticRegression()
reg.fit(X_train,y_train)
print("Accuracy: ", reg.score(X_test,y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Accuracy: ", dt.score(X_test,y_test))

svm = LinearSVC()
svm.fit(X_train,y_train)
print("Accuracy: ", svm.score(X_test,y_test))

knn = KNN()
knn.fit(X_train,y_train)
print("Accuracy: ", knn.score(X_test,y_test))

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)
print("Accuracy: ", rf.score(X_test,y_test))
# ax.scatter(X,Y,Z)

# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)