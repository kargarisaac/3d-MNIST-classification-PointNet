# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:28:39 2018

@author: Isaac
"""

# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import numpy as np
import vispy.scene
from vispy.scene import visuals
import h5py

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# generate data

with h5py.File('./data/train_point_clouds.h5', 'r') as f:
    # Reading digit at zeroth index    
    a = f[str(1)]   
    # Storing group contents of digit a
    data1 = (a["img"][:], a["points"][:], a.attrs["label"])
    

#%%
# create scatter object and fill in the data
scatter = visuals.Markers()

scatter.set_data(data1[:,:3], edge_color=None, size=5)

view.add(scatter)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
