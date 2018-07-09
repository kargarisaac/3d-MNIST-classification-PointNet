from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
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

with h5py.File('./data/train_point_clouds.h5', 'r') as f:
    # Reading digit at zeroth index    
    a = f[str(15)]   
    # Storing group contents of digit a
    digit = (a["img"][:], a["points"][:], a.attrs["label"])
    
    
#%%      visualize 3d point cloud
    
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np


from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
init_notebook_mode()


x, y, z = digit[1][:,0], digit[1][:,1], digit[1][:,2] 

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=2,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
#fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename='3d-scatter-colorscale')

fig = dict( data=data, layout=layout )

plot(fig)  