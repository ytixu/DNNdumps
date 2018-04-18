"""Functions to visualize human poses"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M_POSE_LINES = {'r':[0,1,2,3],
        'g':[0,4,5,6],
        'b':[0,7,8,9,10],
        'm':[8,11,12,13],
        'k':[9,14,15,16]}

class Ax3DPose(object):
  def __init__(self, ax):
    self.ax = ax
    vals = np.zeros((17, 3))

    # Make connection matrix
    self.plots = []
    for i, lines in enumerate(M_POSE_LINES.iteritems()):
      color, line = lines
      x = np.array(vals[line, 0])
      y = np.array(vals[line, 1])
      z = np.array(vals[line, 2])
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=color))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")
    self.ax.set_xlim3d([-1, 1])
    self.ax.set_zlim3d([-1, 1])
    self.ax.set_ylim3d([-1, 1])

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    vals = np.reshape( channels, (-3, 3) )

    for i, lines in enumerate(M_POSE_LINES.iteritems()):
      color, line = lines
      x = np.array(vals[line, 0])
      y = np.array(vals[line, 1])
      z = np.array(vals[line, 2])
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(color)

    self.ax.set_aspect('equal')
