import os
import random
from typing import Tuple, Union
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import j1
from scipy.signal import convolve2d

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonDataModel import vtkPolyData

from visualize import visualize 

x = np.linspace(0.1, 10., num=50)
y = (2 * j1(x) / x) ** 2

plt.plot(x, y)
plt.show()
