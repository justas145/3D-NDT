import numpy as np
from os import listdir
from os.path import isfile, join
from functools import cmp_to_key
import cv2
import pyvista as pv

def compare(item1, item2):
    n1 = [int(s) for s in item1 if s.isdigit()][1:]
    n2 = [int(s) for s in item2 if s.isdigit()][1:]
    n11 = int(''.join(map(str, n1)))
    n22 = int(''.join(map(str, n2)))
    return n11 - n22

# Select cube (1 or 2) and viewing plane
# e.g. Cube1CornerYZ471
cube = ["Cube1Corner", "Cube2Corner"][0]
plane = ["XY", "XZ", "YZ"][0]

folder = "../Processed_Images/Processed_Images/"
files = [f for f in listdir(folder) if isfile(join(folder, f)) and plane in f and cube in f]
files = sorted(files, key=cmp_to_key(compare))

def proc(i):
    return i

imgs = [proc(cv2.imread(folder + f, cv2.IMREAD_GRAYSCALE)).tolist() for f in files]
imgs = np.array(imgs)

# PYVISTA
pl = pv.Plotter()

# Display tomography
grid = pv.UniformGrid()
grid.dimensions = imgs.shape

grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)

grid.point_data["values"] = imgs.flatten(order="F")

pl.add_volume(grid, opacity="sigmoid", cmap="gray")

# Origin
#pl.add_mesh(pv.Sphere(center=(0, 0, 0), radius=8), color='b')

pl.show_axes()
pl.show()
