from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import vtk
from vtkmodules.vtkCommonDataModel import vtkPolyData

from tqdm import tqdm

def read_vtk_file(file_path : str) -> vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader.GetOutput()

def project_points_to_xz_plane(polydata : vtkPolyData) -> np.array:
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    projected_points = np.zeros((num_points, 2))

    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        projected_points[i] = [x, z]
    
    return projected_points

def create_image_from_points(points: np.array,
                             image_height : int = 256 - 50 * 2,
                             image_width : Union[int, None] = None,
                             point_size : int = 6,
                             verbose : bool = False):
    x_min, x_max = np.min(points[:, 0]) - point_size, np.max(points[:, 0]) + point_size
    z_min, z_max = np.min(points[:, 1]) - point_size, np.max(points[:, 1]) + point_size

    true_width = x_max - x_min
    true_height = z_max - z_min

    if image_width is None:
        image_width = int(true_width * (image_height / true_height))
    
    true_min = np.array([z_min, x_min])
    true_max = np.array([z_max, x_max])

    image_size = np.array([image_height, image_width])

    image = np.zeros((image_height + point_size, image_width + point_size))

    transformed_points = (points - true_min) / (true_max - true_min) * image_size
    transformed_points = transformed_points.astype(int)

    assert transformed_points.shape == points.shape

    num_points, _ = transformed_points.shape
    index_iterator = range(num_points)
    if verbose:
        index_iterator = tqdm(index_iterator)
    for index in index_iterator:
        x, z = transformed_points[index, :]
        image[z:z+point_size, x:x+point_size] = 1.
    
    return image

#Returns np array of image with intensities in [0,1]
def get_2d_projection_from_vtk(vtk_file_path : str, 
                               verbose : bool = False) -> np.array:
    if verbose:
        print("Projecting vtk data to 2d numpy array...")

    polydata = read_vtk_file(vtk_file_path)
    projected_points = project_points_to_xz_plane(polydata)
    image_array = create_image_from_points(projected_points, verbose=verbose)
    
    return image_array 

# Test
# if __name__ == '__main__':
#     vtk_file_path = '/Users/karan/Microtubules/tubulaton/Output/NucleusInsideCylinder/NucleusInsideCyl_800.vtk'
#     image = get_2d_projection_from_vtk(vtk_file_path=vtk_file_path,
#                                        verbose=True) 

#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()
