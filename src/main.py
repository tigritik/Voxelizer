import cv2
import xml.etree.ElementTree as ET

import numpy as np

imgs = [
    "../img/cam00_00023_0000008550.png",
    "../img/cam01_00023_0000008550.png",
    "../img/cam02_00023_0000008550.png",
    "../img/cam03_00023_0000008550.png",
    "../img/cam04_00023_0000008550.png",
    "../img/cam05_00023_0000008550.png",
    "../img/cam06_00023_0000008550.png",
    "../img/cam07_00023_0000008550.png"
]

def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name)
        images.append(img)

    return images

def get_silhouettes(images):
    silhouettes = []
    for img in images:
        file_name = f"../silhouettes/silh_{img[7:-4]}.pbm"
        sil = cv2.imread(file_name)
        silhouettes.append(sil)

    return silhouettes

def get_projections(images):
    p_matrices = []
    for img in images:
        file_name = f"../calibrations/{img[7:12]}.xml"
        projection_vals = ET.parse(file_name).getroot().text.split()
        matrix = np.array(projection_vals, dtype=float).reshape((3, 4))
        p_matrices.append(matrix)

    return p_matrices

def setup_voxels(size, center=(0,0,0), voxels_per_side=10):
    min_bound = np.array(center) - np.array(size)/2
    max_bound = np.array(center) + np.array(size)/2
    grid = np.linspace(min_bound, max_bound, voxels_per_side)

    voxel_dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1")
    ])

    voxels_out = np.zeros([voxels_per_side]*3, dtype=voxel_dtype)
    voxels_out["x"] = grid[:, 0].reshape((voxels_per_side, 1, 1))
    voxels_out["y"] = grid[:, 1].reshape((1, voxels_per_side, 1))
    voxels_out["z"] = grid[:, 2].reshape((1, 1, voxels_per_side))
    return voxels_out

print(load_images(imgs))
print(get_silhouettes(imgs))
np.set_printoptions(suppress=True)
print(get_projections(imgs))
print(setup_voxels((5,5,5))[0, 1, 2])

