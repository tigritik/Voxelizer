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

def nearest_neighbor(x, y):
    return int(np.rint(x)), int(np.rint(y))

def project_voxel(voxel, P, interp=nearest_neighbor):
    v = np.array([voxel["x"], voxel["y"], voxel["z"], 1])
    proj = P @ v
    return interp(proj[0]/proj[2], proj[1]/proj[2])

def construct_model(voxels, silhouettes, projections):
    assert len(silhouettes) == len(projections), "Unequal number of silhouettes and projections!"
    n = len(silhouettes)
    count = 0 # count number of voxels that are occupied
    for i, j, k in np.ndindex(voxels.shape):
        if i % 10 == 0: print(f"{i+1}/100")
        voxel = voxels[i, j, k]
        images_matched = 0
        for sil, P in zip(silhouettes, projections):
            pix_x, pix_y = project_voxel(voxel, P)
            try:
                if np.array_equal(sil[pix_y, pix_x], [255, 255, 255]):
                    images_matched += 1
                else:
                    break
            except IndexError:
                break
        if images_matched == n:
            voxels[i, j, k]["r"] = 255
            voxels[i, j, k]["g"] = 255
            voxels[i, j, k]["b"] = 255
            count += 1

    return count

def get_neighbor_offsets():
    for axis in range(3):
        for direction in (-1, 1):
            offset = [0, 0, 0]
            offset[axis] = direction
            yield tuple(offset)

def get_occupied_neighbors(voxels, i, j, k):
    count = 0
    for di, dj, dk in get_neighbor_offsets():
        # check for out of bounds
        if 0 <= i+di < voxels.shape[0] and \
           0 <= j+dj < voxels.shape[1] and \
           0 <= k+dk < voxels.shape[2]:

            v = voxels[i+di, j+dj, k+dk]
            if (v["r"], v["g"], v["b"]) != (0, 0, 0):
                count += 1

    return count

def remove_points(voxels, points):
    for i, j, k in points:
        voxels[i, j, k]["r"] = 0
        voxels[i, j, k]["g"] = 0
        voxels[i, j, k]["b"] = 0

def compute_surface(voxels):
    count = 0
    remove_set = set()
    for i, j, k in np.ndindex(voxels.shape):
        if i % 10 == 0: print(f"{i + 1}/100")
        # Check neighboring voxels
        neighbors = get_occupied_neighbors(voxels, i, j, k)
        v = voxels[i, j, k]
        if neighbors < 6 and (v["r"], v["g"], v["b"]) != (0, 0, 0): # voxel on surface
            count += 1
        else: # voxel not on surface
            remove_set.add((i, j, k))

    # remove voxels not in surface
    remove_points(voxels, remove_set)
    return count

def write_ply(voxels, file_name, count):
    with open(file_name, mode='w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write vertex data
        for voxel in voxels.reshape(-1):
            if (voxel["r"], voxel["g"], voxel["b"]) != (0, 0, 0):
                f.write(f"{voxel['x']} {voxel['y']} {voxel['z']} {voxel['r']} {voxel['g']} {voxel['b']}\n")

sil = get_silhouettes(imgs)
np.set_printoptions(suppress=True)
proj = get_projections(imgs)
v = setup_voxels((5,5,5), voxels_per_side=100)
v_occupied = construct_model(v, sil, proj)
v_occupied = compute_surface(v)
write_ply(v, "../out/surface.ply", v_occupied)

