#!/usr/bin/env python3

"""
This code was used to reconstruct the bell pepper.

It is copied from:

https://github.com/D1rk123/tomosipo_examples/blob/main/flexray_fdk.py

"""

import tifffile
from pathlib import Path
import torch
import tomosipo as ts
import tomosipo.torch_support
import ts_algorithms as tsa
import numpy as np
from tqdm import tqdm
from tiff_handling import load_stack, save_stack

# Reads the scanner settings file into a dictionary
def parse_scan_settings(path):
    contents = {}
    with open(path, "r") as file:
        for line in file:
            split_point =  line.find(":")
            if split_point == -1:
                continue
            else:
                contents[line[:split_point].strip()] = line[split_point+1:].strip()
    return contents


# Preprocesses the projection data without making copies
def preprocess_in_place(y, dark, flat):
    dark = dark[:, None, :]
    flat = flat[:, None, :]
    y -= dark
    y /= (flat - dark)
    torch.log_(y)
    y *= -1


# Loads a tiff file and converts it to a float32 torch tensor
def load_tiff_to_torch(path):
    return torch.from_numpy(tifffile.imread(str(path)).astype(np.float32))


if __name__ == "__main__":
    # Path to the projection data
    data_path = Path("pepper_projections")
    # Path to where the reconstruction will be saved
    save_path = Path("reconstruction")

    # Load the dark field and flat field images separately
    dark_field = load_tiff_to_torch(data_path / "di000000.tif")
    flat_field = load_tiff_to_torch(data_path / "io000000.tif")

    # Load the projection data and apply the log preprocessing
    y = torch.from_numpy(load_stack(data_path, prefix="scan", dtype=np.float32, stack_axis=1, range_stop=-1))
    preprocess_in_place(y, dark_field, flat_field)
    print("Finished loading and preprocessing")

    # The function parse_scan_settings reads the scan settings.txt into a dictionary
    scan_settings = parse_scan_settings(data_path / "scan settings.txt")
    # Read the distances and pixel size from the file
    src_det_dist = float(scan_settings["SDD"])
    src_obj_dist = float(scan_settings["SOD"])
    pixel_size = float(scan_settings["Binned pixel size"])
    # Derive the resolution from the input data
    detector_hor_res = y.shape[2]
    detector_ver_res = y.shape[0]
    num_angles = y.shape[1]

    # Make volume and projection geometries with the parameters
    vg = ts.volume(
        shape=(detector_ver_res, detector_hor_res, detector_hor_res),
        size=np.array((detector_ver_res, detector_hor_res, detector_hor_res))*pixel_size
    )
    pg = ts.cone(
        angles=num_angles,
        shape=(detector_ver_res, detector_hor_res),
        size=np.array((detector_ver_res, detector_hor_res))*pixel_size,
        src_det_dist = src_det_dist,
        src_orig_dist = src_obj_dist
    )
    # Combine the geometries into an operator
    A = ts.operator(vg, pg)

    # Make an FDK reconstruction
    # If you are using large projection data you may want to use overwrite_y=True
    reconstruction = tsa.fdk(A=A, y=y, overwrite_y=True)
    print("Finished reconstruction")
    # Variable y was overwritten with a filtered version of y because overwrite_y=True
    # You probably don't want to use this so delete y to free up memory
    del y

    save_stack(save_path, reconstruction.numpy(), exist_ok=True)
