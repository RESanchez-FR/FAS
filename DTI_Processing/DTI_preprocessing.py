import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 
import os 
import cv2
from cv2 import ximgproc

def process_and_save_nii_slices(nii_file_name, brightness_factor=2, contrast_factor=1, start_slice=None, end_slice=None):
    """
    Process a .nii file, create grayscale images for each slice, and save them.

    Parameters:
    nii_file_name (str): Full path to the .nii file.
    brightness_factor (float): Factor to adjust brightness (default 2).
    contrast_factor (float): Factor to adjust contrast (default 1).
    start_slice (int): Starting slice number (default None, which means start from the first slice).
    end_slice (int): Ending slice number (default None, which means process all slices).

    Returns:
    nii_data: The loaded NIfTI data as a NumPy array.
    """
    # Load the .nii file from the provided full path
    nii_img = nib.load(nii_file_name)
    nii_data = nii_img.get_fdata()

    # Get the range of slices
    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = nii_data.shape[0]
    
    z_slices = range(start_slice, end_slice)

    # Process each slice
    for z_slice in z_slices:
        # Extract the slice data
        data_slice = np.abs(nii_data[z_slice, :, :])

        # Normalize and convert to uint8 with brightness adjustment
        min_val = np.nanmin(data_slice)
        max_val = np.nanmax(data_slice)

        normalized_data = ((data_slice - min_val) / (max_val - min_val) * 255 * brightness_factor).clip(0, 255).astype(np.uint8)

        # Adjust contrast
        normalized_data = cv2.addWeighted(normalized_data, contrast_factor, normalized_data, 0, 0)

        # Ensure the image is in the correct format (uint8, single channel)
        normalized_data = cv2.convertScaleAbs(normalized_data)

        # Convert back to float and rescale to [0, 1]
        grayscale_image = normalized_data.astype(float) / 255.0


    return nii_data
