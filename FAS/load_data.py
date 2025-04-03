import numpy as np 
import scipy as sc
import os
import sys
import scipy.io


""" This Script will focus on Achieving the following sub objectives in the research project

        Objective 1: Rotate negative strain eigenvector and DTI lead eigenvector so that both data sets lie in the positive z quadrants
        Objective 2: compute the x, y, z angle images for negative strain eigenvector
        Objective 3: We repeat the same steps but for the DTI data
        Objectivve 4: Compute the angle Image between negative strain and DTI for each temporal frame
        
"""

# Add parent directory to Python path (go up one level from current file)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from DTI_Processing.DTI_preprocessing import process_and_save_nii_slices

def load_data(data_folder="Data"):
    """
    Reads .mat and .nii files from the specified folder and loads their data into separate variables.
    
    Args:
        data_folder (str): Path to the folder containing the data files (default is "Data").
    
    Returns:
        tuple: Two variables:
            - data_set_1: Data from .mat files (dictionary format for each .mat file).
            - data_set_2: Data from .nii files (3D/4D numpy arrays for each .nii file).
    """
    # Initialize variables to None before the loop
    data_set_1 = None
    data_set_2 = None
    data_set_3 = None

    # Iterate through files in the data folder
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        base_name = os.path.splitext(file)[0]  # Get filename without extension

        # Check if the file is a .mat file
        if file.endswith(".mat"):
            # Load .mat file and store in data_set_1
            mat_data = scipy.io.loadmat(file_path)
            data_set_1 = mat_data['L_vector']
            data_set_1 = data_set_1[1:-1, 2:-2, :, :, :, :]
            data_set_2 = mat_data['m_data']

        # Check if the file is a .nii file
        elif file.endswith(".nii") or file.endswith(".nii.gz"):
            # Call process_and_save_nii_slices for .nii files
            nii_data = process_and_save_nii_slices(
                nii_file_name=file_path
            )
            data_set_3 = nii_data
            data_set_3 = np.swapaxes(data_set_3, 0 , 2) #Not necessary to include but need the dimensions to be in the same format

    # Add warning if data wasn't found
    if data_set_1 is None:
        print("Warning: No .mat files found in the specified folder.")
    if data_set_2 is None:
        print("Warning: No .nii files found in the specified folder.")

   
            

    return data_set_1, data_set_2, data_set_3

# load_data()