import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import cv2.ximgproc as ximgproc
import os
import scipy.io 


""" This Script will take the strain data and properly clean the magnitude image from fat regions to be purely muscle oriented

    Inputs : Strain Data and Magnitude Image Data
    Output: Filtered Image only plotting the Muscle"""


path = os.getcwd()

mat_data_path = path + '\Patient_Data\JH_Data\JH.mat'

mat_data = scipy.io.loadmat(mat_data_path)

L_vector = mat_data['L_vector']
L_vector = L_vector[1:-1, 1:-1, :, :, :, : ]

m_data = mat_data['m_data']

Slice = 13
Frame = 13


def remove_edge_fat(m_data, slice, frame, t_lower=100, t_upper=200, aperture_size=5, 
                    kernel_size=(4,1), fat_threshold=(195, 255), 
                    close_iterations=2, dilate_iterations=3):
    """
    Remove fat from the edges of an MRI slice.
    
    Parameters:
    - m_data: 4D numpy array of MRI data
    - slice: int, slice number
    - frame: int, frame number
    - t_lower: int, lower threshold for Canny edge detection
    - t_upper: int, upper threshold for Canny edge detection
    - aperture_size: int, aperture size for Canny edge detection
    - kernel_size: tuple, size of kernel for morphological operations
    - fat_threshold: tuple, (min, max) intensity values for fat
    - close_iterations: int, number of iterations for morphological closing
    - dilate_iterations: int, number of iterations for dilation
    
    Returns:
    - filtered_image: 2D numpy array, image with edge fat removed
    """
    
    m_data_slice = m_data[:, :, slice, frame]

    # Normalize and convert to uint8
    m_data_slice = ((m_data_slice - m_data_slice.min()) / (m_data_slice.max() - m_data_slice.min()) * 255).astype(np.uint8)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(m_data_slice, t_lower, t_upper, apertureSize=aperture_size)

    # Define kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Apply morphological closing to fill gaps in edges
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # Dilate the edges to create a border region
    dilated_edges = cv2.dilate(closed_edges, kernel, iterations=dilate_iterations)

    # Create a mask for fat in the border region
    fat_mask = ((m_data_slice >= fat_threshold[0]) & (m_data_slice <= fat_threshold[1]) & (dilated_edges > 0)).astype(np.uint8)

    # Dilate the fat mask to connect nearby regions
    fat_mask_dilated = cv2.dilate(fat_mask, kernel, iterations=dilate_iterations)

    # Create the final mask (inverted so fat regions are 0 and non-fat regions are 1)
    final_mask = 1 - fat_mask_dilated

    # Apply the mask to create the filtered image
    filtered_image = m_data_slice * final_mask

    return filtered_image


