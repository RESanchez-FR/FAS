import numpy as np 
import matplotlib.pyplot as plt
from DTI_preprocessing import process_and_save_nii_slices
from Smooth_Data import smooth_anisotropic, smooth_savgol
from Calc_Strain_Stats import calculate_strain_stats
import os
import scipy.io
import cv2
import sys
from scipy import odr



"""
    STEP 1: Fiber Aligned Projection Strain
    This Script is Dedicated to the Fiber Aligned Strain Projection. 
    We look at a region of intrest and determine the Strain Values in the fiber aligned strain
    at each voxel. We calculate the angular deviation between e3 - compressive strain 
    and v1 our principal eigenvector DTI data """

main_path = os.getcwd()
poly_path = os.path.join(os.path.dirname(__file__), main_path + '\smoothing')
# Add the directory to sys.path
sys.path.insert(0, poly_path)



# Global variables
contour_points = []
magnitude_image = None
strain_vec_prev_frame = None 

def draw_contour(event):
    global contour_points, magnitude_image
    
    if event.button == 1:  # Left mouse button
        if event.name == 'button_press_event':
            contour_points = [(event.xdata, event.ydata)]
        elif event.name == 'motion_notify_event':
            contour_points.append((event.xdata, event.ydata))
            x, y = zip(*contour_points)
            plt.gca().plot(x, y, 'r-', linewidth = 1.5)
            plt.draw()
        elif event.name == 'button_release_event':
            print("Contour drawing completed")




def visualize_magnitude_image(m_data, z_slice):
    global magnitude_image
    
    magnitude_image = np.squeeze(m_data[:, :, z_slice, 0])  # Assuming first time point
    
    plt.figure(figsize=(10, 7))
    plt.imshow(magnitude_image, cmap='gray')
    plt.title('Draw contour on the magnitude image')
    plt.connect('button_press_event', draw_contour)
    plt.connect('motion_notify_event', draw_contour)
    plt.connect('button_release_event', draw_contour)
    plt.show()


def compute_fiber_aligned_strain(dti_data, strain_data, contour, z_slice, t, eigenvector_index=0):
    """
    Compute fiber-aligned strain from DTI and strain tensor data within the contour for a single slice and time point.
    
    Parameters:
    dti_data: np.array of shape (X, Y, Z, 3) - principal eigenvector (ν₁)
    strain_data: np.array of shape (X, Y, Z, T, 3, 3) - strain tensor (E)
    contour: list of (x, y) tuples defining the ROI
    z_slice: int, the slice number to analyze
    t: int, the frame we are looking at
    eigenvector_index: int, which eigenvector to use (default is ε₃ = 2)
    
    Returns:
    E_fiber_avg: float - average fiber-aligned strain within ROI
    E_fiber_std: float - standard deviation of fiber-aligned strain within ROI
    theta_avg: float - average angle between ε₃ and ν₁ within ROI (in degrees)
    theta_std: float - standard deviation of angles (in degrees)
    """
    global strain_vec_prev_frame  # Declare global variable to modify it

    # Create a mask from the contour
    mask = np.zeros(dti_data.shape[:2], dtype=np.uint8)
    contour_array = np.array(contour, dtype=np.int32)
    cv2.fillPoly(mask, [contour_array], 1)

    # Normalize DTI principal eigenvector ν₁
    v1 = dti_data[:, :, z_slice]
    v1_norm = np.linalg.norm(v1, axis=2, keepdims=True)
    v1_normalized = np.where(v1_norm > 1e-6, v1 / v1_norm, 0)  # Avoid division by zero

    # Extract strain tensor for the current frame and slice
    E = strain_data[:, :, z_slice, t]  # This reduces to a [x,y,3,3] index now

    
    # Smooth the strain data
    E_smooth = smooth_anisotropic(E , mask )

    # Get mask indices for ROI
    x_idx, y_idx = np.where(mask == 1)

    # Vectorized eigenvalue decomposition for masked pixels
    E_masked = E[x_idx, y_idx]
    E_masked_smooth = E_smooth[x_idx, y_idx]

    # Compute angles between ν₁ and selected strain vector
    v1_masked = v1_normalized[x_idx, y_idx]  # extracts the information based on the contour we drew ROI

    # Directly use the principal eigenvector from strain data
    strain_vec_principal = E_masked_smooth[:, 0, :]  # Assuming E_masked_smooth is your smoothed strain data
    
    # Normalize the strain vector
    norm_strain_vec = strain_vec_principal / np.linalg.norm(strain_vec_principal, axis=1, keepdims=True)

    # Compute angles between ν₁ and selected strain vector
    dot_products = np.sum(norm_strain_vec * v1_masked, axis=1)
    theta = np.arccos(np.clip(np.abs(dot_products), -1.0, 1.0))  # Angle in radians

  
    # Compute fiber-aligned strain projection
    E_fiber = np.matmul(np.matmul(v1_masked, E_masked_smooth), v1_masked.T) #matrix multiplication tool
   

    # Update global variable for next frame's orientation consistency
    if strain_vec_prev_frame is None:
        strain_vec_prev_frame = np.zeros_like(v1_normalized)  # Initialize once globally

    strain_vec_prev_frame[x_idx, y_idx] = strain_vec_principal

    results = calculate_strain_stats(E_fiber, theta, E_masked_smooth)

    avg_strain_x = results['flipped_average_strain_x']
    std_strain_x = results['average_strain_x_std']
    avg_strain_y = results['flipped_average_strain_y']
    std_strain_y = results['average_strain_y_std']
    avg_strain_z = results['flipped_average_strain_z']
    std_strain_z = results['average_strain_z_std']

    total_sum = results['total_flipped_sum']

    # For overall E_avg and theta_avg
    E_avg = results['E_avg']
    E_std = results['E_std']
    theta_avg = results['theta_avg']
    theta_std = results['theta_std']

    print(results.keys())


    return (
        E_avg,               # Average fiber-aligned strain
        E_std,                # Standard deviation of fiber-aligned strain
        theta_avg,     # Average angle in degrees
        theta_std,        # Standard deviation of angles in degrees
        total_sum,         #Average strain for E1xx
        avg_strain_y,  #Average Strain for E1xy
        avg_strain_z,
        std_strain_x, 
        std_strain_y,
        std_strain_z        
    )



#Main execution
if __name__ == "__main__":

    path = os.getcwd()

    mat_data_path = path + '\Patient_Data\JH_Data\JH.mat'

    mat_data = scipy.io.loadmat(mat_data_path)

    L_vector = mat_data['L_vector']
    L_vector = L_vector[1:-1, 2:-2, :, :, :, : ] ## should be 1 instead of two but check later the nii file

    print(L_vector.shape)

    m_data = mat_data['m_data']

    dti_data = process_and_save_nii_slices('20231129_F020Y_2023112916JH_s03_fiber2.nii', 'Images/JH_DTI_Images')

    dti_data = np.swapaxes(dti_data, 0 , 2)
   
    z_slice = 10
    
    # Visualize magnitude image and draw contour
    visualize_magnitude_image(m_data, z_slice = z_slice)  # Adjust slice number as needed


    # Initialize storage
    E_avgs, E_stds = [], []
    theta_avgs, theta_stds = [], []

    # Initialize storage for new metrics
    strain_avgs_x, strain_avgs_y, strain_avgs_z = [], [], []
    strain_x_stds, strain_y_stds, strain_z_stds = [] , [], []

    # Get number of time frames
    num_frames = L_vector.shape[3]  

    # Inside your processing loop:
    for t in range(num_frames):
        e_avg, e_std, theta_avg, theta_std, strain_avg_x, strain_avg_y, strain_avg_z, strain_x_std, strain_y_std, strain_z_std = compute_fiber_aligned_strain(
            dti_data, L_vector, contour_points, z_slice, t)
        
        # Append new metrics
        if t == 0:
            strain_avgs_x.append(0)
            strain_avgs_z.append(0)
            E_avgs.append(0)
            strain_avgs_y.append(0)
            theta_avgs.append(0)
            theta_stds.append(0)
        else:
            strain_avgs_x.append(strain_avg_x)
            E_avgs.append(e_avg)
            strain_avgs_y.append(strain_avg_y)
            strain_avgs_z.append(strain_avg_z)
            theta_avgs.append(theta_avg)
            theta_stds.append(theta_std)
       

        E_stds.append(e_std)
        strain_x_stds.append(strain_x_std)
        strain_y_stds.append(strain_y_std)
        strain_z_stds.append(strain_z_std)



    # Modified plotting section
    plt.figure(figsize=(10, 6))

    # Original Strain Plot
    plt.subplot(2, 2, 1)
     # Plot with error bars
    plt.errorbar(range(num_frames), E_avgs, yerr=E_stds, 
                fmt='o', capsize=10, label='Fiber-aligned Strain')
    plt.xlabel('Time Frame')
    plt.ylabel('E₁₁')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Fiber Aligned Strain with Error Bars')
    plt.grid(False)

    # Angle Plot or strain averages Z
    plt.subplot(2, 2, 2)
    plt.errorbar(range(num_frames), theta_avgs, yerr=theta_stds,
                fmt='o-', capsize=5, color='orange', label='Angle Deviation')
    plt.xlabel('Time Frame')
    plt.ylabel('Angle')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Projected Angle From DTI and Strain')
    plt.grid(False)

    # New Strain Component Plot
    plt.subplot(2, 2, 3)
    plt.errorbar(range(num_frames), strain_avgs_x, yerr = strain_x_stds, 
                    fmt = 'o-' ,  color='green')
    plt.xlabel('Time Frame')
    plt.ylabel('Strain Averages E1xx')
    plt.ylim(-1, 1)
    # Add horizontal line at y=0 for clarity
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Average Strains E1 xx for ROI')
    plt.grid(False)

    # New DTI Magnitude Plot
    plt.subplot(2, 2, 4)
    plt.errorbar(range(num_frames), strain_avgs_y, yerr = strain_y_stds, 
                     fmt = 'o-', color='purple')
     # Add horizontal line at y=0 for clarity
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time Frame')
    plt.ylabel('Strain Averages E1 xy')
    plt.ylim(-1,1)
    plt.title('Average Strains E1 xy in ROI')
    plt.grid(False)

    plt.tight_layout()
    plt.show()

