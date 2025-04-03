from load_data import load_data
import scipy as sc
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import cv2

""" This Script will focus on Achieving the following sub objectives in the research project

        Objective 1: Rotate negative strain eigenvector and DTI lead eigenvector so that both data sets lie in the positive z quadrants
        Objective 2: compute the x, y, z angle images for negative strain eigenvector
        Objective 3: We repeat the same steps but for the DTI data
        Objectivve 4: Compute the angle Image between negative strain and DTI for each temporal frame
        
"""
#Load our two Data Sets we care about
neg_strain_data, m_data, dti_data = load_data()
z_slice = 10
eigenindex = 0


####################################### Takes care of Objective 1 ################################################################

def rotating_data_sets(data_1, data_2, z_slice = z_slice, eigenindex = eigenindex):
    """
    Rotate and reflect data to ensure all coordinates are positive.
    
    Args:
        data (np.ndarray): 6D array (x, y, z, time, eigenindex, last_index)
        z_slice (int): Fixed Z-slice to process
        eigenindex (int): Eigenindex to use
    
    Returns:
        positive_data (np.ndarray): Data with all coordinates >= 0
    """
    # Initialize output array
    positive_data_1 = np.zeros_like(data_1)
    
    "6D Case"
    # Iterate through time and last_index (assumed to be x,y,z components)
    for t in range(data_1.shape[3]):
        # Extract spatial slice: (x, y, z) for current indices
        spatial_slice_1 = data_1[:, :, z_slice, t, eigenindex, :]
        
        # Flatten to points: (N, 3)
        points_1 = spatial_slice_1.reshape(-1, 3)
        
        # Compute principal axis and align with Z+
        centroid_1 = np.mean(points_1, axis=0)
        centered_points_1 = points_1 - centroid_1
        _, _, vh_1 = np.linalg.svd(centered_points_1)
        principal_axis_1 = vh_1[0]
        
        # Align principal axis with Z+ (rotation only)
        rot_1, _ = R.align_vectors([principal_axis_1], [[0, 0, 1]])
        rotated_points_1 = rot_1.apply(points_1)
        
        # Reflect axes if needed to ensure all coordinates are positive
        reflection_1 = np.sign(np.mean(rotated_points_1, axis=0))
        reflected_points_1 = rotated_points_1 * reflection_1
        
        # Ensure no negative values (safety check)
        positive_points_1 = np.abs(reflected_points_1)  # Optional but recommended
        print("still working on rotation...")
        
        # Reshape and save
        positive_data_1[:, :, z_slice, t, eigenindex, :] = positive_points_1.reshape(spatial_slice_1.shape) #only slice and eigenindex are flipped

    print("done with 6D Case, Strain Data")

    "4D Case DTI"
    # Initialize output array (copy original data)
    positive_data_2 = np.copy(data_2)
    
    # Extract the target z-slice with components (x, y, components)
    target_slice_2 = positive_data_2[:, :, z_slice, :]
    x_dim, y_dim, n_components = target_slice_2.shape
    
    # Reshape to 2D array for batch processing (points × components)
    points_2 = target_slice_2.reshape(-1, n_components)
    
    # Compute principal axis (dominant direction) across all points
    centroid_2 = np.mean(points_2, axis=0)
    centered_points_2 = points_2 - centroid_2
    _, _, vh_2 = np.linalg.svd(centered_points_2)
    principal_axis_2 = vh_2[0]  # First singular vector
    
    # Align principal axis with positive Z-axis
    rot_2, _ = R.align_vectors([principal_axis_2], [[0, 0, 1]])
    rotated_points_2 = rot_2.apply(points_2)
    
    # Ensure all values are positive (absolute value)
    positive_points_2 = np.abs(rotated_points_2)
    
    # Reshape back to original slice dimensions and update output
    positive_slice_2 = positive_points_2.reshape(x_dim, y_dim, n_components)
    positive_data_2[:, :, z_slice, :] = positive_slice_2   

    print("done with 4D case DTI")


    return positive_data_1 , positive_data_2


#set up for objective 2
strain_rotated, DTI_rotated = rotating_data_sets(data_1 = neg_strain_data, data_2 = dti_data)
strain_rotated = strain_rotated[:, :, z_slice, :, eigenindex, :] #4D Data now
DTI_rotated = DTI_rotated[:, :, z_slice, :] #3D data now, shape is (160, 78, 3)



####################################### Takes care of Objective 2 ################################################################

# Global variables
contour_points = []
magnitude_image = None

def draw_contour(event):
    global contour_points, magnitude_image
    if event.button == 1:  # Left mouse button
        if event.name == 'button_press_event':
            contour_points = [(int(event.xdata), int(event.ydata))]  # Force integer coordinates
        elif event.name == 'motion_notify_event':
            contour_points.append((int(event.xdata), int(event.ydata)))
            x, y = zip(*contour_points)
            plt.gca().plot(x, y, 'r-', linewidth=1.5)
            plt.draw()
        elif event.name == 'button_release_event':
            print(f"Contour drawing completed with {len(contour_points)} points")

def visualize_magnitude_image(m_data, z_slice):
    global magnitude_image
    magnitude_image = np.squeeze(m_data[:, :, z_slice, 0])
    plt.figure(figsize=(10, 7))
    plt.imshow(magnitude_image, cmap='gray')
    plt.title('Draw contour on the magnitude image')
    plt.connect('button_press_event', draw_contour)
    plt.connect('motion_notify_event', draw_contour)
    plt.connect('button_release_event', draw_contour)
    plt.show()

def get_contour(mag_data, z_slice, min_points=3):
    """Get contour points through visualization (called ONCE)"""
    global contour_points
    contour_points = []
    visualize_magnitude_image(mag_data, z_slice)
    
    if len(contour_points) < min_points:
        raise ValueError(f"Need ≥{min_points} contour points (got {len(contour_points)})")
    
    return np.array(contour_points, dtype=np.int32)

def create_contour_mask(contour_array, data_shape):
    """Create binary mask from pre-defined contour"""
    mask = np.zeros(data_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour_array.reshape((-1, 1, 2))], 1)
    return np.where(mask == 1)



from matplotlib.path import Path



"works for 4D data "
def compute_x_y_z_angles_strain(data_1, x_idx, y_idx):
    """
    Compute angles with standard deviations using global contour_points.
    Returns: (avg_polar, avg_azimuth, std_polar, std_azimuth)
    """

    # Initialize storage with NaNs
    num_frames = data_1.shape[2]
    avg_polar = np.full(num_frames, np.nan)
    avg_azimuth = np.full(num_frames, np.nan)
    std_polar = np.full(num_frames, np.nan)
    std_azimuth = np.full(num_frames, np.nan) #adding a comment

    for t in range(num_frames):
        # Extract vectors in ROI
        vectors = data_1[x_idx, y_idx, t, :]  # Shape: (N, 3)
        if len(vectors) == 0: continue
        
        # Compute individual angles
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid_mask = magnitudes > 1e-6
        if not np.any(valid_mask): continue
        
        # Polar angles
        z_norm = vectors[:, 2] / np.where(valid_mask, magnitudes, 1)
        theta_all = np.degrees(np.arccos(z_norm[valid_mask]))
        
        # Azimuth angles (handle circular nature)
        phi_all = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))[valid_mask]
        phi_all = (phi_all + 360) % 360  # Convert to 0-360 range
        
        # Mean vector calculation
        mean_vec = np.mean(vectors[valid_mask], axis=0)
        mean_mag = np.linalg.norm(mean_vec)
        
        # Store results
        if mean_mag > 1e-6:
            avg_polar[t] = np.degrees(np.arccos(mean_vec[2]/mean_mag))
            avg_azimuth[t] = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
            std_polar[t] = np.std(theta_all)
            std_azimuth[t] = np.std((phi_all - avg_azimuth[t] + 180) % 360 - 180)

    return avg_polar, avg_azimuth, std_polar, std_azimuth


####################################### Takes care of Objective 3 ################################################################

"works for DTI but still unsure"

def compute_x_y_z_angles_DTI(data_2, x_idx, y_idx):
    """
    Compute angles and deviations for DTI data (x, y, components) in a contoured region.
    Returns: (avg_polar, avg_azimuth, std_polar, std_azimuth)
    """
    # Extract vectors from last dimension (components)
    vectors = data_2[x_idx, y_idx, :]  # Shape: (N_points, 3)
    
    # Initialize outputs
    avg_polar, avg_azimuth = np.nan, np.nan
    std_polar, std_azimuth = np.nan, np.nan
    
    if len(vectors) > 0:
        # Compute mean vector
        mean_vec = np.mean(vectors, axis=0)
        magnitude = np.linalg.norm(mean_vec)
        
        if magnitude > 1e-6:
            # Average angles from mean vector
            avg_polar = np.degrees(np.arccos(mean_vec[2] / magnitude))
            avg_azimuth = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
            
            # Individual angles for deviation calculation
            magnitudes = np.linalg.norm(vectors, axis=1)
            valid = magnitudes > 1e-6
            z_norm = vectors[:, 2] / np.where(valid, magnitudes, 1)
            theta_all = np.degrees(np.arccos(z_norm[valid]))
            phi_all = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
            
            # Circular std for azimuth (handle wrap-around)
            phi_diff = (phi_all - avg_azimuth + 180) % 360 - 180
            std_polar = np.std(theta_all)
            std_azimuth = np.std(phi_diff)
    
    return avg_polar, avg_azimuth, std_polar, std_azimuth


####################################### Takes care of Objective 4 ################################################################

def compute_angle_between_strain_DTI(data_1, data_2, x_idx, y_idx):
    """
    Calculate angles between eigenvectors with circular standard deviation
    
    Returns:
        Tuple of (angles_deg, std_dev_deg, avg_strain_vectors, dti_mean_norm)
    """
    # Preprocess DTI vectors
    dti_roi = data_2[x_idx, y_idx, :]
    dti_mean = np.mean(dti_roi, axis=0)
    dti_mean_norm = dti_mean / np.linalg.norm(dti_mean)
    
    num_frames = data_1.shape[2]
    angles_deg = np.full(num_frames, np.nan)
    std_dev_deg = np.full(num_frames, np.nan)  # New std dev storage
    avg_strain_vectors = np.zeros((num_frames, 3))

    for t in range(num_frames):
        strain_vectors = data_1[x_idx, y_idx, t, :]
        if strain_vectors.size == 0:
            continue
            
        # Get valid vectors
        magnitudes = np.linalg.norm(strain_vectors, axis=1)
        valid_mask = magnitudes > 1e-6
        valid_vectors = strain_vectors[valid_mask]
        
        if len(valid_vectors) == 0:
            continue
            
        # Calculate individual angles (vectorized)
        valid_vectors_norm = valid_vectors / magnitudes[valid_mask, None]
        dots = np.dot(valid_vectors_norm, dti_mean_norm)
        dots = np.clip(dots, -1.0, 1.0)
        angles_rad = np.arccos(dots)
        
        # Circular standard deviation [1][2][4]
        C = np.mean(np.cos(angles_rad))
        S = np.mean(np.sin(angles_rad))
        R = np.hypot(S, C)
        
        if R > 1e-6:  # Prevent log(0)
            circ_std = np.sqrt(-2 * np.log(R))
            std_dev_deg[t] = np.degrees(circ_std)
        
        # Store mean angle
        strain_mean = np.mean(valid_vectors, axis=0)
        strain_mean_norm = strain_mean / np.linalg.norm(strain_mean)
        avg_strain_vectors[t] = strain_mean_norm
        
        dot_product = np.dot(strain_mean_norm, dti_mean_norm)
        angles_deg[t] = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

    return angles_deg, std_dev_deg, avg_strain_vectors, dti_mean_norm

# Get contour ONCE
contour_array = get_contour(m_data, z_slice=10)

# Generate reusable indices
x_idx_strain, y_idx_strain = create_contour_mask(contour_array, strain_rotated.shape)
x_idx_DTI, y_idx_DTI = create_contour_mask(contour_array, DTI_rotated.shape)

# Compute all metrics
strain_results = compute_x_y_z_angles_strain(strain_rotated, x_idx_strain, y_idx_strain) #strain_polar, strain_azimuth, strain_polar_std, strain_azimuth_std

DTI_results = compute_x_y_z_angles_DTI(DTI_rotated, x_idx_DTI, y_idx_DTI) #DTI_polar, DTI_azimuth, DTI_polar_std, DTI_azimuth_std

comparison_results = compute_angle_between_strain_DTI(
    strain_rotated, DTI_rotated, x_idx_strain, y_idx_strain
) # projeciton_angles, avg_strain_vectors, dti_mean_norm


# Plot all results using shared contour
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

frames = np.arange(len(strain_results[0]))

# Strain angles
ax[0].errorbar(frames, strain_results[0], yerr=strain_results[2], 
               fmt='o',capsize = 5, color='blue', label='Polar Angle')
ax[0].errorbar(frames, strain_results[1], yerr=strain_results[3], 
               fmt='o', capsize = 5,color='red', label='Azimuth Angle')
ax[0].set_title('Strain Angles')  # Add title to first subplot
ax[0].legend(loc='upper right')  # Add legend to first subplot\
ax[0].grid(True)


# DTI angles
ax[1].errorbar([0], DTI_results[0], yerr=DTI_results[2], 
               fmt='o',capsize = 5, color='blue', label='Polar Angle')
ax[1].errorbar([1], DTI_results[1], yerr=DTI_results[3], 
               fmt='o',capsize = 5, color='red', label='Azimuth Angle')
ax[1].set_title('DTI Angles')  # Add title to second subplot
ax[1].legend(loc='upper right')  # Add legend to second subplot
ax[1].grid(True)

# Strain-DTI comparison
ax[2].errorbar(frames, comparison_results[0], yerr=comparison_results[1], 
               fmt='o',capsize = 5, color='red', label='Polar Angle')
ax[2].set_title('Strain-DTI Projection Angles No DTI Reoriented')  # Add title to third subplot
ax[2].legend(loc='upper right')  # Add legend to third subplot
ax[2].grid(True)

# Adjust layout for better visibility
plt.tight_layout()
plt.savefig('Fig 1.png')
plt.show()



####################################### Takes care of Objective 5 ################################################################

#want to to get the strain projeciton. In other words Lff lambda1.T * F * lambda1 where F is strain tensor, lambda is lead DTI eigenvector 


def compute_Lff_with_std(strain_vector, dti_eigenvector, x_idx, y_idx):
    """
    Computes Lff = lambda1.(F)lambda1 where F is a vector of 3 components.
    
    Parameters:
    strain_vector: 4D array [x, y, time, 3] (x,y,z components)
    dti_eigenvector: 3D array [x, y, 3] (x,y,z components)
    x_idx, y_idx: ROI contour points
    
    Returns:
    Tuple of (mean Lff values, std deviation of Lff values) for each frame
    """
    # Get ROI DTI eigenvectors
    lambda1 = dti_eigenvector[x_idx, y_idx, :]  # [n_points, 3]
    
    # Number of time frames
    num_frames = strain_vector.shape[2]
    mean_results = np.zeros(num_frames)
    std_results = np.zeros(num_frames)
    
    for t in range(num_frames):
        # Extract strain vector F for this frame in ROI
        F = strain_vector[x_idx, y_idx, t, :]  # [n_points, 3]
        
        # For vector F, the operation becomes a simple dot product
        # Compute element-wise product and then sum over components
        Lff_values = np.sum(lambda1 * F, axis=1)
        
        # Calculate mean and standard deviation over ROI
        mean_results[t] = np.mean(Lff_values)
        std_results[t] = np.std(Lff_values)
        
    return mean_results, std_results

def compute_Eff_with_std(strain_vector, dti_eigenvector, x_idx, y_idx):
    """
    Computes E_ff = (1/2)(v^T.C.v - 1) where C is a vector of 3 components.
    
    Parameters:
    strain_vector: 4D array [x, y, time, 3] (x,y,z components)
    dti_eigenvector: 3D array [x, y, 3] (x,y,z components)
    x_idx, y_idx: ROI contour points
    
    Returns:
    Tuple of (mean E_vv values, std deviation of E_ff values) for each frame
    """
    # Get ROI DTI eigenvectors
    v = dti_eigenvector[x_idx, y_idx, :]  # [n_points, 3]
    
    # Number of time frames
    num_frames = strain_vector.shape[2]
    mean_results = np.zeros(num_frames)
    std_results = np.zeros(num_frames)
    
    for t in range(num_frames):
        # Extract strain vector C for this frame in ROI
        C = strain_vector[x_idx, y_idx, t, :]  # [n_points, 3]
        
        # For vector C, the operation becomes a simple dot product
        # Compute v^T.C which is just dot product for vectors
        vCv_values = np.sum(v * C, axis=1)
        
        # Calculate E_vv = (1/2)(v^T.C - 1) for each point in ROI
        Evv_values = 0.5 * (vCv_values - 1)
        
        # Calculate mean and standard deviation over ROI
        mean_results[t] = np.mean(Evv_values)
        std_results[t] = np.std(Evv_values)
        
    return mean_results, std_results

# Calculate metrics with standard deviations
mean_Lff, std_Lff = compute_Lff_with_std(strain_rotated, DTI_rotated, x_idx_strain, y_idx_strain)
mean_Evv, std_Evv = compute_Eff_with_std(strain_rotated, DTI_rotated, x_idx_strain, y_idx_strain)



# Plot all results using shared contour
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

frames = np.arange(len(mean_Evv))

#Lff fiber 
ax[0].errorbar(frames, mean_Lff, yerr= std_Lff, 
               fmt='-o',capsize = 5, color='blue', label='Lff')
ax[0].set_title('Lff, No DTI Reoriented : lambda1.(F)lambda1 ')  # Add title to first subplot
ax[0].legend(loc='upper right')  # Add legend to first subplot
# ax[0].set_ylim([-1, 1])  # Replace y_min and y_max with desired values
ax[0].grid(True)


#Eff fiber
ax[1].errorbar(frames, mean_Evv, yerr= std_Evv, 
               fmt='-o',capsize = 5, color='red', label='Eff')
ax[1].set_title('Eff,  No DTI Reoriented : (1/2)(v^T.C.v - 1)')  # Add title to second subplot
ax[1].legend(loc='upper right')  # Add legend to second subplot
ax[1].set_ylim([-0.5, 0.5])  # Replace y_min and y_max with desired values
ax[1].grid(True)

# Adjust layout for better visibility
plt.tight_layout()
plt.savefig('Fig 2.png')
plt.show()


####################################### Takes care of Objective 6 ################################################################


#then get the projection angles using compoute angle between strain and dti
# get fiber aligned using the Evv

def calculate_displaced_vector(F, e_1):
    """
    Calculate the displaced vector using F(X)e_1 across all points and frames.
    
    Parameters:
    -----------
    F : 4D array of shape (x_dim, y_dim, frames, 3, 3)
        Deformation gradient tensor field
    e_1 : 3D array of shape (x_dim, y_dim, 3)
        Base vector field
    
    Returns:
    --------
    e_1_disp : 4D array of shape (x_dim, y_dim, frames, 3)
        Displaced vectors at each point for each frame
    """
    # Get dimensions
    x_dim, y_dim, num_frames = F.shape[0], F.shape[1], F.shape[2]
    
    # Initialize output array
    e_1_disp = np.zeros((x_dim, y_dim, num_frames, 3))
    
    # Process each frame
    for frame in range(num_frames):
        # For each spatial point, apply the deformation gradient
        for x in range(x_dim):
            for y in range(y_dim):
                # F[x,y,frame] is 3x3, e_1[x,y] is 3D vector
                e_1_disp[x, y, frame] = np.matmul(F[x, y, frame], e_1[x, y])
    
    return e_1_disp


DTI_Disp = calculate_displaced_vector(strain_rotated, DTI_rotated)

DTI_Disp = DTI_Disp[:, :, z_slice, :]
print("DTI Disp Shape", DTI_Disp.shape)

# print(DTI_Disp.shape)
DTI_Strain_Project_Angle_Disp = compute_angle_between_strain_DTI(
    strain_rotated, DTI_Disp, x_idx_strain, y_idx_strain
) # projeciton_angles, avg_strain_vectors, dti_mean_norm

mean_Eff_disp, std_Eff_disp = compute_Eff_with_std(strain_rotated, DTI_Disp, x_idx_strain, y_idx_strain)

mean_Lff_disp, std_Lff_disp = compute_Lff_with_std(strain_rotated, DTI_Disp, x_idx_strain, y_idx_strain)


# Plot all results using shared contour
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

frames = np.arange(len(mean_Evv))

#Lff fiber 
ax[0].errorbar(frames, mean_Lff_disp, yerr= std_Lff_disp, 
               fmt='-o',capsize = 5, color='blue', label='Eff')
ax[0].set_title('Lff W/ Reoriented DTI : lambda1.(F)lambda1')  # Add title to first subplot
ax[0].legend(loc='upper right')  # Add legend to first subplot
# ax[0].set_ylim([-1, 1])  # Replace y_min and y_max with desired values
ax[0].grid(True)

#Lff fiber 
ax[1].errorbar(frames, mean_Eff_disp, yerr= std_Eff_disp, 
               fmt='-o',capsize = 5, color='red', label='Eff')
ax[1].set_title('Eff W/ Reoriented DTI : (1/2)(v^T.C.v - 1)')  # Add title to first subplot
ax[1].legend(loc='upper right')  # Add legend to first subplot
# ax[0].set_ylim([-1, 1])  # Replace y_min and y_max with desired values
ax[1].grid(True)



#Eff fiber
ax[2].errorbar(frames, DTI_Strain_Project_Angle_Disp[0], yerr=DTI_Strain_Project_Angle_Disp[1], 
               fmt='o',capsize = 5, color='green', label='Angle')
ax[2].set_title('Projection Angles W/ DTI Reoriented')  # Add title to second subplot
ax[2].legend(loc='upper right')  # Add legend to second subplot
# ax[1].set_ylim([-0.5, 0.5])  # Replace y_min and y_max with desired values
ax[2].grid(True)

# Adjust layout for better visibility
plt.tight_layout()
plt.savefig('Fig 3.png')
plt.show()