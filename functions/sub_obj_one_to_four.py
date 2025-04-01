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
DTI_rotated = DTI_rotated[:, :, z_slice, :] #3D data now



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

# def compute_angle_between_strain_DTI(data_1, data_2, x_idx, y_idx):
#     """
#     Calculate angles between averaged eigenvectors within a contour-defined ROI
#     across temporal frames.
    
#     Args:
#         strain_4d: 4D array (X, Y, Time, 3) - Strain eigenvectors
#         dti_3d: 3D array (X, Y, 3) - DTI eigenvectors
#         contour_points: List of (x,y) points defining ROI contour
        
#     Returns:
#         Tuple of (angles_deg, avg_strain_vectors, avg_dti_vector)
#     """

#     # Preprocess DTI vectors (static across frames)
#     dti_roi = data_2[x_idx, y_idx, :]  # Shape: (N_points, 3)
#     dti_mean = np.mean(dti_roi, axis=0)
#     dti_mean_norm = dti_mean / np.linalg.norm(dti_mean)
    
#     # Initialize storage
#     num_frames = data_1.shape[2]
#     angles_deg = np.full(num_frames, np.nan)
#     avg_strain_vectors = np.zeros((num_frames, 3))
    
#     for t in range(num_frames):
#         # Extract and average strain vectors in ROI
#         strain_vectors = data_1[x_idx, y_idx, t, :]
        
#         if strain_vectors.size == 0:
#             continue
            
#         strain_mean = np.mean(strain_vectors, axis=0)
#         strain_mag = np.linalg.norm(strain_mean)
        
#         if strain_mag < 1e-6:  # Handle zero vectors
#             continue
            
#         # Normalize and store averaged vector
#         strain_mean_norm = strain_mean / strain_mag
#         avg_strain_vectors[t] = strain_mean_norm
        
#         # Calculate angle between averaged vectors
#         dot_product = np.dot(strain_mean_norm, dti_mean_norm)
#         dot_clipped = np.clip(dot_product, -1.0, 1.0)
#         angles_deg[t] = np.degrees(np.arccos(dot_clipped))

#     return angles_deg, avg_strain_vectors, dti_mean_norm

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
ax[0].errorbar(frames, strain_results[0], yerr = strain_results[2] , 
               fmt = 'o', color='blue', label='Polar Angle')
ax[0].errorbar(frames, strain_results[1], yerr = strain_results[3] , 
               fmt = 'o', color='red', label='Azimuth Angle')

# DTI angles 
ax[1].errorbar([0],  DTI_results[0], yerr = DTI_results[2], 
               fmt = 'o', color='blue', label='Polar Angle')
ax[1].errorbar([1], DTI_results[1], yerr = DTI_results[3], 
               fmt = 'o', color='red',  label='Azimuth Angle')
# ax[1].title('DTI Angle')

# Strain-DTI comparison
ax[2].errorbar(frames,  comparison_results[0], yerr = comparison_results[1], 
               fmt = 'o', color='blue', label='Polar Angle')
# ax[2].title('Strain DTI Projection Angle')

plt.savefig('combined_results.png')

# "works for 4D data "
# def compute_x_y_z_angles_strain(data_1, data_2, mag_data, z_slice):
#     """
#     Compute angles with standard deviations using global contour_points.
#     Returns: (avg_polar, avg_azimuth, std_polar, std_azimuth)
#     """
#     global contour_points
#     contour_points = []
#     visualize_magnitude_image(mag_data, z_slice)
    
#     if len(contour_points) < 3:
#         raise ValueError(f"Need ≥3 contour points (got {len(contour_points)})")
    
#     # Create mask
#     contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
#     mask = np.zeros(data_2.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [contour_array], 1)
#     x_idx, y_idx = np.where(mask == 1)

#     # Initialize storage with NaNs
#     num_frames = data_1.shape[2]
#     avg_polar = np.full(num_frames, np.nan)
#     avg_azimuth = np.full(num_frames, np.nan)
#     std_polar = np.full(num_frames, np.nan)
#     std_azimuth = np.full(num_frames, np.nan) #adding a comment

#     for t in range(num_frames):
#         # Extract vectors in ROI
#         vectors = data_1[x_idx, y_idx, t, :]  # Shape: (N, 3)
#         if len(vectors) == 0: continue
        
#         # Compute individual angles
#         magnitudes = np.linalg.norm(vectors, axis=1)
#         valid_mask = magnitudes > 1e-6
#         if not np.any(valid_mask): continue
        
#         # Polar angles
#         z_norm = vectors[:, 2] / np.where(valid_mask, magnitudes, 1)
#         theta_all = np.degrees(np.arccos(z_norm[valid_mask]))
        
#         # Azimuth angles (handle circular nature)
#         phi_all = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))[valid_mask]
#         phi_all = (phi_all + 360) % 360  # Convert to 0-360 range
        
#         # Mean vector calculation
#         mean_vec = np.mean(vectors[valid_mask], axis=0)
#         mean_mag = np.linalg.norm(mean_vec)
        
#         # Store results
#         if mean_mag > 1e-6:
#             avg_polar[t] = np.degrees(np.arccos(mean_vec[2]/mean_mag))
#             avg_azimuth[t] = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
#             std_polar[t] = np.std(theta_all)
#             std_azimuth[t] = np.std((phi_all - avg_azimuth[t] + 180) % 360 - 180)

#     return avg_polar, avg_azimuth, std_polar, std_azimuth

# # Plot with error bars
# avg_polar_strain, avg_azimuth_strain, std_polar_strain, std_azimuth_strain = compute_x_y_z_angles_strain(
#     data_1=strain_rotated,
#     data_2=DTI_rotated,
#     mag_data=m_data,
#     z_slice=10
# )

# plt.figure(figsize=(12, 6))
# frames = np.arange(len(avg_polar_strain))

# # Polar plot
# plt.errorbar(frames, avg_polar_strain, yerr=std_polar_strain, 
#             fmt='o', color='#1f77b4', markersize=6,
#             capsize=4, capthick=1, label='Polar Angle (θ)')

# # Azimuth plot
# plt.errorbar(frames, avg_azimuth_strain, yerr=std_azimuth_strain,
#             fmt='s', color='#ff7f0e', markersize=5,
#             capsize=4, capthick=1, label='Azimuthal Angle (φ)')

# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Angle (degrees)', fontsize=12)
# plt.title('Strain Angles w/Standard Deviation', fontsize=14)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()

# plt.savefig('Strain Angle Plot.png', dpi=300, bbox_inches='tight', facecolor='white')  # <-- Add this line
# plt.show()


# ####################################### Takes care of Objective 3 ################################################################

# "works for DTI but still unsure"

# def compute_x_y_z_angles_DTI(data_2, mag_data, z_slice):
#     """
#     Compute angles and deviations for DTI data (x, y, components) in a contoured region.
#     Returns: (avg_polar, avg_azimuth, std_polar, std_azimuth)
#     """
#     global contour_points
#     contour_points = []  # Reset contour
    
#     # Draw contour on magnitude image
#     visualize_magnitude_image(mag_data, z_slice)
    
#     # Validate contour
#     if len(contour_points) < 3:
#         raise ValueError(f"Need ≥3 contour points (got {len(contour_points)})")
    
#     # Create mask
#     contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
#     mask = np.zeros(data_2.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [contour_array], 1)
#     x_idx, y_idx = np.where(mask == 1)
    
#     # Extract vectors from last dimension (components)
#     vectors = data_2[x_idx, y_idx, :]  # Shape: (N_points, 3)
    
#     # Initialize outputs
#     avg_polar, avg_azimuth = np.nan, np.nan
#     std_polar, std_azimuth = np.nan, np.nan
    
#     if len(vectors) > 0:
#         # Compute mean vector
#         mean_vec = np.mean(vectors, axis=0)
#         magnitude = np.linalg.norm(mean_vec)
        
#         if magnitude > 1e-6:
#             # Average angles from mean vector
#             avg_polar = np.degrees(np.arccos(mean_vec[2] / magnitude))
#             avg_azimuth = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
            
#             # Individual angles for deviation calculation
#             magnitudes = np.linalg.norm(vectors, axis=1)
#             valid = magnitudes > 1e-6
#             z_norm = vectors[:, 2] / np.where(valid, magnitudes, 1)
#             theta_all = np.degrees(np.arccos(z_norm[valid]))
#             phi_all = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
            
#             # Circular std for azimuth (handle wrap-around)
#             phi_diff = (phi_all - avg_azimuth + 180) % 360 - 180
#             std_polar = np.std(theta_all)
#             std_azimuth = np.std(phi_diff)
    
#     return avg_polar, avg_azimuth, std_polar, std_azimuth

# # Compute angles for 3D data (x, y, components)
# avg_polar_DTI, avg_azim_DTI, std_polar_DTI, std_azim_DTI = compute_x_y_z_angles_DTI(
#     data_2=DTI_rotated,
#     mag_data=m_data,
#     z_slice=10
# )

# # Create error bar plot
# plt.figure(figsize=(8, 5))
# plt.errorbar(
#     [0], [avg_polar_DTI], yerr=[std_polar_DTI],
#     fmt='o', capsize=5, label='Polar Angle (θ)'
# )
# plt.errorbar(
#     [1], [avg_azim_DTI], yerr=[std_azim_DTI],
#     fmt='s', capsize=5, label='Azimuthal Angle (φ)'
# )

# plt.xticks([0, 1], ['Polar', 'Azimuth'])
# plt.ylabel('Angle (degrees)')
# plt.title('DTI Angles w/Standard Deviation in ROI')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('DTI Angle Plot.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()


# ####################################### Takes care of Objective 4 ################################################################

# def compute_angle_between_strain_DTI(data_1, data_2, mag_data, z_slice):
#     """
#     Calculate angles between averaged eigenvectors within a contour-defined ROI
#     across temporal frames.
    
#     Args:
#         strain_4d: 4D array (X, Y, Time, 3) - Strain eigenvectors
#         dti_3d: 3D array (X, Y, 3) - DTI eigenvectors
#         contour_points: List of (x,y) points defining ROI contour
        
#     Returns:
#         Tuple of (angles_deg, avg_strain_vectors, avg_dti_vector)
#     """

#     global contour_points
#     contour_points = []  # Reset contour
    
#     # Draw contour on magnitude image
#     visualize_magnitude_image(mag_data, z_slice)
    
#     # Validate contour
#     if len(contour_points) < 3:
#         raise ValueError(f"Need ≥3 contour points (got {len(contour_points)})")
    
#     # Create mask
#     contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
#     mask = np.zeros(data_2.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [contour_array], 1)
#     x_idx, y_idx = np.where(mask == 1)

#     # Create ROI mask from contour points
#     contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
#     mask = np.zeros(data_1.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [contour_array], 1)
#     x_idx, y_idx = np.where(mask == 1)
    
#     if len(x_idx) == 0:
#         raise ValueError("Contour ROI contains no valid pixels")

#     # Preprocess DTI vectors (static across frames)
#     dti_roi = data_2[x_idx, y_idx, :]  # Shape: (N_points, 3)
#     dti_mean = np.mean(dti_roi, axis=0)
#     dti_mean_norm = dti_mean / np.linalg.norm(dti_mean)
    
#     # Initialize storage
#     num_frames = data_1.shape[2]
#     angles_deg = np.full(num_frames, np.nan)
#     avg_strain_vectors = np.zeros((num_frames, 3))
    
#     for t in range(num_frames):
#         # Extract and average strain vectors in ROI
#         strain_vectors = data_1[x_idx, y_idx, t, :]
        
#         if strain_vectors.size == 0:
#             continue
            
#         strain_mean = np.mean(strain_vectors, axis=0)
#         strain_mag = np.linalg.norm(strain_mean)
        
#         if strain_mag < 1e-6:  # Handle zero vectors
#             continue
            
#         # Normalize and store averaged vector
#         strain_mean_norm = strain_mean / strain_mag
#         avg_strain_vectors[t] = strain_mean_norm
        
#         # Calculate angle between averaged vectors
#         dot_product = np.dot(strain_mean_norm, dti_mean_norm)
#         dot_clipped = np.clip(dot_product, -1.0, 1.0)
#         angles_deg[t] = np.degrees(np.arccos(dot_clipped))

#     return angles_deg, avg_strain_vectors, dti_mean_norm
