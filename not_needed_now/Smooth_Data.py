import numpy as np
import cv2
from cv2.ximgproc import anisotropicDiffusion
from scipy.signal import savgol_filter

def smooth_anisotropic(E, mask, alpha=1.0, K=0.03, niters=8):
    """
    Applies anisotropic diffusion to masked region of tensor components (0,0), (0,1), (0,2)
    
    Parameters:
    - E: 4D tensor slice [x, y, 3, 3]
    - mask: 2D binary mask [x, y] (1 = ROI, 0 = background)
    - alpha: Diffusion coefficient (0.1-0.3)
    - K: Contrast sensitivity (0.05-0.15)
    - niters: Number of iterations (10-20)
    
    Returns:
    - Filtered tensor slice with preserved original values outside mask
    """
    filtered_E = np.copy(E)
    
    # Process each component in first column
    for comp_j in [0, 1, 2]:  # (0,0), (0,1), (0,2)
        # Extract and mask component data
        component = E[:, :, 0, comp_j].copy()
        masked_component = np.where(mask, component, np.nan)
        
        # Normalize within mask region
        gmin = np.nanmin(masked_component)
        gmax = np.nanmax(masked_component)
        norm_range = max(gmax - gmin, 1e-6)  # Prevent division by zero
        
        # Scale to 0-255 and convert to 3-channel
        normalized = ((masked_component - gmin) / norm_range * 255).astype(np.uint8)
        normalized_3ch = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
        
        # Replace NaNs with black (0) after normalization
        normalized_3ch[mask == 0] = 0
        
        # Apply anisotropic diffusion
        filtered = anisotropicDiffusion(normalized_3ch, alpha=alpha, K=K, niters=niters)
        
        # Convert back to single channel and denormalize
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        denormalized = (filtered_gray.astype(float) / 255 * norm_range) + gmin
        
        # Preserve original values outside mask
        filtered_component = np.where(mask, denormalized, component)
        filtered_E[:, :, 0, comp_j] = filtered_component
        
    return filtered_E


def smooth_savgol(data, mask, window_length=15, polyorder=3):
    smoothed_data = np.copy(data)
    x_idx, y_idx = np.where(mask == 1)
    
    for x, y in zip(x_idx, y_idx):
        component_values = data[x, y, 0, :]
        n_points = component_values.shape[0]  # Get actual data length
        
        # Dynamically adjust window length
        adj_window = min(window_length, n_points)
        if adj_window % 2 == 0:  # Ensure odd number
            adj_window = max(3, adj_window - 1)  # Never go below 3
            
        if adj_window > polyorder + 1:
            smoothed_component_values = savgol_filter(
                component_values,
                window_length=adj_window,
                polyorder=polyorder
            )
            smoothed_data[x, y, 0, :] = smoothed_component_values
            
    return smoothed_data