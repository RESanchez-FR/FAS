from typing import Tuple, Dict
import numpy as np

# def calculate_strain_stats(E_fiber: np.ndarray, 
#                           theta: np.ndarray,
#                           E_masked_smooth: np.ndarray,
#                           components: Tuple[int, int, int] = (0, 1, 2)) -> Dict[str, float]:
#     """
#     Calculate strain statistics including averages, standard deviations, and quadrant-adjusted values.
    
#     Parameters:
#     - E_fiber: Array of fiber strain values
#     - theta: Array of angle values in radians
#     - E_masked_smooth: 3D array of smoothed strain values [samples, time, components]
#     - components: Tuple specifying (x, y, z) component indices (default: 0,1,2)
    
#     Returns:
#     Dictionary containing computed statistics
#     """
#     # Unpack component indices
#     component_x, component_y, component_z = components
    
#     # Calculate basic statistics
#     stats = {
#         'E_avg': np.mean(E_fiber),
#         'E_std': np.std(E_fiber),
#         'theta_avg': np.degrees(np.mean(theta)),
#         'theta_std': np.degrees(np.std(theta))
#     }
    
#     # Flip quadrant based on E_avg sign
#     stats['E_avg'] = -stats['E_avg'] if stats['E_avg'] > 0 else stats['E_avg']
    
#     # Component-specific calculations
#     for axis, comp in zip(['x', 'y', 'z'], [component_x, component_y, component_z]):
#         strain_values = E_masked_smooth[:, 0, comp]
#         avg = np.mean(strain_values)
#         std = np.std(strain_values)
        
#         stats[f'average_strain_{axis}'] = avg
#         stats[f'average_strain_{axis}_std'] = std
#         stats[f'flipped_average_strain_{axis}'] = -avg if avg > 0 else avg

#     return stats

def calculate_strain_stats(E_fiber: np.ndarray, 
                          theta: np.ndarray,
                          E_masked_smooth: np.ndarray,
                          components: Tuple[int, int, int] = (0, 1, 2)) -> Dict[str, float]:
    """
    Calculate strain statistics with component sum additions.
    
    New features:
    - Total sum of component averages
    - Total sum of flipped component averages
    - Individual component contributions to sums
    """
    component_x, component_y, component_z = components
    stats = {
        'E_avg': np.mean(E_fiber),
        'E_std': np.std(E_fiber),
        'theta_avg': np.degrees(np.mean(theta)),
        'theta_std': np.degrees(np.std(theta))
    }
    
    # Flip quadrant based on E_avg sign
    stats['E_avg'] = -stats['E_avg'] if stats['E_avg'] > 0 else stats['E_avg']
    
    # Initialize sum tracking
    component_sums = {
        'raw': 0.0,
        'flipped': 0.0,
        'components': []
    }
    
    for axis, comp in zip(['x', 'y', 'z'], [component_x, component_y, component_z]):
        strain_values = E_masked_smooth[:, 0, comp]
        avg = np.mean(strain_values)
        std = np.std(strain_values)
        flipped = -avg if avg > 0 else avg
        
        # Store individual stats
        stats[f'average_strain_{axis}'] = avg
        stats[f'average_strain_{axis}_std'] = std
        stats[f'flipped_average_strain_{axis}'] = flipped
        
        # Accumulate sums
        component_sums['raw'] += avg
        component_sums['flipped'] += flipped
        component_sums['components'].append({
            'axis': axis,
            'average': avg,
            'flipped': flipped,
            'contribution': avg/component_sums['raw'] if component_sums['raw'] != 0 else 0
        })
    
    # Add sum statistics to main output
    stats.update({
        'total_component_sum': component_sums['raw'],
        'total_flipped_sum': component_sums['flipped'],
        'component_breakdown': component_sums['components']
    })
    
    return stats
