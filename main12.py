import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, prange
import os
from mpl_toolkits.mplot3d import Axes3D

# Create output directories
output_dirs = [
    "outputs/heatmaps",
    "outputs/animations",
    "outputs/flow_directions",
    "outputs/hydrographs",
    "outputs/validation",
    "outputs/surface_elevation"
]
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Define model parameters
length = 100.0        
width = 40.0         
resolution = 0.25    
nx = int(length / resolution)  
ny = int(width / resolution)   

# Physical parameters
n_manning = 0.03      
Ks = 1e-6            
min_depth = 0.00001    
total_time = 1800    
rainfall_rate = 0.01 / 3600  
rainfall_duration = 900     
g = 9.81              
CFL = 0.7             
MIN_DT = 0.01        
MAX_DT = 1.0         

# Create the DEM
dem = np.zeros((nx, ny))
slope_x = 1 / 100   
slope_y = 1 / 50    

x_coords = np.arange(nx) * resolution
y_coords = np.arange(ny) * resolution - (ny * resolution) / 2

for i in range(nx):
    for j in range(ny):
        elevation_x = x_coords[i] * slope_x
        elevation_y = abs(y_coords[j]) * slope_y
        dem[i, j] = elevation_y - elevation_x

# Initialize arrays
h = np.zeros((nx, ny))
kernels = [(-1, 0), (0, 1), (1, 0), (0, -1)]
di_array = np.array([k[0] for k in kernels])
dj_array = np.array([k[1] for k in kernels])
time = 0.0
dt = 1.0

# Storage for results
total_volume = []
mass_balance_error = []
frames = []  # For animation
flow_velocities = []

# Plotting functions
def save_heatmap(data, title, filename, cmap='Blues', vmin=None, vmax=None):
    """Save a heatmap plot"""
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(filename)
    plt.close()

def calculate_flow_field(h, dem, Q_out, di_array, dj_array, resolution):
    """
    Calculate flow field vectors more accurately using water surface gradients
    """
    nx, ny = h.shape
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    
    # Calculate water surface elevation
    wse = h + dem
    
    # Calculate gradients
    dx_wse = np.zeros_like(wse)
    dy_wse = np.zeros_like(wse)
    
    # Central difference for interior points
    dx_wse[1:-1, :] = (wse[2:, :] - wse[:-2, :]) / (2 * resolution)
    dy_wse[:, 1:-1] = (wse[:, 2:] - wse[:, :-2]) / (2 * resolution)
    
    # Forward/backward difference for edges
    dx_wse[0, :] = (wse[1, :] - wse[0, :]) / resolution
    dx_wse[-1, :] = (wse[-1, :] - wse[-2, :]) / resolution
    dy_wse[:, 0] = (wse[:, 1] - wse[:, 0]) / resolution
    dy_wse[:, -1] = (wse[:, -1] - wse[:, -2]) / resolution
    
    # Calculate flow directions based on gradients
    magnitude = np.sqrt(dx_wse**2 + dy_wse**2)
    u = np.where(magnitude > 0, -dx_wse/magnitude, 0)
    v = np.where(magnitude > 0, -dy_wse/magnitude, 0)
    
    # Scale by discharge magnitude
    Q_magnitude = np.sqrt(np.sum(Q_out**2, axis=2))
    u = u * Q_magnitude
    v = v * Q_magnitude
    
    return u, v, Q_magnitude

def save_flow_direction(h, dem, Q_out, di_array, dj_array, time, filename, resolution, subsample=5):
    """Enhanced flow direction plot with improved vector field"""
    plt.figure(figsize=(15, 10))
    
    # Calculate flow field
    u, v, Q_magnitude = calculate_flow_field(h, dem, Q_out, di_array, dj_array, resolution)
    
    # Create subsampled grid for clearer visualization
    y, x = np.mgrid[0:h.shape[0]:subsample, 0:h.shape[1]:subsample]
    
    # Create composite plot
    plt.subplot(221)
    plt.imshow(h, cmap='Blues', aspect='auto')
    plt.colorbar(label='Water Depth (m)')
    plt.quiver(x, y, 
              v[::subsample, ::subsample],
              u[::subsample, ::subsample],
              color='red', scale=30)
    plt.title('Flow Vectors with Water Depth')
    
    plt.subplot(222)
    plt.imshow(Q_magnitude, cmap='viridis', aspect='auto')
    plt.colorbar(label='Flow Magnitude (m³/s)')
    plt.title('Flow Magnitude')
    
    plt.subplot(223)
    plt.streamplot(np.arange(h.shape[1]), 
                  np.arange(h.shape[0]), 
                  v, u, color='red',
                  density=1.5)
    plt.imshow(h, cmap='Blues', alpha=0.3, aspect='auto')
    plt.title('Flow Streamlines')
    
    plt.subplot(224)
    flow_direction = np.arctan2(v, u) * 180 / np.pi
    plt.imshow(flow_direction, cmap='hsv', aspect='auto')
    plt.colorbar(label='Flow Direction (degrees)')
    plt.title('Flow Direction Angles')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_flow_magnitude(h, Q_out, time, filename):
    """Save flow magnitude plot"""
    # Calculate total flow magnitude at each cell
    flow_magnitude = np.sqrt(np.sum(Q_out**2, axis=2))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(flow_magnitude, cmap='viridis', aspect='auto')
    plt.colorbar(label='Flow Magnitude (m³/s)')
    plt.title(f'Flow Magnitude at t = {time:.1f} s')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(filename)
    plt.close()

def save_3d_surface(data, title, filename):
    """Save a 3D surface plot"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    surf = ax.plot_surface(X, Y, data, cmap='terrain')
    plt.colorbar(surf)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

def save_hydrograph(times, values, title, ylabel, filename):
    """Save a hydrograph plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(times, values)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_3d_surface_multi_angle(data, title_base, filename_base, dem):
    """Save 3D surface plots with enhanced water visualization"""
    angles = [
        (30, 45),   # Standard view
        (0, 0),     # Front view
        (0, 90),    # Top view
        (90, 0),    # Side view
        (45, 225),  # Alternative angle 1
        (60, 135)   # Alternative angle 2
    ]
    
    # Create custom colormap for better water-terrain distinction
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define colors for terrain (browns) and water (blues)
    terrain_colors = [(0.6, 0.4, 0.2), (0.8, 0.7, 0.5)]  # Brown colors for terrain
    water_colors = [(0.1, 0.3, 0.9), (0.0, 0.0, 0.7)]    # Blue colors for water
    
    # Create separate colormaps
    terrain_cmap = LinearSegmentedColormap.from_list('terrain', terrain_colors)
    water_cmap = LinearSegmentedColormap.from_list('water', water_colors)
    
    for elevation, azimuth in angles:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plot
        Y, X = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        
        # Plot terrain surface
        surf_terrain = ax.plot_surface(X.T, Y.T, dem,  # Transpose X and Y to match the data dimensions
                                     cmap=terrain_cmap,
                                     linewidth=0, 
                                     antialiased=True,
                                     alpha=1.0,
                                     label='Terrain')
        
        # Calculate water depth and create water surface
        water_depth = data - dem
        water_surface = np.where(water_depth > 0.001, data, np.nan)  # Use NaN for dry areas
        
        # Plot water surface
        surf_water = ax.plot_surface(X.T, Y.T, water_surface,
                                   cmap=water_cmap,
                                   linewidth=0,
                                   antialiased=True,
                                   alpha=0.7,
                                   label='Water')
        
        # Set viewing angle
        ax.view_init(elevation, azimuth)
        
        # Add color bars
        terrain_cb = plt.colorbar(surf_terrain, ax=ax, label='Terrain Elevation (m)', pad=0.1)
        water_cb = plt.colorbar(surf_water, ax=ax, label='Water Surface Elevation (m)', pad=0.15)
        
        # Add labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Elevation (m)')
        ax.set_title(f'{title_base}\nView: {elevation}° elevation, {azimuth}° azimuth')
        
        # Save figure
        angle_str = f'elev{elevation}_azim{azimuth}'
        plt.savefig(f'{filename_base}_{angle_str}.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_point_time_series(times, frames, points, dem, resolution, output_dir):
    """
    Create time series plots for specific points in the domain.
    
    Args:
        times: List of simulation timestamps
        frames: List of water depth arrays for each timestamp
        points: List of tuples [(x, y, name), ...] where x,y are coordinates in meters
        dem: Digital elevation model array
        resolution: Grid resolution
        output_dir: Output directory for saving plots
    """
    # Get domain dimensions
    ny, nx = dem.shape
    
    # Convert point coordinates from meters to grid indices
    grid_points = []
    for x, y, name in points:
        i = int(y/resolution + ny/2)  # Adding ny/2 to center the y-coordinate
        j = int(x/resolution)
        if 0 <= i < ny and 0 <= j < nx:
            grid_points.append((i, j, name))
        else:
            print(f"Warning: Point {name} ({x}m, {y}m) falls outside the domain and will be skipped.")
    
    if not grid_points:
        print("Error: No valid observation points within domain.")
        return
    
    # Extract time series for each point
    point_data = {
        'depth': {name: [] for _, _, name in grid_points},
        'velocity': {name: [] for _, _, name in grid_points},
        'wse': {name: [] for _, _, name in grid_points}  # Water surface elevation
    }
    
    # Calculate time series
    for frame in frames:
        # Calculate gradients for velocity estimation
        dy, dx = np.gradient(frame + dem, resolution)
        velocity = np.sqrt(dx**2 + dy**2)
        
        for i, j, name in grid_points:
            point_data['depth'][name].append(frame[i, j])
            point_data['velocity'][name].append(velocity[i, j])
            point_data['wse'][name].append(frame[i, j] + dem[i, j])
    
    # Verify data exists
    if not all(point_data['depth'].values()):
        print("Error: No data collected for observation points.")
        return
    
    # Create plots
    variables = {
        'depth': 'Water Depth (m)',
        'velocity': 'Flow Velocity (m/s)',
        'wse': 'Water Surface Elevation (m)'
    }
    
    # Print debug info
    print(f"Number of timestamps: {len(times)}")
    for name in point_data['depth'].keys():
        print(f"Number of data points for {name}: {len(point_data['depth'][name])}")
    
    for var_name, var_label in variables.items():
        plt.figure(figsize=(12, 6))
        for name in point_data[var_name].keys():
            data = point_data[var_name][name]
            if len(data) == len(times):  # Ensure data lengths match
                plt.plot(times, data, label=f'Point {name}')
                
                # Add statistics annotation
                stats_text = (f'Point {name}:\n'
                            f'Max: {max(data):.3f}\n'
                            f'Min: {min(data):.3f}\n'
                            f'Mean: {np.mean(data):.3f}')
                plt.annotate(
                    stats_text,
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    textcoords='axes fraction',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
        
        plt.xlabel('Time (s)')
        plt.ylabel(var_label)
        plt.title(f'{var_label} Time Series at Selected Points')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'point_timeseries_{var_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def calculate_mass_balance_metrics(h, rainfall_rate, Ks, dt, resolution, prev_volume=None):
    """Calculate mass balance metrics with corrected error calculation"""
    current_volume = np.sum(h) * resolution * resolution
    
    if prev_volume is None:
        prev_volume = current_volume
        return {
            'current_volume': current_volume,
            'volume_change': 0,
            'percent_error': 0,
            'rainfall_volume': 0,
            'infiltration_volume': 0,
            'theoretical_change': 0
        }
    
    # Calculate volume changes from rainfall and infiltration
    domain_area = h.shape[0] * h.shape[1] * resolution * resolution
    rainfall_volume = rainfall_rate * dt * domain_area
    infiltration_volume = Ks * dt * domain_area
    theoretical_change = rainfall_volume - infiltration_volume
    
    # Calculate actual volume change
    actual_change = current_volume - prev_volume
    
    # Calculate error as difference between actual and theoretical change
    volume_error = actual_change - theoretical_change
    
    # Calculate percentage error relative to the current total volume
    if current_volume > 1e-6:  # Avoid division by zero
        percent_error = (volume_error / current_volume) * 100
    else:
        percent_error = 0
    
    return {
        'current_volume': current_volume,
        'volume_change': actual_change,
        'percent_error': percent_error,
        'rainfall_volume': rainfall_volume,
        'infiltration_volume': infiltration_volume,
        'theoretical_change': theoretical_change
    }

def save_mass_balance_plots(times, metrics_history, save_dir):
    """Save mass balance analysis plots with corrected calculations"""
    # Extract time series from metrics history
    volumes = [m['current_volume'] for m in metrics_history]
    errors = [m['percent_error'] for m in metrics_history]
    rainfall_vol = np.cumsum([m['rainfall_volume'] for m in metrics_history])
    infilt_vol = np.cumsum([m['infiltration_volume'] for m in metrics_history])
    
    # Create composite plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Volume over time
    ax1.plot(times, volumes, 'b-')
    ax1.set_title('Water Volume Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Volume (m³)')
    ax1.grid(True)
    
    # Plot 2: Mass Balance Error
    ax2.plot(times, errors, 'r-')
    ax2.set_title('Mass Balance Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (%)')
    ax2.grid(True)
    
    # Plot 3: Cumulative Rainfall vs Infiltration
    ax3.plot(times, rainfall_vol, 'b-', label='Cumulative Rainfall')
    ax3.plot(times, infilt_vol, 'r-', label='Cumulative Infiltration')
    ax3.set_title('Cumulative Water Balance Components')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Volume (m³)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Histogram of errors
    ax4.hist(errors, bins=30, color='gray', alpha=0.7)
    ax4.set_title('Distribution of Mass Balance Errors')
    ax4.set_xlabel('Error (%)')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mass_balance_analysis.png'))
    plt.close()

# Keep the original optimized computation functions
@jit(nopython=True, parallel=True, fastmath=True)
def compute_fluxes_parallel(h, dem, dt, resolution, n_manning, min_depth, di_array, dj_array):
    """Original compute_fluxes_parallel function"""
    # [Previous implementation remains the same]
    nx, ny = h.shape
    h_padded = np.zeros((nx + 2, ny + 2), dtype=h.dtype)
    dem_padded = np.zeros((nx + 2, ny + 2), dtype=dem.dtype)
    
    h_padded[1:-1, 1:-1] = h
    dem_padded[1:-1, 1:-1] = dem
    
    h_padded[0, 1:-1] = h[0, :]    
    h_padded[-1, 1:-1] = h[-1, :]  
    h_padded[:, 0] = h_padded[:, 1]    
    h_padded[:, -1] = h_padded[:, -2]  
    
    dem_padded[0, 1:-1] = dem[0, :]    
    dem_padded[-1, 1:-1] = dem[-1, :]  
    dem_padded[:, 0] = dem_padded[:, 1]    
    dem_padded[:, -1] = dem_padded[:, -2]  
    
    Q_out = np.zeros((nx, ny, len(di_array)), dtype=h.dtype)
    max_velocity = 0.0
    
    for i in prange(nx):
        for j in range(ny):
            for idx in range(len(di_array)):
                di, dj = di_array[idx], dj_array[idx]
                
                h_i = h_padded[i+1, j+1]
                z_i = dem_padded[i+1, j+1]
                H_i = h_i + z_i
                
                h_j = h_padded[i+1+di, j+1+dj]
                z_j = dem_padded[i+1+di, j+1+dj]
                H_j = h_j + z_j
                
                delta_H = H_i - H_j
                S = delta_H / resolution
                
                if S > 0 and h_i > min_depth:
                    R = h_i
                    A = h_i * resolution
                    Q = (A * R**(2/3) * S**0.5) / n_manning
                    velocity = Q / (h_i * resolution)
                    max_velocity = max(max_velocity, velocity)
                    Q_out[i, j, idx] = Q

    return Q_out, max_velocity

@jit(nopython=True, parallel=True, fastmath=True)
def update_water_depths(h, Q_out, dt, resolution, di_array, dj_array):
    """Original update_water_depths function"""
    # [Previous implementation remains the same]
    nx, ny = h.shape
    delta_h = np.zeros_like(h)
    
    V_t = h * resolution * resolution
    Q_total_out = np.sum(Q_out, axis=2)
    c = np.ones_like(h)
    
    for i in prange(nx):
        for j in range(ny):
            if Q_total_out[i, j] > 0:
                c[i, j] = min(1.0, V_t[i, j] / (Q_total_out[i, j] * dt))
    
    for idx in range(len(di_array)):
        Q = Q_out[:, :, idx] * c
        Q_in = np.zeros_like(h)
        
        for i in prange(nx):
            for j in range(ny):
                i_from = (i - di_array[idx]) % nx
                j_from = (j - dj_array[idx]) % ny
                Q_in[i, j] = Q[i_from, j_from]
        
        delta_h += (Q_in - Q) * dt / (resolution * resolution)
    
    return delta_h

# Simulation loop with plot saving
save_interval = 300  # Save plots every 500 seconds
times = []
metrics_history = []
prev_volume = None

while time < total_time:
    # Apply rainfall and infiltration
    if time <= rainfall_duration:
        h += rainfall_rate * dt - Ks * dt
    else:
        h -= Ks * dt
    
    np.maximum(h, 0.0, out=h)
    
    # Compute fluxes and update water depths
    Q_out, max_velocity = compute_fluxes_parallel(h, dem, dt, resolution, n_manning, 
                                                min_depth, di_array, dj_array)
    delta_h = update_water_depths(h, Q_out, dt, resolution, di_array, dj_array)
    
    # Update water depth
    h += delta_h
    np.maximum(h, 0.0, out=h)
    np.minimum(h, np.max(dem) - dem, out=h)
    
    # Update time step
    if max_velocity > 0:
        dt = np.clip(CFL * resolution / max_velocity, MIN_DT, MAX_DT)
    else:
        dt = MAX_DT
    
    # Store results
    time += dt
    times.append(time)
    total_volume.append(np.sum(h) * resolution * resolution)
    flow_velocities.append(max_velocity)
    frames.append(h.copy())

    # Calculate mass balance metrics with the previous volume
    metrics = calculate_mass_balance_metrics(h, rainfall_rate, Ks, dt, resolution, prev_volume)
    prev_volume = metrics['current_volume']  # Update previous volume for next iteration
    metrics_history.append(metrics)

    # Ensure volumes are appended correctly
    # Check if the length of times and total_volume is consistent
    if len(times) != len(total_volume):
        print(f"Warning: Length mismatch - times: {len(times)}, total_volume: {len(total_volume)}")

    # Save plots at intervals
    if int(time) % save_interval == 0:
        timestep_str = f"{int(time):05d}"
        
        # Save water depth heatmap
        save_heatmap(h, f'Water Depth at t = {time:.1f} s',
                    f'outputs/heatmaps/depth_{timestep_str}.png',
                    vmin=0, vmax=np.max(h))
        
        # Save water surface elevation
        save_heatmap(dem + h, f'Water Surface Elevation at t = {time:.1f} s',
                    f'outputs/surface_elevation/elevation_{timestep_str}.png',
                    cmap='terrain')
        
        # Save enhanced flow direction plots
        save_flow_direction(h, dem, Q_out, di_array, dj_array, time,
                          f'outputs/flow_directions/flow_{timestep_str}.png',
                          resolution)
        
        # Save flow magnitude plot
        save_flow_magnitude(h, Q_out, time,
                          f'outputs/flow_directions/magnitude_{timestep_str}.png')

        # Save 3D surface plot
        save_3d_surface_multi_angle(
        dem + h,
        f'Water Surface Profile at t = {time:.1f} s',
        f'outputs/surface_elevation/profile_3d_{timestep_str}',
        dem)

# Example usage with adjusted observation points:
observation_points = [
    (25.0, 0.0, "Upstream"),    # Quarter length, center
    (15.0, 0.0, "Midstream"),   # Middle length, center
    (0.0, 0.0, "Downstream")   # Three-quarter length, center
]

# After simulation completes:
save_point_time_series(times, frames, observation_points, dem, resolution,
                      'outputs/validation')
# Save final mass balance analysis
save_mass_balance_plots(times, metrics_history, 'outputs/validation')

# Hydrograph
save_hydrograph(times, total_volume, 'Global Hydrograph',
                'Total Water Volume (m³)',
                'outputs/hydrographs/global_hydrograph.png')

# Flow velocity over time
save_hydrograph(times, flow_velocities, 'Maximum Flow Velocity',
                'Velocity (m/s)',
                'outputs/hydrographs/flow_velocity.png')

# Mass balance error
initial_volume = total_volume[0]
mass_balance_error = [(v - initial_volume) / initial_volume * 100 for v in total_volume]
save_hydrograph(times, mass_balance_error, 'Mass Balance Error',
                'Error (%)',
                'outputs/validation/mass_balance_error.png')


# Save final state flow visualizations
save_flow_direction(h, dem, Q_out, di_array, dj_array, time,
                   'outputs/flow_directions/final_flow.png', resolution)

save_flow_magnitude(h, Q_out, time,
                   'outputs/flow_directions/final_magnitude.png')

save_heatmap(h, 'Final Water Depth',
             'outputs/heatmaps/final_depth.png',
             vmin=0, vmax=np.max(h))

save_heatmap(dem + h, 'Final Water Surface Elevation',
             'outputs/surface_elevation/final_elevation.png',
             cmap='terrain')

save_3d_surface_multi_angle(dem + h, 'Final Water Surface Profile',
                'outputs/surface_elevation/final_profile_3d.png', dem)

print("Simulation completed! All outputs have been saved to the 'outputs' directory.")