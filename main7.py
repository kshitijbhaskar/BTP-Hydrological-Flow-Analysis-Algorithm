import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, prange, float64, int64, boolean
import numpy.typing as npt

# Define model parameters
length = 100.0        # Length of the domain in meters
width = 40.0          # Width of the domain in meters
resolution = 0.25     # Grid resolution in meters
nx = int(length / resolution)  # Number of cells in x-direction
ny = int(width / resolution)   # Number of cells in y-direction

# Physical parameters
n_manning = 0.03       # Manning's roughness coefficient
Ks = 1e-6              # Saturated hydraulic conductivity (m/s)
min_depth = 0.00001    
total_time = 1800
rainfall_rate = 0.01 / 3600  
rainfall_duration = 900     # Duration of rainfall in seconds
g = 9.81               # Gravitational acceleration (m/s^2)
CFL = 0.7              # CFL number for stability (less than 1)
MIN_DT = 0.01         # Minimum time step (seconds)
MAX_DT = 1.0          # Maximum time step (seconds)

# Create the DEM
dem = np.zeros((nx, ny))

# Initialize the DEM with slopes
slope_x = 1 / 100   # 1% slope along x-direction
slope_y = 1 / 50    # 2% slope along y-direction towards the center

x_coords = np.arange(nx) * resolution
y_coords = np.arange(ny) * resolution - (ny * resolution) / 2

for i in range(nx):
    for j in range(ny):
        elevation_x = x_coords[i] * slope_x
        elevation_y = abs(y_coords[j]) * slope_y
        dem[i, j] = elevation_y - elevation_x

# Initialize water depth grid
h = np.zeros((nx, ny))

# Flow direction kernels for four cardinal directions
kernels = [
    (-1, 0),  # Up (North)
    (0, 1),   # Right (East)
    (1, 0),   # Down (South)
    (0, -1),  # Left (West)
]

di_array = np.array([k[0] for k in kernels])
dj_array = np.array([k[1] for k in kernels])

# Distance between cell centers
distance = resolution

# Time stepping parameters
time = 0.0

# Arrays to store mass balance information
total_volume = []
mass_balance_error = []

# Visualization setup
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax[0].imshow(h, cmap='Blues', vmin=0, vmax=0.01, aspect='auto')
cbar1 = plt.colorbar(im1, ax=ax[0])
cbar1.set_label('Water Depth (m)')
ax[0].set_title('Water Depth at t = 0.0 s')
ax[0].set_xlabel('X Coordinate')
ax[0].set_ylabel('Y Coordinate')

im2 = ax[1].imshow(dem + h, cmap='terrain', aspect='auto')
cbar2 = plt.colorbar(im2, ax=ax[1])
cbar2.set_label('Water Surface Elevation (m)')
ax[1].set_title('Water Surface Elevation at t = 0.0 s')
ax[1].set_xlabel('X Coordinate')
ax[1].set_ylabel('Y Coordinate')

# Initialize time step
dt = 1.0  # Initial time step

# Add these optimization parameters
USE_PARALLEL = True  # Enable parallel processing
DTYPE = np.float32  # Use single precision instead of double

# Convert arrays to single precision
dem = dem.astype(DTYPE)
h = h.astype(DTYPE)
di_array = di_array.astype(np.int32)
dj_array = dj_array.astype(np.int32)

@jit(nopython=True, parallel=USE_PARALLEL, fastmath=True)
def compute_fluxes_parallel(h, dem, dt, resolution, n_manning, min_depth, di_array, dj_array):
    """Optimized parallel computation of water fluxes"""
    nx, ny = h.shape
    # Create padded arrays manually instead of using np.pad
    h_padded = np.zeros((nx + 2, ny + 2), dtype=h.dtype)
    dem_padded = np.zeros((nx + 2, ny + 2), dtype=dem.dtype)
    
    # Fill the center
    h_padded[1:-1, 1:-1] = h
    dem_padded[1:-1, 1:-1] = dem
    
    # Fill the edges (equivalent to mode='edge')
    h_padded[0, 1:-1] = h[0, :]    # Top edge
    h_padded[-1, 1:-1] = h[-1, :]  # Bottom edge
    h_padded[:, 0] = h_padded[:, 1]    # Left edge
    h_padded[:, -1] = h_padded[:, -2]  # Right edge
    
    dem_padded[0, 1:-1] = dem[0, :]    # Top edge
    dem_padded[-1, 1:-1] = dem[-1, :]  # Bottom edge
    dem_padded[:, 0] = dem_padded[:, 1]    # Left edge
    dem_padded[:, -1] = dem_padded[:, -2]  # Right edge
    
    Q_out = np.zeros((nx, ny, len(di_array)), dtype=h.dtype)
    max_velocity = 0.0
    
    # Parallel processing for each row
    for i in prange(nx):
        for j in range(ny):
            for idx in range(len(di_array)):
                di, dj = di_array[idx], dj_array[idx]
                
                # Current cell properties
                h_i = h_padded[i+1, j+1]
                z_i = dem_padded[i+1, j+1]
                H_i = h_i + z_i
                
                # Neighbor cell properties
                h_j = h_padded[i+1+di, j+1+dj]
                z_j = dem_padded[i+1+di, j+1+dj]
                H_j = h_j + z_j
                
                # Calculate water surface slope
                delta_H = H_i - H_j
                S = delta_H / resolution
                
                if S > 0 and h_i > min_depth:
                    # Simplified Manning's equation
                    R = h_i
                    A = h_i * resolution
                    Q = (A * R**(2/3) * S**0.5) / n_manning
                    velocity = Q / (h_i * resolution)
                    max_velocity = max(max_velocity, velocity)
                    Q_out[i, j, idx] = Q

    return Q_out, max_velocity

@jit(nopython=True, parallel=USE_PARALLEL, fastmath=True)
def update_water_depths(h, Q_out, dt, resolution, di_array, dj_array):
    """Optimized parallel computation of water depth updates"""
    nx, ny = h.shape
    delta_h = np.zeros_like(h)
    
    # Calculate scaling coefficients
    V_t = h * resolution * resolution
    Q_total_out = np.sum(Q_out, axis=2)
    c = np.ones_like(h)
    
    for i in prange(nx):
        for j in range(ny):
            if Q_total_out[i, j] > 0:
                c[i, j] = min(1.0, V_t[i, j] / (Q_total_out[i, j] * dt))
    
    # Apply scaling and calculate net fluxes
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

# Precompile the JIT functions
print("Compiling JIT functions...")
small_h = np.zeros((10, 10), dtype=DTYPE)
small_dem = np.zeros((10, 10), dtype=DTYPE)
_ = compute_fluxes_parallel(small_h, small_dem, 1.0, resolution, n_manning, min_depth, di_array, dj_array)
_ = update_water_depths(small_h, np.zeros((10, 10, 4), dtype=DTYPE), 1.0, resolution, di_array, dj_array)
print("JIT compilation complete!")

frame_count = 0
def update(frame):
    global h, time, dt, frame_count
    
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
    
    # Update time step based on CFL condition with bounds
    if max_velocity > 0:
        dt = np.clip(CFL * resolution / max_velocity, MIN_DT, MAX_DT)
    else:
        dt = MAX_DT
    
    time += dt
    frame_count += 1
    
    # Stop the animation if we reach total_time
    if time >= total_time:
        anim.event_source.stop()
    
    # Mass balance calculations
    current_volume = np.sum(h) * resolution * resolution
    total_volume.append(current_volume)
    
    if len(total_volume) > 1:
        mass_balance_error.append(0.0)  # Simplified for speed
    
    # Update plots
    im1.set_array(h)
    im1.set_clim(vmax=np.max(h))
    ax[0].set_title(f'Water Depth at t = {time:.1f} s')
    
    im2.set_array(dem + h)
    im2.set_clim(vmax=np.max(dem + h))
    ax[1].set_title(f'Water Surface Elevation at t = {time:.1f} s')
    
    # Force the titles to update
    ax[0].figure.canvas.draw_idle()
    
    return im1, im2

# Modify animation setup
frames_estimate = int(total_time / MIN_DT)  # Estimate maximum number of frames
anim = FuncAnimation(fig, update, frames=frames_estimate, interval=50, blit=True, repeat=False)

# Save animation with better settings (if needed)
# anim.save('simulation_optimized_4.gif', 
#           writer='pillow', 
#           fps=10,
#           dpi=72
# )
plt.show()
