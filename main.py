import threading
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parking lot dimensions and DEM resolution
length = 100.0  # in meters
width = 40.0    # in meters
resolution = 0.25  # DEM resolution in meters

# Create a grid for the DEM
nx = int(length / resolution)  # Number of grid points along length
ny = int(width / resolution)   # Number of grid points along width

# Initialize the DEM with zeros
dem = np.zeros((nx, ny))

print(f"DEM Grid Size: {dem.shape}")

# Slopes
slope_length = 1 / 100  # Cross slope along length
slope_width = 1 / 50    # Valley-like slope along width

# Apply slopes to the DEM
for i in range(nx):
    for j in range(ny):
        # Calculate elevation based on both slopes
        elevation_length = i * resolution * slope_length
        elevation_width = abs(j - ny / 2) * resolution * slope_width
        dem[i, j] = elevation_width - elevation_length

# DEM now contains the elevations
print(dem)

rainfall_rate = 0.2  # in meters per hour
rainfall_volume = rainfall_rate * (resolution * resolution)  # Volume of rain per cell in cubic meters

# Constants
Ks = 10e-7  # Saturated Hydraulic Conductivity for sand in m/s (example value)
time = 3600  # Time in seconds (1 hour)

# Calculate infiltration per grid cell (assuming uniform infiltration rate)
infiltration_rate = Ks * time  # Convert m/s to m per hour
infiltration_volume = infiltration_rate * (resolution * resolution)  # Infiltration volume per cell in cubic meters

# Calculate runoff as the difference between rainfall and infiltration
runoff_volume = np.maximum(0, rainfall_volume - infiltration_volume)

# Runoff_volume now contains the water available for surface flow in each cell
print(f"Runoff Volume in each cell: {runoff_volume}")

# Initialize a flow direction matrix
flow_direction = np.zeros_like(dem, dtype=int)

# Define the directions: N, NE, E, SE, S, SW, W, NW
directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Iterate over each cell in the DEM
for i in range(1, nx-1):
    for j in range(1, ny-1):
        min_elevation = dem[i, j]
        min_dir = 0
        # Check all 8 neighbors
        for k, (di, dj) in enumerate(directions):
            ni, nj = i + di, j + dj
            if dem[ni, nj] < min_elevation:
                min_elevation = dem[ni, nj]
                min_dir = k
        flow_direction[i, j] = min_dir  # Assign the flow direction index

# Define Manning's roughness coefficient for sand
n_manning = 0.025

# Initialize flow accumulation matrix
flow_accumulation = np.zeros_like(dem)

# Flow velocity and routing calculation
for i in range(1, nx-1):
    for j in range(1, ny-1):
        # Current cell runoff volume
        inflow = runoff_volume
        
        # Flow direction and velocity calculation
        direction = flow_direction[i, j]
        di, dj = directions[direction]
        ni, nj = i + di, j + dj
        
        # Calculate slope between current cell and downstream cell
        slope = (dem[i, j] - dem[ni, nj]) / resolution
        
        # Calculate flow velocity using Manning's equation
        # Assuming hydraulic radius R as depth of flow; simplify to depth = runoff_volume / cell area
        depth = runoff_volume / (resolution * resolution)  # Simplified depth estimation
        velocity = (1 / n_manning) * (depth ** (2/3)) * (slope ** 0.5)
        
        # Calculate the water that will flow from current cell to the next based on velocity
        flow = velocity * resolution * resolution  # Flow rate in cubic meters per second
        
        # Update the flow accumulation for the downstream cell
        flow_accumulation[ni, nj] += inflow + flow

# Determine the exit location (e.g., bottom middle of the parking lot)
exit_cells = [(nx-1, ny//2 + i) for i in range(-5, 5)]  # Example: 10 cells centered at the bottom middle

# Calculate the total flow rate at the exit
flow_rate_at_exit = sum(flow_accumulation[ex, ey] for ex, ey in exit_cells)

print(f"Flow rate at the exit (in cubic meters per hour): {flow_rate_at_exit}")

# Create a figure and axis for the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the DEM
X, Y = np.meshgrid(np.linspace(0, width, ny), np.linspace(0, length, nx))

# Initialize the surface plot
surface = ax.plot_surface(X, Y, dem, cmap='terrain', edgecolor='none')

# Initialize an array to represent water depth and velocity directions
water_depth = np.zeros_like(dem)
flow_velocity = np.zeros_like(dem)

# Add a start time variable
start_time = tm.time()
# Function to update the animation for each frame
def update(frame):
    global water_depth, flow_velocity

    # Simulate water flow by updating water depth according to flow direction and rainfall
    new_water_depth = np.zeros_like(water_depth) + runoff_volume  # Adding rainfall to each cell
    flow_velocity = np.zeros((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            direction = flow_direction[i, j]
            di, dj = directions[direction]
            ni, nj = i + di, j + dj
            
            # Move water based on flow direction (flowing to the downstream cell)
            new_water_depth[ni, nj] += water_depth[i, j]
            
            # Calculate slope between current cell and downstream cell
            slope = (dem[i, j] - dem[ni, nj]) / resolution
            
            # Calculate flow velocity using Manning's equation (with simplified hydraulic radius)
            depth = max(water_depth[i, j], 0.01)  # Use a minimum depth to avoid zero division
            velocity = (1 / n_manning) * (depth ** (2/3)) * (slope ** 0.5)
            flow_velocity[i, j] = velocity

    # Update water depth
    water_depth = new_water_depth

    # Clear the plot
    ax.clear()
    
    # Update the plot by plotting water depth on the surface with terrain
    ax.plot_surface(X, Y, dem, cmap='terrain', edgecolor='none', alpha=0.7)
    
    # Plot water depth separately
    water_surface = ax.plot_surface(X, Y, dem + water_depth, cmap='Blues', edgecolor='none', alpha=0.5)
    
    # Add quiver plot for velocity visualization
    i_indices, j_indices = np.unravel_index(np.arange(0, nx*ny, 10), (nx, ny))
    
    # Calculate the angles for the quiver plot
    angles = np.array([np.arctan2(directions[flow_direction[i, j]][1], 
                                  directions[flow_direction[i, j]][0]) 
                       for i, j in zip(i_indices, j_indices)])
    
    ax.quiver(X[i_indices, j_indices], Y[i_indices, j_indices], dem[i_indices, j_indices], 
              flow_velocity[i_indices, j_indices] * np.cos(angles),
              flow_velocity[i_indices, j_indices] * np.sin(angles),
              0, length=0.2, normalize=True, color='r')
    
    ax.set_zlim(np.min(dem), np.max(dem) + np.max(water_depth))
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_zlabel('Elevation + Water Depth (m)')
    ax.set_title('Water Flow Simulation')

# Create an animation
ani = FuncAnimation(fig, update, frames=None, interval=10000, repeat=False, cache_frame_data=False)
# Save animation
# ani.save('simulation.gif')

plt.show()

