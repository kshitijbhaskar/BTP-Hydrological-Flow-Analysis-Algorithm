import threading
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parking lot dimensions and DEM resolution
length = 100.0  # in meters
width = 40.0    # in meters
resolution = 0.5  # DEM resolution in meters

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

# Initialize flow direction and accumulation matrices
flow_direction = np.zeros_like(dem, dtype=int)
flow_accumulation = np.zeros_like(dem)

# Define directions: N, NE, E, SE, S, SW, W, NW
directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Define Manning's roughness coefficient for sand
n_manning = 0.025

# Time step for simulation (e.g., 1 second)
time_step = 1  # seconds

# Convert the hourly runoff volume to per-second volume
runoff_volume_per_step = runoff_volume / 3600  # As runoff_volume was calculated per hour

# Iterate over each cell in the DEM
for i in range(1, nx-1):
    for j in range(1, ny-1):
        # Determine flow direction based on minimum elevation
        min_elevation = dem[i, j]
        min_dir = 0
        for k, (di, dj) in enumerate(directions):
            ni, nj = i + di, j + dj
            if dem[ni, nj] < min_elevation:
                min_elevation = dem[ni, nj]
                min_dir = k
        flow_direction[i, j] = min_dir

# Flow velocity and routing calculation with consideration of slopes
for i in range(1, nx-1):
    for j in range(1, ny-1):
        # Outflow and inflow initialization
        outflow = 0
        inflow = runoff_volume_per_step

        # Calculate slope and velocity for each neighboring cell
        for k, (di, dj) in enumerate(directions):
            ni, nj = i + di, j + dj
            elevation_change = dem[i, j] - dem[ni, nj]
            
            if elevation_change > 0:  # Outflow only for downhill neighbors
                # Calculate slope and velocity
                slope = elevation_change / resolution
                depth = runoff_volume_per_step / (resolution * resolution)  # Simplified depth estimation
                velocity = (1 / n_manning) * (depth ** (2/3)) * (slope ** 0.5)

                # Flow contribution to the neighbor
                flow_contribution = velocity * resolution * resolution * time_step
                flow_accumulation[ni, nj] += flow_contribution
                outflow += flow_contribution

        # Update inflow for the current cell
        flow_accumulation[i, j] += inflow - outflow

# Determine the exit location (e.g., bottom middle of the parking lot)
exit_cells = [(nx-1, ny//2 + i) for i in range(-5, 5)]  # Example: 10 cells centered at the bottom middle

# Calculate the total flow rate at the exit
flow_rate_at_exit = sum(flow_accumulation[ex, ey] for ex, ey in exit_cells)

print(f"Flow rate at the exit (in cubic meters per hour): {flow_rate_at_exit}")

# Initialize a figure and axis for the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the DEM
X, Y = np.meshgrid(np.linspace(0, width, ny), np.linspace(0, length, nx))

# Initialize the surface plot
surface = ax.plot_surface(X, Y, dem, cmap='terrain', edgecolor='none')

# Initialize flow accumulation with the DEM values
flow_accumulation = np.copy(dem)

# Define Manning's roughness coefficient for sand
n_manning = 0.025

# Function to update the animation for each frame
def update(frame):
    global flow_accumulation

    # Copy the flow accumulation to avoid modifying it during iteration
    new_flow_accumulation = np.copy(flow_accumulation)

    # Flow velocity and routing calculation with consideration of slopes
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Outflow and inflow initialization
            outflow = 0
            inflow = runoff_volume_per_step

            # Calculate slope and velocity for each neighboring cell
            for k, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj

                # Check if the neighbor is within the grid bounds
                if 0 <= ni < nx and 0 <= nj < ny:
                    elevation_change = dem[i, j] - dem[ni, nj]

                    if elevation_change > 0:  # Outflow only for downhill neighbors
                        # Calculate slope and velocity
                        slope = elevation_change / resolution
                        depth = runoff_volume_per_step / (resolution * resolution)  # Simplified depth estimation
                        velocity = (1 / n_manning) * (depth ** (2/3)) * (slope ** 0.5)

                        # Flow contribution to the neighbor
                        flow_contribution = velocity * resolution * resolution * time_step
                        new_flow_accumulation[ni, nj] += flow_contribution
                        outflow += flow_contribution

            # Update inflow for the current cell
            new_flow_accumulation[i, j] += inflow - outflow

    # Update flow accumulation
    flow_accumulation = new_flow_accumulation

    # Clear the plot
    ax.clear()
    
    # Update the plot by plotting flow accumulation on the surface with terrain
    ax.plot_surface(X, Y, dem, cmap='terrain', edgecolor='none', alpha=0.7)
    
    # Plot flow accumulation as a color map
    flow_surface = ax.plot_surface(X, Y, flow_accumulation, cmap='Blues', edgecolor='none', alpha=0.5)
    
    ax.set_zlim(np.min(dem), np.max(flow_accumulation))
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_zlabel('Flow Accumulation (cubic meters)')
    ax.set_title('Flow Accumulation Simulation')

# Create an animation
ani = FuncAnimation(fig, update, frames="None", interval=100, repeat=False, cache_frame_data=False)
# Save animation
# ani.save('flow_accumulation_simulation_multidirection.gif')

plt.show()
