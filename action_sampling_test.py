import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 10  # Number of divisions
sigma_x = 1.0  # Standard deviation for x-direction noise
sigma_y = 1.0  # Standard deviation for y-direction noise

# Points A and B
x_A, y_A = 0, 0
x_B, y_B = 10, 10

# Compute step sizes
dist_x_AB = x_B - x_A
dist_y_AB = y_B - y_A
unit_step_x = dist_x_AB / n
unit_step_y = dist_y_AB / n

# Initialize route
route = [(x_A, y_A)]  # Start with point A

# Generate route points with Gaussian noise
for i in range(1, n):
    # Ideal next point
    x_i = x_A + i * unit_step_x
    y_i = y_A + i * unit_step_y
    
    # Add Gaussian noise
    noise_x = np.random.normal(0, sigma_x)
    noise_y = np.random.normal(0, sigma_y)
    x_sampled = x_i + noise_x
    y_sampled = y_i + noise_y
    
    # Append to route
    route.append((x_sampled, y_sampled))

# Add final point B
route.append((x_B, y_B))

# Display the route points
for idx, point in enumerate(route):
    print(f"Point {idx}: {point}")
    
plt.plot(route)
plt.show()
    
