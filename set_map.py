import numpy as np
import matplotlib.pyplot as plt
from simulator.obstacle import PolygonObstacle

obstacle_data = [
    [(0,10000), (5500,10000), (5300,9000) , (4800,8500), (4200,7300), (4000,5700), (4300, 4900), (4900,4400), (4400,4000), (3200,4100), (2000,4500), (1000,4000), (900,3500), (500,2600), (0,2350)],   # Island 1 
    [(10000,0), (4000,0), (4250, 250), (5000,400), (6000, 900), (8000,1100), (8500,1500), (9000,2250), (9500, 3500), (10000,4000)], # Island 2
    [(5500,5500), (5700,7000), (6200, 8100), (7500, 8000), (7800,7000), (7600, 5500), (6900,4700),(6000,5000)], # Island 3
    [(2000,2000), (2500,2300), (4000,2500), (5000,3000), (4200,2100), (3400,1900)] # Island 4
    ]

obstacle = PolygonObstacle(obstacle_data)

fig_1, axes = plt.subplots(figsize=(10, 10))
# axes = axes.flatten()  # Flatten the 2D array for easier indexing
axes.set_xlim(0, 10000)
axes.set_ylim(0, 10000)
obstacle.plot_obstacle(axes)

plt.show()

# (3000,3500), (2000,3200)