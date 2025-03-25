# Re-run the necessary code since execution state was reset

import numpy as np
import matplotlib.pyplot as plt
import heapq

# Grid size
GRID_SIZE = 10

# A* Pathfinding Algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    """Reconstructs the path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star_find_path(grid, start, goal):
    """Runs A* algorithm and returns the optimal path as a list of waypoints."""
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = []
    heapq.heappush(open_set, (fscore[start], start))
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)  # Return the path
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if grid[neighbor] == 1 or neighbor in close_set:
                    continue  # Obstacle or already visited

                tentative_g_score = gscore[current] + 1
                if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
    
    return []  # No valid path found

# Generate a valid grid
def generate_valid_grid(obstacle_density=0.2):
    """Generates a 10x10 occupancy grid ensuring a valid path exists."""
    while True:
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        start = (0, 0)
        goal = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        
        # Ensure goal is not the start position
        while goal == start:
            goal = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        
        # Randomly place obstacles
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) != start and (i, j) != goal and np.random.rand() < obstacle_density:
                    grid[i, j] = 1  # Mark as obstacle
        
        # Check if valid path exists using A*
        if a_star_find_path(grid, start, goal):
            return grid, start, goal

# Visualize Grid with Path
def plot_grid_with_path(grid, start, goal, path):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap="gray_r")
    plt.scatter(start[1], start[0], color='green', s=100, label="Start")
    plt.scatter(goal[1], goal[0], color='red', s=100, label="Goal")
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='blue', linewidth=2, marker='o', label="A* Path")
    
    plt.legend()
    plt.grid(True)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.title("A* Path on Occupancy Grid")
    plt.show()

# Generate grid, compute path, and visualize
grid, start, goal = generate_valid_grid(obstacle_density=0.2)
path = a_star_find_path(grid, start, goal)
plot_grid_with_path(grid, start, goal, path)
