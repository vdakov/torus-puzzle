import random
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib as mpl 

# Generates a toroidal grid world with a random start and goal position.
def generate_grid(width, height):
    grid = [[0 for _ in range(height)] for _ in range(width)]
    start =(random.randint(0, width - 1), random.randint(0, height - 1))
    goal = (random.randint(0, width - 1), random.randint(0, height - 1))
    return grid, start, goal

# Computes the modular Euclidean distance on a torus grid between a state and the goal.
def policy_euclidean(grid, current, goal):
    left = ((current[0] - 1) % grid.shape[0], current[1])
    right = ((current[0] + 1) % grid.shape[0], current[1])
    up = (current[0], (current[1] - 1) % grid.shape[1])
    down = (current[0], (current[1] + 1) % grid.shape[1])

    moves = [left, right, up, down]
    heuristics = [heuristic_modular_euclidean_distance(grid, left, goal), heuristic_modular_euclidean_distance(grid, right, goal), 
    heuristic_modular_euclidean_distance(grid, up, goal), heuristic_modular_euclidean_distance(grid, down, goal)]

    # Select the move with the smallest heuristic value
    nx, ny =  moves[np.argmin(heuristics)]
    return (nx, ny)

def heuristic_modular_euclidean_distance(maze, state, goal):
    state_x, state_y = state
    goal_x, goal_y = goal
    width, height = maze.shape

    dx = min(abs(goal_x - state_x), width - abs(goal_x - state_x))
    dy = min(abs(goal_y - state_y), height - abs(goal_y - state_y))

    return np.sqrt(dx**2 + dy**2)

counter = 0
step_size = 1
direction = 0
def policy_spiral(grid, current, goal):
    global counter, step_size, direction

    ny, nx = current
    height, width = grid.shape

    if direction == 0:  # Move right
        nx = (nx + 1) % width
    elif direction == 1:  # Move down
        ny = (ny - 1) % height
    elif direction == 2:  # Move left
        nx = (nx - 1) % width
    elif direction == 3:  # Move up
        ny = (ny + 1) % height

    counter += 1

    if counter == step_size:
        counter = 0
        direction = (direction + 1) % 4
        if direction % 2 == 0:  # After right and down, or left and up, increase step size
            step_size += 1
    return ny, nx

def policy_diagonal(grid, current, goal):
    global counter
    height, width = grid.shape

    ny, nx = current
    # Step sizes determined by irrational numbers
    delta_x = (np.sqrt(5) - 1) / 2  # Golden ratio minus 1
    delta_y = 1  # Fixed vertical step size

    if counter == 0:  
        nx = (nx + delta_x) % width
    elif counter == 1:  
        ny = (ny + delta_y) % height

    counter = (counter + 1) % 2
    return ny, nx

def visualize(grid, path=None, start=None, goal=None):
    grid_np = np.array(grid)
    fig = plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('Greys')

    plt.imshow(grid_np, cmap=cmap, interpolation='nearest')
    plt.xticks(np.arange(0, len(grid_np), 1)), plt.yticks(np.arange(0, len(grid_np[0, :]), 1))

    if start:
        plt.text(start[1], start[0], "S", color="green", ha="center", va="center", fontsize=14, fontweight="bold")
    if goal:
        plt.text(goal[1], goal[0], "G", color="red", ha="center", va="center", fontsize=14, fontweight="bold")
    plt.grid()
    if path:
        for  i, step in enumerate(path):
            j = 1.0 * i / len(path)
            plt.scatter(step[1], step[0], color = 'blue', s=100, alpha=0.6)
            plt.text(step[1], step[0], str(i + 1), color="black", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.show() 

# Runs the simulation for a given policy and records the path taken.
def do_run(grid, start, goal, policy):
    global direction, counter, step_size
    visited = set()
    path = []
   
    grid = np.array(grid)
    height, width = grid.shape
    current = (0, 0)
    S = width * height
    ny, nx = 0, 0
    path.append((int(ny), int(nx)))

    while len(visited) != S:
        if policy is policy_euclidean and current == goal:
            break
        
        path.append((int(ny), int(nx)))
        visited.add((int(ny), int(nx)))

        ny, nx = policy(grid, current, goal)
        current = (ny, nx)
        
    direction = 0
    counter = 0
    step_size = 1

    return path 

if __name__ == "__main__":

    num_runs = 1
    results = []
    violations = []    
    grids = [
        (3, 3), (4, 5), (9, 9), (100, 100), (1000, 1000), (1, 10), (1, 1000), 
        (1000, 1), (1, 1000000), (500, 500), (250, 400), (50, 50), (25, 40),
        (2, 5000), (5000, 2), (10, 1000), (1000, 10), (333, 333), (100, 9999), 
        (9999, 100), (123, 456), (77, 1300), (300, 3333), (100, 9999)
    ]
    
    for (width, height) in grids:
        for i in range(num_runs):
            grid, start, goal = generate_grid(width, height)
            S = width * height

            # visualize(grid, start=start, goal=goal)

            # path = do_run(grid, start, goal, policy_euclidean)
            path = do_run(grid, start, goal, policy_diagonal)
            # path = do_run(grid, start, goal, policy_spiral)

            results.append(len(path))
            print((width, height), '|Path length:', len(path), '|S:', S, '|Proportion path-S:', len(path)/ S, 'ln (n)', np.log(S))
            
            # visualize(grid, path=path, start=start, goal=goal)
            violations.append(len(path) > (35 * S))


    violations = np.array(violations)
    print(violations)

