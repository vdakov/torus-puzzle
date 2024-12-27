import random
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib as mpl 

def generate_grid(width, height):
    grid = [[0 for _ in range(height)] for _ in range(width)]
    start =(random.randint(0, width - 1), random.randint(0, height - 1))
    goal = (random.randint(0, width - 1), random.randint(0, height - 1))
    return grid, start, goal


def policy_euclidean(grid, current, goal):
    left = ((current[0] - 1) % grid.shape[0], current[1])
    right = ((current[0] + 1) % grid.shape[0], current[1])
    up = (current[0], (current[1] - 1) % grid.shape[1])
    down = (current[0], (current[1] + 1) % grid.shape[1])

    moves = [left, right, up, down]
    heuristics = [policy(grid, left, goal), policy(grid, right, goal), policy(grid, up, goal), policy(grid, down, goal)]

    nx, ny =  moves[np.argmin(heuristics)]

    return (nx, ny)

def heuristic_modular_euclidean_distance(maze, state, goal):
    state_x, state_y = state
    goal_x, goal_y = goal
    width, height = maze.shape

    dx = min(abs(goal_x - state_x), width - abs(goal_x - state_x))
    dy = min(abs(goal_y - state_y), height - abs(goal_y - state_y))

    # Euclidean distance
    return np.sqrt(dx**2 + dy**2)

counter = 0
step_size = 1
direction = 0
def policy_spiral(grid, current, goal):
    global counter, step_size, direction

    nx, ny = current

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
    return nx, ny

def policy_diagonal(grid, current, goal):
    global counter

    nx, ny = current
    delta_x = (np.sqrt(5) - 1) / 2
    delta_y = 1

    if counter == 0:  
        nx = (nx + delta_x) % width
    elif counter == 1:  
        ny = (ny + delta_y) % height

    counter = (counter + 1) % 2

    return nx, ny



def visualize(grid, path=None, start=None, goal=None):
    grid_np = np.array(grid)
    fig = plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('Greys')

    # Draw grid
    plt.imshow(grid_np, cmap=cmap, interpolation='nearest')
    plt.xticks(np.arange(0, len(grid_np), 1)), plt.yticks(np.arange(0, len(grid_np[0, :]), 1))

    # Mark start and goal
    if start:
        plt.text(start[1], start[0], "S", color="green", ha="center", va="center", fontsize=14, fontweight="bold")
    if goal:
        plt.text(goal[1], goal[0], "G", color="red", ha="center", va="center", fontsize=14, fontweight="bold")
    plt.grid()
    # Draw path
    if path:
        for  i, step in enumerate(path):
            j = 1.0 * i / len(path)
            plt.scatter(step[1], step[0], color= (j, j, j), s=100, alpha=0.6)
            plt.text(step[1], step[0], str(i + 1), color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.show() 

def do_run(grid, start, goal, policy):
    visited = set()
    path = []
    current = (0, 0)
    grid = np.array(grid)
    width, height = grid.shape
    S = width * height
    nx, ny = 0, 0
    path.append((int(nx), int(ny)))

    while len(visited) != S:
  
        path.append((int(nx), int(ny)))
        visited.add((int(nx), int(ny)))

        nx, ny = policy(grid, current, goal)
        current = (nx, ny)



    return path 




def generate_random_AB(limit, shuffle=True):
    sqrt_limit = int(limit**0.5)  # Square root of the limit
    # Generate A close to sqrt_limit
    A = random.randint(max(1, sqrt_limit - 100), min(limit, sqrt_limit + 100))
    max_B = limit // A  # Maximum B such that A * B <= limit
    # Generate B close to sqrt_limit
    B = random.randint(max(1, sqrt_limit - 100), min(max_B, sqrt_limit + 100))

    if shuffle and random.choice([True, False]):  # Optionally shuffle A and B
        A, B = B, A

    return A, B

if __name__ == "__main__":

    num_runs = 1
    results = []
    violations = []    
    grids = [
 

            # Plot the path on the unit square    (10, 10),       # 100
    (4, 5),         # 20
    (20, 20),       # 400
    (100, 100),     # 10,000
    (1000, 1000),   # 1,000,000
    (1, 1000),      # 1,000
    (1000, 1),      # 1,000
    (1, 1000000),   # 1,000,000 
    (500, 500),     # 250,000
    (250, 400),     # 100,000
    (50, 50),       # 2,500
    (25, 40),       # 1,000
    (2, 5000),      # 10,000
    (5000, 2),      # 10,000
    (10, 1000),     # 10,000
    (1000, 10),     # 10,000
    (333, 333),     # 110,889
    (100, 9999),    # 999,900
    (9999, 100),    # 999,900
    (123, 456),     # 56,088
    (77, 1300),     # 100,100
    (300, 3333),    # 999,900
    (100, 9999),
    (983, 1091),
    (10000, 10000)
]


    for (width, height) in grids:
        for i in range(num_runs):
            # width, height = generate_random_AB(1000000, shuffle=True)

            grid, start, goal = generate_grid(width, height)
            S = width * height

            # visualize(grid, start=start, goal=goal)

            # path = do_run(grid, start, goal, heuristic_modular_euclidean_distance)
            path = do_run(grid, start, goal, policy_diagonal)
            # path = do_run(grid, start, goal, policy_spiral)
            results.append(len(path))


            print((width, height), '|Path length:', len(path), '|S:', S, '|Proportion path-S:', len(path)/ S, 'ln (n)', np.log(S))
            
            # visualize(grid, path=path, start=start, goal=goal)
            violations.append(len(path) > (35 * S))


    violations = np.array(violations)

    # Calculate cumulative sum of violations to show trends
    cumulative_violations = np.cumsum(violations)

    # Create a more descriptive plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Bar plot of violations
    plt.subplot(2, 1, 1)
    plt.bar(range(len(violations)), violations, color="red", alpha=0.6, label="Violation")
    plt.xlabel("Run Number")
    plt.ylabel("Violation (1 = Yes, 0 = No)")
    plt.title("Violations Over Runs")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Subplot 2: Cumulative violations over time
    plt.subplot(2, 1, 2)
    plt.plot(range(len(violations)), cumulative_violations, color="blue", label="Cumulative Violations")
    plt.fill_between(range(len(violations)), cumulative_violations, color="blue", alpha=0.3)
    plt.xlabel("Run Number")
    plt.ylabel("Cumulative Violations")
    plt.title("Cumulative Violations Over Time")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()