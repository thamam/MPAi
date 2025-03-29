NUM_SAMPLES = 5000
BATCH_SIZE = 1000

for batch_start in range(0, NUM_SAMPLES, BATCH_SIZE):
    for i in range(BATCH_SIZE):
        index = batch_start + i
        grid, start, goal = generate_valid_grid(obstacle_density=0.2)
        path = a_star_find_path(grid, start, goal)
        save_sample(grid, path, index)
    print(f"Batch {batch_start // BATCH_SIZE + 1} saved.")
