import numpy as np

# Define constants
NUM_EPISODES = 1000
GRID_HEIGHT = 4
GRID_WIDTH = 10
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

START_ROW = GRID_HEIGHT - 1
START_COL = 0
GOAL_ROW = GRID_HEIGHT - 1
GOAL_COL = GRID_WIDTH - 1
CLIFF_ROW = GRID_HEIGHT - 1
CLIFF_START_COL = 1
CLIFF_END_COL = GRID_WIDTH - 2

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
NUM_ACTIONS = len(ACTIONS)

# Initialize Q-values
q_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))


def select_action(row, col):
    if np.random.rand() < EPSILON:
        return np.random.randint(NUM_ACTIONS)  # Explore (choose random action)
    else:
        return np.argmax(q_values[row, col])  # Exploit (choose action with highest Q-value)


def get_next_state(row, col, action):
    next_row = max(0, min(row + ACTIONS[action][0], GRID_HEIGHT - 1))
    next_col = max(0, min(col + ACTIONS[action][1], GRID_WIDTH - 1))
    if next_row == CLIFF_ROW and CLIFF_START_COL <= next_col <= CLIFF_END_COL:
        # Move to the start if falling off the cliff
        next_row = START_ROW
        next_col = START_COL
    return next_row, next_col


def get_reward(row, col):
    if row == GOAL_ROW and col == GOAL_COL:
        return 100  # Goal reached
    elif row == CLIFF_ROW and CLIFF_START_COL <= col <= CLIFF_END_COL:
        return -100  # Cliff penalty
    else:
        return -1  # Default step penalty


# Q-learning training
for episode in range(NUM_EPISODES):
    row, col = START_ROW, START_COL

    while not (row == GOAL_ROW and col == GOAL_COL):
        action = select_action(row, col)
        next_row, next_col = get_next_state(row, col, action)
        reward = get_reward(next_row, next_col)
        max_q_value_next_state = np.max(q_values[next_row, next_col])
        q_values[row, col, action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q_value_next_state -
                                                       q_values[row, col, action])
        row, col = next_row, next_col

# Extracted policy
policy = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=str)
for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH):
        best_action = np.argmax(q_values[row, col])
        if best_action == 0:
            policy[row, col] = '^'
        elif best_action == 1:
            policy[row, col] = 'v'
        elif best_action == 2:
            policy[row, col] = '<'
        elif best_action == 3:
            policy[row, col] = '>'
        else:
            policy[row, col] = '?'

# Print policy
print("Extracted Policy:")
for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH):
        print(policy[row, col], end=' ')
    print()
