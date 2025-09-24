"""
robot_cleaner.py
Implements a GridWorld Markov Decision Process (MDP) for a robot cleaner.
The robot navigates a grid, avoiding walls and cleaning dirty cells using value iteration.
Classes:
    GridWorldMDP_RobotCleaner: Defines the MDP, state transitions, rewards, and value iteration.
"""
import numpy as np

class GridWorldMDP_RobotCleaner:
    NON_ENTERABLE_COORDS = ['b7', 'f7', 'a3']
    VALUABLE_COORD = 'b7'
    DOOR_COORD = 'b5'
    SWITCH_COORD = 'a3'
    DIRTY_CELL_COORD = 'f7'  # Dirty cell is always at f7
    actions = ['up', 'down', 'left', 'right', 'clean', 'toggle_switch', 'pick_up']

    def __init__(self, grid, start, mdp_type, gamma=0.9):
        """
        Initialize the GridWorldMDP_RobotCleaner.
        Args:
            grid (list of list of int): 2D grid, 0 for empty cell, 1 for wall.
            start (str): Starting position in chess notation (e.g., 'b1').
            dirty_cells (list of str): List of positions of dirty cells in chess notation.
            gamma (float): Discount factor for future rewards.
        """
        self.grid = grid  # 2D list: 0-empty, 1-wall
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.dirty_cell_exists = True
        self.mdp_type = mdp_type
        self.gamma = gamma
        # Generate all possible states using chess notation, door state, valuable item state, and dirty cell existence
        self.states = []
        for r in range(self.rows):
            for c in range(self.cols):
                coord = self.idx_to_chess(r, c, self.rows)
                if self.grid[r][c] == 0 and coord not in self.NON_ENTERABLE_COORDS:
                    for door_open in [True, False]:
                        for valuable_picked in [True, False]:
                            for dirty_cell_exists in [True, False]:
                                self.states.append((coord, door_open, valuable_picked, dirty_cell_exists))

    def can_pick_valuable(self, coord, valuable_picked):
        # Can pick up valuable if on b6 and not already picked
        return coord == 'b6' and not valuable_picked

    def can_clean_dirt(self, coord, dirty_cell_exists):
        # Can clean dirt in f7 if on f6
        return coord == 'f6' and dirty_cell_exists
    
    def can_close_door(self, coord, door_open):
        # Can close the door only from b3 if it's currently open
        return coord == 'b3' and door_open
    
    @staticmethod
    def idx_to_chess(row, col, rows):
        """
        Convert grid indices to chess notation.
        Args:
            row (int): Row index (0 at top)
            col (int): Column index (0 at left)
            rows (int): Total number of rows
        Returns:
            str: Chess coordinate (e.g., 'a1')
        """
        col_letter = chr(ord('a') + col)
        row_number = str(rows - row)
        return f"{col_letter}{row_number}"

    @staticmethod
    def chess_to_idx(chess_coord, rows):
        """
        Convert chess notation to grid indices.
        Args:
            chess_coord (str): Chess coordinate (e.g., 'a1')
            rows (int): Total number of rows
        Returns:
            tuple: (row, col)
        """
        col = ord(chess_coord[0]) - ord('a')
        row = rows - int(chess_coord[1:])
        return (row, col)

    def is_terminal(self, state):
        """
        Check if the state is terminal based on mdp_type.
        Args:
            state (tuple): (coord, door_open, valuable_picked, dirty_cell_exists)
        Returns:
            bool: True if terminal, else False.
        """
        _, door_open, valuable_picked, dirty_cell_exists = state
        if self.mdp_type in ['original', 'legibility']:
            return not dirty_cell_exists
        elif self.mdp_type == 'destroy':
            return valuable_picked
        elif self.mdp_type == 'close':
            return not door_open
        return False

    def get_next_state(self, state, action):
        """
        Compute the next state given current state and action.
        Args:
            state (tuple): Current state (coord, door_open, valuable_picked, dirty_cell_exists)
            action (str): Action to take ('up', 'down', 'left', 'right', 'clean', 'toggle_switch', 'pick_up')
        Returns:
            tuple: Next state (coord, door_open, valuable_picked, dirty_cell_exists)
        """
        coord, door_open, valuable_picked, dirty_cell_exists = state
        r, c = self.chess_to_idx(coord, self.rows)
        next_door_open = door_open
        next_valuable_picked = valuable_picked
        next_dirty_cell_exists = dirty_cell_exists
        # Clean dirt in f7 from f6
        if action == 'clean' and self.can_clean_dirt(coord, dirty_cell_exists):
            next_dirty_cell_exists = False
        # Pick up valuable item in b7 from b6
        elif action == 'pick_up' and self.can_pick_valuable(coord, valuable_picked) and not valuable_picked:
            next_valuable_picked = True
        # Toggle switch: can only close the door from b3 if open
        elif action == 'toggle_switch' and self.can_close_door(coord, door_open):
            next_door_open = False
        elif action == 'up' and r > 0 and self.grid[r-1][c] == 0:
            next_r, next_c = r-1, c
            next_coord = self.idx_to_chess(next_r, next_c, self.rows)
            if next_coord in self.NON_ENTERABLE_COORDS:
                next_coord = coord
            elif next_coord == self.DOOR_COORD and not door_open:
                next_coord = coord
            else:
                r = next_r
        elif action == 'down' and r < self.rows-1 and self.grid[r+1][c] == 0:
            next_r, next_c = r+1, c
            next_coord = self.idx_to_chess(next_r, next_c, self.rows)
            if next_coord in self.NON_ENTERABLE_COORDS:
                next_coord = coord
            elif next_coord == self.DOOR_COORD and not door_open:
                next_coord = coord
            else:
                r = next_r
        elif action == 'left' and c > 0 and self.grid[r][c-1] == 0:
            next_r, next_c = r, c-1
            next_coord = self.idx_to_chess(next_r, next_c, self.rows)
            if next_coord in self.NON_ENTERABLE_COORDS:
                next_coord = coord
            elif next_coord == self.DOOR_COORD and not door_open:
                next_coord = coord
            else:
                c = next_c
        elif action == 'right' and c < self.cols-1 and self.grid[r][c+1] == 0:
            next_r, next_c = r, c+1
            next_coord = self.idx_to_chess(next_r, next_c, self.rows)
            if next_coord in self.NON_ENTERABLE_COORDS:
                next_coord = coord
            elif next_coord == self.DOOR_COORD and not door_open:
                next_coord = coord
            else:
                c = next_c
        next_coord = self.idx_to_chess(r, c, self.rows)
        return (next_coord, next_door_open, next_valuable_picked, next_dirty_cell_exists)

    def get_reward(self, state, action, next_state, Q_destroy=None):
        """
        Get the reward for taking an action in a state.
        Args:
            state (tuple): Current state.
            action (str): Action taken.
            next_state (tuple): Resulting state.
        Returns:
            int: Reward value.
        """
        # Reward logic based on mdp_type
        if self.mdp_type == 'original':
            if action == 'clean' and self.can_clean_dirt(state[0], state[3]):
                return 10  # reward for cleaning dirt
            if action == 'pick_up' and self.can_pick_valuable(state[0], state[2]) and not state[2]:
                return -20  # penalty for picking up valuable item
            if action == 'toggle_switch' and self.can_close_door(state[0], state[1]):
                return -1  # small penalty for toggling
            return -1  # penalty for movement
        elif self.mdp_type == 'destroy':
            if action == 'pick_up' and self.can_pick_valuable(state[0], state[2]) and not state[2]:
                return 20  # reward for picking up valuable item
            if action == 'clean' and self.can_clean_dirt(state[0], state[3]):
                return -10  # penalty for cleaning dirt
            if action == 'toggle_switch' and self.can_close_door(state[0], state[1]):
                return -1
            return -1
        elif self.mdp_type == 'close':
            if action == 'toggle_switch' and self.can_close_door(state[0], state[1]):
                return 20  # reward for toggling switch (closing door)
            if action == 'clean' and self.can_clean_dirt(state[0], state[3]):
                return -10
            if action == 'pick_up' and self.can_pick_valuable(state[0], state[2]) and not state[2]:
                return -10
            return -1
        elif self.mdp_type == 'legibility':
            # 1st Formula
            # Legibility: Reward_legibility = -Q_destroy + Reward for mdp_type original
            # Q_destroy must be provided as a dict
            if Q_destroy is None:
                raise ValueError("Q_destroy must be provided for legibility reward calculation.")
            # Calculate original reward
            original_reward = GridWorldMDP_RobotCleaner(grid=self.grid, start=self.start, mdp_type='original', gamma=self.gamma).get_reward((state[0], state[1], state[2], state[3]), action, next_state)
            # Get Q_destroy value for this state-action
            q_destroy_val = Q_destroy.get((state, action), 0)
            return -q_destroy_val + original_reward
        else:
            return -1

    def value_iteration(self, theta=1e-4, Q_destroy=None):
        """
        Perform value iteration to compute the optimal value function and policy.
        Args:
            theta (float): Convergence threshold.
        Returns:
            tuple: (Value function dict, Policy dict)
        """
        V = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    coord = self.idx_to_chess(r, c, self.rows)
                    for door_open in [True, False]:
                        for valuable_picked in [True, False]:
                            for dirty_cell_exists in [True, False]:
                                V[(coord, door_open, valuable_picked, dirty_cell_exists)] = 0
        i = 0
        while True:
            i += 1
            # print(f"Iteration {i}")
            delta = 0
            for state in V:
                if self.is_terminal(state):
                    continue
                v = V[state]
                values = []
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    if self.mdp_type == 'legibility':
                        reward = self.get_reward(state, action, next_state, Q_destroy=Q_destroy)
                    else:
                        reward = self.get_reward(state, action, next_state)
                    values.append(reward + self.gamma * V.get(next_state, 0))
                V[state] = max(values)
                # print("State:", state, "Value:", V[state])
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        # Extract policy
        policy = {}
        for state in V:
            if self.is_terminal(state):
                policy[state] = None
                continue
            values = []
            for action in self.actions:
                next_state = self.get_next_state(state, action)
                if self.mdp_type == 'legibility':
                    reward = self.get_reward(state, action, next_state, Q_destroy=Q_destroy)
                else:
                    reward = self.get_reward(state, action, next_state)
                values.append((reward + self.gamma * V.get(next_state, 0), action))
            policy[state] = max(values)[1]
        # Calculate Q-values for each state-action pair
        Q = {}
        for state in V:
            if self.is_terminal(state):
                continue
            for action in self.actions:
                next_state = self.get_next_state(state, action)
                if self.mdp_type == 'legibility':
                    reward = self.get_reward(state, action, next_state, Q_destroy=Q_destroy)
                else:
                    reward = self.get_reward(state, action, next_state)
                Q[(state, action)] = reward + self.gamma * V.get(next_state, 0)
        return V, policy, Q
    

# Visualization function for chess grid mapping
def print_chess_grid(grid):
    rows = len(grid)
    cols = len(grid[0])
    print("Robot Cleaner Grid Mapping:")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:W]")  # Wall
            elif coord == GridWorldMDP_RobotCleaner.DOOR_COORD:
                row_str.append(f"[{coord}:D]")  # Door
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:S]")  # Switch
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:V]")  # Valuable item
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:X]")  # Special cell, not wall, not enterable
            else:
                row_str.append(f"[{coord}  ]")  # Empty cell
        print(' '.join(row_str))

# Visualization function for value function mapping
def print_value_grid(grid, V, door_open, valuable_picked=False, dirty_cell_exists=True):
    rows = len(grid)
    cols = len(grid[0])
    print(f"Value Function Grid (door_open={door_open}, valuable_picked={valuable_picked}, dirty_cell_exists={dirty_cell_exists}):")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:    W]")
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:    S]")
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:    V]")
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:    X]")
            else:
                value = V.get((coord, door_open, valuable_picked, dirty_cell_exists), None)
                if value is not None:
                    row_str.append(f"[{coord}:{value:5.1f}]")
                else:
                    row_str.append(f"[{coord}:·]")
        print(' '.join(row_str))
    print()

# Visualization function for policy mapping
def print_policy_grid(grid, policy, door_open, valuable_picked=False, dirty_cell_exists=True):
    rows = len(grid)
    cols = len(grid[0])
    action_symbols = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→',
        'clean': 'C',
        'toggle_switch': 'T',
        'pick_up': 'P',
        None: '·'
    }
    print(f"Policy Grid Mapping (door_open={door_open}, valuable_picked={valuable_picked}, dirty_cell_exists={dirty_cell_exists}):")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:W]")
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:S]")
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:V]")
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:X]")
            else:
                action = policy.get((coord, door_open, valuable_picked, dirty_cell_exists), None)
                symbol = action_symbols.get(action, '?')
                row_str.append(f"[{coord}:{symbol}]")
        print(' '.join(row_str))
    print()

# Visualization function for Q-value
def print_qvalue_grid(grid, Q, door_open, valuable_picked=False, dirty_cell_exists=True):
    rows = len(grid)
    cols = len(grid[0])
    action_labels = ['Up', 'Down', 'Left', 'Right', 'Clean', 'Toggle', 'Pick']
    print(f"Q-Value Grid (door_open={door_open}, valuable_picked={valuable_picked}, dirty_cell_exists={dirty_cell_exists}):")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:    W]")
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:    S]")
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:    V]")
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:    X]")
            else:
                state = (coord, door_open, valuable_picked, dirty_cell_exists)
                q_vals = [Q.get((state, a), None) for a in GridWorldMDP_RobotCleaner.actions]
                q_strs = []
                for label, q in zip(action_labels, q_vals):
                    if q is not None:
                        q_strs.append(f"{label}:{q:5.1f}")
                    else:
                        q_strs.append(f"{label}: ·  ")
                row_str.append(f"[{coord}:" + ", ".join(q_strs) + "]")
        print('\n'.join(row_str))
    print()

# Trajectory generation function
def generate_trajectory(mdp, policy, start_state, max_steps=50, Q_destroy=None):
    """
    Generate a trajectory following the optimal policy from a start state.
    Args:
        mdp: The MDP instance
        policy: The optimal policy dictionary
        start_state: Initial state (coord, door_open, valuable_picked, dirty_cell_exists)
        max_steps: Maximum number of steps to prevent infinite loops
    Returns:
        trajectory: List of (state, action, reward, next_state) tuples
    """
    trajectory = []
    current_state = start_state    
    for step in range(max_steps):
        # Check if current state is terminal
        if mdp.is_terminal(current_state):
            trajectory.append((current_state, None, 0, current_state))
            break    
        # Get optimal action from policy
        action = policy.get(current_state, None)
        if action is None:
            break        
        # Get next state and reward
        next_state = mdp.get_next_state(current_state, action)
        if mdp.mdp_type == 'legibility':
            reward = mdp.get_reward(current_state, action, next_state, Q_destroy=Q_destroy)
        else:
            reward = mdp.get_reward(current_state, action, next_state)
        trajectory.append((current_state, action, reward, next_state))
        current_state = next_state        
        # Break if state doesn't change (stuck)
        if len(trajectory) > 1 and trajectory[-1][0] == trajectory[-1][3]:
            break    
    return trajectory

def print_trajectory(trajectory, mdp_name):
    """
    Print the trajectory in a readable format.
    Args:
        trajectory: List of (state, action, reward, next_state) tuples
        mdp_name: Name of the MDP for display
    """
    print(f"\n=== TRAJECTORY for {mdp_name} ===")
    total_reward = 0    
    for i, (state, action, reward, next_state) in enumerate(trajectory):
        coord, door_open, valuable_picked, dirty_cell_exists = state
        if action is None:  # Terminal state
            print(f"Step {i}: {coord} (door_open={door_open}, valuable_picked={valuable_picked}, dirty_cell_exists={dirty_cell_exists}) - TERMINAL")
        else:
            total_reward += reward
            print(f"Step {i}: {coord} (door_open={door_open}, valuable_picked={valuable_picked}, dirty_cell_exists={dirty_cell_exists}) -> Action: {action} -> Reward: {reward}")    
    print(f"Total steps: {len(trajectory)}")
    print(f"Total reward: {total_reward}")
    print("=" * 50)

def calculate_legibility_mdp(Q_original, Q_destroy, grid, start, weight):
    """
    Calculate legibility MDP using the formula: Q_legibility = weight * Q_destroy + Q_original
    Args:
        Q_original: Q-values from original MDP
        Q_destroy: Q-values from destroy MDP
        grid: Grid layout
        start: Starting position
        weight: Weight multiplier for Q_destroy
    Returns:
        tuple: (V_legibility, policy_legibility, Q_legibility)
    """
    print("Using Q-values from mdp_destroy and mdp_original to compute legibility rewards.")
    print(f"The formula is: Q_legibility = {weight} * Q_destroy + Q_original")
    # Calculate Q_legibility using the formula: Q_legibility = weight * Q_destroy + Q_original
    Q_legibility = {}
    for state_action in Q_original:
        if state_action in Q_destroy:
            Q_legibility[state_action] = weight * Q_destroy[state_action] + Q_original[state_action]
            # print(f"State-Action: {state_action}, Q_destroy: {Q_destroy[state_action]:.2f}, Q_original: {Q_original[state_action]:.2f}, Q_legibility: {Q_legibility[state_action]:.2f}")
        else:
            Q_legibility[state_action] = Q_original[state_action] 
    # Extract policy from Q_legibility
    policy_legibility = {}
    V_legibility = {}    
    # Get all states from Q_original
    all_states = set()
    for (state, action), q_val in Q_legibility.items():
        all_states.add(state)    
    # Calculate value function and policy from Q_legibility
    for state in all_states:
        # Check if terminal state
        mdp_legibility = GridWorldMDP_RobotCleaner(grid, start, 'legibility')
        if mdp_legibility.is_terminal(state):
            policy_legibility[state] = None
            V_legibility[state] = 0
        else:
            # Find best action and value for this state
            best_q = float('-inf')
            best_action = None
            for action in GridWorldMDP_RobotCleaner.actions:
                if (state, action) in Q_legibility:
                    if Q_legibility[(state, action)] > best_q:
                        best_q = Q_legibility[(state, action)]
                        best_action = action
            policy_legibility[state] = best_action
            V_legibility[state] = best_q if best_q != float('-inf') else 0    
    return V_legibility, policy_legibility, Q_legibility


# 0-empty, 1-wall
grid = [
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
# Chess coordinates for a 7x6 grid:
# a7 b7 c7 d7 e7 f7
# a6 b6 c6 d6 e6 f6
# a5 b5 c5 d5 e5 f5
# a4 b4 c4 d4 e4 f4
# a3 b3 c3 d3 e3 f3
# a2 b2 c2 d2 e2 f2
# a1 b1 c1 d1 e1 f1
print_chess_grid(grid)
print("\nLegend: W=Wall, D=Door, S=Switch, V=Valuable Item, X=Non-enterable cell, C=Clean, T=Toggle Switch, P=Pick Up")
start = 'b1'
door_open = True  # Set initial door state for visualization
valuable_picked = False  # Set initial valuable picked state for visualization
dirty_cell_exists = True  # Set initial dirty cell existence state for visualization
initial_state = (start, door_open, valuable_picked, dirty_cell_exists)
# mdp_type = ['original', 'destroy' -> it will pick up the valuable item, 'close' -> it will close the door, 'legibility' -> it will avoid the door and still clean the dirty cell]

# 1. mdp_original -> the robot will clean the dirty cell
print("\nMDP Type: Original (Clean the dirty cell)")    
mdp_type = 'original'
mdp_original = GridWorldMDP_RobotCleaner(grid, start, mdp_type)
V_original, policy_original, Q_original = mdp_original.value_iteration()
print_value_grid(grid, V_original, door_open, valuable_picked, dirty_cell_exists)
print_policy_grid(grid, policy_original, door_open, valuable_picked, dirty_cell_exists)
print_qvalue_grid(grid, Q_original, door_open, valuable_picked, dirty_cell_exists)
# Generate and print trajectory for original MDP
trajectory_original = generate_trajectory(mdp_original, policy_original, initial_state)
print_trajectory(trajectory_original, "Original MDP")

# 2. mdp_destroy -> the robot will pick up the valuable item
print("\nMDP Type: Destroy (Pick up the valuable item)")
mdp_type = 'destroy'
mdp_destroy = GridWorldMDP_RobotCleaner(grid, start, mdp_type)
V_destroy, policy_destroy, Q_destroy = mdp_destroy.value_iteration()
print_value_grid(grid, V_destroy, door_open, valuable_picked, dirty_cell_exists)
print_policy_grid(grid, policy_destroy, door_open, valuable_picked, dirty_cell_exists)
print_qvalue_grid(grid, Q_destroy, door_open, valuable_picked, dirty_cell_exists)
# Generate and print trajectory for destroy MDP
trajectory_destroy = generate_trajectory(mdp_destroy, policy_destroy, initial_state)
print_trajectory(trajectory_destroy, "Destroy MDP")

# 3. mdp_close -> the robot will close the door
print("\nMDP Type: Close (Close the door)")
mdp_type = 'close'
mdp_close = GridWorldMDP_RobotCleaner(grid, start, mdp_type)
V_close, policy_close, Q_close = mdp_close.value_iteration()
print_value_grid(grid, V_close, door_open, valuable_picked, dirty_cell_exists)
print_policy_grid(grid, policy_close, door_open, valuable_picked, dirty_cell_exists)
print_qvalue_grid(grid, Q_close, door_open, valuable_picked, dirty_cell_exists)
# Generate and print trajectory for close MDP
trajectory_close = generate_trajectory(mdp_close, policy_close, initial_state)
print_trajectory(trajectory_close, "Close MDP")

# 4. mdp_legibility -> the robot will choose path that avoids the door, and still clean the dirty cell
print("\nMDP Type: Legibility (Avoid the door and clean the dirty cell)")
# # 1st Formula
# print("Using Q-values from mdp_destroy to compute legibility rewards.")
# print("The formula is: Reward_legibility = -Q_destroy + Reward_original")
# mdp_type = 'legibility'
# # Pass Q_destroy to legibility MDP
# mdp_legibility = GridWorldMDP_RobotCleaner(grid, start, mdp_type)
# V_legibility, policy_legibility, Q_legibility = mdp_legibility.value_iteration(Q_destroy=Q_destroy)

# 2nd Formula - Calculate Legibility MDP
V_legibility, policy_legibility, Q_legibility = calculate_legibility_mdp(Q_original, Q_destroy, grid, start, weight=-1)

# Print visualization grids for legibility MDP
print_value_grid(grid, V_legibility, door_open, valuable_picked, dirty_cell_exists)
print_policy_grid(grid, policy_legibility, door_open, valuable_picked, dirty_cell_exists)
print_qvalue_grid(grid, Q_legibility, door_open, valuable_picked, dirty_cell_exists)

# # Generate and print trajectory for legibility MDP (note: for 1st Formula)
# trajectory_legibility = generate_trajectory(mdp_legibility, policy_legibility, initial_state, Q_destroy=Q_destroy)
# print_trajectory(trajectory_legibility, "Legibility MDP")