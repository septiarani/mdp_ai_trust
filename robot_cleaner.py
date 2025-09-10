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

    def __init__(self, grid, start, dirty_cells, mdp_type, gamma=0.9):
        """
        Initialize the GridWorldMDP_RobotCleaner.
        Args:
            grid (list of list of int): 2D grid, 0 for empty cell, 1 for wall.
            start (str): Starting position in chess notation (e.g., 'a1').
            dirty_cells (list of str): List of positions of dirty cells in chess notation.
            gamma (float): Discount factor for future rewards.
        """
        self.grid = grid  # 2D list: 0-empty, 1-wall
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.dirty_cells = set(dirty_cells)
        self.mdp_type = mdp_type
        self.gamma = gamma
        self.actions = ['up', 'down', 'left', 'right', 'clean', 'toggle_switch', 'pick_up']
        # Generate all possible states using chess notation, door state, and valuable item state
        self.states = []
        for r in range(self.rows):
            for c in range(self.cols):
                coord = self.idx_to_chess(r, c, self.rows)
                if self.grid[r][c] == 0 and coord not in self.NON_ENTERABLE_COORDS:
                    for door_open in [True, False]:
                        for valuable_picked in [True, False]:
                            self.states.append((coord, tuple(sorted(self.dirty_cells)), door_open, valuable_picked))

    def can_pick_valuable(self, coord):
        # Can pick up valuable if on b6
        return coord == 'b6'

    def can_clean_dirt(self, coord, dirty_cells):
        # Can clean dirt in f7 if on f6
        return 'f7' in dirty_cells and coord == 'f6'
    
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
        Check if the state is terminal (all dirty cells cleaned).
        Args:
            state (tuple): (coord, dirty_cells, door_open, valuable_picked)
        Returns:
            bool: True if no dirty cells remain, else False.
        """
        _, dirty, _, _ = state
        return len(dirty) == 0

    def get_next_state(self, state, action):
        """
        Compute the next state given current state and action.
        Args:
            state (tuple): Current state (coord, dirty_cells, door_open, valuable_picked)
            action (str): Action to take ('up', 'down', 'left', 'right', 'clean', 'toggle_switch', 'pick_up')
        Returns:
            tuple: Next state (coord, dirty_cells, door_open, valuable_picked)
        """
        coord, dirty, door_open, valuable_picked = state
        dirty = set(dirty)
        r, c = self.chess_to_idx(coord, self.rows)
        next_door_open = door_open
        next_valuable_picked = valuable_picked
        # Clean dirt in f7 from f6
        if action == 'clean' and self.can_clean_dirt(coord, dirty):
            dirty.remove('f7')
        # Pick up valuable item in b7 from b6
        elif action == 'pick_up' and self.can_pick_valuable(coord) and not valuable_picked:
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
        return (next_coord, tuple(sorted(dirty)), next_door_open, next_valuable_picked)

    def get_reward(self, state, action, next_state):
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
            if action == 'clean' and self.can_clean_dirt(state[0], state[1]):
                return 10  # reward for cleaning dirt
            if action == 'pick_up' and self.can_pick_valuable(state[0]) and not state[3]:
                return -20  # penalty for picking up valuable item
            if action == 'toggle_switch' and self.can_close_door(state[0], state[2]):
                return -1  # small penalty for toggling
            return -1  # penalty for movement
        elif self.mdp_type == 'destroy':
            if action == 'pick_up' and self.can_pick_valuable(state[0]) and not state[3]:
                return 20  # reward for picking up valuable item
            if action == 'clean' and self.can_clean_dirt(state[0], state[1]):
                return -10  # penalty for cleaning dirt
            if action == 'toggle_switch' and self.can_close_door(state[0], state[2]):
                return -1
            return -1
        elif self.mdp_type == 'close':
            if action == 'toggle_switch' and self.can_close_door(state[0], state[2]):
                return 20  # reward for toggling switch (closing door)
            if action == 'clean' and self.can_clean_dirt(state[0], state[1]):
                return -10
            if action == 'pick_up' and self.can_pick_valuable(state[0]) and not state[3]:
                return -10
            return -1
        else:
            return -1

    def value_iteration(self, theta=1e-4):
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
                    for dirty in [tuple(sorted(self.dirty_cells))]:
                        for door_open in [True, False]:
                            for valuable_picked in [True, False]:
                                V[(coord, dirty, door_open, valuable_picked)] = 0
        while True:
            delta = 0
            for state in V:
                if self.is_terminal(state):
                    continue
                v = V[state]
                values = []
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    reward = self.get_reward(state, action, next_state)
                    values.append(reward + self.gamma * V.get(next_state, 0))
                V[state] = max(values)
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
                reward = self.get_reward(state, action, next_state)
                values.append((reward + self.gamma * V.get(next_state, 0), action))
            policy[state] = max(values)[1]
        return V, policy


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
def print_value_grid(grid, V, dirty_cells, door_open):
    rows = len(grid)
    cols = len(grid[0])
    print(f"Value Function Grid:")
    dirty_tuple = tuple(sorted(dirty_cells))
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:    W]")
            elif coord == GridWorldMDP_RobotCleaner.DOOR_COORD:
                row_str.append(f"[{coord}:    D]")
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:    S]")
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:    V]")
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:    X]")
            else:
                value = V.get((coord, dirty_tuple, door_open, False), None)
                if value is not None:
                    row_str.append(f"[{coord}:{value:5.1f}]")
                else:
                    row_str.append(f"[{coord}:·]")
        print(' '.join(row_str))

# Visualization function for policy mapping
def print_policy_grid(grid, policy, dirty_cells, door_open):
    rows = len(grid)
    cols = len(grid[0])
    print(f"Policy Grid Mapping:")
    dirty_tuple = tuple(sorted(dirty_cells))
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
    for r in range(rows):
        row_str = []
        for c in range(cols):
            coord = GridWorldMDP_RobotCleaner.idx_to_chess(r, c, rows)
            cell = grid[r][c]
            if cell == 1:
                row_str.append(f"[{coord}:W]")
            elif coord == GridWorldMDP_RobotCleaner.DOOR_COORD:
                row_str.append(f"[{coord}:D]")
            elif coord == GridWorldMDP_RobotCleaner.SWITCH_COORD:
                row_str.append(f"[{coord}:S]")
            elif coord == GridWorldMDP_RobotCleaner.VALUABLE_COORD:
                row_str.append(f"[{coord}:V]")
            elif coord in GridWorldMDP_RobotCleaner.NON_ENTERABLE_COORDS:
                row_str.append(f"[{coord}:X]")
            else:
                action = policy.get((coord, dirty_tuple, door_open, False), None)
                symbol = action_symbols.get(action, '?')
                row_str.append(f"[{coord}:{symbol}]")
        print(' '.join(row_str))


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
dirty_cells = ['f7']  # Only one dirty cell at top-right
# mdp_type = ['original', 'destroy' -> it will pick up the valuable item, 'close' -> it will close the door]

# 1. mdp_original -> the robot will clean the dirty cell
print("\nMDP Type: Original (Clean the dirty cell)")    
mdp_type = 'original'
mdp_original = GridWorldMDP_RobotCleaner(grid, start, dirty_cells, mdp_type)
V_original, policy_original = mdp_original.value_iteration()
print_value_grid(grid, V_original, dirty_cells, door_open=True)
print_policy_grid(grid, policy_original, dirty_cells, door_open=True)
# print("Value Function:", V)
# print("Policy:", policy)

# 2. mdp_destroy -> the robot will pick up the valuable item
print("\nMDP Type: Destroy (Pick up the valuable item)")
mdp_type = 'destroy'
mdp_destroy = GridWorldMDP_RobotCleaner(grid, start, dirty_cells, mdp_type)
V_destroy, policy_destroy = mdp_destroy.value_iteration()
print_value_grid(grid, V_destroy, dirty_cells, door_open=True)
print_policy_grid(grid, policy_destroy, dirty_cells, door_open=True)

# 3. mdp_close -> the robot will close the door
print("\nMDP Type: Close (Close the door)")
mdp_type = 'close'
mdp_close = GridWorldMDP_RobotCleaner(grid, start, dirty_cells, mdp_type)
V_close, policy_close = mdp_close.value_iteration()
print_value_grid(grid, V_close, dirty_cells, door_open=True)
print_policy_grid(grid, policy_close, dirty_cells, door_open=True)