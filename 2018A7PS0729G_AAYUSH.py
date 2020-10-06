#!/usr/bin/env python3
import time
import pickle
import heapq
from itertools import permutations, chain, combinations

# ********************************************************************

#Global Variables
int_max = 1000000000
known_states = dict()
conflict_value_table = None

class State():
    """
    A State object represents a state of the 15-puzzle board.
    The board is a 4x4 grid with tiles Numbered 1-15 and the 
    blank space is represented with a 0 on the board.
    """
    def __init__(self, board):
        self.board = tuple(tuple(i) for i in board)
        self.fval = int_max
        self.gval = int_max
        self.parent = None

    def clone_board(self):
        """
        :return: (list) clone of State.board.
        """
        b = self.board
        return [
            [b[0][0], b[0][1], b[0][2], b[0][3]],
            [b[1][0], b[1][1], b[1][2], b[1][3]],
            [b[2][0], b[2][1], b[2][2], b[2][3]],
            [b[3][0], b[3][1], b[3][2], b[3][3]],
        ]
        # a more pythonic approach: return list(list(i) for i in self.board) 
        # but it has more delay(confirmed with cProfile)

    def neighbors(self):
        """
        :return: (list) of possible States after moving the blank space to a valid position.
        """
        i0, j0 = find_zero(self.board)
        neighbor_list = []

        #possibilities[i]::(condition, next_row, next_col)
        possibilities = ((j0<3, i0, j0+1), (j0>0, i0, j0-1), (i0>0, i0-1, j0), (i0<3, i0+1, j0))

        for condition, i, j in possibilities:
            if condition == True:
                new_board = self.clone_board()
                new_board[i][j], new_board[i0][j0] = new_board[i0][j0], new_board[i][j]
                new_state = make_state(new_board)
                neighbor_list.append(new_state)

        return neighbor_list
 
def make_state(board):
    """
    Generates and returns a new State for the given board 
    arrangement or returns the State if it's already known.

    :param board: (list) the board we want to get State of.
    :return: (State) the State corresponding to board.
    """
    global known_states
    new_board =   tuple(tuple(i) for i in board) 
 
    if new_board in known_states:
        return 	known_states[new_board]
    else:
        new_state = State(board)
        known_states[new_board] = new_state
        return new_state

class PriorityQueue():
    """
    PriorityQueue to save the frontier set.
    Each element of the PriorityQueue consists of (State.fval, State.board)
    """
    def __init__(self):
        """
        Data Structure which contains the smallest element(First element of tuple in this case) at index 0.
        """
        self.queue_length = 0
        self.qheap = []
        heapq.heapify(self.qheap)
 
    def push(self, new_state):
        heapq.heappush(self.qheap,(new_state.fval,new_state.board))
        self.queue_length += 1
 
    def pop(self):
        if self.queue_length < 1:
            return None
        fval, board = heapq.heappop(self.qheap)
        self.queue_length -= 1
        global known_states
        state = known_states[board]
        
        # State.fval might get updated after pushing (State.fval, State.board in the PriorityQueue)
        # Therefore we check and return the state only if the information is consistent.
        if state.fval == fval:
            return state
        else:
            return self.pop()

def get_path(current):
    """
    :param current: (State) the final state who's path is to be found.
    :return: (list) of Actions(Up, Down, Right, Left) which lead to the currentState.
    """
    path = []
    curr_i, curr_j = find_zero(current.board)
    while current.parent != None:
        next_i, next_j = find_zero(current.parent.board)
        if curr_i > next_i:
            path.append('Down')
        elif curr_i < next_i:
            path.append('Up')
        elif curr_j > next_j:
            path.append('Right')
        else:
            path.append('Left')
        current = current.parent
        curr_i, curr_j = next_i, next_j
    return path[::-1]
 
def find_zero(board):
    """
    :param board: (list) board in which 0 has to be found.
    :return: (tuple) of row, col where 0 is present in given board.
    """
    for row in range(4):
        for col in range(4):
            if board[row][col] == 0:
                return (row, col)
  

def build_conflict_values_table():
    """
    Builds a dictionary which contains (Key: String, Value: int)
    The String key represents the goal_pattern or the conflict type.
    The int value is the number of moves to add to the estimate of f(n) for the given goal_pattern.
    """
    global conflict_value_table
    conflict_value_table = dict()

    file = open('2018A7PS0729G_AAYUSH.dat', 'rb')
    conflict_value_table = pickle.load(file)
    file.close()


def get_linear_conflicts(start_list,goal_list): 
    """
    :param start_list: (tuple) representing a row/col of the board at start.
    :param goal_list: (list) representing the same row/col of the board at goal. 
    :return: (int) the number of moves to add to the estimate of the moves to get from start to goal based on the number
    of conflicts on a given row or column
    """

    # Find which of the tiles in start_list have their goals on this row/col
    # build a pattern to use in a lookup table of this form:
    # g0, g1, g2, g3 if they don't have a goal in this row/col put an x in the pattern at their place.

    goal_pattern = ['x', 'x', 'x', 'x']
    # initially all 'x' until we file a tile whose goal is in this line

    #Find the goal pattern
    for g in range(4):
        for s in range(4):
            start_tile_num = start_list[s]
            if start_tile_num == goal_list[g] and start_tile_num != 0:
                goal_pattern[s] = f'g{g}' 
 
    global conflict_value_table
 
    tup_goal_pattern = tuple(goal_pattern)
    
    #Lookup and return the value for goal_pattern that we found.
    if tup_goal_pattern in conflict_value_table:
        return conflict_value_table[tuple(goal_pattern)]
    else:
        return 0
 
class LinearConflict_map(dict):
    """
    LinearConflict_map is a dictionary which returns 0 is the key is not found.
    """
    def __missing__(self, key):
        return 0
        
def list_possible_conflicts(goal_list):
    """
    :param goal_list: (list) representing a row/col of the goal State.
    :return: (LinearConflict_map) which maps the various conflicts which can occur with the given goal_list.
    """
    all_pieces = [i for i in range(16)]    
    non_goal_pieces = []  #Pieces which don't have a final position in the goal_list
 
    for t in all_pieces:
        if t not in goal_list:
            non_goal_pieces.append(t) 
    
    conflicts = LinearConflict_map()

    #num_x: Number of non_goal_pieces.
    for num_x in [0,1,2]:
        for x_pieces in combinations(non_goal_pieces, num_x):
            for g_pieces in combinations(goal_list, 4-num_x):
                for start_list in permutations(chain(g_pieces, x_pieces)):
                    conflictadd = get_linear_conflicts(start_list,goal_list)
                    if conflictadd > 0:
                        conflicts[start_list]=conflictadd   

    return conflicts

class Heuristic(): 
    """
    Heuristic Class is used to find out the heuristic value for a given goal state.
    """

    def __init__(self, goal):
        """
        :param goal: (State) the goal state that is to be achieved.
        """
        self.goal_lists = goal.board

        #precompute conflict values
        build_conflict_values_table()

        #precompute for manhattan distance
        self.goal_map = [i for i in range(16)]
        for row in range(4):
            for col in range(4):
                self.goal_map[goal.board[row][col]] = (row, col)
 
        self.goal_map = tuple(self.goal_map)
  
        #precompute All possible linearConflicts in rows.
        self.row_conflicts = []
        for row in range(4):
            t = goal.board[row]
            conf_dict = list_possible_conflicts([t[0],t[1],t[2],t[3]])
            self.row_conflicts.append(conf_dict)
 
        #precompute All possible linearConflicts in columns.
        self.col_conflicts = []
        for col in range(4):
            col_list =[]
            for row in range(4):
                col_list.append(goal.board[row][col])
            conf_dict = list_possible_conflicts(col_list)
            self.col_conflicts.append(conf_dict)

    def heuristic(self, start):
        """
        This calculates the sum of manhattan distances and also the linear conflicts between the start state and the goal state. Except for the 0 tile.
        :param start: (State) the starting State.
        :return: (int) the heuristic value for the given startState with resepect to the goal state of Heuristic object.
        """
        distance = 0

        t = start.board
        g = self.goal_map
        rc = self.row_conflicts
        cc = self.col_conflicts

        # calculate manhattan distance

        for row in range(4):
            for col in range(4):
                start_tilenum = t[row][col]
                if start_tilenum != 0:
                    (goal_row, goal_col) = g[start_tilenum]
                    distance += abs(row - goal_row) + abs(col - goal_col)


        # add linear conflicts 

        for row in range(4):
            curr_row = t[row]
            distance += rc[row][curr_row]

        for col in range(4):
            col_tuple = (t[0][col], t[1][col], t[2][col], t[3][col])
            distance += cc[col][col_tuple]

        return distance


def a_star(start_board, goal_board):
    """
    Solves the start_board with the A* approch supplemented by the Manhattan Distance and Linear Conflicts heuristic function.
    :param start_board: (list) representing the initialState.
    :param goal_board: (list) representing the goalState.
    :return: (tuple) Path and nodes generated in the process.
    """
    start = make_state(start_board)
    goal = make_state(goal_board)
    nodesGenerated = 0

    h = Heuristic(goal)

    frontier_set = PriorityQueue()

    start.fval = h.heuristic(start)
    start.gval = 0
    frontier_set.push(start)
    nodesGenerated += 1
  
    while frontier_set.queue_length > 0:
        current = frontier_set.pop()
        if current == None: 
            break
 
        if current == goal:
            path = get_path(current)
            return path, nodesGenerated
 
        for neighbor in current.neighbors():
            gval = current.gval + 1
            if gval < neighbor.gval: 
                if neighbor.gval != int_max:
                    neighbor.fval = neighbor.fval - neighbor.gval + gval
                else:
                    neighbor.fval = gval + h.heuristic(neighbor)
                neighbor.parent = current
                neighbor.gval = gval
                frontier_set.push(neighbor) 
                nodesGenerated += 1

def FindMinimumPath(initialState,goalState):
    """
    Finds the minimum Path from initial to goal state.
    :param initialState: (list) representing initialState.
    :param goalState: (list) representing goalState.
    :return: (tuple) minimumPath and nodes generated.
    """
    #We convert the state from hexadecimal(0-F) to decimal(0-9) for conveniency.
    initialState = [list(map(lambda x: int(x, 16), i)) for i in initialState]
    goalState = [list(map(lambda x: int(x, 16), i)) for i in goalState]
    return a_star(initialState, goalState)

#**************   DO NOT CHANGE ANY CODE BELOW THIS LINE *****************************


def ReadInitialState():
    with open("initial_state4.txt", "r") as file: #IMP: If you change the file name, then there will be an error when
                                                        #               evaluators test your program. You will lose 2 marks.
        initialState = [[x for x in line.split()] for i,line in enumerate(file) if i<4]

    return initialState

def ShowState(state,heading=''):
    print(heading)
    for row in state:
        print(*row, sep = " ")

def main():
    initialState = ReadInitialState()
    ShowState(initialState,'Initial state:')
    goalState = [['0','1','2','3'],['4','5','6','7'],['8','9','A','B'],['C','D','E','F']]
    ShowState(goalState,'Goal state:')
    start = time.time()
    minimumPath, nodesGenerated = FindMinimumPath(initialState,goalState)
    timeTaken = time.time() - start
    
    if len(minimumPath)==0:
        minimumPath = ['Up','Right','Down','Down','Left']
        print('Example output:')
    else:
        print('Output:')

    print('   Minimum path cost : {0}'.format(len(minimumPath)))
    print('   Actions in minimum path : {0}'.format(minimumPath))
    print('   Nodes generated : {0}'.format(nodesGenerated))
    print('   Time taken : {0} s'.format(round(timeTaken,4)))

if __name__== '__main__':
    main()
