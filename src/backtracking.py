from copy import deepcopy

import numpy as np


rows = "ABCDEFGHIJKLMNOPQRSTUVWXY"
cols = "123456789abcdefghijklmnopqrstuvwxyz"
values = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s + t for s in A for t in B]

def row_units(nb_elements):
    col_elements = cols[: nb_elements]
    row_elements = rows[: nb_elements]
    
    return [cross(row_element, col_elements) for row_element in row_elements]

def column_units(nb_elements):
    col_elements = cols[: nb_elements]
    row_elements = rows[: nb_elements]
    
    return [cross(row_elements, col_element) for col_element in col_elements]

def square_units(nb_elements):
    col_elements = cols[: nb_elements]
    row_elements = rows[: nb_elements]
    
    side_square = int(np.sqrt(nb_elements))
    
    rows_per_square = [row_elements[i * side_square: (i + 1) * side_square] for i in range(side_square)]
    columns_per_square = [col_elements[i * side_square: (i + 1) * side_square] for i in range(side_square)]
    
    return [cross(row_element, col_element) for row_element in rows_per_square for col_element in columns_per_square]


class Sudoku():
    
    def __init__(self, string: str) -> None:
                
        assert np.sqrt(len(string)) % 1 == 0
        
        self.n = int(np.sqrt(len(string)))
        
        self.row_units = row_units(self.n)
        self.column_units = column_units(self.n)
        self.square_units = square_units(self.n)
        
        self.boxes = cross(rows[: self.n], cols[: self.n])
        
        unitlist = self.row_units + self.column_units + self.square_units
        units = dict((s, [u for u in unitlist if s in u]) for s in self.boxes)
        self.peers = dict((s, set(sum(units[s], [])) - set([s])) for s in self.boxes)
        
        self.keys = cross(rows[: self.n], cols[: self.n])
        
        self.values = dict()
        for i, key in enumerate(self.keys):
            if string[i] != "0":
                self.values[key] = string[i]
            else:
                self.values[key] = values[: self.n]
        
        self.representation = dict()
        for i, key in enumerate(self.keys):
            self.representation[key] = string[i]
        
        self.quiz = deepcopy(self.values)
        
        self._eliminate()
        
        self.box_dict = None
        
    def set_state(self, state):
        self.values = dict()
        for i, key in enumerate(self.keys):
            if state[i] != "0":
                self.values[key] = state[i]
            else:
                self.values[key] = values[: self.n]
        
        self.representation = dict()
        for i, key in enumerate(self.keys):
            self.representation[key] = state[i]
        
        self.quiz = deepcopy(self.values)
        
        self._eliminate()
        
        self.box_dict = None
        
    def _eliminate(self):
        solved_values = [box for box in self.keys if len(self.values[box]) == 1]
        for box in solved_values:
            digit = self.values[box]
            for peer in self.peers[box]:
                self.values[peer] = self.values[peer].replace(digit, '')
                
    def _is_valid(self):
        return all([len(self.values[x]) != 0 for x in self.keys])
    
    def _solve_puzzle(self):
        """
        Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
        If the sudoku is solved, return the sudoku.
        If after an iteration of both functions, the sudoku remains the same, return the sudoku.
        Input: A sudoku in dictionary form.
        Output: The resulting sudoku in dictionary form.
        """
        if self._is_valid():
            if not self.box_dict:
                self.box_dict = {box: len(self.values[box]) for box in self.boxes if len(self.values[box]) > 1}

            if len(self.box_dict) == 0:
                return self.values
            
            sorted_box_dict = sorted(self.box_dict.keys(), key=lambda x: self.box_dict[x])
            
            key = sorted_box_dict[0]
            prev_values = self.values[key]
            del self.box_dict[key]
            
            for value in prev_values:
                invalid = False
                removed_from_peer = []
                removed = {}
                self.values[key] = value
                
                for peer in self.peers[key]:
                    if value in self.values[peer]:
                        removed_from_peer.append(peer)
                        self.values[peer] = self.values[peer].replace(value, "")
                        if peer in self.box_dict.keys():
                            self.box_dict[peer] -= 1
                        else:
                            invalid = True
                            break
                
                if not invalid:
                    solution = self._solve_puzzle()
                    if solution:
                        return solution
                    
                for peer in set(removed_from_peer) - set(removed.keys()):
                    if len(self.values[peer]) > 0:
                        self.box_dict[peer] += 1
                    self.values[peer] += value
                    
                for peer in removed.keys():
                    self.values[peer] = removed[peer]
                    self.box_dict[peer] = len(removed[peer])
            
            self.values[key] = prev_values
            self.box_dict[key] = len(self.values[key])
            
        return False
        
    def solve(self):
        self._eliminate()
        if self._solve_puzzle():
            return self.values
        else:
            print("No solution was found!")
            return None
        
    def _print_dev(self):
        representation = ""
        width = 1 + max(len(self.values[s]) for s in self.keys)
        sqrt = int(np.sqrt(self.n))
        line = '+'.join(['-' * (width * sqrt)] * sqrt)
        add_vline = [cols[i * sqrt - 1] for i in range(1, sqrt)]
        add_hline = [rows[i * sqrt - 1] for i in range(1, sqrt)]
        for r in rows[: self.n]:
            representation += ''.join(self.values[r + c].center(width) + ('|' if c in add_vline else '') for c in cols[: self.n]) + "\n"
            if r in add_hline: representation += line + "\n"
        return representation + "\n"
    
    def __repr__(self):
        if self.is_solution():
            self.representation = self.values
        representation = ""
        sqrt = int(np.sqrt(self.n))
        width = 1
        outer = ['-' * ((width + 1) * sqrt)]
        inner = ['-' * ((width + 1) * sqrt + 1)]
        line = []
        line += outer
        for _ in range(sqrt - 2):
            line += inner
        line += outer
        line = "+".join(line)

        add_vline = [cols[i * sqrt - 1] for i in range(1, sqrt)]
        add_hline = [rows[i * sqrt - 1] for i in range(1, sqrt)]

        for r in rows[: self.n]:
            representation += ' '.join(self.representation[r + c].center(width) + (' |' if c in add_vline else '') for c in cols[: self.n]) + "\n"
            if r in add_hline: representation += line + "\n"
        return representation + "\n"
        
    def _check_per_unit(self, units):
        for unit in units:
            set_of_values = set()
            for key in unit:
                v = len(set_of_values)
                set_of_values.add(self.values[key])
                if v == len(set_of_values):
                    return False
                
            if len(set_of_values) != self.n:
                return False
        return True
        
    def is_solution(self) -> bool:
            
        for key in self.keys:
            if len(self.quiz[key]) == 1 and self.quiz[key] != self.values[key]:
                return False
                
        if not self._check_per_unit(row_units(self.n) + column_units(self.n) + square_units(self.n)):
            return False
        
        return True
    
    def get_state(self) -> str:
        if self.is_solution():
            self.representation = self.values
        s = ""
        for key in self.keys:
            s += self.representation[key]
        return s
    
    def set_solution(self, state):
        self.values = dict()
        for i, key in enumerate(self.keys):
            if state[i] != "0":
                self.values[key] = state[i]
            else:
                self.values[key] = values[: self.n]
        
        self.representation = dict()
        for i, key in enumerate(self.keys):
            self.representation[key] = state[i]
    
