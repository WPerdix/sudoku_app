import math
import numpy as np

def print_cover_matrix(h):
    representation = ""
    
    N = 0
    c = h.R
    while c != h:
        c = c.R
        N += 1
    
    if N == 0:
        print(0)
        return representation
    
    columns_done = set()
    
    current_c = h.R
    while current_c not in columns_done:
        line = ""
        c = h.R
        while c != current_c:
            c = c.R
            line += "0"
        line += "1"
        start = line
        r = c.D
        while r != c:
            line = start
            first_r = r
            rr = r
            cc = c
            while rr.R != first_r:
                rr = rr.R
                counter = -1
                while cc != rr.C:
                    cc = cc.R
                    counter += 1
                line += "0" * counter + "1"
                columns_done.add(rr.C)
            
            if len(line) < N:
                line += "0" * (N - len(line))
            
            line += "\n"
            representation += f"{current_c.N} {r.N}\t" + line
            r = r.D
        current_c = current_c.R
        columns_done.add(current_c.L)
            
    print(N, len(representation.split('\n')) - 1)
    return representation

# x
class DataObject():
    def __init__(self, N: str, C):
        self.L = None
        self.R = None
        self.U = None
        self.D = None
        self.C = C
        self.N = N # name
        
# y
class ColumnObject():
    def __init__(self, N: str):
        self.L = None
        self.R = None
        self.U = None
        self.D = None
        self.C = None
        self.S = 0 # size
        self.N = N # name
    
    def add(self, name: str):
        self.S += 1
        r = DataObject(name, self)
        if self.D is None:
            self.U = r
            self.D = r
            r.U = self
            r.D = self
        else:
            self.U.D = r
            r.U = self.U
            r.D = self
            self.U = r
        return r
    
    def add_with_row(self, row: DataObject, name: str=''):
        r = self.add(name)
        if row.R is None:
            row.R = r
            row.L = r
            r.R = row
            r.L = row
        else:
            r.L = row
            r.R = row.R
            row.R.L = r
            row.R = r
        return r

# h
class RootObject():
    L: ColumnObject
    R: ColumnObject
    def __init__(self):
        self.L = None
        self.R = None
    
    def add(self, name: str):
        c = ColumnObject(name)
        if self.R is None:
            self.L = c
            self.R = c
            c.L = self
            c.R = self
        else:
            self.L.R = c
            c.L = self.L
            c.R = self
            self.L = c
        return c
        
def cover(c: ColumnObject):
    c.R.L = c.L
    c.L.R = c.R
    i = c.D
    while i != c:
        j = i.R
        while j != i:
            j.D.U = j.U
            j.U.D = j.D
            
            j.C.S -= 1
            j = j.R
        i = i.D
            
def uncover(c: ColumnObject):
    i = c.U
    while i != c:
        j = i.L
        while j != i:
            j.C.S += 1
            j.D.U = j
            j.U.D = j
            
            j = j.L
        i = i.U
    c.R.L = c
    c.L.R = c

def map_to_solution(solution: np.ndarray, name: str):
    row, col, entry = name.split(",")
    solution[int(row) * int(math.sqrt(len(solution))) + int(col)] = entry
        
def search(h: RootObject, solution: np.ndarray, k: int=0):
    if h.R == h:
        yield solution
        return 
    
    # Choose a column object c
    j: ColumnObject = h.R
    s = math.inf
    while j != h:
        if j.S < s:
            c = j
            s = j.S
        j = j.R
            
    # Cover column c
    cover(c)
    
    r = c.D
    while r != c:
        O = r
        map_to_solution(solution, O.N)
        j = r.R
        while j != r:
            # cover column j
            cover(j.C)
            
            j = j.R
        
        yield from search(h, solution, k + 1)
        
        r = O
        c = r.C
        
        j = r.L
        while j != r:
            # uncover column j
            uncover(j.C)
    
            j = j.L
            
        r = r.D
            
    # uncover column c
    uncover(c)
    
class Sudoku():
    def __init__(self, state):
        if math.sqrt(math.sqrt(len(state))) % 1 != 0 or (type(state) != str and type(state) != list):
            raise Exception('The input should be a string and the amount of inputs should be an integer after taking its quartic root.')
                
        self.characters = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.set_state(state)
        
    def set_state(self, state):
        self.state = state
        
        n = int(math.sqrt(len(state)))
        
        characters = self.characters[: n]
        
        # Construct cover matrix
        self.cover_matrix = RootObject()
        
        # Add cell constraints
        for i in range(n):
            for j in range(n):
                start_column_row = self.cover_matrix.add(f'cell r{i}c{j}')
        
        # Add row constraints
        for i in range(n):
            for j in range(n):
                start_column_column = self.cover_matrix.add(f'row r{i}c{j}')
                
        start_column_row = start_column_row.R
        
        # Add column constraints
        for i in range(n):
            for j in range(n):
                start_column_box = self.cover_matrix.add(f'col r{i}c{j}')
                
        start_column_column = start_column_column.R
        
        # Add box constraints
        for i in range(n):
            for j in range(n):
                self.cover_matrix.add(f'box r{i}c{j}')
                
        start_column_box = start_column_box.R
        start_column_box_nested = start_column_box
        
        column_cell: ColumnObject = self.cover_matrix.R
        column_box: ColumnObject = start_column_box 
        
        # Add rows cover matrix
        for i in range(n):
            column_column: ColumnObject = start_column_column
            
            if i % int(math.sqrt(n)) == 0 and i != 0:
                start_column_box = column_box
                
            start_column_box_nested = start_column_box
            
            for j in range(n):
                column_row: ColumnObject = start_column_row
                
                if j % int(math.sqrt(n)) == 0 and j != 0:
                    start_column_box_nested = column_box
                    
                column_box = start_column_box_nested
                
                for k in range(n):
                    name = f'{i},{j},{characters[k]}'
                    # Add cell constraint
                    r = column_cell.add(name)#characters[k])
                    
                    # Add row constraint
                    r = column_row.add_with_row(r, name)
                    
                    # Add column constraint
                    r = column_column.add_with_row(r, name)
                    
                    # Add box constraint
                    column_box.add_with_row(r, name)
                    
                    # Change columns for constraints
                    column_row = column_row.R
                        
                    column_column = column_column.R
                    
                    column_box = column_box.R
                    
                column_cell = column_cell.R
            
            start_column_row = column_row
        
        # cover given inputs
        column_cell: ColumnObject = self.cover_matrix.R
        
        try:
            self.valid = True
            for i in range(n):
                for j in range(n):
                    if self.state[i * n + j] != "0":
                        # iterate over rows in column until we find row corresponding to the entry
                        r = column_cell.D
                        for k in range(n):
                            if self.state[i * n + j] == r.N.split(',')[-1]:
                                break
                            r = r.D
                        else:
                            self.valid = False
                            
                        # cover all columns associated with this row
                        cover(r.C)
                        
                        rr = r.R
                        while rr != r:
                            # cover column j
                            cover(rr.C)
                            
                            rr = rr.R

                    column_cell = column_cell.R
        except:
            self.valid = False
        
    def solve(self):
        if not self.valid:
            return None
        solution = search(self.cover_matrix, np.array([s for s in self.state]))
        yield from solution
    
    def __repr__(self):
        representation = ""
        n = int(math.sqrt(len(self.state)))
        sqrt = int(math.sqrt(n))
        width = 1
        outer = ['-' * ((width + 1) * sqrt)]
        inner = ['-' * ((width + 1) * sqrt + 1)]
        line = []
        line += outer
        for _ in range(sqrt - 2):
            line += inner
        line += outer
        line = "+".join(line)

        add_vline = [i * sqrt - 1 for i in range(1, sqrt)]
        add_hline = [i * sqrt - 1 for i in range(1, sqrt)]

        for r in range(n):
            representation += ' '.join(self.state[c + n * r].center(width) + (' |' if c in add_vline else '') for c in range(n)) + "\n"
            if r in add_hline: representation += line + "\n"
        return representation + "\n"
    
    def print_state(self, state):
        representation = ""
        n = int(math.sqrt(len(state)))
        sqrt = int(math.sqrt(n))
        width = 1
        outer = ['-' * ((width + 1) * sqrt)]
        inner = ['-' * ((width + 1) * sqrt + 1)]
        line = []
        line += outer
        for _ in range(sqrt - 2):
            line += inner
        line += outer
        line = "+".join(line)

        add_vline = [i * sqrt - 1 for i in range(1, sqrt)]
        add_hline = [i * sqrt - 1 for i in range(1, sqrt)]

        for r in range(n):
            representation += ' '.join(state[c + n * r].center(width) + (' |' if c in add_vline else '') for c in range(n)) + "\n"
            if r in add_hline: representation += line + "\n"
        return representation + "\n"
    
    def _print_cover_matrix(self):
        representation = ""
        
        for i in range(len(self.state)):
            line = ""
            c = self.cover_matrix.R
            for _ in range(i):
                c = c.R
                line += "0"
            line += "1"
            start = line
            r = c.D
            while r != c:
                line = start
                first_r = r
                rr = r
                cc = c
                while rr.R != first_r:
                    rr = rr.R
                    counter = -1
                    while cc != rr.C:
                        cc = cc.R
                        counter += 1
                    line += "0" * counter + "1"
                
                if len(line) < 4 * len(self.state):
                    line += "0" * (4 * len(self.state) - len(line))
                
                for i in range(1, 4):
                    line = f"{line[: i * len(self.state) + i - 1]} {line[i * len(self.state) + i - 1:]}"
                
                line += "\n"
                representation += line
                r = r.D
                
        return representation
    
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
        
    def is_solution(self, state) -> bool:
            
        for i, s in enumerate(self.state):
            if s != "0":
                if s != state[i]:
                    return False
        return True
    
    