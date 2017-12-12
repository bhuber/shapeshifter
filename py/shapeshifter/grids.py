# -*- coding: utf-8 -*-

class GridSymbol:
    class _GridSymbol:
        def __init__(self, value):
            if value >= GridSymbol.number_of_symbols:
                raise Exception("Can only have up to %i symbols" % GridSymbol.number_of_symbols)
            self.value = value

        def next_symbol(self):
            return GridSymbol((self.value + 1) % GridSymbol.number_of_symbols)

    # Change this manually to support different numbers of symbols
    number_of_symbols = 3

    def __new__(cls, value):
        return cls._cache[value]

GridSymbol._cache = [GridSymbol._GridSymbol(i) for i in range(GridSymbol.number_of_symbols)]

class Grid:
    def __init__(self, rows, cols=None, grid_state=None):
        if cols == None:
            cols = rows
        self.rows = rows
        self.cols = cols
        self.state = grid_state or self._default_state()
        self._hash = None

    @staticmethod
    def create_from_ints(int_list):
        rows = len(int_list)
        cols = len(int_list[0])
        assert all([len(l) == cols for l in int_list])
        return Grid(rows, cols, [GridSymbol(e) for row in int_list for e in row])

    def _default_state(self):
        return [GridSymbol(0) for i in range(self.rows) for j in range(self.cols)]

    def get_cell(self, x, y):
        assert(0 <= x < self.cols)
        assert(0 <= y < self.rows)
        return self.state[y*self.cols + x]

    def apply_shape(self, x, y, shape):
        assert(x + shape.cols <= self.cols)
        assert(y + shape.rows <= self.rows)
        new_state = []
        for j in range(self.rows):
            for i in range(self.cols):
                cell = self.get_cell(i, j)
                new_state.append(cell.next_symbol() if shape.in_shape(i - x, j - y) else cell)

        return Grid(self.rows, self.cols, new_state)

    def apply_shape_everywhere(self, shape):
        for y in range(self.rows - shape.rows + 1):
            for x in range(self.cols - shape.cols + 1):
                yield (x, y, self.apply_shape(x, y, shape))

    def __str__(self):
        state = "\n".join([
            str([self.get_cell(i, j).value for i in range(self.cols)])
            for j in range(self.rows)])
        return "{'rows': %s, 'cols': %s, 'state':\n%s}" % (self.rows, self.cols, state)

    __repr__ = __str__

    def __eq__(self, other):
        return (self.rows == other.rows
               and self.cols == other.cols
               and self.state == other.state)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.rows, self.cols, self.state))
        return self._hash

class Shape:
    def __init__(self, rows, cols, mask):
        assert(rows * cols == len(mask))
        self.rows = rows
        self.cols = cols
        self.mask = [bool(i) for i in mask]
        self._hash = None

    def in_shape(self, x, y):
        return (0 <= x < self.cols and 0 <= y < self.rows) and self.mask[y*self.cols + x]

    def __eq__(self, other):
        return (self.rows == other.rows
               and self.cols == other.cols
               and self.mask == other.mask)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.rows, self.cols, self.mask))
        return self._hash

def find_shape_path_inner(start_grid, shapes, end_grid, current_path):
   # if start_grid == end_grid:
   #     print(current_path)
    if len(shapes) == 0:
        return current_path if start_grid == end_grid else None
    for next_step in start_grid.apply_shape_everywhere(shapes[0]):
        result = yield from find_shape_path_inner(next_step[2], shapes[1:], end_grid, current_path + [next_step])
        if result is not None:
            yield result
    return None

def find_shape_path(start_grid, shapes, end_grid):
    return find_shape_path_inner(start_grid, shapes, end_grid, [])

_usage = """
# Cup: 0
# Crown: 1
# Sword: 2
start = Grid.create_from_ints([
    [2, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 0, 2]
])
end = Grid.create_from_ints(4 * [4 * [2]])
shape1 = Shape(3, 3, [
    0, 1, 1,
    0, 1, 0,
    1, 1, 0
])
shapes = [
    Shape(3, 3, [
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    ]),
    Shape(1, 2, [1, 1]),
    shape1,
    shape1,
    Shape(3, 3, [
        1, 0, 0,
        1, 1, 0,
        0, 1, 1
    ]),
    Shape(2, 1, [
        1,
        1
    ]),
    Shape(3, 3, [
        1, 1, 0,
        0, 1, 0,
        0, 1, 1
    ]),
    shape1,
    Shape(1, 3, [1, 1, 1]),
    shape1,
    Shape(3, 3, [
        1, 1, 0,
        1, 1, 0,
        0, 1, 1
    ])
]


next(find_shape_path(start, shapes, end))
"""
