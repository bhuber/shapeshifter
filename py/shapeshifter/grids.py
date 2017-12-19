# -*- coding: utf-8 -*-

# From http://stevenloria.com/lazy-evaluated-properties-in-python/
def lazy(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

class GridSymbol:
    # Change this manually to support different numbers of symbols
    number_of_symbols = 3

    def __new__(cls, value):
        return cls._cache[value]

    class _GridSymbol:
        def __init__(self, value):
            if value >= GridSymbol.number_of_symbols:
                raise Exception("Can only have up to %i symbols" % GridSymbol.number_of_symbols)
            self.value = value

        def next_symbol(self):
            return GridSymbol((self.value + 1) % GridSymbol.number_of_symbols)

        def __add__(self, other):
            return GridSymbol((self.value + other.value) % GridSymbol.number_of_symbols)

        def __sub__(self, other):
            return GridSymbol((self.value - other.value) % GridSymbol.number_of_symbols)

GridSymbol._cache = [GridSymbol._GridSymbol(i) for i in range(GridSymbol.number_of_symbols)]


class GenericGrid:
    def __init__(self, cols, rows=None, grid_state=None):
        rows = rows or cols
        self.rows = rows
        self.cols = cols
        self.state = grid_state or self._default_state()
        assert(rows * cols == len(self.state))

    def get_cell(self, x, y):
        assert(0 <= x < self.cols)
        assert(0 <= y < self.rows)
        return self.state[y*self.cols + x]

    def get_cell_primitive(self, cell):
        return cell

    @lazy
    def _state_str(self):
        return "\n".join([
            ",".join([str(self.get_cell_primitive(self.get_cell(i, j))) for i in range(self.cols)])
            for j in range(self.rows)])

    def __str__(self):
        state = self._state_str
        return "{'rows': %s, 'cols': %s, 'state':\n%s}\n" % (self.rows, self.cols, state)

    __repr__ = __str__

    def __eq__(self, other):
        return (self.rows == other.rows
               and self.cols == other.cols
               and self.state == other.state)

    def __hash__(self):
        return self._hash

    @lazy
    def _hash(self):
        if self._hash is None:
            self._hash = hash((self.rows, self.cols, self.state))
        return self._hash

    def __add__(self, other):
        assert(isinstance(other, self.__class__))
        assert(self.cols == other.cols)
        assert(self.rows == other.rows)
        return Grid(self.cols, self.rows, [i[0] + i[1] for i in zip(self.state, other.state)])

    def __sub__(self, other):
        assert(isinstance(other, self.__class__))
        assert(self.cols == other.cols)
        assert(self.rows == other.rows)
        return Grid(self.cols, self.rows, [i[0] - i[1] for i in zip(self.state, other.state)])

    @lazy
    def sum_of_cells(self):
        return None if len(self.state) == 0 else sum(self.state[1:], self.state[0])


class Grid(GenericGrid):
    def __init__(self, cols, rows=None, grid_state=None):
        super().__init__(cols, rows, grid_state or self._default_state(cols, rows))

    @staticmethod
    def create_from_ints(int_list):
        rows = len(int_list)
        cols = len(int_list[0])
        assert all([len(l) == cols for l in int_list])
        return Grid(cols, rows, [GridSymbol(e) for row in int_list for e in row])

    def apply_shape(self, x, y, shape):
        assert(x + shape.cols <= self.cols)
        assert(y + shape.rows <= self.rows)
        new_state = []
        for j in range(self.rows):
            for i in range(self.cols):
                cell = self.get_cell(i, j)
                new_state.append(cell.next_symbol() if shape.in_shape(i - x, j - y) else cell)

        return Grid(self.cols, self.rows, new_state)

    def apply_shape_everywhere(self, shape):
        for y in range(self.rows - shape.rows + 1):
            for x in range(self.cols - shape.cols + 1):
                yield (x, y, self.apply_shape(x, y, shape))

    def get_cell_primitive(self, cell):
        return cell.value

    @lazy
    def sum_of_cells(self):
        return sum(self.get_cell_primitive(i) for i in self.state)

    def _default_state(self, cols, rows):
        return [GridSymbol(0) for i in range((rows or cols) * cols)]

    def __str__(self):
        return "Game Grid:[\n%s\n]" % self._state_str

    __repr__ = __str__


class Shape(GenericGrid):
    def __init__(self, cols, rows, mask):
        super().__init__(cols, rows, [bool(i) for i in mask])
        self.cell_count = sum(1 if i else 0 for i in self.state)

    def in_shape(self, x, y):
        return (0 <= x < self.cols and 0 <= y < self.rows) and self.state[y*self.cols + x]

    def get_cell_primitive(self, cell):
        return int(cell)

    def __str__(self):
        return "Shape: [\n%s\n]" % self._state_str

    __repr__ = __str__

class ShapePathFinder:
    class ShapeWrapper(Shape):
        def __init__(self, other, original_idx, sorted_idx, cells_remaining):
            super().__init__(other.cols, other.rows, other.state)
            self.original_idx = original_idx
            self.sorted_idx = sorted_idx
            self.cells_remaining = cells_remaining

    def __init__(self, start_grid, shapes, end_grid):
        self.start_grid = start_grid
        self.original_shapes = shapes
        self.end_grid = end_grid
        self._precompute_shape_info()
        self.metrics_count_short_circuit = 0
        self.metrics_mod_short_circuit = 0
        self.metrics_total_invocations = 0

    def _precompute_shape_info(self):
        shapes = self.original_shapes
        n_shapes = len(shapes)

        shapes_sorted_idx = sorted(
            zip(shapes, range(n_shapes)), key=lambda i: i[0].rows * i[0].cols, reverse=True)

        # Shape objects sorted by descending number of cells
        sorted_shapes = [i[0] for i in shapes_sorted_idx]
        self.sorted_shapes = sorted_shapes

        # key: index in self.sorted_shapes
        # value: index in self.original_shapes
        self.sorted_to_original = dict((i, shapes_sorted_idx[i][1]) for i in range(n_shapes))

        # Index: sorted shape index
        # Value: number of cells between current shape and end of list, inclusive
        shapes_count_index = [0] * n_shapes
        total_cells = 0
        for i in range(n_shapes):
            shape_idx = n_shapes - i - 1
            total_cells += sorted_shapes[shape_idx].cell_count
            shapes_count_index[shape_idx] = total_cells

        self.shapes_count_index = shapes_count_index

        for i in range(n_shapes):
            shape = sorted_shapes[i]
            sorted_shapes[i] = ShapePathFinder.ShapeWrapper(
                shape, self.sorted_to_original[i], i, shapes_count_index[i])

    def _find_shape_path_inner(self, start_grid, shapes, end_grid, current_path):
        self.metrics_total_invocations += 1
        if len(shapes) == 0:
            return current_path if start_grid == end_grid else None

        current_shape = shapes[0]
        grid_diff = end_grid - start_grid

        # Short circuit: no solution if grid difference is greater than number of cells remaining
        if grid_diff.sum_of_cells > current_shape.cells_remaining:
            self.metrics_count_short_circuit += 1
            return None

        for next_step in start_grid.apply_shape_everywhere(current_shape):
            result = yield from self._find_shape_path_inner(next_step[2], shapes[1:], end_grid, current_path + [next_step])
            if result is not None:
                yield result
        return None

    def find_shape_path(self):
        return (self._reshuffle_shapes(path)
            for path in
            self._find_shape_path_inner(self.start_grid, self.sorted_shapes, self.end_grid, []))

    def _reshuffle_shapes(self, path):
        n_shapes = len(self.original_shapes)
        new_path = [None] * n_shapes
        for i in range(n_shapes):
            new_path[self.sorted_to_original[i]] = path[i]

        current_grid = self.start_grid
        for i in range(n_shapes):
            path_e = new_path[i]
            x = path_e[0]
            y = path_e[1]
            shape = self.original_shapes[i]
            new_path[i] = (x, y, shape, current_grid.apply_shape(x, y, shape))

        return new_path



"""
Theorem: if a valid shape application exists for a start grid, shape list, and end grid, then the
sum of the cell difference between start and end is equal to the total number of cells in the shape
list, modulo the number of cell symbols.
i.e.:
sum(cell value, end - start) = sum(number of cells in shape for shape in shape_list) mod number_of_symbols

Proof by Induction:
Let ns = number of symbols
Let s = current shape
Let start = start grid
Let end = end grid
Base case: start == end, shapes list is empty.  0 = 0 mod ns
Hypothesis: For some value start != end, non empty shapes list, 
sum(cell value, end - start) = sum(number of cells in s for s in shape_list) mod ns
Inductive step:
Apply an arbitrary shape s to end to start yielding end' .  Our equation is now
sum(cell value, cellCount(s) + end - start) = sum(number of cells in s for s in shape_list) + cellCount(s) mod ns
Subtracting cellCount(s) from both sides yields the original equation, so it's true.

Example where this is useful (in one dimension):
ns = 3
start = [2, 2, 0], end = [0, 0, 0], sl = [Shape([1]), Shape([1, 1]), Shape([1])]
end - start = [1, 1, 0], sum(end - start) % 3 = 2
countCells(sl) = 4 = 1 % 3
Therefore No solution exists.


Theorem: If sum(cell value, end - start) > countCells(shape_list), then no solution exists.

Proof: The minimum number of shape cells needed to get from start to end is
sum(cell value, end - start), if we don't have at least that many we can never get from start to
end.

"""

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
