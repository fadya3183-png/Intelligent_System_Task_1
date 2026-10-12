"""
Warehouse Delivery Robot - Search Algorithms
Uninformed: Uniform-Cost Search (UCS)
Informed:   A* with Manhattan distance

HOW TO RUN
    python warehouse_robot.py            -> runs both algorithms on MAP_A, MAP_B, MAP_C
                                            and prints a summary table.

HOW TO CHANGE THE MAP
    Change the GRID variable below (e.g. GRID = MAP_B) and set RUN_ALL = False
    to run only that map. You can also paste your own map as a list of strings.
    Symbols: S start, G goal, . normal floor (cost 1), ~ rough floor (cost 3), # wall.
"""
import time
import heapq

# ----------------------------------------------------------------------
# 1. Maps (copied from the task PDF)
# ----------------------------------------------------------------------
MAP_A = [
    "S..#....",
    "##.#.##.",
    "....~~..",
    ".####.#.",
    "...~..#G",
]

MAP_B = [
    "S~~~~~~~G",
    ".#######.",
    ".........",
    ".#.#.#.#.",
    ".........",
]

MAP_C = [
    "S..#....",
    ".#.#.##.",
    "...#..#.",
    ".###.##.",
    "....#..G",
]

GRID = MAP_A      # map used when RUN_ALL is False
RUN_ALL = True    # True: run all three maps and print the table


# ----------------------------------------------------------------------
# 2. Helper functions
# ----------------------------------------------------------------------
def find_start_goal(grid):
    """Scan the grid to find S and G as (row, col)."""
    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 'S':
                start = (r, c)
            elif cell == 'G':
                goal = (r, c)
    return start, goal


def step_cost(cell):
    """Cost of ENTERING a cell. S is never entered, G costs 1."""
    return 3 if cell == '~' else 1


def neighbors(pos, grid):
    """Valid neighbours (Up, Down, Left, Right) that are inside the grid and not walls."""
    r, c = pos
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] != '#':
            result.append((nr, nc))
    return result


def manhattan(a, b):
    """Heuristic for A*: admissible and consistent (min step cost is 1)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def print_path_on_map(grid, path):
    """Print the map with path cells marked by '*'."""
    g = [list(row) for row in grid]
    for r, c in path:
        if g[r][c] not in ('S', 'G'):
            g[r][c] = '*'
    for row in g:
        print("".join(row))


# ----------------------------------------------------------------------
# 3. Search algorithms
# Both return the same dict: path, cost, expanded, max_frontier, time
# ----------------------------------------------------------------------
def search(grid, use_heuristic):
    """
    Best-first search on f = g + h.
      use_heuristic = False -> h = 0  -> Uniform-Cost Search (UCS)
      use_heuristic = True  -> h = Manhattan distance -> A*
    """
    t0 = time.perf_counter()
    start, goal = find_start_goal(grid)

    def h(p):
        return manhattan(p, goal) if use_heuristic else 0

    counter = 0                                   # tie-breaker for the heap
    frontier = [(h(start), counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded = 0
    max_frontier = 1
    found = False

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, _, current = heapq.heappop(frontier)
        if current in closed:                     # stale heap entry
            continue
        closed.add(current)
        expanded += 1
        if current == goal:
            found = True
            break
        for nxt in neighbors(current, grid):
            new_g = g_score[current] + step_cost(grid[nxt[0]][nxt[1]])
            if nxt not in g_score or new_g < g_score[nxt]:
                g_score[nxt] = new_g
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (new_g + h(nxt), counter, nxt))
                # Update maximum frontier size after adding nodes
    max_frontier = max(max_frontier, len(frontier))

    elapsed = time.perf_counter() - t0
    if not found:
        return {"path": None, "cost": None, "expanded": expanded,
                "max_frontier": max_frontier, "time": elapsed}
    return {"path": reconstruct_path(came_from, goal), "cost": g_score[goal],
            "expanded": expanded, "max_frontier": max_frontier, "time": elapsed}


def ucs(grid):
    return search(grid, use_heuristic=False)


def a_star(grid):
    return search(grid, use_heuristic=True)


# ----------------------------------------------------------------------
# 4. Run experiments
# ----------------------------------------------------------------------
def run(map_name, grid, rows):
    print("=" * 50)
    print(f"Testing on {map_name}")
    print("=" * 50)
    for algo_name, algo in [("UCS (Uninformed)", ucs), ("A* (Informed)", a_star)]:
        res = algo(grid)
        print(f"\n--- Algorithm: {algo_name} ---")
        if res["path"] is None:
            print("No path found")
            print(f"Nodes Expanded: {res['expanded']}")
            length = cost = "-"
        else:
            print(f"Path: {res['path']}")
            print(f"Total Cost: {res['cost']}")
            print(f"Nodes Expanded: {res['expanded']}")
            print("Path on map:")
            print_path_on_map(grid, res["path"])
            length = len(res["path"]) - 1          # number of moves
            cost = res["cost"]
        rows.append((map_name, algo_name.split()[0], length, cost,
                     res["expanded"], res["max_frontier"], res["time"]))
    print()


def print_table(rows):
    print("=" * 78)
    print(f"{'Map':<6}{'Algo':<6}{'PathLen':>8}{'Cost':>6}{'Expanded':>10}"
          f"{'MaxFrontier':>13}{'Time (ms)':>12}")
    print("-" * 78)
    for m, a, length, cost, exp, mf, t in rows:
        print(f"{m:<6}{a:<6}{str(length):>8}{str(cost):>6}{exp:>10}{mf:>13}{t * 1000:>12.4f}")
    print("=" * 78)


if __name__ == "__main__":
    rows = []
    if RUN_ALL:
        for name, g in [("MAP_A", MAP_A), ("MAP_B", MAP_B), ("MAP_C", MAP_C)]:
            run(name, g, rows)
    else:
        run("GRID", GRID, rows)
    print_table(rows)