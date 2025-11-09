def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False

    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row, n, solutions):
    if row == n:
        solutions.append([row[:] for row in board])
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens_util(board, row + 1, n, solutions)
            board[row][col] = 0  # Backtrack

def solve_n_queens(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    solutions = []
    
    solve_n_queens_util(board, 0, n, solutions)
    
    if not solutions:
        print("No solution exists.")
    else:

        seen_boards = set()
        unique_solutions = []

        for sol in solutions:
            board_tuple = tuple(tuple(row) for row in sol)
            if board_tuple not in seen_boards:
                seen_boards.add(board_tuple)
                unique_solutions.append(sol)
        
       
        print("\nUnique Solutions:")
        for sol in unique_solutions:
            for row in sol:
                print(row)
            print()
    print(f"Total overall solutions: {len(solutions)}")
    print(f"Total unique solutions: {len(unique_solutions)}")

n = int(input("Enter the number of queens (n): "))
solve_n_queens(n)
